"""Versioned CLI/GUI contract. No dependency on the GUI or its Python environment."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)


class Smoothing(StrictModel):
    enabled: StrictBool = True
    frequency: float = Field(default=250, gt=0, le=250)
    max_velocity: float = Field(default=4, gt=0, le=7)
    max_acceleration: float = Field(default=20, gt=0, le=20)
    response_time: float = Field(default=.04, gt=0)
    stale_timeout: float = Field(default=1, gt=.6)


class ModelOptions(StrictModel):
    action_steps: int = Field(default=32, ge=1, le=32, strict=True)
    action_head: Literal['checkpoint', 'fm', 'ar'] = 'fm'
    fm_steps: int = Field(default=10, ge=1, le=50, strict=True)
    weights_bf16: StrictBool = True
    attention: Literal['sdpa', 'eager'] = 'sdpa'
    request_timeout: float = Field(default=60, gt=0, le=300)
    max_cycles: int = Field(default=0, ge=0, strict=True)
    max_initial_joint_step: float = Field(default=.1, gt=0, le=.5)
    max_joint_step: float = Field(default=.1, gt=0, le=.5)
    enforce_joint_step: StrictBool = False
    gripper_boundary_tolerance: float = Field(default=.02, ge=0, le=.1)
    smoothing: Smoothing = Field(default_factory=Smoothing)

    @model_validator(mode='after')
    def bounded(self):
        if not self.enforce_joint_step and not self.smoothing.enabled:
            raise ValueError('Disabling consecutive checks requires bounded smoothing')
        return self

    def overrides(self):
        values = [f'model.model_weights_to_bf16={str(self.weights_bf16).lower()}',
                  f'model.model_arch.attn_implementation={self.attention}',
                  f'model.model_arch.fm.num_inference_steps={self.fm_steps}']
        if self.action_head != 'checkpoint':
            fm = self.action_head == 'fm'
            values += [f'model.model_arch.discrete_action={str(not fm).lower()}',
                       f'model.model_arch.continuous_action={str(fm).lower()}',
                       f'model.model_arch.return_continuous_action={str(fm).lower()}']
        return values


class Execution(StrictModel):
    lead_steps: int = Field(default=12, ge=2, le=24, strict=True)
    transition_steps: int = Field(default=4, ge=2, le=12, strict=True)
    latency_margin_s: float = Field(default=.05, ge=0, le=.3)
    max_control_lateness_s: float = Field(default=.1, gt=0, le=.25)
    rtc_guidance_weight: float = Field(default=1, gt=0, le=10)
    rtc_soft_steps: int = Field(default=4, ge=0, le=16, strict=True)


class Camera(StrictModel):
    type: Literal['opencv', 'realsense'] = 'opencv'
    serial: str = ''
    index_or_path: str = ''
    width: int = 640
    height: int = 480
    fps: int = 30


class Hardware(StrictModel):
    left_can: str
    right_can: str
    cameras: dict[str, Camera]

    @model_validator(mode='after')
    def layout(self):
        if not self.left_can or not self.right_can or self.left_can == self.right_can:
            raise ValueError('Select two distinct CAN interfaces')
        if set(self.cameras) != {'top_rgb','left_wrist_rgb','right_wrist_rgb'}:
            raise ValueError('Three trained RGB views are required')
        if any((c.width,c.height,c.fps)!=(640,480,30) for c in self.cameras.values()):
            raise ValueError('Camera specification differs from trained 640x480@30 RGB')
        return self


def default_state_root():
    return Path(os.environ.get('XDG_STATE_HOME', str(Path.home()/'.local/state')))/'a1z-g05'


class RunConfig(StrictModel):
    schema_version: Literal[1] = 1
    engine: Literal['baseline','sync-fast','async','rtc'] = 'baseline'
    bundle: Path
    model_python: Path
    model_source: Path | None = None
    output_root: Path = Field(default_factory=default_state_root)
    task: str = Field(min_length=1)
    duration: float = Field(default=600, gt=0, le=3600)
    fps: Literal[30] = 30
    seed: int = Field(default=7, ge=0, le=2**31-1)
    safety_confirmed: StrictBool = False
    model: ModelOptions = Field(default_factory=ModelOptions)
    execution: Execution = Field(default_factory=Execution)
    hardware: Hardware | None = None

    @model_validator(mode='after')
    def compatibility(self):
        if self.engine in ('rtc','async'):
            if self.model.action_head != 'fm' or self.model.action_steps != 32:
                raise ValueError('Async/RTC experiments require FM-only and full 32-step predictions')
            if not self.model.smoothing.enabled:
                raise ValueError('Async/RTC require bounded smoothing')
        return self

    def resolve(self, base: Path):
        for name in ('bundle','model_python','model_source','output_root'):
            value = getattr(self,name)
            if value is not None:
                # Do not resolve a venv Python symlink into its base interpreter.
                setattr(self,name, value.absolute() if value.is_absolute() else (base/value).absolute())
        return self
