"""A1Z adapter, imported only after explicit execution; no dependency on the GUI."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
from .version import ROOT
from .wire import split,vector


def official_client():
    path=ROOT/'a1z_g05_deploy/client'
    sys.path.insert(0,str(path))
    spec=importlib.util.spec_from_file_location('g05_official_hardware',path/'dual_arm_client.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def rgb_device(camera):
    if camera.type=='opencv':
        value=camera.index_or_path
        if not value:
            raise ValueError('Camera device is not selected')
        return int(value) if value.isdecimal() else str(Path(value).resolve(strict=True))
    import subprocess
    import pyrealsense2 as rs
    usb=None
    for device in rs.context().query_devices():
        if device.get_info(rs.camera_info.serial_number)==camera.serial:
            port=Path(device.get_info(rs.camera_info.physical_port))
            if not port.is_absolute() or port.parent.name!='video4linux':
                raise ValueError('Unsupported RealSense USB identity')
            usb=port.parent.parent.parent.resolve()
            break
    if usb is None:
        raise ValueError(f'RealSense not found: {camera.serial}')
    matches=[]
    for node in Path('/sys/class/video4linux').glob('video*'):
        if (node/'device').resolve().parent==usb:
            path='/dev/'+node.name
            formats=subprocess.check_output(['v4l2-ctl','-d',path,'--list-formats-ext'],text=True,timeout=5)
            if "'YUYV'" in formats:
                matches.append(path)
    if len(matches)!=1:
        raise ValueError(f'Expected one RGB node for {camera.serial}: {matches}')
    return matches[0]


class A1ZHardware:
    def __init__(self,cfg):
        self.cfg=cfg;self.robot=None

    def open(self):
        if not self.cfg.safety_confirmed or self.cfg.hardware is None:
            raise ValueError('Explicit execution confirmation and hardware configuration required')
        h=self.cfg.hardware
        settings={**self.cfg.model.model_dump(mode='json'),'left_can':h.left_can,'right_can':h.right_can,
                  'image_size':[640,480],'control_frequency':30}
        fields={'top_rgb':'camera_head','left_wrist_rgb':'camera_left_wrist','right_wrist_rgb':'camera_right_wrist'}
        for name,field in fields.items():
            settings[field]=rgb_device(h.cameras[name])
        identities=[str(Path(f'/dev/video{x}' if isinstance(x,int) else x).resolve())
                    for x in (settings[f] for f in fields.values())]
        if len(set(identities))!=3:
            raise ValueError('Three distinct RGB cameras required')
        official=official_client();official.validate_smoothing_config(settings)
        self.robot=official.Hardware(settings,execute=True)
        self.robot.open()

    def observe(self):
        return {'images':self.robot.images(),'state':self.robot.state(),'task':self.cfg.task,
                'frequency':self.cfg.fps,'embodiment_type':'galaxea_a1z'}

    def send(self,target):
        self.robot.send(split(target))

    def feedback(self):
        return vector(self.robot.state()).tolist()

    def hold(self):
        if self.robot:
            self.robot.hold()

    def close(self):
        if self.robot:
            self.robot.close()
