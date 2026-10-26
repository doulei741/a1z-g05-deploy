# Standalone G0.5 runtime: durable project memory

Created 2026-10-27 by explicit user request. Canonical organization/resume instructions:
https://github.com/doulei741/A1Z_Lerobot/blob/main/docs/G05_PROJECT_MEMORY.md

Repositories: GUI/product `doulei741/A1Z_Lerobot`; this standalone runtime Fork
`doulei741/a1z-g05-deploy`; model Fork/submodule `doulei741/GalaxeaVLA`.
Upstream vendor repositories are read-only development references. GUI and CLI run
the same runtime; model and hardware environments remain separate.

Original local checkouts were frozen without modification into
`/home/dj/DJ/g05-experiments/management/source_snapshot_20261027` (GUI, runtime,
model, unrelated data-collection patches). Baseline model commit is
`5ea8c7fc9d46486b6c0e943c075b64a56eb2a77f`; baseline runtime tag is
`g05-fm-only-baseline-20261027`. Collection application patches are archived only;
the inference runtime imports the installed A1Z SDK directly and does not depend on
the modified collection application.

Work proceeds through common `feat/runtime-base`, then independent `exp/rtc`,
`exp/async`, `exp/sync-fast`. Training-time RTC is explicitly excluded.
Current algorithm availability is declared by runtime.json; never equate a branch
name with implementation/testing or silently fall back to baseline.

Contract: dual A1Z, 14D grouped absolute arm radians and normalized gripper openings,
three RGB 640x480@30 views. Model inference does relative restoration once.
RTC prefix is absolute at the transport boundary, reprocessed against the new
observation through the original trained processor before FM guidance.
Model parameters are not trained/updated by inference-time latent gradients.

Standalone commands are `python -m g05_runtime.cli --capabilities`, `--config FILE
--check`, `--offline` (recorded observations and real GPU), `--stub` (CPU synthetic),
or `--execute` (requires explicit human confirmation and configured hardware).
Capabilities/checks never open hardware. Runtime writes independent per-run folders;
shared user-scoped hardware/GPU leases protect CLI and GUI experiment instances.

Real test baseline: 30000 FM10, 32-step window, BF16/SDPA, request median164ms/P99
203ms, inside-chunk target interval33.6ms but boundary198ms, first two targets0.31ms.
250Hz bounded smoother was already enabled. Offline jitter/continuity and timing
must be tested; hardware outcomes remain pending user validation.

Do not activate old cloud-training automations, resume training, replace 30000/37000/
50000 weights, or start cameras/CAN/motors from development tasks. Preserve first
target checks, finite-value/layout checks, bounded smoothing, SDK fault handling,
and stopping both arms. Never increase motion limits to conceal inference pauses.

## 当前实验分支 rtc

方案①：异步执行与FM采样期RTC；完整VJP、已承诺前缀投影、软后缀引导。模型依赖固定RTC分支提交。

实现与验证入口见 docs/experiment.md，方案测试状态由主产品experiments/g05/status.json追踪。
