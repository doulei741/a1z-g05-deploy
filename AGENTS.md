# G0.5 runtime development and experiment management

Read `PROJECT_MEMORY.md`, `runtime.json`, and the parent GUI project's
`docs/G05_PROJECT_MEMORY.md` before continuing. This is the user's maintained Fork,
not an unmodified vendor checkout. Preserve upstream licenses and notices.

- Runtime/CLI must not import the GUI. GUI delegates configuration and lifecycle;
  all action scheduling and model interaction live here.
- `feat/runtime-base` contains reusable interfaces/infrastructure; each experiment
  implementation belongs only to its `exp/rtc`, `exp/async`, or `exp/sync-fast` branch.
- Model source is the pinned `GalaxeaVLA` submodule from doulei741/GalaxeaVLA.
  Publish a child commit before publishing a parent reference. Never push to vendor upstream.
- Keep original /home/dj/DJ/a1z-g05-deploy untouched. Do not copy weights or datasets
  into Git; select a read-only deployment bundle and dedicated run output directory.
- No unattended hardware execution or training. CPU stubs and recorded-observation
  GPU replay are authorized; real robot validation is performed by the user.
- A run must record actual source versions, dirty flags, checkpoint SHA, effective
  parameters, timing and stop reason. Offline finite output is not physical success.
- No `git add -A`, destructive reset, credential logging, or global Python installation.
- Validate meaningful scheduling/cancellation/underflow/RTC tests. Do not count skipped
  historical prefetch tests as evidence that a new implementation works.
