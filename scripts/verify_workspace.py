# This file is part of G05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

"""Read-only source/version/config check; never opens hardware."""
import argparse
import json
import hashlib
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def git_output(folder, *args):
    """Return git output, or None for the vendored source snapshot."""
    repo = ROOT / folder
    # Do not let Git walk up to the monorepo root for vendored components.
    # A real checkout can have either a .git directory or a worktree .git file.
    if not (repo / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-model-assets", action="store_true")
    args = parser.parse_args()
    errors = []
    for folder, expected in json.loads((ROOT / "versions.json").read_text()).items():
        if folder == "$comment":
            continue
        if not (ROOT / folder).is_dir():
            errors.append(f"Missing source folder: {folder}")
            continue
        actual = git_output(folder, "rev-parse", "HEAD")
        if actual is None:
            if folder != "a1z_g05_deploy":
                errors.append(f"Submodule not initialized: {folder}; run python3 scripts/bootstrap.py")
            else:
                print(f"OK {folder} snapshot (baseline {expected[:12]})")
        elif actual != expected:
            errors.append(f"Version mismatch: {folder}")
        else:
            print(f"OK {folder} {expected[:12]}")
    sdk = ROOT / "a1z-data-collect/a1z-t/GALAXEA-A1Z"
    if not (sdk / "a1z/__init__.py").is_file():
        errors.append("Nested SDK is missing; run python3 scripts/bootstrap.py")
    memory_config = ROOT / "GalaxeaVLA/configs/model/g05_mem.yaml"
    if memory_config.is_file() and hashlib.sha256(memory_config.read_bytes()).hexdigest() == "b4514cb901c355bcbe33fb1fc9553e24b1ee82218e47b2bcd963e160cb35dc6f":
        errors.append("Upstream checkpoint path is not configured; run python3 scripts/bootstrap.py")
    # Submodules normally use detached HEAD; verify their pinned commits above.
    series = json.loads((ROOT / "patches/series.json").read_text())
    if isinstance(series, dict):
        series = series["patches"]
    for component, filename in series:
        patch = ROOT / "patches" / filename
        result = subprocess.run(["git", "-C", str(ROOT / component),
                                 "apply", "--reverse", "--check", str(patch)],
                                capture_output=True, text=True)
        if result.returncode:
            errors.append(f"Missing or modified integration patch: {component}; run python3 scripts/bootstrap.py")
    for name in ("configs/collect_single.example.yaml", "configs/collect_dual.example.yaml", "configs/deployment/pc-single.example.json", "scripts/train_single_lora.sh", "scripts/train_single_full.sh", "a1z_g05_deploy/client/single_client.py", "configs/collect.example.yaml", "configs/collect_single_left.example.yaml", "configs/deployment/pc-direct.example.json", "configs/deployment/pc-single-left.example.json", "GalaxeaVLA/configs/task/a1z.yaml", "GalaxeaVLA/configs/data/a1z.yaml", "a1z_g05_deploy/client/dual_arm_client.py", "a1z_g05_deploy/client/single_left_client.py"):
        if not (ROOT / name).is_file():
            errors.append(f"Missing integration file: {name}")
    for name in ("checkpoints/g05-base/.hydra/config.yaml", "checkpoints/g05-base/dataset_stats.json", "checkpoints/g05-base/checkpoints/model_state_dict.pt", "checkpoints/action_tokenizer.pt", "checkpoints/qwen3_5_2b_base_processor"):
        if not (ROOT / name).exists():
            message = f"Model asset not downloaded: {name}"
            print(message)
            if args.require_model_assets:
                errors.append(message)
    print("Deployment: PC direct. Configure camera/CAN in configs/deployment/pc-direct.local.json before hardware tests.")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("Source integration OK. This is not a GPU or physical-robot test.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
