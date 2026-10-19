#!/usr/bin/env python3
# This file is part of G05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

"""Initialize pinned upstream sources and apply the tracked G05_A1Z adaptations."""
from pathlib import Path
import json
import hashlib
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = ("a1z-data-collect", "GalaxeaVLA")
NESTED_COMPONENTS = (
    ("a1z-data-collect", "a1z-t"),
    ("a1z-data-collect/a1z-t", "GALAXEA-A1Z"),
)


def git(*args, check=True):
    return subprocess.run(["git", "-c", "fetch.recurseSubmodules=false", "-C", str(ROOT), *args], check=check,
                          text=True, capture_output=True)


def main():
    git("submodule", "sync", "--", *COMPONENTS)
    git("submodule", "update", "--init", "--", *COMPONENTS)
    versions = json.loads((ROOT / "versions.json").read_text())
    # Initialize only the Leader and its SDK; both are pinned in the manifest.
    for parent, child in NESTED_COMPONENTS:
        git("-C", parent, "submodule", "sync", "--", child)
        git("-C", parent, "submodule", "update", "--init", "--", child)
        folder = f"{parent}/{child}"
        revision = versions[folder]
        if git("-C", folder, "rev-parse", "HEAD").stdout.strip() != revision:
            if git("-C", folder, "status", "--porcelain").stdout.strip():
                raise RuntimeError(f"Local edits found in {folder}; preserve them before updating")
            if git("-C", folder, "cat-file", "-e", revision + "^{commit}", check=False).returncode:
                git("-C", folder, "fetch", "origin", revision)
            git("-C", folder, "checkout", "--detach", revision)
    folders = (*COMPONENTS, *(f"{parent}/{child}" for parent, child in NESTED_COMPONENTS))
    for name in folders:
        actual = git("-C", name, "rev-parse", "HEAD").stdout.strip()
        if actual != versions[name]:
            raise RuntimeError(f"Pinned version mismatch for {name}: {actual}")
    # Replace the upstream machine path only in the unchanged pinned config.
    memory_config = ROOT / "GalaxeaVLA/configs/model/g05_mem.yaml"
    if memory_config.is_file() and hashlib.sha256(memory_config.read_bytes()).hexdigest() == "b4514cb901c355bcbe33fb1fc9553e24b1ee82218e47b2bcd963e160cb35dc6f":
        lines = memory_config.read_text().splitlines(keepends=True)
        memory_config.write_text("".join(
            "pretrained_ckpt: ${oc.env:G05_MEM_CKPT,checkpoints/g05_pretrain_mem.pt}\n"
            if line.startswith("pretrained_ckpt:") else line for line in lines
        ))
    # Upgrade recognized generated configs; preserve user edits.
    config = ROOT / "GalaxeaVLA/configs/data/a1z.yaml"
    if config.is_file() and hashlib.sha256(config.read_bytes()).hexdigest() == "c1fd8c487266d01caabd6e9af64d53c9a661eeff38ce0c68eecfde2f3e3da390":
        config.write_text(config.read_text().replace("lerobot_ds_version: '3.0'", "lerobot_ds_version: '2.1'"))
    # Keep step-based training independent of epoch limits.
    task = ROOT / "GalaxeaVLA/configs/task/a1z.yaml"
    if task.is_file() and hashlib.sha256(task.read_bytes()).hexdigest() == "4e18ee65fa1f38f95294be226fc6ae78a76b812c6390420411707a7c488856a7":
        task.write_text(task.read_text().replace("  max_steps: 10000", "  max_epochs: null\n  max_steps: 10000"))
    # Add notices only to recognized, unedited integration-generated files.
    # Keep Hydra package directives first; upstream source notices are retained.
    integration_headers = {
        "configs/data/a1z.yaml": "21794e276541441bdeb8d9414bead341ccac37c30628862185221dd78e8b8b4e",
        "configs/data/a1z_single_left.yaml": "8a86a1c8fa0cb4df708ba90c28d5b5242214a8906161d2506fc369f9adbf4d1f",
        "configs/task/a1z.yaml": "e81a522c09563238715d47b610cd30f7b27a9582fb8d49cd270200c9d0cc2098",
        "configs/task/a1z_lora.yaml": "e3ed6c52f55069457aaab6527acad500ddfaed4258017dcea3914b1ce8aa0b4f",
        "configs/task/a1z_single_left_lora.yaml": "94eece1be06f65f543f137aa65734170507e2add24e3a7227d3940bc78d60d9c",
        "configs/task/a1z_single_left_full.yaml": "9495872c9c3c8d48f7cfc476c59badd1d128a88635ce858be640d33f97803fcc",
        "src/g05/models/g05/lora.py": "495b802b7f0cdeaf3748f2f31d8fb26997dce4e9b16a7ddb54e95f43ebada106"
    }
    header = (
        "# This file is part of G05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.\n"
        "# See LICENSE.txt in the project root; upstream components retain their applicable licenses.\n\n"
    )
    for relative, expected in integration_headers.items():
        path = ROOT / "GalaxeaVLA" / relative
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
            text = path.read_text()
            if text.startswith("# @package "):
                first, rest = text.split("\n", 1)
                text = first + "\n" + header + rest
            else:
                text = header + text
            path.write_text(text)
    pending = []
    series = json.loads((ROOT / "patches/series.json").read_text())
    if isinstance(series, dict):
        series = series["patches"]
    for name, filename in series:
        patch = str(ROOT / "patches" / filename)
        if git("-C", name, "apply", "--reverse", "--check", patch, check=False).returncode == 0:
            print(f"Already applied: {name}")
        elif git("-C", name, "apply", "--check", patch, check=False).returncode == 0:
            pending.append((name, patch))
        else:
            raise RuntimeError(f"Patch conflict in {name}; preserve your local edits and resolve manually")
    for name, patch in pending:
        git("-C", name, "apply", patch)
        print(f"Applied: {name}")
    print("Submodules ready. Run python3 scripts/verify_workspace.py next.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(exc.stderr or str(exc), file=sys.stderr)
        sys.exit(1)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
