# This file is part of G05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

"""Export an intermediate G0.5 training checkpoint for inference on a 24 GB GPU.

Removes optimizer/scheduler state and stores eligible weights as BF16. Parameters
that the policy restores to FP32 for inference stay FP32 in the export.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "GalaxeaVLA" / "src"))


def export(source: Path, destination: Path) -> tuple[int, int]:
    if source.resolve() == destination.resolve():
        raise ValueError("Source and destination must differ")
    if destination.exists():
        raise FileExistsError(destination)
    from g05.models.g05.g05_policy_qwen35 import G05PolicyQwen35

    patterns = G05PolicyQwen35.fp32_param_patterns.fget(None)
    checkpoint = torch.load(source, map_location="cpu", mmap=True, weights_only=False)
    weights = checkpoint.get("model_state_dict")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("Expected a training checkpoint with model_state_dict")
    result = {}
    converted = 0
    for name, tensor in weights.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Non-tensor model_state_dict entry: {name}")
        if (tensor.dtype == torch.float32 and tensor.numel() * tensor.element_size() >= 1024**2
                and not any(pattern in name for pattern in patterns)):
            tensor = tensor.to(torch.bfloat16)
            converted += 1
        result[name] = tensor
    payload = {"model_state_dict": result, "step": checkpoint.get("step"),
               "export_dtype": "bf16_for_inference"}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return len(weights), converted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Training step checkpoint (.pt)")
    parser.add_argument("destination", type=Path, help="New inference checkpoint (.pt)")
    args = parser.parse_args()
    count, converted = export(args.source, args.destination)
    print(f"Exported {count} model tensors ({converted} BF16) to {args.destination}")


if __name__ == "__main__":
    main()
