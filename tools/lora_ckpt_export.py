#!/usr/bin/env python3
"""Export a tensor-only LoRA/DoRA checkpoint with a PyTorch environment.

The native converter consumes the resulting safetensors plus JSON configuration.
The original checkpoint is never modified, and existing outputs are not replaced.
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file


def export_checkpoint(checkpoint, output):
    # Adapter checkpoints need only tensors and primitive configuration values.
    # Never fall back to unrestricted pickle execution for an incompatible file.
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(data, dict):
        raise ValueError("checkpoint must contain a state_dict and lora_config")
    state, config = data.get("state_dict"), data.get("lora_config")
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint state_dict must be a nonempty tensor dictionary")
    if not isinstance(config, dict) or not config:
        raise ValueError("checkpoint needs a nonempty lora_config")
    config_json = json.dumps(config, indent=2, allow_nan=False)
    flat = {}
    for key, tensor in state.items():
        if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("state_dict must contain only named tensors")
        if tensor.layout != torch.strided:
            raise ValueError("adapter tensors must use strided storage")
        value = tensor.detach().cpu()
        if value.is_floating_point():
            value = value.float()
        # Clone shared storage so safetensors can preserve every named tensor.
        flat[key] = value.contiguous().clone()

    output = Path(output)
    targets = [output.with_suffix(".safetensors"), output.with_suffix(".json")]
    created = []
    with tempfile.TemporaryDirectory(prefix=".sa3-export-", dir=output.parent) as staging:
        staged_tensor = Path(staging) / "adapter.safetensors"
        save_file(flat, str(staged_tensor), metadata={"lora_config": config_json})
        try:
            with targets[0].open("xb") as destination:
                created.append(targets[0])
                with staged_tensor.open("rb") as source:
                    shutil.copyfileobj(source, destination)
            with targets[1].open("xb") as destination:
                created.append(targets[1])
                destination.write(config_json.encode("utf-8"))
        except BaseException:
            for path in created:
                path.unlink()
            raise
    return len(flat), config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True, help="output basename (.safetensors + .json)")
    args = parser.parse_args()
    count, config = export_checkpoint(args.ckpt, args.out)
    print(f"exported {count} tensors + config -> {args.out}.safetensors / .json")
    print(f"  adapter_type={config.get('adapter_type')} rank={config.get('rank')} alpha={config.get('alpha')}")


if __name__ == "__main__":
    sys.exit(main())
