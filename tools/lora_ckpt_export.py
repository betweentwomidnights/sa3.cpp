#!/usr/bin/env python3
"""Export a LoRA/DoRA .ckpt (torch pickle) to a flat safetensors + json config.

The gguf converter (tools/convert_lora.py) runs in the .venv which has no torch,
so this stage (run with the SA3 venv) just unpickles the checkpoint and re-saves the
raw adapter tensors + the lora_config as torch-free formats.

Usage (SA3 venv):
  .../services/sa3/env/Scripts/python.exe tools/lora_ckpt_export.py \
      --ckpt loras/kev.ckpt --out loras/kev
"""
import argparse, json, sys
from pathlib import Path
import torch
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True, help="output basename (writes <out>.safetensors + <out>.json)")
    args = ap.parse_args()

    d = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd, cfg = d["state_dict"], d["lora_config"]
    flat = {k: v.detach().cpu().float().contiguous() for k, v in sd.items()}

    out = Path(args.out)
    save_file(flat, str(out.with_suffix(".safetensors")))
    out.with_suffix(".json").write_text(json.dumps(cfg, indent=2, default=str))
    print(f"exported {len(flat)} tensors + config -> {out}.safetensors / .json")
    print(f"  adapter_type={cfg.get('adapter_type')} rank={cfg.get('rank')} alpha={cfg.get('alpha')}")


if __name__ == "__main__":
    sys.exit(main())
