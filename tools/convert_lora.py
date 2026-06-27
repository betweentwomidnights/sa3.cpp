#!/usr/bin/env python3
"""Convert an exported LoRA/DoRA (safetensors + json from lora_ckpt_export.py) to a
sa3.cpp LoRA gguf.

Each target module's {lora_A, lora_B, magnitude} is renamed to match the base DiT
gguf weight it adapts (reusing convert_dit's mapping) and written as
`<base>.lora_A` / `.lora_B` / `.magnitude`. Adapter metadata (type, rank, alpha) goes
in the gguf KV store. The C++ apply pass recomputes W_eff = f(W0; A,B,magnitude,strength)
per the adapter type, so the base gguf is never modified.

Run with the converter .venv (has gguf + safetensors):
  .venv/Scripts/python.exe tools/convert_lora.py --in loras/kev --out models/lora-kev-f32.gguf
"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file
from gguf import GGUFWriter

sys.path.insert(0, str(Path(__file__).parent))
from convert_dit import rename as dit_rename   # rename("model.model.<...>.weight") -> "dit....weight" | None

PARAM_RE = re.compile(r"^(?P<mod>.+)\.parametrizations\.weight\.0\.(?P<kind>lora_A|lora_B|magnitude|"
                      r"magnitude_r|magnitude_c|U|V|M_xs)$")


def base_name(module):
    """LoRA module name -> base DiT weight name (without .weight), or None if not a DiT weight."""
    if not module.startswith("model."):
        return None                                  # e.g. conditioners.* — not in the DiT gguf
    full = "model.model." + module[len("model."):] + ".weight"
    n = dit_rename(full)
    return n[:-len(".weight")] if n and n.endswith(".weight") else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="basename of the exported .safetensors/.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    inp = Path(args.inp)
    cfg = json.loads(inp.with_suffix(".json").read_text())
    sd = load_file(str(inp.with_suffix(".safetensors")))

    # group adapter tensors by target module
    mods = {}
    for k, v in sd.items():
        m = PARAM_RE.match(k)
        if not m:
            print(f"  ! unrecognized key skipped: {k}"); continue
        mods.setdefault(m.group("mod"), {})[m.group("kind")] = v

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    w = GGUFWriter(str(out), arch="sa3-lora")
    w.add_name(f"sa3 {cfg.get('adapter_type','lora')} adapter")
    w.add_string("lora.adapter_type", str(cfg.get("adapter_type", "lora")))
    w.add_uint32("lora.rank", int(cfg["rank"]))
    w.add_float32("lora.alpha", float(cfg["alpha"]))

    n_mapped, n_skipped, mapped_names = 0, 0, []
    for module, t in sorted(mods.items()):
        base = base_name(module)
        if base is None:
            n_skipped += 1; continue
        for kind, arr in t.items():
            w.add_tensor(f"{base}.{kind}", np.ascontiguousarray(arr.astype(np.float32)))
        n_mapped += 1; mapped_names.append(base)
    w.add_uint32("lora.n_targets", n_mapped)

    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"wrote {out}  ({n_mapped} targets mapped, {n_skipped} non-DiT skipped) "
          f"type={cfg.get('adapter_type')} rank={cfg['rank']} alpha={cfg['alpha']}")
    print(f"  e.g. {mapped_names[0]} .. {mapped_names[-1]}")


if __name__ == "__main__":
    sys.exit(main())
