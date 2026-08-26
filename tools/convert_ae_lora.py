#!/usr/bin/env python3
"""Convert an autoencoder-targeted LoRA (decoder or encoder) to a sa3.cpp LoRA gguf.

The DiT converter's sibling. Two things differ, and both come from where these
checkpoints are produced:

  * They carry their config in the SAFETENSORS METADATA (a `lora_config` JSON
    blob written by save_lora_safetensors), not in a sidecar .json. So there is
    no --in basename to pair up; you point at the file.
  * They declare which half of the autoencoder they adapt via `target`
    ("decoder" | "encoder"). That field is what routes them at load time, both
    upstream and here, so it is copied into the gguf KV as `lora.target` rather
    than inferred from the tensor names.

Module names are mapped with tools/convert.py's own rename(), the same function
that produced the base AE gguf, so the two cannot drift apart. Every nn.Linear
in either half maps 1:1 with no special cases -- the only layer convert.py
treats specially is the weight_norm'd `mapping` conv, which cannot carry a LoRA
anyway (torch's deprecated weight_norm leaves `weight` a hook product, so
register_parametrization refuses it) and therefore never appears here.

Run with the converter .venv (has gguf + safetensors):
  .venv/Scripts/python.exe tools/convert_ae_lora.py \
      --in squeakfix_v3.safetensors --out models/lora-squeakfix-v3.gguf
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFWriter
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).parent))
from convert import SRC_PREFIX, rename as ae_rename  # noqa: E402

PARAM_RE = re.compile(
    r"^(?P<mod>.+)\.parametrizations\.weight\.(?P<slot>\d+)\.(?P<kind>lora_A|lora_B|"
    r"magnitude|magnitude_r|magnitude_c|U|V|M_xs)$"
)


def base_name(module, target):
    """Adapter module name -> base AE gguf weight name (without `.weight`).

    `module` is relative to the half it adapts (e.g. "layers.3.transformers.0.ff.ff.2"),
    because the upstream loader attaches the adapter to model.<target> directly.
    convert.py's rename() wants the full checkpoint key, so put the prefix back.
    """
    full = f"{SRC_PREFIX}{target}.{module}.weight"
    n = ae_rename(full)
    return n[: -len(".weight")] if n and n.endswith(".weight") else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="AE LoRA .safetensors")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    inp = Path(args.inp)
    with safe_open(str(inp), framework="np") as f:
        meta = f.metadata() or {}
        sd = {k: f.get_tensor(k) for k in f.keys()}

    if "lora_config" not in meta:
        sys.exit(
            f"{inp} has no `lora_config` metadata. Autoencoder adapters are "
            f"written by save_lora_safetensors, which stores rank/alpha/"
            f"adapter_type/target there; a checkpoint without it is either a "
            f"DiT export (use convert_lora.py) or was not produced by this "
            f"pipeline."
        )
    cfg = json.loads(meta["lora_config"])

    target = cfg.get("target", "dit")
    if target not in ("decoder", "encoder"):
        sys.exit(
            f"{inp} declares target={target!r}. This converter handles the "
            f"autoencoder halves only; a DiT adapter goes through "
            f"convert_lora.py, which maps onto the DiT gguf instead."
        )

    # group adapter tensors by target module
    mods, slots = {}, set()
    for k, v in sd.items():
        m = PARAM_RE.match(k)
        if not m:
            print(f"  ! unrecognized key skipped: {k}")
            continue
        slots.add(m.group("slot"))
        mods.setdefault(m.group("mod"), {})[m.group("kind")] = v

    # A checkpoint saved while stacked behind another adapter would carry a
    # non-zero slot, and its A/B would then be the SECOND adapter's -- correct
    # tensors, wrong provenance. Refuse rather than silently convert one.
    if slots - {"0"}:
        sys.exit(
            f"{inp} has parametrization slots {sorted(slots)}; expected only "
            f"slot 0. This checkpoint was saved from a stacked adapter list, so "
            f"which adapter these tensors belong to is ambiguous."
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    w = GGUFWriter(str(out), arch="sa3-lora")
    w.add_name(f"sa3 {cfg.get('adapter_type', 'lora')} {target} adapter")
    w.add_string("lora.adapter_type", str(cfg.get("adapter_type", "lora")))
    w.add_uint32("lora.rank", int(cfg["rank"]))
    w.add_float32("lora.alpha", float(cfg["alpha"]))
    # Which half this adapts. The C++ side needs it to decide whether an adapter
    # belongs on the encode pass or the decode pass; the tensor names alone
    # would answer it, but only by string-matching "ae.enc"/"ae.dec", which
    # silently answers "neither" for an adapter that mapped nothing.
    w.add_string("lora.target", target)
    if "base_model" in cfg:
        # Which AE this was trained against. SAME-L and SAME-S share a tensor
        # NAMESPACE ("ae.dec.<i>...") at different widths, so a cross-load
        # matches by name on the blocks both models have -- 25 of SAME-L's 49
        # land on SAME-S, 24 at the wrong width. lora.h refuses that on the
        # factor shapes; this key is so the message can say which file it was.
        w.add_string("lora.base_model", str(cfg["base_model"]))

    n_mapped, n_skipped, mapped = 0, 0, []
    for module, t in sorted(mods.items()):
        base = base_name(module, target)
        if base is None:
            print(f"  ! unmapped module skipped: {module}")
            n_skipped += 1
            continue
        for kind, arr in t.items():
            w.add_tensor(f"{base}.{kind}", np.ascontiguousarray(arr.astype(np.float32)))
        n_mapped += 1
        mapped.append(base)
    w.add_uint32("lora.n_targets", n_mapped)

    if n_mapped == 0:
        sys.exit(
            f"{inp}: no module mapped onto the {target} of the AE gguf. The "
            f"adapter would load and do nothing."
        )
    # Every module in an AE adapter should map; convert.py covers both halves
    # completely. A skip means the checkpoint carries something the base gguf
    # does not, which is worth stopping for rather than shipping a partial merge.
    if n_skipped:
        sys.exit(
            f"{inp}: {n_skipped} module(s) did not map onto the AE gguf (see "
            f"above). Converting would produce a partially-applied adapter."
        )

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(
        f"wrote {out}  ({n_mapped} targets mapped) target={target} "
        f"type={cfg.get('adapter_type', 'lora')} rank={cfg['rank']} "
        f"alpha={cfg['alpha']}"
    )
    print(f"  e.g. {mapped[0]} .. {mapped[-1]}")


if __name__ == "__main__":
    sys.exit(main())
