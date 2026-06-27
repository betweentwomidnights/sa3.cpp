#!/usr/bin/env python3
"""Validate the weight-space transform for every adapter_type against PyTorch's
LoRAParametrization, on synthetic random weights. Pins the exact math the C++
apply pass (src/lora.h) must reproduce, since we have no trained BoRA/-xs ckpt.

Run with the SA3 venv:
  .../env/Scripts/python.exe tools/lora_spec_test.py
"""
import sys
import numpy as np
import torch
from stable_audio_3.models.lora.model import LoRAParametrization


def numpy_apply(adapter_type, W0, sc, t):
    """Mirror of the intended C++ apply: t holds the adapter tensors as numpy."""
    if "xs" in adapter_type:
        delta = t["U"] @ t["M_xs"] @ t["V"].T            # U[out,r] @ M[r,r] @ V[in,r].T
    else:
        delta = t["lora_B"] @ t["lora_A"]                # B[out,r] @ A[r,in]
    V = W0 + sc * delta
    if adapter_type in ("lora", "lora-xs"):
        return V
    if adapter_type in ("dora-rows", "dora-rows-xs"):    # norm per row (over in), magnitude[out]
        return t["magnitude"][:, None] * V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    if adapter_type in ("dora-cols", "dora-cols-xs"):    # norm per col (over out), magnitude[in]
        return t["magnitude"][None, :] * V / (np.linalg.norm(V, axis=0, keepdims=True) + 1e-12)
    if adapter_type in ("bora", "bora-xs"):
        Vr = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
        inter = t["magnitude_r"][:, None] * Vr
        Hc = inter / (np.linalg.norm(inter, axis=0, keepdims=True) + 1e-12)
        return t["magnitude_c"][None, :] * Hc
    raise ValueError(adapter_type)


def main():
    types = ["lora", "dora-rows", "dora-cols", "bora",
             "lora-xs", "dora-rows-xs", "dora-cols-xs", "bora-xs"]
    in_f, out_f, rank, alpha, strength = 12, 9, 4, 6, 0.7
    ok = True
    for at in types:
        torch.manual_seed(0)
        W0 = torch.randn(out_f, in_f)
        layer = torch.nn.Linear(in_f, out_f, bias=False)
        layer.weight.data = W0.clone()
        p = LoRAParametrization.from_linear(layer, rank=rank, lora_alpha=alpha, adapter_type=at)
        with torch.no_grad():
            for nm in ("lora_A", "lora_B", "M_xs"):
                if hasattr(p, nm): getattr(p, nm).copy_(torch.randn_like(getattr(p, nm)))
            for nm in ("magnitude", "magnitude_r", "magnitude_c"):
                if hasattr(p, nm): getattr(p, nm).copy_(torch.rand_like(getattr(p, nm)) + 0.5)
        p.lora_strength.fill_(strength)
        W_pt = p.forward(W0).detach().numpy()

        t = {nm: getattr(p, nm).detach().numpy() for nm in
             ("lora_A", "lora_B", "M_xs", "U", "V", "magnitude", "magnitude_r", "magnitude_c")
             if hasattr(p, nm)}
        W_np = numpy_apply(at, W0.numpy(), alpha / rank * strength, t)

        a, b = W_pt.ravel(), W_np.ravel()
        cs = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        md = float(np.abs(W_pt - W_np).max())
        flag = "OK " if (cs > 0.999999 and md < 1e-4) else "FAIL"
        if flag == "FAIL": ok = False
        print(f"  {flag} {at:14s} cossim={cs:.6f} maxdiff={md:.2e}")
    print("ALL MATCH" if ok else "MISMATCH — fix formula before porting")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
