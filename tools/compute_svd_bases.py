#!/usr/bin/env python3
"""Precompute frozen SVD bases (U, V) for LoRA-XS training with sa3-train.

sa3-train computes these bases natively (randomized SVD) by default. This tool is the
optional exact-parity path: it runs torch.linalg.svd on each DiT weight and writes a
GGUF that `sa3-train --svd-bases bases.gguf` loads instead of recomputing. The tensor
names/shapes match docs/lora_checkpoint_contract.md (`<stem>.U` = ne[rank,out],
`<stem>.V` = ne[rank,in]), and the sign convention matches the reference
_canonicalize_svd_signs (largest-magnitude entry of each U column positive).

Requires: pip install gguf torch numpy
"""
import argparse
import numpy as np
import torch
import gguf


def canonicalize_svd_signs(U, Vh):
    max_abs_idx = U.abs().argmax(dim=0)
    signs = U[max_abs_idx, torch.arange(U.shape[1])].sign()
    signs[signs == 0] = 1
    return U * signs.unsqueeze(0), Vh * signs.unsqueeze(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dit", required=True, help="path to the DiT GGUF (source of base weights)")
    ap.add_argument("--out", required=True, help="output bases GGUF path")
    ap.add_argument("--rank", type=int, required=True, help="LoRA-XS rank (columns of U/V to keep)")
    args = ap.parse_args()

    reader = gguf.GGUFReader(args.dit)
    writer = gguf.GGUFWriter(args.out, "sa3-lora")
    writer.add_uint32("lora.rank", args.rank)

    n = 0
    for t in reader.tensors:
        name = t.name
        if not name.startswith("dit.") or not name.endswith(".weight"):
            continue
        W = torch.from_numpy(np.array(t.data)).float()  # gguf ne=[in,out] -> numpy [out, in]
        if W.ndim != 2:
            continue
        out_dim, in_dim = W.shape
        if args.rank > min(out_dim, in_dim):
            continue  # infeasible rank for this target; sa3-train rejects these too
        U_full, _S, Vh_full = torch.linalg.svd(W, full_matrices=False)
        U_full, Vh_full = canonicalize_svd_signs(U_full, Vh_full)
        U = U_full[:, : args.rank].contiguous()        # [out, rank]  -> ne[rank, out]
        V = Vh_full[: args.rank, :].T.contiguous()     # [in, rank]   -> ne[rank, in]
        stem = name[: -len(".weight")]
        writer.add_tensor(stem + ".U", U.numpy())
        writer.add_tensor(stem + ".V", V.numpy())
        n += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {n} target bases (rank {args.rank}) -> {args.out}")


if __name__ == "__main__":
    main()
