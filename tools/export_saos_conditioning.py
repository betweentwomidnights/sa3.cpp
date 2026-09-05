#!/usr/bin/env python3
"""Export the two SAOS conditioning inputs as portable raw float32 buffers.

This is the temporary bridge for end-to-end GGML DiT/VAE validation while the
native T5 tokenizer/encoder packaging is completed.  It loads only public T5-base
plus the three learned seconds_total tensors from the SAOS checkpoint, not the
PyTorch diffusion model.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer, T5EncoderModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--seconds", type=float, default=11.0)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    ccfg = cfg["model"]["conditioning"]
    prompt_cfg = next(x["config"] for x in ccfg["configs"] if x["id"] == "prompt")
    sec_cfg = next(x["config"] for x in ccfg["configs"] if x["id"] == "seconds_total")
    max_len = int(prompt_cfg["max_length"])
    model_name = prompt_cfg["t5_model_name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoded = tokenizer([a.prompt], truncation=True, max_length=max_len,
                        padding="max_length", return_tensors="pt")
    ids = encoded["input_ids"].to(a.device)
    mask = encoded["attention_mask"].to(a.device)
    dtype = torch.float16 if a.device.startswith("cuda") else torch.float32
    t5 = T5EncoderModel.from_pretrained(model_name, torch_dtype=dtype).eval().to(a.device)
    with torch.inference_mode():
        prompt = t5(input_ids=ids, attention_mask=mask).last_hidden_state.float()
        prompt = prompt * mask.unsqueeze(-1).float()
    del t5

    p = "conditioner.conditioners.seconds_total.embedder.embedding."
    with safe_open(a.checkpoint, framework="pt", device="cpu") as f:
        learned = f.get_tensor(p + "0.weights").float()
        weight = f.get_tensor(p + "1.weight").float()
        bias = f.get_tensor(p + "1.bias").float()
    value = min(max(float(a.seconds), float(sec_cfg["min_val"])), float(sec_cfg["max_val"]))
    value = (value - float(sec_cfg["min_val"])) / (float(sec_cfg["max_val"]) - float(sec_cfg["min_val"]))
    freqs = torch.tensor([value])[:, None] * learned[None, :] * (2.0 * math.pi)
    features = torch.cat((torch.tensor([[value]]), freqs.sin(), freqs.cos()), dim=-1)
    seconds = torch.nn.functional.linear(features, weight, bias).reshape(1, 1, -1)

    cross = torch.cat((prompt.cpu(), seconds), dim=1).contiguous().numpy().astype(np.float32)
    global_cond = seconds.reshape(-1).contiguous().numpy().astype(np.float32)
    prefix = Path(a.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    cross.tofile(str(prefix) + ".cross.f32")
    global_cond.tofile(str(prefix) + ".global.f32")
    meta = {"prompt": a.prompt, "seconds": a.seconds, "cross_tokens": int(cross.shape[1]),
            "cond_dim": int(cross.shape[2]), "t5_model": model_name}
    Path(str(prefix) + ".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {prefix}.cross.f32 {cross.shape} and {prefix}.global.f32 {global_cond.shape}")


if __name__ == "__main__":
    main()
