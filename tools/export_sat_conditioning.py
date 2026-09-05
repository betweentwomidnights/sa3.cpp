#!/usr/bin/env python3
"""Export classic stable-audio-tools conditioning as portable float32 buffers."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer, T5EncoderModel


def number_condition(source, conditioner_id, config, value):
    prefix = f"conditioner.conditioners.{conditioner_id}.embedder.embedding."
    learned = source.get_tensor(prefix + "0.weights").float()
    weight = source.get_tensor(prefix + "1.weight").float()
    bias = source.get_tensor(prefix + "1.bias").float()
    minimum, maximum = float(config["min_val"]), float(config["max_val"])
    normalized = (min(max(float(value), minimum), maximum) - minimum) / (maximum - minimum)
    freqs = torch.tensor([normalized])[:, None] * learned[None, :] * (2.0 * math.pi)
    features = torch.cat((torch.tensor([[normalized]]), freqs.sin(), freqs.cos()), dim=-1)
    return torch.nn.functional.linear(features, weight, bias).reshape(1, 1, -1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seconds-start", type=float, default=0.0)
    parser.add_argument("--seconds", type=float, default=11.0)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    conditioner_config = config["model"]["conditioning"]
    configs = {item["id"]: item for item in conditioner_config["configs"]}
    prompt_config = configs["prompt"]["config"]
    max_length = int(prompt_config["max_length"])
    model_name = prompt_config["t5_model_name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoded = tokenizer([args.prompt], truncation=True, max_length=max_length,
                        padding="max_length", return_tensors="pt")
    ids = encoded["input_ids"].to(args.device)
    mask = encoded["attention_mask"].to(args.device)
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    t5 = T5EncoderModel.from_pretrained(model_name, torch_dtype=dtype).eval().to(args.device)
    with torch.inference_mode():
        prompt = t5(input_ids=ids, attention_mask=mask).last_hidden_state.float()
        prompt = prompt * mask.unsqueeze(-1).float()
    del t5

    values = {"seconds_start": args.seconds_start, "seconds_total": args.seconds}
    conditions = {"prompt": prompt.cpu()}
    with safe_open(args.checkpoint, framework="pt", device="cpu") as source:
        for conditioner_id, value in values.items():
            if conditioner_id in configs:
                conditions[conditioner_id] = number_condition(
                    source, conditioner_id, configs[conditioner_id]["config"], value)

    diffusion = config["model"]["diffusion"]
    cross = torch.cat([conditions[key] for key in diffusion["cross_attention_cond_ids"]],
                      dim=1).contiguous().numpy().astype(np.float32)
    global_cond = torch.cat([conditions[key] for key in diffusion["global_cond_ids"]],
                            dim=-1).reshape(-1).contiguous().numpy().astype(np.float32)
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    cross.tofile(str(prefix) + ".cross.f32")
    global_cond.tofile(str(prefix) + ".global.f32")
    Path(str(prefix) + ".json").write_text(json.dumps({
        "prompt": args.prompt, "seconds_start": args.seconds_start, "seconds": args.seconds,
        "cross_tokens": int(cross.shape[1]), "cond_dim": int(cross.shape[2]),
        "global_dim": int(global_cond.size), "t5_model": model_name,
    }, indent=2), encoding="utf-8")
    print(f"wrote {prefix}.cross.f32 {cross.shape} and "
          f"{prefix}.global.f32 {global_cond.shape}")


if __name__ == "__main__":
    main()
