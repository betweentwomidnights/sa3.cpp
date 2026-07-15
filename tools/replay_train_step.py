#!/usr/bin/env python3
"""Replay a native sa3-train capture through the PyTorch trainer model.

The C++ side writes a bundle with SA3_DUMP_STEP and raw gradients with
SA3_DUMP_GRADS. This script bypasses dataset, crop, RNG, conditioner, timestep,
noise, and mask generation: both frameworks receive the identical DiT inputs.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


DIRECT = {
    "preprocess_conv": "dit.pre_conv",
    "postprocess_conv": "dit.post_conv",
    "to_cond_embed.0": "dit.cond_embed.0",
    "to_cond_embed.2": "dit.cond_embed.2",
    "to_global_embed.0": "dit.global_embed.0",
    "to_global_embed.2": "dit.global_embed.2",
    "to_timestep_embed.0": "dit.time_embed.0",
    "to_timestep_embed.2": "dit.time_embed.2",
    "transformer.global_cond_embedder.0": "dit.gce.0",
    "transformer.global_cond_embedder.2": "dit.gce.2",
    "transformer.project_in": "dit.proj_in",
    "transformer.project_out": "dit.proj_out",
}

LAYER = {
    "self_attn.to_qkv": "self.qkv",
    "self_attn.to_out": "self.out",
    "cross_attn.to_q": "cross.q",
    "cross_attn.to_kv": "cross.kv",
    "cross_attn.to_out": "cross.out",
    "ff.ff.0.proj": "ff.proj",
    "ff.ff.2": "ff.out",
    "to_scale_shift_gate": "ssg",
    "to_local_embed.0": "local.0",
    "to_local_embed.2": "local.2",
}


def cpp_stem(name: str):
    # get_lora_layers(model.model) reports names under DiTWrapper, beginning
    # with its child "model" (the DiffusionTransformer).
    if name.startswith("model."):
        name = name[len("model."):]
    if name in DIRECT:
        return DIRECT[name]
    prefix = "transformer.layers."
    if name.startswith(prefix):
        layer, rest = name[len(prefix):].split(".", 1)
        mapped = LAYER.get(rest)
        if mapped:
            return f"dit.{layer}.{mapped}"
    return None


def read_f32(bundle: Path, name: str, count=None):
    data = np.fromfile(bundle / f"{name}.f32", dtype=np.float32)
    if count is not None and data.size != count:
        raise ValueError(f"{name}: expected {count} floats, found {data.size}")
    return data


def time_major(bundle: Path, name: str, frames: int, channels: int, device: str):
    data = read_f32(bundle, name, frames * channels).reshape(frames, channels).T.copy()
    return torch.from_numpy(data).unsqueeze(0).to(device)


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0 if na == nb else 0.0
    return float(np.dot(a, b) / (na * nb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--cpp-grads")
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--adapter",
                    help="PyTorch safetensors for the same shared adapter state loaded by C++")
    ap.add_argument("--service-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--padding-mask", choices=("all", "none"), default="all")
    args = ap.parse_args()

    service_root = Path(args.service_root).resolve()
    sys.path.insert(0, str(service_root))
    from underfit.backends import get_backend
    from underfit.training.lora import apply_lora_from_config, load_lora_resume
    from stable_audio_3.models.lora import get_lora_layers

    bundle = Path(args.bundle)
    cpp_grads = Path(args.cpp_grads) if args.cpp_grads else None
    out_dir = Path(args.out)
    grad_dir = out_dir / "grads"
    grad_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((bundle / "meta.json").read_text())
    frames, io = int(meta["frames"]), int(meta["io"])
    cond_dim, ctx_len = int(meta["cond_dim"]), int(meta["ctx_len"])
    local_dim = int(meta["local_dim"])

    config = json.loads(Path(args.model_config).read_text())
    lora_cfg = dict(config["training"]["lora_config"])
    exclude = list(lora_cfg.get("exclude") or [])
    if "seconds_total" not in exclude:
        exclude.append("seconds_total")
    lora_cfg["exclude"] = exclude

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backend = get_backend("sa3")
    print(f"loading base model on {device}", flush=True)
    model, _ = backend.load_model(args.model_config, args.base_ckpt, device=device, half=(device == "cuda"))
    state = None
    if args.adapter:
        state, _ = load_lora_resume(backend, args.adapter)
    lora_params, _ = apply_lora_from_config(
        backend, model, lora_cfg, lora_state_dict=state,
        base_precision=config["training"].get("base_precision"),
        svd_bases_path=config.get("svd_bases_path"),
    )
    for p in lora_params:
        p.data = p.data.float()
        p.requires_grad_(True)

    layers = {}
    unmapped = []
    for name, layer in get_lora_layers(model.model):
        stem = cpp_stem(name)
        if stem is None:
            unmapped.append(name)
        else:
            layers[stem] = layer
    expected = {x.strip() for x in (bundle / "targets.txt").read_text().splitlines() if x.strip()}
    actual = set(layers)
    if unmapped or actual != expected:
        raise RuntimeError(
            f"target manifest mismatch: unmapped={unmapped}, "
            f"missing={sorted(expected-actual)}, extra={sorted(actual-expected)}"
        )
    print(f"target manifest matched: {len(actual)} layers", flush=True)

    activations = {}
    hooks = []
    def capture(name):
        def fn(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            activations[name] = value.detach().float().cpu().numpy().copy().reshape(-1)
        return fn
    def capture_input(name):
        def fn(_module, inputs):
            activations[name] = inputs[0].detach().float().cpu().numpy().copy().reshape(-1)
        return fn
    dit = model.model.model
    hooks.append(dit.to_cond_embed.register_forward_hook(capture("context")))
    hooks.append(dit.transformer.global_cond_embedder.register_forward_hook(capture("gcond")))
    hooks.append(dit.transformer.layers[0].register_forward_pre_hook(capture_input("block_0")))
    for i, layer in enumerate(dit.transformer.layers):
        hooks.append(layer.register_forward_hook(capture(f"block_{i + 1}")))

    x_t = time_major(bundle, "x_t", frames, io, device)
    target = time_major(bundle, "target", frames, io, device)
    cross_np = read_f32(bundle, "cross", ctx_len * cond_dim).reshape(ctx_len, cond_dim).copy()
    cross = torch.from_numpy(cross_np).unsqueeze(0).to(device)
    global_embed = torch.from_numpy(read_f32(bundle, "global", cond_dim).copy()).unsqueeze(0).to(device)
    local = time_major(bundle, "local", frames, local_dim, device) if local_dim else None
    weights = time_major(bundle, "loss_weight", frames, io, device) if local_dim else None
    t = torch.tensor([float(meta["t"])], device=device, dtype=torch.float32)
    padding = torch.ones((1, frames), device=device, dtype=torch.bool) if args.padding_mask == "all" else None

    model.zero_grad(set_to_none=True)
    amp = torch.amp.autocast("cuda", dtype=torch.float16) if device == "cuda" else torch.no_grad()
    # torch.no_grad is replaced below on CPU; CUDA is the parity target used by the reference.
    if device != "cuda":
        from contextlib import nullcontext
        amp = nullcontext()
    with amp:
        velocity = model.model.model(
            x_t, t,
            cross_attn_cond=cross,
            global_embed=global_embed,
            local_add_cond=local,
            padding_mask=padding,
            cfg_dropout_prob=0.0,
        )
        mse = (velocity.float() - target.float()).square()
        loss = (mse * weights).sum() if weights is not None else mse.mean()
    loss.backward()
    for hook in hooks:
        hook.remove()

    vel_np = velocity.detach().float().cpu().numpy()[0].T.copy().reshape(-1)
    vel_np.tofile(out_dir / "velocity_pytorch.f32")
    grad_metrics = []
    for stem in sorted(layers):
        layer = layers[stem]
        for attr, suffix in (("lora_A", "gA"), ("lora_B", "gB"), ("magnitude", "gmag")):
            param = getattr(layer, attr, None)
            if param is None or param.grad is None:
                continue
            py = param.grad.detach().float().cpu().numpy().copy().reshape(-1)
            py.tofile(grad_dir / f"{stem}.{suffix}.f32")
            if cpp_grads is not None:
                cpp_path = cpp_grads / f"{stem}.{suffix}.f32"
                if not cpp_path.exists():
                    raise RuntimeError(f"missing C++ gradient {cpp_path.name}")
                cpp = np.fromfile(cpp_path, np.float32)
                if cpp.shape != py.shape:
                    raise RuntimeError(f"gradient shape mismatch {stem}.{suffix}: {cpp.shape} vs {py.shape}")
                denom = max(np.linalg.norm(cpp), 1e-30)
                grad_metrics.append({
                    "name": f"{stem}.{suffix}",
                    "cos": cosine(cpp, py),
                    "rel_l2": float(np.linalg.norm(py - cpp) / denom),
                    "cpp_norm": float(np.linalg.norm(cpp)),
                    "py_norm": float(np.linalg.norm(py)),
                    "max_abs": float(np.max(np.abs(py - cpp))),
                })

    vel_cpp = read_f32(bundle, "velocity_cpp", frames * io)
    vel_cos = cosine(vel_cpp, vel_np)
    vel_rel = float(np.linalg.norm(vel_np - vel_cpp) / max(np.linalg.norm(vel_cpp), 1e-30))
    activation_metrics = []
    for name, py in activations.items():
        py.tofile(out_dir / f"{name}_pytorch.f32")
        cpp_path = bundle / f"{name}_cpp.f32"
        if not cpp_path.exists():
            continue
        cpp = np.fromfile(cpp_path, np.float32)
        if cpp.shape != py.shape:
            raise RuntimeError(f"activation shape mismatch {name}: {cpp.shape} vs {py.shape}")
        activation_metrics.append({
            "name": name,
            "cos": cosine(cpp, py),
            "rel_l2": float(np.linalg.norm(py - cpp) / max(np.linalg.norm(cpp), 1e-30)),
        })

    report = {
        "loss_cpp": float(meta["loss_cpp"]),
        "loss_pytorch": float(loss.detach().cpu()),
        "velocity_cos": vel_cos,
        "velocity_rel_l2": vel_rel,
        "padding_mask": args.padding_mask,
        "gradient_count": len(grad_metrics),
        "gradient_cos_median": float(np.median([x["cos"] for x in grad_metrics])) if grad_metrics else None,
        "gradient_cos_min": float(np.min([x["cos"] for x in grad_metrics])) if grad_metrics else None,
        "gradient_rel_l2_median": float(np.median([x["rel_l2"] for x in grad_metrics])) if grad_metrics else None,
        "activations": sorted(activation_metrics, key=lambda x: int(x["name"].split("_")[1]) if x["name"].startswith("block_") else -1),
        "worst_cos": sorted(grad_metrics, key=lambda x: x["cos"])[:25],
        "worst_rel_l2": sorted(grad_metrics, key=lambda x: x["rel_l2"], reverse=True)[:25],
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if not k.startswith("worst_")}, indent=2))
    print("worst gradient cosines:")
    for row in report["worst_cos"][:12]:
        print(f"  {row['cos']:.7f} rel={row['rel_l2']:.4g} {row['name']}")
    if activation_metrics:
        print("activation boundary cosines:")
        for row in report["activations"]:
            print(f"  {row['cos']:.7f} rel={row['rel_l2']:.4g} {row['name']}")


if __name__ == "__main__":
    main()
