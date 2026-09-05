#!/usr/bin/env python3
"""Convert the classic stable-audio-tools continuous DiT to GGUF.

This intentionally targets the small, well-defined architecture used by Stable
Audio Open 1.0 and Stable Audio Open Small.  The graph contract lives in
src/sat/dit.h; keeping this converter separate from SA3's MM-DiT converter avoids
model-family branches in either implementation.
"""

import argparse
from contextlib import contextmanager
import json
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFWriter
from safetensors import safe_open

import gguf_meta


class ConversionError(ValueError):
    pass


@contextmanager
def open_tensor_source(path):
    """Open safetensors or a PyTorch/Lightning checkpoint without executing pickle code."""
    path = Path(path)
    if path.suffix.lower() == ".safetensors":
        with safe_open(str(path), framework="numpy") as source:
            yield set(source.keys()), source.get_tensor
        return

    try:
        import torch
    except ImportError as exc:
        raise ConversionError("PyTorch is required to convert .ckpt files") from exc
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:  # mmap is unavailable on older torch versions
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise ConversionError("checkpoint does not contain a tensor state_dict")

    # Lightning training snapshots wrap the inference model under `diffusion.`.
    # Prefer EMA when present, matching Gary's stable-audio model loader.
    ema_prefix = "diffusion_ema.ema_model."
    diffusion_prefix = "diffusion."
    if any(k.startswith(ema_prefix) for k in state):
        tensors = {}
        for key, value in state.items():
            if key.startswith(ema_prefix):
                tail = key[len(ema_prefix):]
                # EMA owns the inner DiffusionTransformer, so its `model.*`
                # corresponds to the wrapper's `model.model.*` namespace.
                tensors[tail if tail.startswith("model.model.") else "model." + tail] = value
            elif key.startswith(diffusion_prefix + "conditioner."):
                # The frozen conditioner is not part of the EMA object.
                tensors[key[len(diffusion_prefix):]] = value
    elif any(k.startswith(diffusion_prefix) for k in state):
        tensors = {k[len(diffusion_prefix):]: v for k, v in state.items()
                   if k.startswith(diffusion_prefix)}
    else:
        tensors = state

    def get_tensor(name):
        value = tensors[name]
        if not hasattr(value, "detach"):
            return np.asarray(value)
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            value = value.float()
        return value.numpy()

    try:
        yield set(tensors), get_tensor
    finally:
        del tensors, state, checkpoint


def read_spec(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        d = cfg["model"]["diffusion"]
        c = d["config"]
        seconds = next(x["config"] for x in cfg["model"]["conditioning"]["configs"]
                       if x["id"] == "seconds_total")
    except (KeyError, TypeError) as exc:
        raise ConversionError("not a stable-audio-tools conditional diffusion config") from exc
    if d.get("type") != "dit" or c.get("transformer_type") != "continuous_transformer":
        raise ConversionError("only the classic continuous_transformer DiT is supported")
    qk = c.get("attn_kwargs", {}).get("qk_norm")
    if qk != "ln":
        raise ConversionError(f"expected qk_norm='ln', got {qk!r}")
    return cfg, {
        "sample_rate": int(cfg["sample_rate"]),
        "sample_size": int(cfg["sample_size"]),
        "io_channels": int(c["io_channels"]),
        "width": int(c["embed_dim"]),
        "depth": int(c["depth"]),
        "heads": int(c["num_heads"]),
        "cond_dim": int(c["cond_token_dim"]),
        "global_dim": int(c["global_cond_dim"]),
        "objective": str(d["diffusion_objective"]),
        "qk_norm": qk,
        "seconds_min": float(seconds["min_val"]),
        "seconds_max": float(seconds["max_val"]),
    }


def convert(src, config, out, model_id="stable-audio-open-small", weight_type="f16"):
    _, spec = read_spec(config)
    if spec["width"] % spec["heads"]:
        raise ConversionError("DiT width must be divisible by the head count")
    if weight_type not in ("f16", "f32"):
        raise ConversionError("weight_type must be f16 or f32")
    wdtype = np.float16 if weight_type == "f16" else np.float32
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    writer = GGUFWriter(str(out), arch="sat-dit")
    gguf_meta.add_general(writer, basename=f"{model_id}-dit", name=f"{model_id} classic DiT",
                          finetune=model_id, license_id="stabilityai-community")
    writer.add_string("sat.architecture", "stable-audio-tools")
    writer.add_uint32("sat.format_version", 1)
    writer.add_string("sat.model_id", model_id)
    writer.add_uint32("sat.sample_rate", spec["sample_rate"])
    writer.add_uint32("sat.sample_size", spec["sample_size"])
    writer.add_uint32("sat.dit.io_channels", spec["io_channels"])
    writer.add_uint32("sat.dit.width", spec["width"])
    writer.add_uint32("sat.dit.depth", spec["depth"])
    writer.add_uint32("sat.dit.heads", spec["heads"])
    writer.add_uint32("sat.dit.cond_dim", spec["cond_dim"])
    writer.add_uint32("sat.dit.global_dim", spec["global_dim"])
    rotary_dims = max(spec["width"] // spec["heads"] // 2, 32)
    if rotary_dims > spec["width"] // spec["heads"]:
        raise ConversionError("rotary width exceeds the attention head width")
    writer.add_uint32("sat.dit.rotary_dims", rotary_dims)
    writer.add_string("sat.dit.qk_norm", spec["qk_norm"])
    writer.add_string("sat.diffusion_objective", spec["objective"])
    writer.add_string("sat.dit.weight_type", weight_type.upper())
    writer.add_float32("sat.conditioner.seconds_total.min", spec["seconds_min"])
    writer.add_float32("sat.conditioner.seconds_total.max", spec["seconds_max"])

    consumed = set()
    count = params = 0
    prefix = "model.model."
    with open_tensor_source(src) as (keys, get_tensor):

        def take(name):
            key = prefix + name
            if key not in keys:
                raise ConversionError(f"missing DiT tensor: {key}")
            consumed.add(key)
            return get_tensor(key)

        def emit(dst, src_name, weight=False, squeeze_kernel=False):
            nonlocal count, params
            a = take(src_name)
            if squeeze_kernel:
                if a.ndim != 3 or a.shape[-1] != 1:
                    raise ConversionError(f"expected a 1x1 Conv1d at {src_name}, got {a.shape}")
                a = a[..., 0]
            a = np.ascontiguousarray(a.astype(wdtype if weight else np.float32))
            writer.add_tensor(dst, a)
            count += 1
            params += int(a.size)

        def emit_full(dst, key, weight=False):
            nonlocal count, params
            if key not in keys:
                raise ConversionError(f"missing conditioner tensor: {key}")
            a = np.ascontiguousarray(get_tensor(key).astype(wdtype if weight else np.float32))
            writer.add_tensor(dst, a)
            count += 1
            params += int(a.size)

        emit("dit.pre.weight", "preprocess_conv.weight", True, True)
        emit("dit.post.weight", "postprocess_conv.weight", True, True)
        emit("dit.time_fourier.weight", "timestep_features.weight")
        for stem in ("to_timestep_embed", "to_cond_embed", "to_global_embed"):
            short = {"to_timestep_embed": "time", "to_cond_embed": "cond", "to_global_embed": "global"}[stem]
            for layer in (0, 2):
                emit(f"dit.{short}.{layer}.weight", f"{stem}.{layer}.weight", True)
                bias = f"{stem}.{layer}.bias"
                if prefix + bias in keys:
                    emit(f"dit.{short}.{layer}.bias", bias)
        emit("dit.in.weight", "transformer.project_in.weight", True)
        emit("dit.out.weight", "transformer.project_out.weight", True)

        # Rotary frequencies are deterministic, but consume and validate the source
        # tensor so an architecture drift cannot silently pass conversion.
        inv = take("transformer.rotary_pos_emb.inv_freq").astype(np.float32).reshape(-1)
        expected = 1.0 / (10000.0 ** (np.arange(0, rotary_dims, 2, dtype=np.float32) /
                                      rotary_dims))
        if inv.shape != expected.shape or not np.allclose(inv, expected, rtol=2e-5, atol=1e-7):
            raise ConversionError("unexpected rotary_pos_emb frequencies")

        for i in range(spec["depth"]):
            s = f"transformer.layers.{i}."
            d = f"dit.blocks.{i}."
            for norm, out_name in (("pre_norm", "pre_norm"),
                                   ("cross_attend_norm", "cross_norm"),
                                   ("ff_norm", "ff_norm")):
                emit(d + out_name + ".gamma", s + norm + ".gamma")
                emit(d + out_name + ".beta", s + norm + ".beta")

            emit(d + "self.qkv.weight", s + "self_attn.to_qkv.weight", True)
            emit(d + "self.out.weight", s + "self_attn.to_out.weight", True)
            emit(d + "cross.q.weight", s + "cross_attn.to_q.weight", True)
            emit(d + "cross.kv.weight", s + "cross_attn.to_kv.weight", True)
            emit(d + "cross.out.weight", s + "cross_attn.to_out.weight", True)
            for attn in ("self_attn", "cross_attn"):
                tag = "self" if attn == "self_attn" else "cross"
                for qk in ("q", "k"):
                    emit(d + f"{tag}.{qk}_norm.weight", s + f"{attn}.{qk}_norm.weight")
                    emit(d + f"{tag}.{qk}_norm.bias", s + f"{attn}.{qk}_norm.bias")
            emit(d + "ff.in.weight", s + "ff.ff.0.proj.weight", True)
            emit(d + "ff.in.bias", s + "ff.ff.0.proj.bias")
            emit(d + "ff.out.weight", s + "ff.ff.2.weight", True)
            emit(d + "ff.out.bias", s + "ff.ff.2.bias")

        sec = "conditioner.conditioners.seconds_total.embedder.embedding."
        emit_full("conditioner.seconds_total.fourier.weight", sec + "0.weights")
        emit_full("conditioner.seconds_total.linear.weight", sec + "1.weight")
        emit_full("conditioner.seconds_total.linear.bias", sec + "1.bias")

        model_keys = {k for k in keys if k.startswith(prefix)}
        unexpected = sorted(model_keys - consumed)
        if unexpected:
            raise ConversionError("unrecognized DiT tensors: " + ", ".join(unexpected[:8]))

    writer.add_uint64("sat.dit.parameter_count", params)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return {"output": str(out), "tensor_count": count, "parameter_count": params, "spec": spec}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-id", default="stable-audio-open-small")
    ap.add_argument("--out-type", choices=("f16", "f32"), default="f16")
    a = ap.parse_args()
    try:
        r = convert(a.src, a.config, a.out, a.model_id, a.out_type)
    except (ConversionError, OSError, KeyError, json.JSONDecodeError) as exc:
        sys.exit(f"error: {exc}")
    print(f"wrote {r['output']} ({r['tensor_count']} tensors, "
          f"{r['parameter_count']:,} parameters, {a.out_type.upper()})")


if __name__ == "__main__":
    main()
