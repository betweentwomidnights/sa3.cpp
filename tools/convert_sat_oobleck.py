#!/usr/bin/env python3
"""Convert a stable-audio-tools Oobleck decoder from safetensors to GGUF.

The source may be a complete Stable Audio checkpoint (the usual case) or a
standalone autoencoder state dict. Conversion is lazy and never imports Torch.

Weight normalization is fused as PyTorch applies it at inference. Transposed
convolutions are additionally rearranged for the GEMM + ggml_col2im_1d graph in
src/sat/oobleck.h. Unknown or missing decoder tensors are fatal: silently emitting
a partial decoder produces plausible-looking GGUF files which fail much later.

Usage:
  .venv/Scripts/python.exe tools/convert_sat_oobleck.py \
      --src model.safetensors --config model_config.json \
      --out models/stable-audio-open-small-oobleck-decoder-F16.gguf
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFWriter
from safetensors import safe_open

import gguf_meta


class ConversionError(ValueError):
    pass


PREFIX_CANDIDATES = (
    "pretransform.model.",
    "model.pretransform.model.",
    "",  # standalone AudioAutoencoder state dict
)


def fuse_weight_norm(g, v):
    """PyTorch legacy weight_norm(dim=0): g * v / norm(v, dims=1..N)."""
    if v.ndim < 2 or g.shape[0] != v.shape[0]:
        raise ConversionError(f"invalid weight_norm shapes: g={g.shape}, v={v.shape}")
    axes = tuple(range(1, v.ndim))
    vf = v.astype(np.float64)
    norm = np.sqrt(np.sum(vf * vf, axis=axes, keepdims=True))
    if np.any(norm == 0):
        raise ConversionError("cannot fuse a zero-norm weight_v slice")
    try:
        fused = g.astype(np.float64) * vf / norm
    except ValueError as exc:
        raise ConversionError(
            f"weight_g does not broadcast over weight_v: g={g.shape}, v={v.shape}"
        ) from exc
    return np.ascontiguousarray(fused.astype(np.float32))


def conv_transpose_for_col2im(fused):
    """[in, out, kernel] -> numpy [out*kernel, in] -> GGML [in, kernel*out]."""
    if fused.ndim != 3:
        raise ConversionError(f"ConvTranspose1d weight must be rank 3, got {fused.shape}")
    return np.ascontiguousarray(fused.reshape(fused.shape[0], -1).T)


def find_source_prefix(keys):
    keyset = set(keys)
    marker = "decoder.layers.0.weight_v"
    found = [p for p in PREFIX_CANDIDATES if p + marker in keyset]
    if len(found) != 1:
        where = ", ".join(repr(p) for p in found) if found else "none"
        raise ConversionError(
            f"could not uniquely locate the Oobleck decoder (matching prefixes: {where})"
        )
    return found[0]


def read_oobleck_config(config_path):
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    try:
        pre = cfg["model"]["pretransform"]
        ae = pre["config"]
        dec_entry = ae["decoder"]
        dec = dec_entry["config"]
        enc = ae["encoder"]["config"]
    except (KeyError, TypeError) as exc:
        raise ConversionError(f"{config_path}: not a Stable Audio autoencoder config") from exc
    if pre.get("type") != "autoencoder" or dec_entry.get("type") != "oobleck":
        raise ConversionError("pretransform decoder is not Oobleck")

    c_mults = [int(v) for v in dec["c_mults"]]
    strides = [int(v) for v in dec["strides"]]
    if not c_mults or len(c_mults) != len(strides):
        raise ConversionError("Oobleck c_mults and strides must have the same non-zero length")
    ratio = int(np.prod(strides, dtype=np.int64))
    declared_ratio = int(ae["downsampling_ratio"])
    if ratio != declared_ratio:
        raise ConversionError(
            f"Oobleck stride product {ratio} disagrees with downsampling_ratio {declared_ratio}"
        )
    latent = int(ae["latent_dim"])
    if int(dec["latent_dim"]) != latent:
        raise ConversionError("decoder latent_dim disagrees with autoencoder latent_dim")

    return cfg, {
        "sample_rate": int(cfg["sample_rate"]),
        "sample_size": int(cfg["sample_size"]),
        "audio_channels": int(ae["io_channels"]),
        "base_channels": int(dec["channels"]),
        "channel_multipliers": c_mults,
        "strides": strides,
        "encoder_out_channels": int(enc["latent_dim"]),
        "latent_channels": latent,
        "use_snake": bool(dec.get("use_snake", False)),
        "final_tanh": bool(dec.get("final_tanh", True)),
        "downsampling_ratio": declared_ratio,
    }


def convert(src, config, out, model_id="stable-audio-open-small", weight_type="f16"):
    _, spec = read_oobleck_config(config)
    if not spec["use_snake"]:
        raise ConversionError("the current GGML Oobleck graph requires Snake activations")
    if weight_type not in ("f16", "f32"):
        raise ConversionError("weight_type must be f16 or f32")
    weight_dtype = np.float16 if weight_type == "f16" else np.float32

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    consumed = set()
    tensor_count = 0
    parameter_count = 0

    writer = GGUFWriter(str(out), arch="sat-oobleck")
    writer.add_string("sat.architecture", "stable-audio-tools")
    writer.add_uint32("sat.format_version", 1)
    writer.add_string("sat.model_id", model_id)
    writer.add_uint32("sat.sample_rate", spec["sample_rate"])
    writer.add_uint32("sat.sample_size", spec["sample_size"])
    writer.add_uint32("sat.ae.audio_channels", spec["audio_channels"])
    writer.add_uint32("sat.ae.base_channels", spec["base_channels"])
    writer.add_array("sat.ae.channel_multipliers", spec["channel_multipliers"])
    writer.add_array("sat.ae.encoder_strides", spec["strides"])
    writer.add_uint32("sat.ae.encoder_out_channels", spec["encoder_out_channels"])
    writer.add_uint32("sat.ae.latent_channels", spec["latent_channels"])
    writer.add_uint32("sat.ae.downsampling_ratio", spec["downsampling_ratio"])
    writer.add_bool("sat.ae.use_snake", spec["use_snake"])
    writer.add_bool("sat.ae.final_tanh", spec["final_tanh"])
    writer.add_string("sat.ae.weight_type", weight_type.upper())

    with safe_open(str(src), framework="numpy") as f:
        keys = list(f.keys())
        prefix = find_source_prefix(keys)

        def source(name):
            key = prefix + name
            if key not in keys:
                raise ConversionError(f"missing decoder tensor: {key}")
            consumed.add(key)
            return f.get_tensor(key)

        def emit(name, array, weights=False):
            nonlocal tensor_count, parameter_count
            a = np.ascontiguousarray(array.astype(weight_dtype if weights else np.float32))
            writer.add_tensor(name, a)
            tensor_count += 1
            parameter_count += int(a.size)

        def emit_bias(src_stem, dst):
            emit(dst, source(src_stem + ".bias").reshape(-1))

        def emit_snake(src_stem, dst_stem):
            alpha = source(src_stem + ".alpha").astype(np.float32).reshape(-1)
            beta = source(src_stem + ".beta").astype(np.float32).reshape(-1)
            emit(dst_stem + ".alpha_exp", np.exp(alpha))
            emit(dst_stem + ".beta_recip", np.exp(-beta))

        def emit_weight_norm(src_stem, dst, transpose=False):
            fused = fuse_weight_norm(
                source(src_stem + ".weight_g"),
                source(src_stem + ".weight_v"),
            )
            if transpose:
                fused = conv_transpose_for_col2im(fused)
            emit(dst, fused, weights=True)

        emit_weight_norm("decoder.layers.0", "ae.decoder.in.weight")
        emit_bias("decoder.layers.0", "ae.decoder.in.bias")

        stages = len(spec["strides"])
        for block in range(stages):
            layer = block + 1
            sp = f"decoder.layers.{layer}.layers."
            dp = f"ae.decoder.blocks.{block}."
            emit_snake(sp + "0", dp + "snake")
            emit_weight_norm(sp + "1", dp + "up.weight_col2im", transpose=True)
            emit_bias(sp + "1", dp + "up.bias")
            for residual in range(3):
                sr = sp + f"{residual + 2}.layers."
                dr = dp + f"res.{residual}."
                emit_snake(sr + "0", dr + "snake1")
                emit_weight_norm(sr + "1", dr + "conv1.weight")
                emit_bias(sr + "1", dr + "conv1.bias")
                emit_snake(sr + "2", dr + "snake2")
                emit_weight_norm(sr + "3", dr + "conv2.weight")
                emit_bias(sr + "3", dr + "conv2.bias")

        final_snake_layer = stages + 1
        final_conv_layer = stages + 2
        emit_snake(f"decoder.layers.{final_snake_layer}", "ae.decoder.out_snake")
        emit_weight_norm(f"decoder.layers.{final_conv_layer}", "ae.decoder.out.weight")

        decoder_keys = {k for k in keys if k.startswith(prefix + "decoder.")}
        unexpected = sorted(decoder_keys - consumed)
        if unexpected:
            preview = ", ".join(unexpected[:5])
            suffix = f" (+{len(unexpected) - 5} more)" if len(unexpected) > 5 else ""
            raise ConversionError(f"unrecognized decoder tensors: {preview}{suffix}")

    writer.add_uint64("sat.ae.parameter_count", parameter_count)
    gguf_meta.add_general(
        writer,
        basename=f"{model_id}-oobleck",
        name=f"{model_id} Oobleck decoder",
        finetune=model_id,
        license_id="stabilityai-community",
    )
    gguf_meta.add_source(
        writer, "Stable Audio Open Small", "Stability AI",
        "https://huggingface.co/stabilityai/stable-audio-open-small",
        "dc620d91535857b72ebb59b4ca45978db6d417f5", "model.safetensors")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return {
        "output": str(out),
        "source_prefix": prefix,
        "tensor_count": tensor_count,
        "parameter_count": parameter_count,
        "spec": spec,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Stable Audio .safetensors checkpoint")
    ap.add_argument("--config", required=True, help="Stable Audio model_config.json")
    ap.add_argument("--out", required=True, help="output decoder .gguf")
    ap.add_argument("--model-id", default="stable-audio-open-small")
    ap.add_argument("--out-type", default="f16", choices=("f16", "f32"))
    args = ap.parse_args()
    try:
        result = convert(args.src, args.config, args.out, args.model_id, args.out_type)
    except (ConversionError, OSError, KeyError, json.JSONDecodeError) as exc:
        sys.exit(f"error: {exc}")
    spec = result["spec"]
    print(
        f"wrote {result['output']} ({result['tensor_count']} tensors, "
        f"{result['parameter_count']:,} parameters, {args.out_type.upper()})"
    )
    print(
        f"  prefix={result['source_prefix']!r} latent={spec['latent_channels']} "
        f"ratio={spec['downsampling_ratio']}x strides={spec['strides']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
