#!/usr/bin/env python3
"""Synthetic topology/coverage tests for convert_sat_dit.py."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
DEPENDENCY_ERROR = None
try:
    import numpy as np
    from gguf import GGUFReader
    from safetensors.numpy import save_file
    from convert_sat_dit import ConversionError, convert
except ImportError as exc:
    DEPENDENCY_ERROR = str(exc)


def config(qk_norm="ln", seconds_start=False, objective="rf_denoiser"):
    conditioners = [{"id": "seconds_total", "type": "number",
                     "config": {"min_val": 0, "max_val": 256}}]
    if seconds_start:
        conditioners.insert(0, {"id": "seconds_start", "type": "number",
                                "config": {"min_val": 0, "max_val": 512}})
    dit_config = {"io_channels": 4, "embed_dim": 64, "depth": 1, "num_heads": 1,
                  "cond_token_dim": 8, "global_cond_dim": 16 if seconds_start else 8,
                  "transformer_type": "continuous_transformer"}
    if qk_norm is not None:
        dit_config["attn_kwargs"] = {"qk_norm": qk_norm}
    return {"model_type": "diffusion_cond", "sample_rate": 44100, "sample_size": 128,
            "model": {"conditioning": {"configs": conditioners},
            "diffusion": {"type": "dit", "diffusion_objective": objective,
            "config": dit_config}}}


def state(qk_norm=True, seconds_start=False):
    p = "model.model."
    sd = {}
    def add(name, shape):
        n = int(np.prod(shape))
        sd[p + name] = np.linspace(-0.1, 0.1, n, dtype=np.float32).reshape(shape)
    add("preprocess_conv.weight", (4, 4, 1)); add("postprocess_conv.weight", (4, 4, 1))
    add("timestep_features.weight", (128, 1))
    for stem, inp in (("to_timestep_embed", 256), ("to_cond_embed", 8), ("to_global_embed", 8)):
        add(stem + ".0.weight", (64, inp)); add(stem + ".2.weight", (64, 64))
        if stem == "to_timestep_embed":
            add(stem + ".0.bias", (64,)); add(stem + ".2.bias", (64,))
    add("transformer.project_in.weight", (64, 4)); add("transformer.project_out.weight", (4, 64))
    sd[p + "transformer.rotary_pos_emb.inv_freq"] = (
        1.0 / (10000.0 ** (np.arange(0, 32, 2, dtype=np.float32) / 32)))
    b = "transformer.layers.0."
    for norm in ("pre_norm", "cross_attend_norm", "ff_norm"):
        add(b + norm + ".gamma", (64,)); add(b + norm + ".beta", (64,))
    add(b + "self_attn.to_qkv.weight", (192, 64)); add(b + "self_attn.to_out.weight", (64, 64))
    add(b + "cross_attn.to_q.weight", (64, 64)); add(b + "cross_attn.to_kv.weight", (128, 64))
    add(b + "cross_attn.to_out.weight", (64, 64))
    if qk_norm:
        for attn in ("self_attn", "cross_attn"):
            for qk in ("q", "k"):
                add(b + f"{attn}.{qk}_norm.weight", (64,)); add(b + f"{attn}.{qk}_norm.bias", (64,))
    add(b + "ff.ff.0.proj.weight", (512, 64)); add(b + "ff.ff.0.proj.bias", (512,))
    add(b + "ff.ff.2.weight", (64, 256)); add(b + "ff.ff.2.bias", (64,))
    sec = "conditioner.conditioners.seconds_total.embedder.embedding."
    sd[sec + "0.weights"] = np.linspace(-1, 1, 128, dtype=np.float32)
    sd[sec + "1.weight"] = np.zeros((8, 257), dtype=np.float32)
    sd[sec + "1.bias"] = np.zeros(8, dtype=np.float32)
    if seconds_start:
        start = "conditioner.conditioners.seconds_start.embedder.embedding."
        sd[start + "0.weights"] = np.linspace(-1, 1, 128, dtype=np.float32)
        sd[start + "1.weight"] = np.zeros((8, 257), dtype=np.float32)
        sd[start + "1.bias"] = np.zeros(8, dtype=np.float32)
    return sd


@unittest.skipIf(DEPENDENCY_ERROR is not None, f"converter dependencies unavailable: {DEPENDENCY_ERROR}")
class SatDitConverterTest(unittest.TestCase):
    def fixture(self, sd, out_type="f16", config_data=None):
        temp = tempfile.TemporaryDirectory(); root = Path(temp.name)
        src, cfg_path, out = root / "m.safetensors", root / "m.json", root / "m.gguf"
        save_file(sd, str(src))
        cfg_path.write_text(json.dumps(config() if config_data is None else config_data),
                            encoding="utf-8")
        try:
            result = convert(src, cfg_path, out, weight_type=out_type)
        except Exception:
            temp.cleanup()
            raise
        return temp, result, GGUFReader(str(out))

    def test_complete_conversion_and_types(self):
        temp, result, reader = self.fixture(state())
        self.addCleanup(temp.cleanup)
        tensors = {str(t.name): np.array(t.data) for t in reader.tensors}
        self.assertEqual(result["tensor_count"], 39)
        self.assertEqual(len(tensors), 39)
        self.assertEqual(tensors["dit.blocks.0.self.qkv.weight"].dtype, np.float16)
        self.assertEqual(tensors["dit.blocks.0.self.q_norm.weight"].dtype, np.float32)
        self.assertEqual(tensors["dit.time.0.bias"].dtype, np.float32)
        self.assertEqual(tensors["dit.pre.weight"].shape, (4, 4))

    def test_missing_and_unknown_are_fatal(self):
        sd = state(); del sd["model.model.transformer.layers.0.ff.ff.2.bias"]
        with self.assertRaisesRegex(ConversionError, "missing DiT tensor"):
            temp, _, _ = self.fixture(sd); temp.cleanup()

    def test_sao1_topology_without_qk_norm_and_with_both_timing_conditions(self):
        temp, result, reader = self.fixture(
            state(qk_norm=False, seconds_start=True),
            config_data=config(qk_norm=None, seconds_start=True, objective="v"))
        self.addCleanup(temp.cleanup)
        tensors = {str(t.name): np.array(t.data) for t in reader.tensors}
        self.assertEqual(result["spec"]["objective"], "v")
        self.assertEqual(result["spec"]["qk_norm"], "none")
        self.assertEqual(result["tensor_count"], 34)
        self.assertIn("conditioner.seconds_start.linear.weight", tensors)
        self.assertNotIn("dit.blocks.0.self.q_norm.weight", tensors)
        sd = state(); sd["model.model.surprise"] = np.ones(1, np.float32)
        with self.assertRaisesRegex(ConversionError, "unrecognized DiT tensors"):
            temp, _, _ = self.fixture(sd); temp.cleanup()

    def test_lightning_checkpoint_state_dict(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is unavailable")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src, cfg, out = root / "m.ckpt", root / "m.json", root / "m.gguf"
            torch.save({"state_dict": {k: torch.from_numpy(v) for k, v in state().items()}}, src)
            cfg.write_text(json.dumps(config()), encoding="utf-8")
            result = convert(src, cfg, out)
            self.assertEqual(result["tensor_count"], 39)
            self.assertEqual(len(GGUFReader(str(out)).tensors), 39)

    def test_lightning_ema_checkpoint(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is unavailable")
        wrapped = {}
        for key, value in state().items():
            if key.startswith("model.model."):
                key = "diffusion_ema.ema_model." + key[len("model."):]
            else:
                key = "diffusion." + key
            wrapped[key] = torch.from_numpy(value)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src, cfg, out = root / "ema.ckpt", root / "m.json", root / "m.gguf"
            torch.save({"state_dict": wrapped}, src)
            cfg.write_text(json.dumps(config()), encoding="utf-8")
            result = convert(src, cfg, out)
            self.assertEqual(result["tensor_count"], 39)


if __name__ == "__main__":
    unittest.main()
