#!/usr/bin/env python3
"""Synthetic coverage and numerical tests for convert_sat_oobleck.py."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

DEPENDENCY_ERROR = None
try:
    import numpy as np
    from gguf import GGUFReader
    from safetensors.numpy import save_file
    from convert_sat_oobleck import ConversionError, convert, fuse_weight_norm
except ImportError as exc:  # Clean source builds do not require converter dependencies.
    DEPENDENCY_ERROR = str(exc)


def tiny_config():
    return {
        "model_type": "diffusion_cond",
        "sample_rate": 44100,
        "sample_size": 128,
        "model": {
            "pretransform": {
                "type": "autoencoder",
                "config": {
                    "encoder": {"type": "oobleck", "config": {
                        "in_channels": 2, "channels": 2, "c_mults": [1, 2],
                        "strides": [2, 4], "latent_dim": 4, "use_snake": True,
                    }},
                    "decoder": {"type": "oobleck", "config": {
                        "out_channels": 2, "channels": 2, "c_mults": [1, 2],
                        "strides": [2, 4], "latent_dim": 2, "use_snake": True,
                        "final_tanh": False,
                    }},
                    "bottleneck": {"type": "vae"},
                    "latent_dim": 2,
                    "downsampling_ratio": 8,
                    "io_channels": 2,
                },
            }
        },
    }


def add_wn(sd, stem, out_or_in, other, kernel, transpose=False, scale=1.0):
    shape = (out_or_in, other, kernel)
    v = np.arange(1, np.prod(shape) + 1, dtype=np.float32).reshape(shape)
    v *= scale
    g = np.linspace(0.5, 1.5, out_or_in, dtype=np.float32).reshape(out_or_in, 1, 1)
    sd[stem + ".weight_g"] = g
    sd[stem + ".weight_v"] = v
    return g, v


def add_snake(sd, stem, channels):
    sd[stem + ".alpha"] = np.linspace(-0.2, 0.2, channels, dtype=np.float32)
    sd[stem + ".beta"] = np.linspace(-0.3, 0.3, channels, dtype=np.float32)


def synthetic_state(prefix="pretransform.model."):
    sd = {}
    add_wn(sd, prefix + "decoder.layers.0", 4, 2, 7)
    sd[prefix + "decoder.layers.0.bias"] = np.arange(4, dtype=np.float32)
    channels = [(4, 2, 4), (2, 2, 2)]  # in, out, stride
    for block, (inc, outc, stride) in enumerate(channels):
        layer = block + 1
        p = prefix + f"decoder.layers.{layer}.layers."
        add_snake(sd, p + "0", inc)
        add_wn(sd, p + "1", inc, outc, 2 * stride, transpose=True,
               scale=block + 1)
        sd[p + "1.bias"] = np.arange(outc, dtype=np.float32)
        for residual in range(3):
            rp = p + f"{residual + 2}.layers."
            add_snake(sd, rp + "0", outc)
            add_wn(sd, rp + "1", outc, outc, 7)
            sd[rp + "1.bias"] = np.arange(outc, dtype=np.float32)
            add_snake(sd, rp + "2", outc)
            add_wn(sd, rp + "3", outc, outc, 1)
            sd[rp + "3.bias"] = np.arange(outc, dtype=np.float32)
    add_snake(sd, prefix + "decoder.layers.3", 2)
    add_wn(sd, prefix + "decoder.layers.4", 2, 2, 7)
    return sd


@unittest.skipIf(DEPENDENCY_ERROR is not None,
                 f"converter dependencies unavailable: {DEPENDENCY_ERROR}")
class SatOobleckConverterTest(unittest.TestCase):
    def convert_fixture(self, sd, weight_type="f32"):
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name)
        src, config, out = root / "model.safetensors", root / "model.json", root / "ae.gguf"
        save_file(sd, str(src))
        config.write_text(json.dumps(tiny_config()), encoding="utf-8")
        result = convert(src, config, out, weight_type=weight_type)
        return temp, result, GGUFReader(str(out))

    def test_complete_conversion_and_transforms(self):
        sd = synthetic_state()
        temp, result, reader = self.convert_fixture(sd)
        self.addCleanup(temp.cleanup)
        tensors = {str(t.name): np.array(t.data) for t in reader.tensors}
        self.assertEqual(result["source_prefix"], "pretransform.model.")
        self.assertEqual(result["tensor_count"], 61)
        self.assertEqual(set(tensors), {
            "ae.decoder.in.weight", "ae.decoder.in.bias",
            "ae.decoder.out_snake.alpha_exp", "ae.decoder.out_snake.beta_recip",
            "ae.decoder.out.weight",
            *{
                f"ae.decoder.blocks.{b}.{suffix}"
                for b in range(2)
                for suffix in (
                    "snake.alpha_exp", "snake.beta_recip", "up.weight_col2im", "up.bias",
                    *tuple(
                        f"res.{r}.{leaf}"
                        for r in range(3)
                        for leaf in (
                            "snake1.alpha_exp", "snake1.beta_recip", "conv1.weight", "conv1.bias",
                            "snake2.alpha_exp", "snake2.beta_recip", "conv2.weight", "conv2.bias",
                        )
                    ),
                )
            },
        })

        prefix = "pretransform.model.decoder.layers.1.layers."
        expected = fuse_weight_norm(sd[prefix + "1.weight_g"], sd[prefix + "1.weight_v"])
        expected = expected.reshape(expected.shape[0], -1).T
        np.testing.assert_allclose(tensors["ae.decoder.blocks.0.up.weight_col2im"], expected)
        np.testing.assert_allclose(
            tensors["ae.decoder.blocks.0.snake.alpha_exp"],
            np.exp(sd[prefix + "0.alpha"]),
        )
        np.testing.assert_allclose(
            tensors["ae.decoder.blocks.0.snake.beta_recip"],
            np.exp(-sd[prefix + "0.beta"]),
        )

    def test_default_f16_keeps_small_state_f32(self):
        temp, _, reader = self.convert_fixture(synthetic_state(), weight_type="f16")
        self.addCleanup(temp.cleanup)
        tensors = {str(t.name): np.array(t.data) for t in reader.tensors}
        self.assertEqual(tensors["ae.decoder.in.weight"].dtype, np.float16)
        self.assertEqual(tensors["ae.decoder.blocks.0.up.weight_col2im"].dtype, np.float16)
        self.assertEqual(tensors["ae.decoder.in.bias"].dtype, np.float32)
        self.assertEqual(tensors["ae.decoder.blocks.0.snake.alpha_exp"].dtype, np.float32)

    def test_missing_tensor_is_fatal(self):
        sd = synthetic_state()
        del sd["pretransform.model.decoder.layers.2.layers.1.bias"]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src, config, out = root / "model.safetensors", root / "model.json", root / "ae.gguf"
            save_file(sd, str(src))
            config.write_text(json.dumps(tiny_config()), encoding="utf-8")
            with self.assertRaisesRegex(ConversionError, "missing decoder tensor"):
                convert(src, config, out)

    def test_unknown_decoder_tensor_is_fatal(self):
        sd = synthetic_state()
        sd["pretransform.model.decoder.surprise"] = np.ones(1, dtype=np.float32)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src, config, out = root / "model.safetensors", root / "model.json", root / "ae.gguf"
            save_file(sd, str(src))
            config.write_text(json.dumps(tiny_config()), encoding="utf-8")
            with self.assertRaisesRegex(ConversionError, "unrecognized decoder tensors"):
                convert(src, config, out)


if __name__ == "__main__":
    unittest.main()
