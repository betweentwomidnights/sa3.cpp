#!/usr/bin/env python3
"""Synthetic coverage tests for convert_sat_t5.py."""

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
    from convert_sat_t5 import ConversionError, convert
except ImportError as exc:
    DEPENDENCY_ERROR = str(exc)


def fixture_state():
    sd = {"shared.weight": np.zeros((8, 4), np.float32),
          "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight": np.zeros((4, 2), np.float32),
          "encoder.block.0.layer.0.layer_norm.weight": np.ones(4, np.float32),
          "encoder.block.0.layer.1.layer_norm.weight": np.ones(4, np.float32),
          "encoder.block.0.layer.1.DenseReluDense.wi.weight": np.zeros((6, 4), np.float32),
          "encoder.block.0.layer.1.DenseReluDense.wo.weight": np.zeros((4, 6), np.float32),
          "encoder.final_layer_norm.weight": np.ones(4, np.float32)}
    for name in ("q", "k", "v", "o"):
        sd[f"encoder.block.0.layer.0.SelfAttention.{name}.weight"] = np.zeros((4, 4), np.float32)
    return sd


def config():
    return {"d_model": 4, "num_layers": 1, "num_heads": 2, "d_kv": 2, "d_ff": 6,
            "vocab_size": 8, "relative_attention_num_buckets": 4,
            "layer_norm_epsilon": 1e-6, "pad_token_id": 0, "eos_token_id": 1}


def tokenizer():
    return {"model": {"unk_id": 2, "vocab": [["<pad>", 0.0], ["</s>", 0.0],
            ["<unk>", 0.0], ["▁", -1.0], ["a", -2.0], ["▁a", -0.5]]},
            "added_tokens": [{"id": 0, "content": "<pad>"}, {"id": 1, "content": "</s>"}]}


@unittest.skipIf(DEPENDENCY_ERROR is not None, f"converter dependencies unavailable: {DEPENDENCY_ERROR}")
class SatT5ConverterTest(unittest.TestCase):
    def run_fixture(self, state):
        td = tempfile.TemporaryDirectory(); root = Path(td.name)
        src, cfg, tok, out = root/"m.safetensors", root/"config.json", root/"tokenizer.json", root/"m.gguf"
        save_file(state, str(src)); cfg.write_text(json.dumps(config()), encoding="utf-8")
        tok.write_text(json.dumps(tokenizer()), encoding="utf-8")
        try:
            result = convert(src, cfg, tok, out)
        except Exception:
            td.cleanup(); raise
        return td, result, GGUFReader(str(out))

    def test_complete_encoder_and_tokenizer(self):
        td, result, reader = self.run_fixture(fixture_state()); self.addCleanup(td.cleanup)
        tensors = {str(t.name): np.array(t.data) for t in reader.tensors}
        self.assertEqual(result["tensor_count"], 11)
        self.assertEqual(result["token_count"], 6)
        self.assertEqual(tensors["te.0.q.weight"].dtype, np.float16)
        self.assertEqual(tensors["te.0.attn_norm.weight"].dtype, np.float32)

    def test_unknown_encoder_tensor_is_fatal(self):
        sd = fixture_state(); sd["encoder.surprise"] = np.ones(1, np.float32)
        with self.assertRaisesRegex(ConversionError, "unrecognized encoder tensors"):
            self.run_fixture(sd)


if __name__ == "__main__":
    unittest.main()
