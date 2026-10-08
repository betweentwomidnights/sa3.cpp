"""Run with a PyTorch + safetensors environment; no models or GPU needed."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from lora_ckpt_export import export_checkpoint


def write_marker(path):
    Path(path).write_bytes(b"pickle executed")
    return {}


class PicklePayload:
    def __init__(self, path):
        self.path = str(path)

    def __reduce__(self):
        return write_marker, (self.path,)


class ExportTests(unittest.TestCase):
    def test_export_preserves_original_config_shared_tensors_and_integer_dtype(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "original.ckpt"
            shared = torch.arange(4, dtype=torch.float16)
            config = {"adapter_type": "dora", "rank": 16, "target": "decoder", "label": "koan ♭"}
            torch.save({"state_dict": {"a": shared, "b": shared, "index": torch.tensor([2])},
                        "lora_config": config}, source)
            original = source.read_bytes()
            count, actual = export_checkpoint(source, root / "copy")
            self.assertEqual((count, actual), (3, config))
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(json.loads((root / "copy.json").read_bytes()), config)
            with safe_open(root / "copy.safetensors", framework="pt", device="cpu") as tensors:
                self.assertEqual(json.loads(tensors.metadata()["lora_config"]), config)
                self.assertEqual(tensors.get_tensor("a").dtype, torch.float32)
                self.assertEqual(tensors.get_tensor("index").dtype, torch.int64)
                torch.testing.assert_close(tensors.get_tensor("a"), shared.float())
                torch.testing.assert_close(tensors.get_tensor("a"), tensors.get_tensor("b"))
            with self.assertRaises(FileExistsError):
                export_checkpoint(source, root / "copy")
            self.assertEqual(source.read_bytes(), original)

    def test_existing_sidecar_is_preserved_and_own_partial_tensor_removed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "original.ckpt"
            torch.save({"state_dict": {"a": torch.ones(1)}, "lora_config": {"rank": 16}}, source)
            (root / "copy.json").write_bytes(b"curated")
            with self.assertRaises(FileExistsError):
                export_checkpoint(source, root / "copy")
            self.assertEqual((root / "copy.json").read_bytes(), b"curated")
            self.assertFalse((root / "copy.safetensors").exists())
            self.assertFalse(list(root.glob(".sa3-export-*")))

    def test_rejects_pickle_execution_and_invalid_config_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "unsafe.ckpt"
            torch.save(PicklePayload(root / "marker"), source)
            with self.assertRaises(Exception):
                export_checkpoint(source, root / "copy")
            self.assertFalse((root / "marker").exists())
            for data in [{}, {"state_dict": {"a": "not a tensor"}, "lora_config": {"rank": 16}},
                         {"state_dict": {"a": torch.ones(1)}, "lora_config": {"rank": float("nan")}}]:
                torch.save(data, source)
                with self.assertRaises(ValueError):
                    export_checkpoint(source, root / "copy")
                self.assertFalse((root / "copy.safetensors").exists())
                self.assertFalse((root / "copy.json").exists())


if __name__ == "__main__":
    unittest.main()
