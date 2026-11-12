"""Packaging boundary checks; no model, compiler, or GPU required."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

SPEC = importlib.util.spec_from_file_location("package_linux", Path(__file__).parents[1] / "ci/package-linux.py")
PACKAGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACKAGE)


class LinuxPackageTests(unittest.TestCase):
    def test_build_ownership_rejects_source_and_existing_user_data(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            with patch.object(PACKAGE, "ROOT", root):
                for value in (".", "../outside", "src/build", "ggml/build"):
                    with self.subTest(value=value), self.assertRaises(ValueError):
                        PACKAGE.owned_directory(value, "build")
                user = root / "user-data"
                user.mkdir()
                (user / "keep").write_text("preserve me")
                with self.assertRaises(ValueError):
                    PACKAGE.owned_directory("user-data", "build")
                self.assertEqual((user / "keep").read_text(), "preserve me")
                owned = PACKAGE.owned_directory("build-dist-linux", "build")
                (owned / "cached-object").write_text("cache")
                self.assertEqual(PACKAGE.owned_directory("build-dist-linux", "build"), owned)

    def test_archive_rejects_escape_before_extracting(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            archive = root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as writer:
                writer.writestr("../outside", "must not escape")
            into = root / "into"
            into.mkdir()
            with self.assertRaises(ValueError):
                PACKAGE.unpack(archive, into)
            self.assertFalse((root / "outside").exists())


if __name__ == "__main__":
    unittest.main()
