#!/usr/bin/env python3
"""Standard-library tests for published model naming and download manifests."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from model_artifacts import build_download_plan, dit_filename, dit_identity
from stage_training_base_repos import notice_text


class ModelArtifactsTest(unittest.TestCase):
    def test_training_base_identity_is_unambiguous(self):
        identity = dit_identity("medium", training_base=True)
        self.assertEqual(identity["basename"], "stable-audio-3-medium-base-dit")
        self.assertEqual(identity["name"], "stable-audio-3-medium-base DiT")
        self.assertEqual(identity["finetune"], "medium-base")
        self.assertEqual(identity["upstream_revision"], "b32993f73c3bdc3864043a72d8032606bba737c8")
        self.assertEqual(
            dit_filename("small-music", "f16", training_base=True),
            "stable-audio-3-small-music-base-dit-0.5B-v1.0-F16.gguf",
        )

    def test_inference_plan_is_unchanged(self):
        plan = build_download_plan("thepatch", "medium", "f16")
        self.assertEqual(len(plan), 2)
        self.assertEqual(plan[0][0], "thepatch/stable-audio-3-medium-GGUF")
        self.assertIn("stable-audio-3-medium-dit-1.5B-v1.0-F16.gguf", plan[0][1])
        self.assertEqual(plan[-1][0], "thepatch/t5gemma-b-b-ul2-GGUF")

    def test_training_plan_adds_base_dit(self):
        plan = build_download_plan("thepatch", "small-sfx", "f32", training_base=True)
        self.assertEqual(len(plan), 3)
        self.assertEqual(plan[1][0], "thepatch/stable-audio-3-small-sfx-base-GGUF")
        self.assertEqual(
            plan[1][1],
            ["stable-audio-3-small-sfx-base-dit-0.5B-v1.0-F32.gguf"],
        )

    def test_notice_retains_required_attribution(self):
        notice = notice_text("medium")
        self.assertIn(
            "This Stability AI Model is licensed under the Stability AI Community License, "
            "Copyright © Stability AI Ltd. All Rights Reserved",
            notice,
        )
        self.assertIn("Powered by Stability AI", notice)
        self.assertIn("The model was not retrained.", notice)


if __name__ == "__main__":
    unittest.main()
