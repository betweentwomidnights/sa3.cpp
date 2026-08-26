#!/usr/bin/env python3
"""Standard-library tests for published model naming and download manifests."""

import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(REPO_ROOT / "tools"))

from model_artifacts import (TEXT_ENCODER_ENCODINGS, build_download_plan, dit_filename,
                             dit_identity, text_encoder_filename)
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

    def test_text_encoder_defaults_to_f16(self):
        # F16 is equivalent to F32 for this encoder at half the size, so it is what a plain
        # download gets; the shared repo's file list is the thing that decides it.
        plan = build_download_plan("thepatch", "medium", "f16")
        self.assertEqual(plan[-1][0], "thepatch/t5gemma-b-b-ul2-GGUF")
        self.assertIn("t5gemma-b-b-ul2-encoder-0.3B-v1.0-F16.gguf", plan[-1][1])
        self.assertIn("t5gemma-b-b-ul2-v1.0-vocab.gguf", plan[-1][1])

    def test_text_encoder_encoding_is_independent_of_the_dit(self):
        # The combination worth having on a small device: quantized DiT, F16 encoder.
        plan = build_download_plan("thepatch", "medium", "q4_k_m", text_encoding="f16")
        self.assertIn("stable-audio-3-medium-dit-1.5B-v1.0-Q4_K_M.gguf", plan[0][1])
        self.assertIn("t5gemma-b-b-ul2-encoder-0.3B-v1.0-F16.gguf", plan[-1][1])

    def test_every_published_text_encoder_encoding_resolves(self):
        for enc in TEXT_ENCODER_ENCODINGS:
            plan = build_download_plan("thepatch", "medium", "f16", text_encoding=enc)
            self.assertIn(f"t5gemma-b-b-ul2-encoder-0.3B-v1.0-{enc}.gguf", plan[-1][1])

    def test_q4_text_encoder_is_not_published(self):
        # Excluded on measurement: slower than F32 for the encoder AND far worse prompt fidelity.
        with self.assertRaises(ValueError) as ctx:
            text_encoder_filename("q4_k_m")
        self.assertIn("f16", str(ctx.exception))

    def test_training_plan_adds_base_dit(self):
        plan = build_download_plan("thepatch", "small-sfx", "f32", training_base=True)
        self.assertEqual(len(plan), 3)
        self.assertEqual(plan[1][0], "thepatch/stable-audio-3-small-sfx-base-GGUF")
        self.assertEqual(
            plan[1][1],
            ["stable-audio-3-small-sfx-base-dit-0.5B-v1.0-F32.gguf"],
        )

    def test_quantized_encoding_resolves_the_dit_but_not_the_same(self):
        # `encoding` used to pick the DiT AND the autoencoder, so asking for a quantized DiT
        # fetched a quantized SAME too -- and since nothing else landed on disk, the resolver then
        # had nothing better to prefer. SAME is the last net the audio crosses (and audio2audio
        # crosses it twice per iteration) and the cheap one to keep -- 413 MB at F32 for SAME-S
        # against 72 MB at Q4_K_M -- so it defaults to F32 on its own axis now.
        for enc, suffix in (("q4_k_m", "Q4_K_M"), ("q5_k_m", "Q5_K_M"), ("q8_0", "Q8_0")):
            plan = build_download_plan("thepatch", "medium", enc)
            files = plan[0][1]
            self.assertIn(f"stable-audio-3-medium-dit-1.5B-v1.0-{suffix}.gguf", files)
            self.assertIn("stable-audio-3-medium-same-l-v1.0-F32.gguf", files)
            self.assertNotIn(f"stable-audio-3-medium-same-l-v1.0-{suffix}.gguf", files)
            # the conditioner is quality-critical and tiny: always F32, never quantized
            self.assertIn("stable-audio-3-medium-conditioner-v1.0-F32.gguf", files)

    def test_ae_encoding_is_honoured_when_asked_for(self):
        # Quantized autoencoders stay available. What changed is that they have to be REQUESTED
        # rather than arriving as a side effect of the DiT's tier.
        plan = build_download_plan("thepatch", "medium", "q4_k_m", ae_encoding="q4_k_m")
        self.assertIn("stable-audio-3-medium-same-l-v1.0-Q4_K_M.gguf", plan[0][1])
        self.assertIn("stable-audio-3-medium-dit-1.5B-v1.0-Q4_K_M.gguf", plan[0][1])

        plan = build_download_plan("thepatch", "medium", "f16", ae_encoding="f16")
        self.assertIn("stable-audio-3-medium-same-l-v1.0-F16.gguf", plan[0][1])

    def test_bad_ae_encoding_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            build_download_plan("thepatch", "medium", "f16", ae_encoding="nonsense")
        self.assertIn("ae_encoding", str(ctx.exception))

    def test_q4_request_gets_a_q4_training_base(self):
        # Training on a quantized base works on every backend, so a Q4_K_M request resolves to a
        # Q4_K_M base rather than substituting F16 -- the whole point of publishing the quant base.
        plan = build_download_plan("thepatch", "medium", "q4_k_m", training_base=True)
        self.assertIn("stable-audio-3-medium-dit-1.5B-v1.0-Q4_K_M.gguf", plan[0][1])
        self.assertEqual(
            plan[1][1],
            ["stable-audio-3-medium-base-dit-1.5B-v1.0-Q4_K_M.gguf"],
        )

    def test_unpublished_base_tier_falls_back_to_f16(self):
        # Only Q4_K_M is published as a base. Q5_K_M/Q8_0 plus --training-base must not ask for a
        # file that does not exist, so the base DiT -- and only the base DiT -- falls back to F16.
        for enc in ("q5_k_m", "q8_0"):
            with self.subTest(encoding=enc):
                plan = build_download_plan("thepatch", "medium", enc, training_base=True)
                suffix = enc.upper()
                self.assertIn(f"stable-audio-3-medium-dit-1.5B-v1.0-{suffix}.gguf", plan[0][1])
                self.assertEqual(
                    plan[1][1],
                    ["stable-audio-3-medium-base-dit-1.5B-v1.0-F16.gguf"],
                )

    def test_unknown_encoding_is_rejected_and_names_the_valid_ones(self):
        with self.assertRaises(ValueError) as ctx:
            build_download_plan("thepatch", "medium", "q3_k_s")
        self.assertIn("q4_k_m", str(ctx.exception))

    def test_notice_retains_required_attribution(self):
        notice = notice_text("medium")
        self.assertIn(
            "This Stability AI Model is licensed under the Stability AI Community License, "
            "Copyright © Stability AI Ltd. All Rights Reserved",
            notice,
        )
        self.assertIn("Powered by Stability AI", notice)
        self.assertIn("The model was not retrained.", notice)

    def assert_downloader_plan(self, command):
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        output = result.stdout.replace("\\", "/")
        self.assertEqual(output.count("[plan]"), 6)
        self.assertIn(
            "thepatch/stable-audio-3-small-sfx-base-GGUF/resolve/main/"
            "stable-audio-3-small-sfx-base-dit-0.5B-v1.0-F32.gguf",
            output,
        )
        self.assertIn("t5gemma-b-b-ul2-v1.0-vocab.gguf", output)

    # models.sh stays a RELATIVE path: shutil.which("bash") may resolve to the WSL shim in
    # WindowsApps, whose filesystem namespace has no C:\ — an absolute Windows path fails there
    # with exit 127, while a relative one resolves against the cwd WSL translates for us.
    def test_shell_downloader_training_base_plan(self):
        bash = shutil.which("bash")
        if not bash:
            self.skipTest("bash is not installed")
        self.assert_downloader_plan([
            bash, "models.sh", "--variant", "small-sfx", "--encoding", "f32",
            "--training-base", "--dry-run", "--out", "test-models",
        ])

    # models.cmd, by contrast, is named ABSOLUTELY: `cmd.exe /c models.cmd` resolves the script
    # through the executable search, and a process environment carrying
    # NoDefaultCurrentDirectoryInExePath=1 drops the current directory from that search (agent and
    # tool shells set it; ordinary shells and CI do not). The test would then fail with
    # "'models.cmd' is not recognized" despite cwd being correct — a false alarm about the
    # downloader, so pin the path rather than depend on the lookup.
    @unittest.skipUnless(os.name == "nt", "Windows command script")
    def test_cmd_downloader_training_base_plan(self):
        self.assert_downloader_plan([
            "cmd.exe", "/d", "/c", str(REPO_ROOT / "models.cmd"), "--variant", "small-sfx",
            "--encoding", "f32", "--training-base", "--dry-run", "--out", "test-models",
        ])


if __name__ == "__main__":
    unittest.main()
