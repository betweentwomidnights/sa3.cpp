#!/usr/bin/env python3
"""Stage the SAOS GGUF Hugging Face repository without creating or uploading it."""

import argparse
import hashlib
import os
import shutil
import urllib.request
from pathlib import Path

from model_artifacts import (
    SAOS_ENCODINGS,
    SAOS_REPO,
    SAOS_VARIANTS,
    saos_dit_filename,
    saos_oobleck_filename,
    saos_t5_filename,
)

SAOS_REVISION = "dc620d91535857b72ebb59b4ca45978db6d417f5"
SAOS_LICENSE_SHA256 = "d6f6b1a4dce5c852bd6d7d9482d002baf0ccdb71e662250b73be9eec8764ee8d"
APACHE_LICENSE_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def link_or_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size or sha256(destination) != sha256(source):
            raise FileExistsError(f"refusing to replace different file: {destination}")
        return "existing"
    try:
        os.link(source, destination)
        return "hard-link"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def source_path(models, quant, component, variant, encoding):
    if component == "dit" and encoding == "F16":
        if variant == "arc":
            return models / "stable-audio-open-small-dit-f16.gguf"
        if variant == "kickbass":
            return models / "finetunes" / "kickbass" / "kickbass-v1-e257-dit-f16.gguf"
        return models / "finetunes" / "jerry_grunge" / "jerry-grunge-bs64-step3000-dit-f16.gguf"
    if component == "dit":
        prefix = {"arc": "stable-audio-open-small", "kickbass": "kickbass",
                  "jerry-grunge": "jerry-grunge"}[variant]
        return quant / f"{prefix}-dit-{encoding}.gguf"
    if component == "t5":
        return models / "t5-base-encoder-f16.gguf" if encoding == "F16" else quant / f"t5-base-{encoding}.gguf"
    return models / "stable-audio-open-small-oobleck-f16.gguf" if encoding == "F16" else quant / f"oobleck-{encoding}.gguf"


def notice_text():
    return """This Stability AI Model is licensed under the Stability AI Community License, Copyright © Stability AI Ltd. All Rights Reserved

Powered by Stability AI

This repository contains GGUF conversions derived from Stable Audio Open Small and two topology-compatible finetunes. KickBass is redistributed with direct permission from S3Sound. Jerry Grunge is published by thepatch. Full source repositories and pinned revisions are recorded in README.md.

Modification notice: model tensors were renamed and serialized as GGUF for sa3.cpp. Selected tensors were converted to F16 or quantized to Q8_0, Q5_K_M, or Q4_K_M. The models were not retrained as part of this conversion.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--quant-dir", type=Path, default=Path("models/quantization/saos"))
    parser.add_argument("--canonical-dir", type=Path,
                        help="directory already containing the canonical repository-relative files")
    parser.add_argument("--out", type=Path, required=True,
                        help="parent staging directory; the repository directory is created inside")
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit('missing dependency: python -m pip install -U "huggingface_hub"')

    destination = args.out / SAOS_REPO
    destination.mkdir(parents=True, exist_ok=True)
    cards = Path(__file__).resolve().parents[1] / "docs" / "model-cards"
    shutil.copy2(cards / f"{SAOS_REPO}.md", destination / "README.md")

    stability_license = Path(hf_hub_download(
        "stabilityai/stable-audio-open-small", "LICENSE", revision=SAOS_REVISION))
    if sha256(stability_license) != SAOS_LICENSE_SHA256:
        raise RuntimeError(f"unexpected upstream Stability license: {stability_license}")
    shutil.copy2(stability_license, destination / "LICENSE.md")

    apache_license = destination / "LICENSE_T5.md"
    if not apache_license.exists():
        urllib.request.urlretrieve("https://www.apache.org/licenses/LICENSE-2.0.txt", apache_license)
    if sha256(apache_license) != APACHE_LICENSE_SHA256:
        raise RuntimeError(f"unexpected Apache license content: {apache_license}")

    (destination / "NOTICE").write_text(notice_text(), encoding="utf-8")
    (destination / ".gitattributes").write_text(
        "*.gguf filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8")

    sums = []
    for encoding in SAOS_ENCODINGS:
        for variant in SAOS_VARIANTS:
            relative = saos_dit_filename(variant, encoding)
            source = (args.canonical_dir / relative if args.canonical_dir else
                      source_path(args.models_dir, args.quant_dir, "dit", variant, encoding))
            if not source.is_file():
                raise FileNotFoundError(source)
            mode = link_or_copy(source, destination / relative)
            sums.append(f"{sha256(source)}  {relative}")
            print(f"[{mode}] {relative}")
        for component, relative in (
            ("t5", saos_t5_filename(encoding)),
            ("oobleck", saos_oobleck_filename(encoding)),
        ):
            source = (args.canonical_dir / relative if args.canonical_dir else
                      source_path(args.models_dir, args.quant_dir, component, "arc", encoding))
            if not source.is_file():
                raise FileNotFoundError(source)
            mode = link_or_copy(source, destination / relative)
            sums.append(f"{sha256(source)}  {relative}")
            print(f"[{mode}] {relative}")
    (destination / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="ascii")
    print(f"staged {len(sums)} GGUF files in {destination}")


if __name__ == "__main__":
    main()
