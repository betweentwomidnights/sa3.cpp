#!/usr/bin/env python3
"""Stage the three training-base Hugging Face repositories without uploading.

The GGUFs are hard-linked when the staging directory is on the same volume, so
staging the F16/F32 set does not consume another ~14 GB. The exact upstream
license is fetched at the pinned source revision and verified before use.
"""

import argparse
import hashlib
import os
import shutil
from pathlib import Path

from model_artifacts import (
    STABILITY_LICENSE_SHA256,
    UPSTREAM_BASE_REVISIONS,
    VARIANTS,
    dit_filename,
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def link_or_copy(source, destination):
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


def notice_text(variant):
    upstream = f"https://huggingface.co/stabilityai/stable-audio-3-{variant}-base"
    return f"""This Stability AI Model is licensed under the Stability AI Community License, Copyright © Stability AI Ltd. All Rights Reserved

Powered by Stability AI

This repository contains a GGUF format conversion of the Stable Audio 3 {variant} base DiT from:
{upstream}

Modification notice: the model tensors were renamed and serialized as GGUF for sa3.cpp. Selected two-dimensional tensors in the F16 artifact were converted from F32 to F16. The model was not retrained.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--variant", choices=["all", *VARIANTS], default="all")
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit('missing dependency: python -m pip install -U "huggingface_hub"')

    variants = VARIANTS if args.variant == "all" else [args.variant]
    cards = Path(__file__).resolve().parents[1] / "docs" / "model-cards"
    args.out.mkdir(parents=True, exist_ok=True)

    for variant in variants:
        repo_name = f"stable-audio-3-{variant}-base-GGUF"
        destination = args.out / repo_name
        destination.mkdir(parents=True, exist_ok=True)

        revision = UPSTREAM_BASE_REVISIONS[variant]
        upstream_id = f"stabilityai/stable-audio-3-{variant}-base"
        license_path = Path(hf_hub_download(upstream_id, "LICENSE.md", revision=revision))
        if sha256(license_path) != STABILITY_LICENSE_SHA256:
            raise RuntimeError(f"unexpected upstream license content: {license_path}")

        shutil.copy2(cards / f"{repo_name}.md", destination / "README.md")
        shutil.copy2(license_path, destination / "LICENSE.md")
        (destination / "NOTICE").write_text(notice_text(variant), encoding="utf-8")
        (destination / ".gitattributes").write_text(
            "*.gguf filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8"
        )

        sums = []
        for encoding in ("F16", "F32"):
            filename = dit_filename(variant, encoding, training_base=True)
            source = args.gguf_dir / filename
            if not source.is_file():
                raise FileNotFoundError(source)
            mode = link_or_copy(source, destination / filename)
            digest = sha256(source)
            sums.append(f"{digest}  {filename}")
            print(f"[{repo_name}] {mode}: {filename}")
        (destination / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="ascii")

    print(f"staged {len(variants)} repo(s) in {args.out}")


if __name__ == "__main__":
    main()
