#!/usr/bin/env python3
"""Stage self-contained SAO 1.0 and Foundation-1 GGUF repositories locally."""

import argparse
import shutil
import urllib.request
from pathlib import Path

from model_artifacts import (
    SAOS_ENCODINGS,
    SAT_LARGE_MODELS,
    SAT_LARGE_REPOS,
    sat_large_dit_filename,
    sat_oobleck_filename,
    sat_t5_128_filename,
)
from stage_saos_repo import link_or_copy, sha256

SOURCES = {
    "stable-audio-open-1.0": (
        "stabilityai/stable-audio-open-1.0",
        "f21265c1e2710b3bd2386596943f0007f55f802e",
        "model.safetensors",
    ),
    "foundation-1": (
        "RoyalCities/Foundation-1",
        "d3160956fa13a8f861d3f608ed24075d44e98554",
        "Foundation_1.safetensors",
    ),
}
STABILITY_LICENSE_SHA256 = "d6f6b1a4dce5c852bd6d7d9482d002baf0ccdb71e662250b73be9eec8764ee8d"
FOUNDATION_NOTICE_SHA256 = "5b82bfdcdbff48cbf72da490db9404a6ec343c5ccc3ce9d54ff55bdd309f02bc"
APACHE_LICENSE_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"


def component_source(models, model, component, encoding):
    if component == "dit":
        if encoding == "F16":
            folder = "sao1-source" if model == "stable-audio-open-1.0" else "foundation-source"
            stem = "stable-audio-open-1.0" if model == "stable-audio-open-1.0" else "foundation-1"
            return models / folder / f"{stem}-dit-f16.gguf"
        folder = "sao1" if model == "stable-audio-open-1.0" else "foundation"
        stem = "stable-audio-open-1.0" if model == "stable-audio-open-1.0" else "foundation-1"
        return models / "quantization" / folder / f"{stem}-dit-{encoding}.gguf"
    if component == "t5":
        if encoding == "F16":
            return models / "sao1-source" / "t5-base-encoder-128-f16.gguf"
        return models / "quantization" / "foundation" / f"t5-base-encoder-128-{encoding}.gguf"
    return (models / "publication-v2" /
            f"stable-audio-open-small-oobleck-v1.0-{encoding}.gguf")


def notice_text(model):
    repo, revision, source_file = SOURCES[model]
    return f"""This Stability AI Model is licensed under the Stability AI Community License, Copyright © Stability AI Ltd. All Rights Reserved

Powered by Stability AI

This repository contains GGUF conversions derived from {repo}, revision {revision}, file {source_file}.

Modification notice: model tensors were renamed and serialized as GGUF for sa3.cpp. Selected tensors were converted to F16 or quantized to Q8_0, Q5_K_M, or Q4_K_M. The model was not retrained as part of this conversion.
"""


def stage_model(model, args, hf_hub_download):
    repo_name = SAT_LARGE_REPOS[model]
    destination = args.out / repo_name
    destination.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    shutil.copy2(root / "docs" / "model-cards" / f"{repo_name}.md",
                 destination / "README.md")

    # The full Stability license is canonical for both repositories. Foundation's
    # source-repository license note is retained as an additional file.
    stability = Path(hf_hub_download(
        "stabilityai/stable-audio-open-1.0", "LICENSE.md",
        revision=SOURCES["stable-audio-open-1.0"][1]))
    if sha256(stability) != STABILITY_LICENSE_SHA256:
        raise RuntimeError(f"unexpected upstream Stability license: {stability}")
    shutil.copy2(stability, destination / "LICENSE.md")
    if model == "foundation-1":
        foundation_notice = Path(hf_hub_download(
            SOURCES[model][0], "LICENSE.md", revision=SOURCES[model][1]))
        if sha256(foundation_notice) != FOUNDATION_NOTICE_SHA256:
            raise RuntimeError(f"unexpected Foundation license notice: {foundation_notice}")
        shutil.copy2(foundation_notice, destination / "LICENSE_FOUNDATION.md")

    apache = destination / "LICENSE_T5.md"
    if not apache.exists():
        urllib.request.urlretrieve("https://www.apache.org/licenses/LICENSE-2.0.txt", apache)
    if sha256(apache) != APACHE_LICENSE_SHA256:
        raise RuntimeError(f"unexpected Apache license content: {apache}")
    (destination / "NOTICE").write_text(notice_text(model), encoding="utf-8")
    (destination / ".gitattributes").write_text(
        "*.gguf filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8")

    sums = []
    for encoding in SAOS_ENCODINGS:
        entries = (
            ("dit", sat_large_dit_filename(model, encoding)),
            ("t5", sat_t5_128_filename(encoding)),
            ("oobleck", sat_oobleck_filename(encoding)),
        )
        for component, relative in entries:
            source = component_source(args.models_dir, model, component, encoding)
            if not source.is_file():
                raise FileNotFoundError(source)
            mode = link_or_copy(source, destination / relative)
            sums.append(f"{sha256(source)}  {relative}")
            print(f"[{mode}] {repo_name}/{relative}")
    (destination / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="ascii")
    print(f"staged {len(sums)} GGUF files in {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("all",) + SAT_LARGE_MODELS, default="all")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit('missing dependency: python -m pip install -U "huggingface_hub"')
    selected = SAT_LARGE_MODELS if args.model == "all" else (args.model,)
    for model in selected:
        stage_model(model, args, hf_hub_download)


if __name__ == "__main__":
    main()
