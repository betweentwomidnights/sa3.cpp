#!/usr/bin/env python3
"""Shared GGUF `general.*` metadata stamping for the sa3.cpp converters.

Implements the naming convention in docs/DISTRIBUTION.md so every emitted gguf carries
the basename / version / license (and size_label where it applies) the HF cards lean on.
The `general.architecture` value is still set by each converter's GGUFWriter(arch=...) call
(our loaders key off it) — this only adds the descriptive/catalog metadata on top.
"""

VERSION = "v1.0"
LICENSE = "stabilityai-community"

# variant tag -> (autoencoder suffix, human label)
VARIANTS = {
    "medium":      ("same-l", "medium"),
    "small-music": ("same-s", "small-music"),
    "small-sfx":   ("same-s", "small-sfx"),
}


def size_label(n_params):
    """Param-count class per the gguf convention. Sub-billion models render as '0.xB'
    (matching the HF/acestep norm, e.g. Qwen3-Embedding-0.6B) down to 100M; smaller use M/K.
    e.g. 1.45e9 -> '1.5B', 4.6e8 -> '0.5B', 2.8e8 -> '0.3B'."""
    if n_params >= 1e8:
        s = f"{n_params / 1e9:.1f}B"
    elif n_params >= 1e5:
        s = f"{n_params / 1e6:.0f}M"
    else:
        s = f"{n_params / 1e3:.0f}K"
    return s.replace(".0B", "B")


def add_general(w, basename, name, finetune=None, n_params=None, license_id=LICENSE):
    """Stamp the convention's catalog metadata. basename e.g. 'stable-audio-3-medium-dit';
    finetune is the variant tag; pass n_params on model-like components (DiT, encoder) to
    emit size_label (omit for the autoencoder + tokenizer, which are convention-exempt).
    license_id defaults to the SA3 community license; the shared T5Gemma encoder + tokenizer
    pass 'gemma' since they're Google's under the Gemma Terms of Use."""
    w.add_name(name)
    w.add_string("general.basename", basename)
    w.add_string("general.version", VERSION)
    w.add_string("general.license", license_id)
    if finetune:
        w.add_string("general.finetune", finetune)
    if n_params is not None:
        w.add_string("general.size_label", size_label(n_params))


def add_source(w, name, organization, repo_url, revision, source_file=None):
    """Record the exact upstream checkpoint identity using standard base-model fields."""
    w.add_base_model_count(1)
    w.add_base_model_name(0, name)
    w.add_base_model_organization(0, organization)
    w.add_base_model_repo_url(0, repo_url)
    w.add_base_model_version(0, revision)
    if source_file:
        w.add_string("general.source.file", source_file)
