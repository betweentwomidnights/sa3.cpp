#!/usr/bin/env python3
import hashlib
import json
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path.home() / "repos" / "bako" / "song-book" / "datasets" / "mnesia-audio-v1"
CHUNKS = SOURCE / "lyric-section-chunks"
OUTPUT = ROOT.parent / "datasets" / "mnesia-audio-chunks-v1"

TEST_TRACKS = {
    "album-causality-04-show-me",
    "album-context-08-machine",
    "single-fluke",
    "single-losing-you",
    "single-synesthesia",
}

EVALUATION_TRACKS = {
    "album-causality-10-youll-be-sorry",
    "album-context-05-patterns",
    "single-moment",
    "single-no-one-told-me",
}


def read_jsonl(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        out[item["id"]] = item
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def split_for(track_id: str) -> str:
    if track_id in EVALUATION_TRACKS:
        return "evaluation"
    if track_id in TEST_TRACKS:
        return "test"
    return "train"


def compact_list(values: list[str]) -> str:
    return ", ".join(v for v in values if v)


def build_caption(chunk: dict, source_meta: dict | None) -> str:
    metadata = (source_meta or {}).get("metadata") or {}
    title = chunk.get("title") or (source_meta or {}).get("title") or chunk["track_id"]
    artist = chunk.get("artist") or (source_meta or {}).get("artist") or "MNESIA"
    section_labels = compact_list(chunk.get("section_labels") or [])
    production = metadata.get("production_vision") or (
        "electronic pop production built from synths, synth bass, drums, and percussion; "
        "professionally mixed and mastered with full stereo FX and full compression; "
        "a consistent female lead vocalist"
    )
    mood = metadata.get("mood_energy") or metadata.get("emotional_tone") or metadata.get("vibe")
    tempo = metadata.get("tempo_vibe") or metadata.get("tempo")
    themes = compact_list(metadata.get("themes") or [])

    parts = [
        f"{artist} - {title}.",
        production,
        f"Song section chunk: {section_labels}." if section_labels else "",
        f"Mood: {mood}." if mood else "",
        f"Tempo/vibe: {tempo}." if tempo else "",
        f"Themes: {themes}." if themes else "",
    ]
    return " ".join(part.strip() for part in parts if part).strip() + "\n"


def write_split(split: str, records: list[dict]) -> None:
    split_dir = OUTPUT / split
    audio_dir = split_dir / "audio"
    lyrics_dir = split_dir / "lyrics"
    audio_dir.mkdir(parents=True, exist_ok=True)
    lyrics_dir.mkdir(parents=True, exist_ok=True)

    with (split_dir / "filelist.txt").open("w", encoding="utf-8") as filelist:
        for record in records:
            filelist.write(record["audio_path"] + "\n")

    with (split_dir / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    (split_dir / "metadata.json").write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    if not (CHUNKS / "index.json").exists():
        raise SystemExit(f"missing chunk manifest: {CHUNKS / 'index.json'}")
    if not (SOURCE / "metadata.jsonl").exists():
        raise SystemExit(f"missing source metadata: {SOURCE / 'metadata.jsonl'}")

    chunk_index = json.loads((CHUNKS / "index.json").read_text(encoding="utf-8"))
    source_metadata = read_jsonl(SOURCE / "metadata.jsonl")

    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)

    all_records: list[dict] = []
    splits: dict[str, list[dict]] = {"train": [], "test": [], "evaluation": []}

    for chunk in chunk_index["chunks"]:
        split = split_for(chunk["track_id"])
        chunk_id = chunk["id"]
        src_audio = CHUNKS / chunk["audio_path"]
        src_lyrics = CHUNKS / chunk["lyrics_path"]
        dst_audio_rel = f"audio/{chunk_id}.mp3"
        dst_caption_rel = f"audio/{chunk_id}.txt"
        dst_lyrics_rel = f"lyrics/{chunk_id}.txt"

        split_dir = OUTPUT / split
        dst_audio = split_dir / dst_audio_rel
        dst_caption = split_dir / dst_caption_rel
        dst_lyrics = split_dir / dst_lyrics_rel

        link_or_copy(src_audio, dst_audio)
        dst_caption.parent.mkdir(parents=True, exist_ok=True)
        dst_caption.write_text(build_caption(chunk, source_metadata.get(chunk["track_id"])), encoding="utf-8")
        dst_lyrics.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_lyrics, dst_lyrics)

        record = {
            "id": chunk_id,
            "split": split,
            "artist": chunk.get("artist"),
            "title": chunk.get("title"),
            "audio_format": "mp3",
            "audio_path": dst_audio_rel,
            "caption_path": dst_caption_rel,
            "lyrics_path": dst_lyrics_rel,
            "audio_sha256": sha256(dst_audio),
            "audio_size_bytes": dst_audio.stat().st_size,
            "duration_seconds": chunk["duration_seconds"],
            "source_track_id": chunk["track_id"],
            "source_audio_path": chunk["source_audio_path"],
            "start_time": chunk["start_time"],
            "end_time": chunk["end_time"],
            "section_indexes": chunk.get("section_indexes") or [],
            "section_labels": chunk.get("section_labels") or [],
            "warnings": chunk.get("warnings") or [],
            "metadata": (source_metadata.get(chunk["track_id"]) or {}).get("metadata") or {},
        }
        splits[split].append(record)
        all_records.append({**record, "audio_path": f"{split}/{dst_audio_rel}", "caption_path": f"{split}/{dst_caption_rel}", "lyrics_path": f"{split}/{dst_lyrics_rel}"})

    for split, records in splits.items():
        write_split(split, records)

    summary = {
        "dataset": "mnesia-audio-chunks-v1",
        "source_dataset": str(CHUNKS),
        "chunk_count": len(all_records),
        "splits": {split: len(records) for split, records in splits.items()},
        "test_source_tracks": sorted(TEST_TRACKS),
        "evaluation_source_tracks": sorted(EVALUATION_TRACKS),
        "skipped_source_tracks": chunk_index.get("skipped_tracks") or [],
    }
    (OUTPUT / "metadata.json").write_text(json.dumps(all_records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUTPUT / "metadata.jsonl").write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in all_records), encoding="utf-8")
    (OUTPUT / "dataset_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUTPUT / "README.md").write_text(
        "# MNESIA Audio Chunks v1\n\n"
        "Chunk-level LoRA dataset built from lyric-section chunks in song-book.\n"
        "Audio is split at lyric section boundaries, with every emitted chunk at least 30 seconds.\n"
        "Captions describe production/style; `lyrics_path` points to the chunk-specific canonical lyrics.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
