"""Exercise SAT HTTP request validation without downloading model weights."""

import base64
import io
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import wave


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def request(port, path, body=None, raw=None):
    data = raw if raw is not None else None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=3) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def check(executable, model, cases):
    port = free_port()
    process = subprocess.Popen(
        [executable, "--model", model, "--models-dir", "missing-sat-models",
         "--port", str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    try:
        for _ in range(50):
            if process.poll() is not None:
                raise AssertionError(f"{model} server exited early: {process.stderr.read()!r}")
            try:
                status, body = request(port, "/health")
                break
            except (OSError, TimeoutError):
                time.sleep(0.1)
        else:
            raise AssertionError(f"{model} server did not start")
        assert status == 503 and body["status"] == "model_missing", (status, body)
        for path, payload, expected, fragment in cases:
            status, body = request(port, path, payload)
            assert status == expected and fragment in body.get("error", ""), (status, body)
        if model == "foundation-1":
            check_randomize(port)
        if model == "saos":
            check_saos_catalog(port)
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        process.stderr.close()


def check_randomize(port):
    """/randomize keeps gary4local's Python contract and needs no model weights."""
    status, body = request(port, "/randomize", {"seed": 42, "mode": "standard"})
    assert status == 200 and body["success"], (status, body)
    for key in ("family", "subfamily", "descriptor_knob_a", "descriptor_knob_b", "descriptor_knob_c",
                "reverb_amount", "delay_type", "distortion_amount", "phaser_amount", "bitcrush_amount",
                "speed", "structure", "prompt", "mode", "variant"):
        assert isinstance(body[key], str), (key, body)
    for key in ("descriptors_extra", "behavior_tags", "rhythm", "contour", "density", "spatial_tags",
                "band_tags", "wave_tech_tags", "style_tags", "all_tags"):
        assert isinstance(body[key], list), (key, body)
    for fx in ("reverb", "delay", "distortion", "phaser", "bitcrush"):
        assert isinstance(body[f"{fx}_enabled"], bool), (fx, body)
    assert body["seed"] == 42 and body["variant"] == "M1" and body["family"] in body["prompt"], body
    status, again = request(port, "/randomize", {"seed": 42, "mode": "standard"})
    assert again == body, "the same seed must give the same preset"
    status, mix = request(port, "/randomize", {"seed": 7, "variant": "T1", "family_hint": "synth"})
    assert status == 200 and mix["variant"] == "T1" and mix["family"] == "Synth", mix
    for _ in range(20):
        status, drawn = request(port, "/randomize", {"seed": -1})
        assert status == 200 and 0 <= drawn["seed"] < 2**31, drawn
    status, empty = request(port, "/randomize", raw=b"")
    assert status == 200 and empty["success"], empty
    status, bad = request(port, "/randomize", {"family_hint": "Drums"})
    assert status == 400 and "unknown Foundation family" in bad["error"], bad


def check_saos_catalog(port):
    """With nothing installed, SAOS lists no models and selects none."""
    status, body = request(port, "/models/status")
    assert status == 200 and body["model_details"] == {} and body["cache_status"]["usage_order"] == [], body
    status, body = request(port, "/models/prompts")
    assert status == 200 and body["success"] and body["prompts"] is None, body


def tiny_wav():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(bytes(4 * 4800))
    return base64.b64encode(buffer.getvalue()).decode()


def main():
    binary = sys.argv[1]
    wav = tiny_wav()
    check(binary, "foundation-1", [
        ("/audio2audio", {"host_bpm": 120}, 400, "audio_data (base64 WAV) is required"),
        ("/audio2audio", {"audio_data": wav, "prompt": "Synth"}, 400, "host_bpm is required"),
        ("/audio2audio", {"audio_data": "not*base64", "host_bpm": 120, "prompt": "Synth"}, 400, "not valid base64"),
        ("/audio2audio", {"audio_data": base64.b64encode(b"not a wav").decode(), "host_bpm": 120,
                          "prompt": "Synth"}, 400, "RIFF/WAVE"),
        ("/audio2audio", {"audio_data": wav, "host_bpm": 120, "prompt": "Synth", "init_noise_level": 0.3,
                          "sigma_min": 0.5}, 400, "sigma_min must be below"),
        ("/audio2audio", {"audio_data": wav, "host_bpm": 123, "prompt": "Synth"}, 503, "missing"),
        ("/generate", {"prompt": "Synth", "bars": 5, "bpm": 120}, 400, "4 or 8 bars"),
        ("/generate", {"prompt": "Synth", "bars": 4, "host_bpm": 1000}, 400, "between 20 and 999"),
        ("/generate", {"prompt": "Synth", "bars": 4, "host_bpm": 123}, 503, "missing"),
        ("/generate", {"prompt": "Synth", "bars": 4, "bpm": 120}, 503, "missing"),
    ])
    kickbass = {"model_type": "finetune", "finetune_repo": "S3Sound/kickbass",
                "finetune_checkpoint": "kickbass_v1_saos_e257_s18870.ckpt"}
    check(binary, "saos", [
        ("/generate", {"prompt": "drums", "seconds_total": 0}, 400, "seconds_total must be >= 1"),
        ("/generate", {"prompt": "drums", "steps": 0}, 400, "steps must be"),
        ("/generate", {"prompt": "drums", "cfg_scale": 21}, 400, "cfg_scale must be"),
        ("/generate", {"prompt": "drums", "sampler_type": "dpmpp-3m-sde"}, 400, "sampler_type must be"),
        ("/generate", {"prompt": "drums", "return_format": "mp3"}, 400, "return_format"),
        ("/generate", {"prompt": "drums", "model_type": "remix"}, 400, "model_type must be"),
        ("/generate", {"prompt": "drums", "model_type": "finetune"}, 400, "finetune_repo and finetune_checkpoint"),
        ("/generate", {"prompt": "drums", "model_type": "finetune", "finetune_repo": "thepatch/jerry_grunge",
                       "finetune_checkpoint": "jerry_encoded_bs64_epoch=374-step=3000.ckpt"}, 400,
         "no native SAOS finetune"),
        ("/generate/loop", {"prompt": "drums"}, 400, "BPM must be specified"),
        ("/generate/loop", {"prompt": "drums 100bpm", "bars": 8}, 400, "exceeds"),
        ("/generate/loop", {"prompt": "drums 120bpm", "loop_type": "vocals"}, 400, "loop_type must be"),
        ("/generate", {"prompt": "drums", "seconds_total": 30, "model_type": "standard"}, 503, "missing"),
        ("/generate/loop", dict(kickbass, prompt="kick 120bpm"), 503, "kickbass"),
        ("/randomize", {}, 404, "Foundation-1 route"),
        ("/audio2audio", {"prompt": "drums", "host_bpm": 120, "audio_data": wav}, 400, "Foundation-1 route"),
    ])


if __name__ == "__main__":
    main()
