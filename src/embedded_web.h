#pragma once
// embedded_web.h — web UI assets embedded in the binary at build time.
// AUTO-GENERATED from web/index.html, app.js, studio.js, and studio.css. Do not edit by hand.
// Rebuild with: python3 tools/gen_embedded_web.py

#include <string>

namespace embedded_web {

inline const std::string index_html =
    std::string(R"sa3web(<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>sa3.cpp studio — inference</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg: #090909; --surface: #171313; --border: #493030;
    --text: #f4eded; --muted: #b6a5a5; --accent: #e23b42;
    --green: #6cc58b; --red: #ed6268; --orange: #d9a36c;
    --radius: 6px; --mono: 'SF Mono', 'Cascadia Code', 'Fira Code', monospace;
  }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.5;
    max-width: 860px; margin: 0 auto; padding: 20px 16px 60px;
  }
  h1 { font-size: 1.5rem; font-weight: 600; margin-bottom: 4px; }
  h1 small { font-weight: 400; font-size: 0.85rem; color: var(--muted); }
  h2 { font-size: 1.05rem; font-weight: 600; margin-bottom: 10px; margin-top: 20px;
       padding-bottom: 6px; border-bottom: 1px solid var(--border); }
  h2 .badge { font-weight: 400; font-size: 0.7rem; background: var(--surface);
              padding: 1px 7px; border-radius: 9px; color: var(--muted); margin-left: 6px; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
          padding: 16px; margin-bottom: 12px; }
  .row { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }
  .row.gapped { margin-bottom: 10px; }
  .col { flex: 1; min-width: 120px; }
  label { display: block; font-size: 0.78rem; color: var(--muted); margin-bottom: 3px; font-weight: 500; }
  input, select, textarea {
    font-family: inherit; font-size: 0.88rem; background: #100d0d; color: var(--text);
    border: 1px solid var(--border); border-radius: var(--radius); padding: 6px 10px;
    width: 100%; outline: none; transition: border .15s;
  }
  input:focus, select:focus, textarea:focus { border-color: var(--accent); }
  textarea { resize: vertical; min-height: 48px; font-family: var(--mono); font-size: 0.82rem; }
  input[type=number] { font-family: var(--mono); font-size: 0.82rem; }
  select { cursor: pointer; }
  button {
    font-family: inherit; font-size: 0.85rem; cursor: pointer; border: 1px solid var(--border);
    border-radius: var(--radius); padding: 7px 16px; background: #251a1a; color: var(--text);
    transition: background .15s, border-color .15s; white-space: nowrap;
  }
  button:hover { background: #392222; }
  button.primary { background: #b6222b; border-color: #d3323b; color: #fff; font-weight: 600; }
  button.primary:hover { background: #d3323b; }
  button.primary:disabled { opacity: .5; cursor: not-allowed; }
  button.loop { background: #8e2027; border-color: #b42c34; color: #fff; font-weight: 600; }
  button.loop:hover { background: #b42c34; }
  button.loop:disabled { opacity: .5; cursor: not-allowed; }
  button.small { padding: 2px 8px; font-size: 0.78rem; }
  button.danger { border-color: var(--red); color: var(--red); }
  button.danger:hover { background: var(--red); color: #fff; }
  .inline-label { display: inline-flex; align-items: center; gap: 6px; font-size: 0.85rem; cursor: pointer; }
  .inline-label input[type=checkbox] { width: auto; accent-color: var(--accent); }
  #server-status { font-size: 0.85rem; font-weight: 500; }
  #server-status.ok { color: var(--green); }
  #server-status.err { color: var(--red); }
  #model-info { font-size: 0.82rem; color: var(--muted); }
  .collapse-toggle { cursor: pointer; user-select: none; background: none; border: none;
    color: var(--accent); font-size: 0.82rem; padding: 0; }
  .collapse-toggle::before { content: '▾ '; }
  .collapse-toggle.collapsed::before { content: '▸ '; }
  .collapse-body { overflow: hidden; transition: max-height .2s; max-height: 2000px; }
  .collapse-body.collapsed { max-height: 0; }
  .param-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 10px; }
  .range-row { display: flex; )sa3web") +
    std::string(R"sa3web(align-items: center; gap: 8px; }
  .range-row input[type=range] { flex: 1; accent-color: var(--accent); background: transparent; border: none; padding: 0; }
  .range-row input[type=number] { width: 70px; }
  #progress-wrap { margin-top: 10px; }
  #progress-bar { height: 6px; background: var(--accent); border-radius: 3px; width: 0%; transition: width .3s; }
  #progress-label { font-size: 0.82rem; color: var(--muted); margin-top: 4px; }
  #result-section { margin-top: 12px; }
  #result-section audio { width: 100%; }
  #seed-info { font-size: 0.8rem; color: var(--muted); margin-top: 4px; }
  .lora-tag { display: inline-flex; align-items: center; gap: 4px; background: #2b181a;
    border: 1px solid #593338; border-radius: 4px; padding: 3px 8px; font-size: 0.82rem; margin: 2px; }
  .lora-tag .lora-str { color: var(--muted); }
  .lora-tag button { padding: 0 4px; font-size: 1rem; line-height: 1; background: none; border: none; color: var(--muted); }
  .lora-tag button:hover { color: var(--red); }
  #active-loras { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 6px; min-height: 26px; }
  #error-msg { color: var(--red); font-size: 0.85rem; margin-top: 6px; min-height: 1.2em; }
  .small-note { font-size: 0.75rem; color: var(--muted); margin-top: 2px; }
  hr { border: none; border-top: 1px solid var(--border); margin: 12px 0; }
  .song-entry { display:flex; align-items:center; gap:8px; padding:8px 0; border-bottom:1px solid var(--border); }
  .song-entry:last-child { border-bottom:none; }
  .song-entry .song-name { flex:0 0 auto; font-size:0.85rem; font-weight:500; min-width:150px; padding-right:40px; }
  .song-entry audio { height:32px; flex:0 0 200px; }
  .song-entry .song-params { flex:1; font-size:0.75rem; color:var(--muted); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; min-width:0; }
  .song-entry .song-actions { display:flex; gap:3px; flex:0 0 auto; }
  #past-songs { max-height:none; }
  #past-songs:empty::after { content:"No past takes yet"; display:block; font-size:0.82rem; color:var(--muted); padding:12px 0; }
  .studio-nav { display:flex; gap:8px; margin: 12px 0 18px; }
  .studio-nav a { color:var(--muted); text-decoration:none; padding:7px 14px; border:1px solid var(--border); border-radius:var(--radius); }
  .studio-nav a[aria-current="page"] { color:var(--text); border-color:var(--accent); background:#38171b; }
  .studio-nav a:hover { color:var(--text); border-color:var(--accent); }
  .studio-credit { margin-top:24px; color:var(--muted); font-size:.78rem; }
  .studio-credit a { color:var(--accent); }
  @media (max-width: 600px) {
    .row { flex-direction: column; }
    .col { min-width: 100%; }
    .param-grid { grid-template-columns: 1fr; }
  }
</style>
<link rel="stylesheet" href="studio.css?v=2">
</head>
<body>

<div id="studio-root"></div>
<div id="legacy-root">

<h1><a href="https://github.com/betweentwomidnights/sa3.cpp" target="_blank" rel="noopener" style="color:var(--accent);text-decoration:none">SA3.CPP</a> <small>studio</small></h1>
<nav class="studio-nav" aria-label="Studio views"><a href="/" aria-current="page">Inference</a><a id="training-link" href="http://127.0.0.1:8016/">LoRA training</a></nav>

<!-- ─── Server Status & Presets ──────────────────────────────────────── -->
<div id="top-bar" class="card" style="padding:8px 16px;display:flex;align-items:center;gap:12px;flex-wrap:wrap">
  <div style="display:flex;align-items:center;gap:10px;flex:1;min-width:180px">
    <span id="server-status" style="display:none;font-weight:600;font-size:0.85rem"></span>
    <span id="model-info" style="display:none;font-size:0.82rem;color:var(--muted)"></span>
    <label class="inline-label" style="font-size:0.82rem"><input id="keep-models" type="checkbox"> Keep Models Resident</label>
  </div>
  <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap">
    <button id="save-config-btn" class="small" title="Save current config">Save preset</button>
    <button id="load-config-btn" class="smal)sa3web") +
    std::string(R"sa3web(l" title="Load config file">Load preset</button>
    <span id="config-filename" style="font-size:0.78rem;color:var(--muted);max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap"></span>
    <input id="load-config-input" type="file" accept=".json" style="display:none">
  </div>
</div>

<!-- ─── Prompt ────────────────────────────────────────────────────────── -->
<div class="card">
  <div class="row gapped">
    <div class="col" style="flex:3">
      <label for="prompt">Prompt</label>
      <textarea id="prompt" rows="4" placeholder="e.g. upbeat funk groove with slap bass, bright horns, tight drums">upbeat funk groove with slap bass, bright horns, tight drums</textarea>
    </div>
  </div>
  <div class="row gapped">
    <div class="col" style="flex:1">
      <label for="negative-prompt">Negative Prompt (CFG)</label>
      <textarea id="negative-prompt" rows="1" placeholder="optional"></textarea>
    </div>
  </div>
</div>

<!-- ─── Basic Params ──────────────────────────────────────────────────── -->
<div class="card">
  <div class="param-grid">
    <div>
      <label for="duration">Duration (seconds)</label>
      <div class="range-row">
        <input id="duration" type="range" min="0.1" max="300" step="0.1" value="30">
        <input id="duration-num" type="number" min="0.1" max="300" step="0.1" value="30">
      </div>
    </div>
    <div>
      <label for="steps">Sampling Steps</label>
      <div class="range-row">
        <input id="steps" type="range" min="1" max="100" step="1" value="8">
        <input id="steps-num" type="number" min="1" max="100" step="1" value="8">
      </div>
    </div>
    <div>
      <label for="seed">Seed (<span style="color:var(--muted)">-1 = random</span>)</label>
      <input id="seed" type="number" value="-1">
    </div>
    <div>
      <label for="duration-padding">Duration Padding (s)</label>
      <div class="range-row">
        <input id="duration-padding" type="range" min="0" max="30" step="0.5" value="6">
        <input id="duration-padding-num" type="number" min="0" max="30" step="0.5" value="6">
      </div>
    </div>
  </div>
</div>

<!-- ─── Advanced: CFG ────────────────────────────────────────────────── -->
<h2>Advanced <span class="badge">collapsible</span></h2>

<div class="card">
  <button class="collapse-toggle collapsed" data-target="cfg-section" type="button">Classifier-Free Guidance</button>
  <div id="cfg-section" class="collapse-body collapsed">
    <div class="param-grid" style="margin-top:10px">
      <div><label for="cfg-scale">CFG Scale</label><input id="cfg-scale" type="number" step="0.1" min="0" value="1.0"></div>
      <div><label for="cfg-rescale">CFG Rescale</label><input id="cfg-rescale" type="number" step="0.05" min="0" max="1" value="0.0"></div>
      <div><label for="apg-scale">APG Scale</label><input id="apg-scale" type="number" step="0.05" min="0" max="1" value="1.0"></div>
      <div><label for="cfg-norm-threshold">CFG Norm Threshold</label><input id="cfg-norm-threshold" type="number" step="0.1" min="0" value="0.0"></div>
      <div><label for="cfg-interval-min">CFG Interval Min</label><input id="cfg-interval-min" type="number" step="0.05" min="0" max="1" value="0.0"></div>
      <div><label for="cfg-interval-max">CFG Interval Max</label><input id="cfg-interval-max" type="number" step="0.05" min="0" max="1" value="1.0"></div>
    </div>
  </div>
</div>

<div class="card">
  <button class="collapse-toggle collapsed" data-target="ds-section" type="button">Distribution Shift</button>
  <div id="ds-section" class="collapse-body collapsed">
    <div class="row gapped" style="margin-top:10px">
      <div class="col" style="flex:0 0 140px">
        <label for="dist-shift">Type</label>
        <select id="dist-shift">
          <option value="LogSNR" selected>LogSNR</option>
          <option value="Flux">Flux</option>
          <option value="Full">Full</option>
          <option value="None">None</option>
        </select>
      </div>
   )sa3web") +
    std::string(R"sa3web(   <div class="col"><label for="dsp1">p1</label><input id="dsp1" type="number" step="any"></div>
      <div class="col"><label for="dsp2">p2</label><input id="dsp2" type="number" step="any"></div>
      <div class="col"><label for="dsp3">p3</label><input id="dsp3" type="number" step="any"></div>
      <div class="col"><label for="dsp4">p4</label><input id="dsp4" type="number" step="any"></div>
    </div>
    <div class="small-note">Params auto-fill per-type defaults. Edit to override.</div>
  </div>
</div>

<div class="card">
  <button class="collapse-toggle collapsed" data-target="chunk-section" type="button">Chunked SAME Encode/Decode</button>
  <div id="chunk-section" class="collapse-body collapsed">
    <div class="param-grid" style="margin-top:10px">
      <div><label for="encode-chunk-size">Encode Chunk Size</label><input id="encode-chunk-size" type="number" min="0" step="1" value="0"></div>
      <div><label for="encode-overlap">Encode Overlap</label><input id="encode-overlap" type="number" min="0" step="1" value="32"></div>
      <div><label for="decode-chunk-size">Decode Chunk Size</label><input id="decode-chunk-size" type="number" min="0" step="1" value="0"></div>
      <div><label for="decode-overlap">Decode Overlap</label><input id="decode-overlap" type="number" min="0" step="1" value="32"></div>
    </div>
    <div class="small-note">0 = monolithic. Overlap must be &lt; chunk size when chunk size &gt; 0.</div>
  </div>
</div>

<div class="card">
  <button class="collapse-toggle collapsed" data-target="loud-section" type="button">Loudness / Output Processing</button>
  <div id="loud-section" class="collapse-body collapsed">
    <div class="param-grid" style="margin-top:10px">
      <div><label for="latent-rescale">Latent Rescale</label><input id="latent-rescale" type="number" step="0.05" min="0"></div>
      <div><label for="latent-shift">Latent Shift</label><input id="latent-shift" type="number" step="0.05"></div>
      <div><label for="latent-target-std">Latent Target Std <span class="small-note">(blank=off)</span></label><input id="latent-target-std" type="number" step="0.05" min="0" placeholder="off"></div>
      <div><label for="latent-adapt-min">Adapt Min</label><input id="latent-adapt-min" type="number" step="0.05" min="0"></div>
      <div><label for="latent-adapt-max">Adapt Max</label><input id="latent-adapt-max" type="number" step="0.05" min="0"></div>
      <div><label for="peak-normalize-db">Peak Normalize dB <span class="small-note">(blank=off)</span></label><input id="peak-normalize-db" type="number" step="0.5" placeholder="off"></div>
      <div><label for="limiter-ceiling-db">Limiter Ceiling dB <span class="small-note">(blank=off)</span></label><input id="limiter-ceiling-db" type="number" step="0.1" placeholder="off"></div>
      <div><label for="limiter-knee">Limiter Knee</label><input id="limiter-knee" type="number" step="0.05" min="0" max="1"></div>
    </div>
  </div>
</div>

<!-- ─── Init Audio / Inpaint ─────────────────────────────────────────── -->
<div class="card">
  <button class="collapse-toggle collapsed" data-target="init-audio-section" type="button">Init Audio &amp; Inpainting</button>
  <div id="init-audio-section" class="collapse-body collapsed">
  <div class="param-grid">
    <div style="grid-column:1/-1">
      <div class="row" style="gap:6px;align-items:end">
        <div style="flex:3">
          <label for="init-audio-select">Active init audio (audio-in/)</label>
          <div class="row" style="gap:6px">
            <select id="init-audio-select" style="flex:1">
              <option value="">-- none (text-to-music) --</option>
            </select>
            <button id="init-audio-refresh-btn" class="small" title="Refresh file list">Refresh</button>
          </div>
        </div>
        <div style="flex:2">
          <label for="init-audio-upload">Upload WAV</label>
          <div class="row" style="gap:6px">
            <input id="init-audio-upload" type="file" accept=")sa3web") +
    std::string(R"sa3web(.wav,.WAV" style="flex:1;padding:4px 0">
            <button id="init-audio-upload-btn" class="small">Upload</button>
          </div>
        </div>
      </div>
    </div>
    <input id="init-path" type="hidden" value="">
    <div><label for="init-noise-level">Init Noise Level</label><input id="init-noise-level" type="number" step="0.05" min="0" max="1" value="0.85"></div>
    <div><label for="inpaint-start">Inpaint Start (s)</label><input id="inpaint-start" type="number" step="0.5" value="-1"></div>
    <div><label for="inpaint-end">Inpaint End (s)</label><input id="inpaint-end" type="number" step="0.5" value="-1"></div>
  </div>
  <div class="small-note">Set inpaint_start >= 0 to enable inpainting (requires init audio + local-cond DiT). -1 = disabled.</div>
  </div>
</div>

<!-- ─── LoRAs ─────────────────────────────────────────────────────────── -->
<div class="card">
  <button class="collapse-toggle collapsed" data-target="lora-section" type="button">Creative LoRAs</button>
  <div id="lora-section" class="collapse-body collapsed">
  <div class="row gapped">
    <div class="col" style="flex:2">
      <label for="lora-select">Name</label>
      <div class="row" style="gap:6px">
        <select id="lora-select" style="flex:1"></select>
        <button id="lora-add-btn" class="small">Add</button>
      </div>
    </div>
  </div>
  <div id="active-loras"></div>
  <div class="small-note">Add a creative adapter, then set its strength. Decoder corrections are in Models.</div>
  </div>
</div>

<!-- ─── Generate Buttons ─────────────────────────────────────────────── -->
<h2>Generate</h2>
<div class="card">
  <div class="row gapped">
    <div class="col"><button id="gen-btn" class="primary" style="width:100%">Generate</button></div>
    <div class="col">
      <button id="loop-btn" class="loop" style="width:100%">Generate Loop</button>
    </div>
  </div>
  <div class="row gapped" style="margin-top:8px">
    <div class="col" style="flex:0 0 100px"><label for="loop-bpm">BPM</label><input id="loop-bpm" type="number" min="20" max="300" value="120"></div>
    <div class="col" style="flex:0 0 100px"><label for="loop-bars">Bars</label><select id="loop-bars"><option>4</option><option selected>8</option><option>16</option><option>32</option></select></div>
    <div class="col"><div class="small-note" style="margin-top:20px">Ctrl+Enter to generate</div></div>
  </div>
  <div id="error-msg"></div>
</div>
<!-- ─── Progress & Results ───────────────────────────────────────────── -->
<h2>Progress &amp; Results</h2>
<div class="card">
  <div id="progress-wrap">
    <div id="progress-bar"></div>
    <div id="progress-label">idle</div>
  </div>
  <hr>
  <div id="result-section" style="display:none">
    <div style="display:flex;align-items:center;gap:8px">
      <audio id="result-audio"></audio>
      <button id="delete-current-btn" class="small danger" style="flex:none">✕ Delete</button>
    </div>
    <div id="seed-info"></div>
  </div>
</div>

<!-- ─── Past Songs ──────────────────────────────────────────── -->
<h2 style="display:flex;align-items:center;gap:8px">Past Songs <span id="past-count" class="badge">0</span>
  <button id="clear-all-btn" class="small danger" style="margin-left:auto">Clear history</button>
</h2>
<div id="past-songs" class="card"></div>

<p class="studio-credit">Inference and training web interfaces by <a href="https://github.com/pillopaus-project/sa3.cpp" target="_blank" rel="noopener">pillopaus-project</a>; integrated into sa3.cpp.</p>
</div>
<script>document.getElementById('training-link').href = location.protocol + '//' + location.hostname + ':8016/';</script>
<script src="app.js?v=6"></script>
<script src="studio.js?v=9"></script>
</body>
</html>
)sa3web");

inline const std::string app_js =
    std::string(R"sa3web("use strict";
// ─── Types matching the sa3-server HTTP API ────────────────────────────────
let currentConfigFilename = "";
const CONFIG_EXT = ".json";
// ─── Dist-shift default profiles ───────────────────────────────────────────
const DIST_SHIFT_DEFAULTS = {
    LogSNR: [2000, -6.2, 0, 2],
    Flux: [256, 4096, 6.93, 6.93],
    Full: [0.5, 1.15, 256, 4096],
    None: [0, 0, 0, 0],
};
const DIST_SHIFT_LABELS = {
    LogSNR: ["anchor_length", "anchor_logsnr", "rate", "logsnr_end"],
    Flux: ["min_length", "max_length", "alpha_min", "alpha_max"],
    Full: ["base_shift", "max_shift", "min_length", "max_length"],
    None: ["—", "—", "—", "—"],
};
// ─── State ──────────────────────────────────────────────────────────────────
const server = { host: "127.0.0.1", port: 8006 };
let loraList = [];
let activeLoras = [];
let pastSongs = [];
let lastGenParams = null;
let lastGenSeed = 0;
let uploadInProgress = false;
let currentResult = null;
// ─── DOM helpers ────────────────────────────────────────────────────────────
const $ = (s) => document.querySelector(s);
function val(s) {
    return $(s).value;
}
function num(s) {
    return parseFloat(val(s));
}
function int(s) {
    return parseInt(val(s), 10);
}
function isChecked(s) {
    return $(s).checked;
}
function setVal(s, v) {
    const el = $(s);
    if (el.type === "checkbox")
        el.checked = v;
    else
        el.value = String(v);
}
function apiBase() {
    return location.origin;
}
// ─── API calls ──────────────────────────────────────────────────────────────
async function apiGet(path) {
    const r = await fetch(`${apiBase()}${path}`);
    if (!r.ok)
        throw new Error(`HTTP ${r.status}: ${r.statusText}`);
    return r.json();
}
async function apiPost(path, body) {
    const r = await fetch(`${apiBase()}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    const data = await r.json();
    if (!r.ok)
        throw new Error(data.error || `HTTP ${r.status}: ${r.statusText}`);
    return data;
}
// ─── Slider-number sync ─────────────────────────────────────────────────────
function syncSliderToNum(sliderId, numId) {
    const slider = $(sliderId);
    const numInput = $(numId);
    if (!slider || !numInput)
        return;
    slider.addEventListener("input", () => { numInput.value = slider.value; });
    numInput.addEventListener("input", () => { slider.value = numInput.value; });
    numInput.addEventListener("change", () => { slider.value = numInput.value; });
}
// ─── Read form into request object ─────────────────────────────────────────
function readForm() {
    const distShift = val("#dist-shift");
    const dsParams = [
        num("#dsp1"), num("#dsp2"), num("#dsp3"), num("#dsp4"),
    ];
    const hasLatentTarget = val("#latent-target-std").trim().length > 0;
    const hasPeakNorm = val("#peak-normalize-db").trim().length > 0;
    const hasLimiter = val("#limiter-ceiling-db").trim().length > 0;
    const initPath = val("#init-path").trim();
    const negPrompt = val("#negative-prompt").trim();
    return {
        prompt: val("#prompt"),
        duration: num("#duration"),
        steps: int("#steps"),
        seed: int("#seed"),
        ...(initPath ? { init_path: initPath } : {}),
        init_noise_level: num("#init-noise-level"),
        inpaint_start: num("#inpaint-start"),
        inpaint_end: num("#inpaint-end"),
        ...(negPrompt ? { negative_prompt: negPrompt } : {}),
        cfg_scale: num("#cfg-scale"),
        cfg_rescale: num("#cfg-rescale"),
        apg_scale: num("#apg-scale"),
        cfg_norm_threshold: num("#cfg-norm-threshold"),
        cfg_interval_min: num("#cfg-interval-min"),
        cfg_interval_max: num("#cfg-interval-max"),
        dist_shift: distShift,
        dist_shift_params: dsParams,
        duration_padding_sec: num("#duration-padding"),
        keep_models: isChecked("#keep-models"),
        loras: [...activeLoras, ...(window.st)sa3web") +
    std::string(R"sa3web(udioDecoderAdapter?.() || [])],
        encode_chunk_size: int("#encode-chunk-size"),
        encode_overlap: int("#encode-overlap"),
        decode_chunk_size: int("#decode-chunk-size"),
        decode_overlap: int("#decode-overlap"),
        latent_rescale: num("#latent-rescale"),
        latent_shift: num("#latent-shift"),
        ...(hasLatentTarget ? { latent_target_std: num("#latent-target-std") } : { latent_target_std: null }),
        latent_adapt_min: num("#latent-adapt-min"),
        latent_adapt_max: num("#latent-adapt-max"),
        ...(hasPeakNorm ? { peak_normalize_db: num("#peak-normalize-db") } : { peak_normalize_db: null }),
        ...(hasLimiter ? { limiter_ceiling_db: num("#limiter-ceiling-db") } : { limiter_ceiling_db: null }),
        limiter_knee: num("#limiter-knee"),
    };
}
// ─── Health ─────────────────────────────────────────────────────────────────
async function checkHealth() {
    const statusEl = $("#server-status");
    const modelInfo = $("#model-info");
    statusEl.textContent = "Connecting…";
    statusEl.className = "";
    statusEl.style.display = "";
    try {
        const h = await apiGet("/health");
        statusEl.textContent = "✓ Connected";
        statusEl.className = "ok";
        modelInfo.textContent = `${h.model} / ${h.encoding} ${h.loaded ? "(loaded)" : "(unloaded)"}`;
        modelInfo.style.display = "";
        loadLoras();
        loadInitAudioList();
        applyLoudnessDefaults(h.loudness_defaults);
    }
    catch {
        statusEl.textContent = "✗ Server unreachable";
        statusEl.className = "err";
        modelInfo.style.display = "none";
    }
}
function applyLoudnessDefaults(defaults) {
    if (defaults.latent_rescale != null)
        setVal("#latent-rescale", defaults.latent_rescale);
    if (defaults.latent_shift != null)
        setVal("#latent-shift", defaults.latent_shift);
    if (defaults.latent_adapt_min != null)
        setVal("#latent-adapt-min", defaults.latent_adapt_min);
    if (defaults.latent_adapt_max != null)
        setVal("#latent-adapt-max", defaults.latent_adapt_max);
    if (defaults.limiter_knee != null)
        setVal("#limiter-knee", defaults.limiter_knee);
    if (defaults.latent_target_std != null) {
        setVal("#latent-target-std", defaults.latent_target_std);
    }
    if (defaults.peak_normalize_db != null) {
        setVal("#peak-normalize-db", defaults.peak_normalize_db);
    }
    if (defaults.limiter_ceiling_db != null) {
        setVal("#limiter-ceiling-db", defaults.limiter_ceiling_db);
    }
}
// ─── Loras ──────────────────────────────────────────────────────────────────
async function loadLoras() {
    try {
        const r = await apiGet("/loras");
        loraList = r.loras;
        activeLoras = activeLoras.filter((l) => !loraList.some((known) => known.name === l.name && known.target === "decoder"));
        renderLoraDropdown();
        window.studioLorasReady?.();
    }
    catch {
        // server not connected yet
    }
}
function renderLoraDropdown() {
    const sel = $("#lora-select");
    sel.innerHTML = '<option value="">— select —</option>';
    for (const l of loraList.filter((item) => item.target !== "decoder" && item.target !== "encoder")) {
        const opt = document.createElement("option");
        opt.value = l.name;
        opt.textContent = l.name;
        sel.appendChild(opt);
    }
    renderActiveLoras();
}
function addLora() {
    const sel = $("#lora-select");
    const name = sel.value;
    if (!name)
        return;
    if (activeLoras.some((l) => l.name === name))
        return;
    activeLoras.push({ name, strength: 1 });
    sel.value = "";
    renderActiveLoras();
}
function removeLora(name) {
    activeLoras = activeLoras.filter((l) => l.name !== name);
    renderActiveLoras();
}
function renderActiveLoras() {
    const container = $("#active-loras");
    container.innerHTML = "";
    for (const l of activeLoras) {
        const row = document.createElement("div"); row.classNam)sa3web") +
    std::string(R"sa3web(e = "studio-lora-row";
        const name = document.createElement("span"); name.textContent = l.name;
        const slider = document.createElement("input"); slider.type = "range"; slider.min = "0"; slider.max = "2"; slider.step = "0.05"; slider.value = String(l.strength);
        slider.setAttribute("aria-label", `${l.name} strength`);
        const value = document.createElement("output"); value.textContent = Number(l.strength).toFixed(2);
        slider.addEventListener("input", () => { l.strength = Number(slider.value); value.textContent = l.strength.toFixed(2); });
        const remove = document.createElement("button"); remove.type = "button"; remove.className = "small"; remove.textContent = "Remove";
        remove.addEventListener("click", () => removeLora(l.name));
        row.append(name, slider, value, remove); container.appendChild(row);
    }
}
// ─── Init Audio ────────────────────────────────────────────────────────────
async function loadInitAudioList() {
    try {
        const r = await apiGet("/init-audio");
        const sel = $("#init-audio-select");
        sel.innerHTML = '<option value="">-- none (text-to-music) --</option>';
        for (const f of r.files) {
            const opt = document.createElement("option");
            opt.value = f.path;
            opt.textContent = f.name;
            sel.appendChild(opt);
        }
    }
    catch {
        // server not available
    }
}
function onInitAudioSelect() {
    const sel = $("#init-audio-select");
    const path = sel.value;
    setVal("#init-path", path);
}
async function uploadInitAudio() {
    const input = $("#init-audio-upload");
    const file = input.files?.[0];
    if (!file)
        return;
    const btn = $("#init-audio-upload-btn");
    btn.disabled = true;
    btn.textContent = "Uploading…";
    try {
        const form = new FormData();
        form.append("file", file);
        const r = await fetch(`${apiBase()}/init-audio/upload`, {
            method: "POST",
            body: form,
        });
        if (!r.ok)
            throw new Error(`Upload failed: ${r.status}`);
        await loadInitAudioList();
        showError("");
    }
    catch (e) {
        showError(e instanceof Error ? e.message : "Upload failed");
    }
    finally {
        btn.disabled = false;
        btn.textContent = "Upload";
        input.value = "";
    }
}
// ─── Dist-shift parameter defaults ──────────────────────────────────────────
function onDistShiftChange() {
    const type = val("#dist-shift");
    const labels = DIST_SHIFT_LABELS[type] || ["p1", "p2", "p3", "p4"];
    const defaults = DIST_SHIFT_DEFAULTS[type] || [0, 0, 0, 0];
    for (let i = 0; i < 4; i++) {
        const input = $(`#dsp${i + 1}`);
        const label = document.querySelector(`label[for="dsp${i + 1}"]`);
        if (label)
            label.textContent = labels[i];
        // Only reset to defaults if the user hasn't manually edited this param
        if (input.dataset.userEdited === undefined) {
            input.value = String(defaults[i]);
        }
        input.disabled = type === "None";
    }
}
// ─── Past Songs ────────────────────────────────────────────────────────────
function pushPastSong(entry) {
    pastSongs.push(entry);
    try { localStorage.setItem("sa3-past-songs", JSON.stringify(pastSongs)); }
    catch { /* Large WAV data can exceed browser storage; keep this session's history. */ }
    renderPastSongs();
}
function renderPastSongs() {
    const container = $("#past-songs");
    const countEl = $("#past-count");
    if (countEl)
        countEl.textContent = String(pastSongs.length);
    container.innerHTML = "";
    for (let i = pastSongs.length - 1; i >= 0; i--) {
        const s = pastSongs[i];
        const div = document.createElement("div");
        div.className = "song-entry";
        div.innerHTML = `<span class="song-name" title="${escapeHtml(s.prompt || "")}">${escapeHtml((s.prompt || "(no prompt)").slice(0, 30))}</span>
      <span class=")sa3web") +
    std::string(R"sa3web(song-params">seed: ${s.seed}</span>
      <span class="song-actions">
        <button class="small open-song-btn" data-index="${i}" title="Open in waveform">Open</button>
        <button class="small load-params-btn" data-index="${i}" title="Open audio and restore generation settings">Reuse</button>
        <button class="small download-song-btn" data-index="${i}" title="Download WAV">Download</button>
        <button class="small danger delete-song-btn" data-index="${i}" title="Delete">Remove</button>
      </span>`;
        container.appendChild(div);
    }
    for (const btn of container.querySelectorAll(".open-song-btn")) {
        btn.addEventListener("click", () => {
            const s = pastSongs[parseInt(btn.dataset.index || "0", 10)];
            if (s) window.studioLoadTake?.(s);
        });
    }
    for (const btn of container.querySelectorAll(".delete-song-btn")) {
        btn.addEventListener("click", () => {
            const idx = parseInt(btn.dataset.index || "0", 10);
            pastSongs.splice(idx, 1);
            try { localStorage.setItem("sa3-past-songs", JSON.stringify(pastSongs)); }
            catch { /* Keep in-memory history if storage is full. */ }
            renderPastSongs();
        });
    }
    for (const btn of container.querySelectorAll(".download-song-btn")) {
        btn.addEventListener("click", () => {
            const idx = parseInt(btn.dataset.index || "0", 10);
            const s = pastSongs[idx];
            if (!s)
                return;
            const ts = new Date(s.timestamp).toISOString().replace(/[:.]/g, "-").replace("T", "_").slice(0, 19);
            const a = document.createElement("a");
            a.href = s.audioUrl;
            a.download = `sa3-${s.seed}-${ts}.wav`;
            a.click();
        });
    }
    for (const btn of container.querySelectorAll(".load-params-btn")) {
        btn.addEventListener("click", () => {
            const idx = parseInt(btn.dataset.index || "0", 10);
            const s = pastSongs[idx];
            if (!s)
                return;
            if (s.params)
                loadParamsFromSnapshot(s.params);
            window.studioLoadTake?.(s);
        });
    }
}
function loadParamsFromSnapshot(params) {
    const set = (id, val) => {
        if (val != null)
            setVal(id, val);
    };
    set("#prompt", params.prompt);
    set("#negative-prompt", params.negative_prompt || "");
    set("#duration", params.duration);
    set("#duration-num", params.duration);
    set("#steps", params.steps);
    set("#steps-num", params.steps);
    set("#seed", params.seed);
    set("#duration-padding", params.duration_padding_sec);
    set("#duration-padding-num", params.duration_padding_sec);
    set("#cfg-scale", params.cfg_scale);
    set("#cfg-rescale", params.cfg_rescale);
    set("#apg-scale", params.apg_scale);
    set("#cfg-norm-threshold", params.cfg_norm_threshold);
    set("#cfg-interval-min", params.cfg_interval_min);
    set("#cfg-interval-max", params.cfg_interval_max);
    set("#dist-shift", params.dist_shift);
    const dsp = params.dist_shift_params;
    if (dsp) {
        for (let i = 0; i < 4 && i < dsp.length; i++) {
            const inp = $(`#dsp${i + 1}`);
            inp.value = String(dsp[i]);
            inp.dataset.userEdited = "true";
        }
    }
    onDistShiftChange();
    set("#keep-models", params.keep_models);
    set("#encode-chunk-size", params.encode_chunk_size);
    set("#encode-overlap", params.encode_overlap);
    set("#decode-chunk-size", params.decode_chunk_size);
    set("#decode-overlap", params.decode_overlap);
    set("#latent-rescale", params.latent_rescale);
    set("#latent-shift", params.latent_shift);
    const lts = params.latent_target_std;
    set("#latent-target-std", lts != null && lts !== false ? String(lts) : "");
    set("#latent-adapt-min", params.latent_adapt_min);
    set("#latent-adapt-max", params.latent_adapt_max);
    const pndb = params.peak_normalize_db;
    set(")sa3web") +
    std::string(R"sa3web(#peak-normalize-db", pndb != null && pndb !== false ? String(pndb) : "");
    const lcdb = params.limiter_ceiling_db;
    set("#limiter-ceiling-db", lcdb != null && lcdb !== false ? String(lcdb) : "");
    set("#limiter-knee", params.limiter_knee);
    set("#init-path", params.init_path || "");
    set("#init-noise-level", params.init_noise_level);
    set("#inpaint-start", params.inpaint_start);
    set("#inpaint-end", params.inpaint_end);
    const bpm = params.bpm;
    if (bpm != null)
        set("#loop-bpm", bpm);
    const bars = params.bars;
    if (bars != null)
        set("#loop-bars", bars);
    // restore LoRAs
    const loras = params.loras;
    if (loras) {
        activeLoras = loras.filter((l) => !loraList.some((known) => known.name === l.name && known.target === "decoder")).map((l) => ({ ...l }));
        renderActiveLoras();
    }
}
function clearPastSongs() {
    pastSongs = [];
    localStorage.removeItem("sa3-past-songs");
    renderPastSongs();
}
function deleteCurrentSong() {
    if (!currentResult)
        return;
    currentResult = null;
    const resultSection = $("#result-section");
    const resultAudio = $("#result-audio");
    resultSection.style.display = "none";
    resultAudio.src = "";
}
function loadPastSongs() {
    try {
        const saved = localStorage.getItem("sa3-past-songs");
        if (saved) {
            pastSongs = JSON.parse(saved);
            renderPastSongs();
        }
    }
    catch {
        // ignore corrupt data
    }
}
// ─── Generate ───────────────────────────────────────────────────────────────
let pollTimer = null;
async function generate(overrides = {}) {
    clearPolling();
    if (currentResult) {
        pushPastSong(currentResult);
        currentResult = null;
    }
    const body = { ...readForm(), ...overrides };
    lastGenParams = { ...body };
    const btn = $("#gen-btn");
    btn.disabled = true;
    btn.textContent = "Generating…";
    try {
        const r = await apiPost("/generate", body);
        lastGenSeed = r.seed;
        startPolling(r.session_id);
    }
    catch (e) {
        lastGenParams = null;
        showError(e instanceof Error ? e.message : "Request failed");
        btn.disabled = false;
        btn.textContent = "Generate";
    }
}
async function generateLoop(overrides = {}) {
    clearPolling();
    if (currentResult) {
        pushPastSong(currentResult);
        currentResult = null;
    }
    const body = {
        ...readForm(), ...overrides,
        bpm: num("#loop-bpm"),
        bars: int("#loop-bars"),
    };
    lastGenParams = { ...body };
    const btn = $("#loop-btn");
    btn.disabled = true;
    btn.textContent = "Generating loop…";
    try {
        const r = await apiPost("/generate/loop", body);
        lastGenSeed = r.seed;
        startPolling(r.session_id);
    }
    catch (e) {
        lastGenParams = null;
        showError(e instanceof Error ? e.message : "Request failed");
        btn.disabled = false;
        btn.textContent = "Generate Loop";
    }
}
function startPolling(sessionId) {
    const progressBar = $("#progress-bar");
    const progressLabel = $("#progress-label");
    const resultAudio = $("#result-audio");
    const resultSection = $("#result-section");
    const seedInfo = $("#seed-info");
    progressBar.style.width = "0%";
    progressLabel.textContent = "queued";
    resultSection.style.display = "none";
    resultAudio.src = "";
    seedInfo.textContent = "";
    showError("");
    pollTimer = setInterval(async () => {
        try {
            const r = await apiGet(`/poll_status/${sessionId}`);
            progressBar.style.width = `${r.progress}%`;
            if (r.status === "queued") {
                progressLabel.textContent = "queued…";
            }
            else if (r.status === "generating" || r.status === "encoding" || r.status === "decoding" || r.status === "finalizing") {
                progressLabel.textContent = `${r.status} step ${r.step}/${r.total_steps} (${r.progres)sa3web") +
    std::string(R"sa3web(s}%)`;
            }
            else if (r.status === "completed") {
                progressLabel.textContent = `completed (${r.progress}%)`;
                if (r.audio_data) {
                    resultAudio.src = `data:audio/wav;base64,${r.audio_data}`;
                    resultSection.style.display = "block";
                }
                const resolvedSeed = r.meta?.seed ?? lastGenSeed;
                const metaParts = [];
                if (resolvedSeed != null)
                    metaParts.push(`Seed: ${resolvedSeed}`);
                if (r.meta?.loudness) {
                    const lm = r.meta.loudness;
                    if (lm.final_peak != null)
                        metaParts.push(`Peak: ${Number(lm.final_peak).toFixed(3)}`);
                    if (lm.decoded_peak != null)
                        metaParts.push(`Decoded: ${Number(lm.decoded_peak).toFixed(3)}`);
                }
                seedInfo.textContent = metaParts.join(" · ");
                if (lastGenParams && r.audio_data) {
                    currentResult = {
                        timestamp: Date.now(),
                        seed: resolvedSeed,
                        audioUrl: `data:audio/wav;base64,${r.audio_data}`,
                        params: { ...lastGenParams },
                        prompt: lastGenParams.prompt || "",
                    };
                    window.studioResultReady?.(currentResult);
                    lastGenParams = null;
                }
                clearPolling();
                enableButtons();
            }
            else if (r.status === "failed") {
                progressLabel.textContent = `failed: ${r.error || "unknown error"}`;
                clearPolling();
                enableButtons();
            }
        }
        catch {
            progressLabel.textContent = "poll error";
            clearPolling();
            enableButtons();
        }
    }, 500);
}
function clearPolling() {
    if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
    }
}
function enableButtons() {
    const genBtn = $("#gen-btn");
    genBtn.disabled = false;
    genBtn.textContent = "Generate";
    const loopBtn = $("#loop-btn");
    loopBtn.disabled = false;
    loopBtn.textContent = "Generate Loop";
    window.studioSetBusy?.(false);
}
function showError(msg) {
    const el = $("#error-msg");
    if (el)
        el.textContent = msg;
}
function readFormAsConfig() {
    const dsParams = [
        num("#dsp1"), num("#dsp2"), num("#dsp3"), num("#dsp4"),
    ];
    const ltRaw = val("#latent-target-std").trim();
    const pnRaw = val("#peak-normalize-db").trim();
    const lcRaw = val("#limiter-ceiling-db").trim();
    return {
        version: 1,
        prompt: val("#prompt"),
        negative_prompt: val("#negative-prompt").trim(),
        duration: num("#duration"),
        steps: int("#steps"),
        seed: int("#seed"),
        duration_padding_sec: num("#duration-padding"),
        cfg_scale: num("#cfg-scale"),
        cfg_rescale: num("#cfg-rescale"),
        apg_scale: num("#apg-scale"),
        cfg_norm_threshold: num("#cfg-norm-threshold"),
        cfg_interval_min: num("#cfg-interval-min"),
        cfg_interval_max: num("#cfg-interval-max"),
        dist_shift: val("#dist-shift"),
        dist_shift_params: dsParams,
        keep_models: isChecked("#keep-models"),
        encode_chunk_size: int("#encode-chunk-size"),
        encode_overlap: int("#encode-overlap"),
        decode_chunk_size: int("#decode-chunk-size"),
        decode_overlap: int("#decode-overlap"),
        latent_rescale: num("#latent-rescale"),
        latent_shift: num("#latent-shift"),
        latent_target_std: ltRaw.length > 0 ? num("#latent-target-std") : null,
        latent_adapt_min: num("#latent-adapt-min"),
        latent_adapt_max: num("#latent-adapt-max"),
        peak_normalize_db: pnRaw.length > 0 ? num("#peak-normalize-db") : null,
        limiter_ceiling_db: lcRaw.length > 0)sa3web") +
    std::string(R"sa3web( ? num("#limiter-ceiling-db") : null,
        limiter_knee: num("#limiter-knee"),
        init_path: val("#init-path").trim(),
        init_noise_level: num("#init-noise-level"),
        inpaint_start: num("#inpaint-start"),
        inpaint_end: num("#inpaint-end"),
        loop_bpm: num("#loop-bpm"),
        loop_bars: int("#loop-bars"),
        loras: activeLoras.map((l) => ({ ...l })),
    };
}
function applyConfig(cfg) {
    setVal("#prompt", cfg.prompt);
    setVal("#negative-prompt", cfg.negative_prompt || "");
    setVal("#duration", cfg.duration);
    setVal("#duration-num", cfg.duration);
    setVal("#steps", cfg.steps);
    setVal("#steps-num", cfg.steps);
    setVal("#seed", cfg.seed);
    setVal("#duration-padding", cfg.duration_padding_sec);
    setVal("#duration-padding-num", cfg.duration_padding_sec);
    setVal("#cfg-scale", cfg.cfg_scale);
    setVal("#cfg-rescale", cfg.cfg_rescale);
    setVal("#apg-scale", cfg.apg_scale);
    setVal("#cfg-norm-threshold", cfg.cfg_norm_threshold);
    setVal("#cfg-interval-min", cfg.cfg_interval_min);
    setVal("#cfg-interval-max", cfg.cfg_interval_max);
    setVal("#dist-shift", cfg.dist_shift);
    if (cfg.dist_shift_params) {
        for (let i = 0; i < 4; i++) {
            const inp = $(`#dsp${i + 1}`);
            inp.value = String(cfg.dist_shift_params[i]);
            inp.dataset.userEdited = "true";
        }
    }
    onDistShiftChange();
    setVal("#keep-models", cfg.keep_models);
    setVal("#encode-chunk-size", cfg.encode_chunk_size);
    setVal("#encode-overlap", cfg.encode_overlap);
    setVal("#decode-chunk-size", cfg.decode_chunk_size);
    setVal("#decode-overlap", cfg.decode_overlap);
    setVal("#latent-rescale", cfg.latent_rescale);
    setVal("#latent-shift", cfg.latent_shift);
    setVal("#latent-target-std", cfg.latent_target_std != null ? String(cfg.latent_target_std) : "");
    setVal("#latent-adapt-min", cfg.latent_adapt_min);
    setVal("#latent-adapt-max", cfg.latent_adapt_max);
    setVal("#peak-normalize-db", cfg.peak_normalize_db != null ? String(cfg.peak_normalize_db) : "");
    setVal("#limiter-ceiling-db", cfg.limiter_ceiling_db != null ? String(cfg.limiter_ceiling_db) : "");
    setVal("#limiter-knee", cfg.limiter_knee);
    setVal("#init-path", cfg.init_path || "");
    setVal("#init-noise-level", cfg.init_noise_level);
    setVal("#inpaint-start", cfg.inpaint_start);
    setVal("#inpaint-end", cfg.inpaint_end);
    setVal("#loop-bpm", cfg.loop_bpm);
    setVal("#loop-bars", cfg.loop_bars);
    activeLoras = cfg.loras.map((l) => ({ ...l }));
    renderActiveLoras();
}
function saveConfig() {
    const suggested = currentConfigFilename || "sa3-config.json";
    const name = prompt("Save config as:", suggested);
    if (!name)
        return;
    const finalName = name.endsWith(CONFIG_EXT) ? name : name + CONFIG_EXT;
    currentConfigFilename = finalName;
    const cfg = readFormAsConfig();
    const blob = new Blob([JSON.stringify(cfg, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = finalName;
    a.click();
    URL.revokeObjectURL(url);
    const fnEl = $("#config-filename");
    if (fnEl)
        fnEl.textContent = currentConfigFilename;
}
function loadConfig() {
    const input = $("#load-config-input");
    input.value = "";
    input.click();
}
function onConfigFileSelected(e) {
    const file = e.target.files?.[0];
    if (!file)
        return;
    const reader = new FileReader();
    reader.onload = () => {
        try {
            const cfg = JSON.parse(reader.result);
            if (cfg.version !== 1) {
                showError("Unsupported config version");
                return;
            }
            applyConfig(cfg);
            currentConfigFilename = file.name;
            const fnEl = $("#config-filename");
            if (fnEl)
                fnEl.textContent = currentConfigFilename;
            sho)sa3web") +
    std::string(R"sa3web(wError("");
        }
        catch {
            showError("Invalid config file");
        }
    };
    reader.readAsText(file);
}
// ─── Helpers ────────────────────────────────────────────────────────────────
function escapeHtml(s) {
    const d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
}
// ─── Toggle collapsible sections ────────────────────────────────────────────
function setupCollapsibles() {
    for (const btn of document.querySelectorAll(".collapse-toggle")) {
        btn.addEventListener("click", () => {
            const target = document.getElementById(btn.getAttribute("data-target") || "");
            if (target) {
                target.classList.toggle("collapsed");
                btn.classList.toggle("collapsed");
            }
        });
    }
}
// ─── Init ───────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    setupCollapsibles();
    loadPastSongs();
    // Sync range sliders with their number companions
    syncSliderToNum("#duration", "#duration-num");
    syncSliderToNum("#steps", "#steps-num");
    syncSliderToNum("#duration-padding", "#duration-padding-num");
    // Dist-shift defaults
    onDistShiftChange();
    // Event listeners
    $("#gen-btn").addEventListener("click", generate);
    $("#loop-btn").addEventListener("click", generateLoop);
    $("#lora-add-btn").addEventListener("click", addLora);
    $("#dist-shift").addEventListener("change", onDistShiftChange);
    $("#save-config-btn").addEventListener("click", saveConfig);
    $("#load-config-btn").addEventListener("click", loadConfig);
    $("#load-config-input").addEventListener("change", onConfigFileSelected);
    $("#init-audio-refresh-btn").addEventListener("click", loadInitAudioList);
    $("#init-audio-upload-btn").addEventListener("click", uploadInitAudio);
    $("#init-audio-select").addEventListener("change", onInitAudioSelect);
    $("#clear-all-btn").addEventListener("click", clearPastSongs);
    $("#delete-current-btn").addEventListener("click", deleteCurrentSong);
    // Mark dist-shift params as user-edited on first input
    for (let i = 1; i <= 4; i++) {
        const inp = $(`#dsp${i}`);
        inp.addEventListener("input", () => { inp.dataset.userEdited = "true"; });
        inp.addEventListener("change", () => { inp.dataset.userEdited = "true"; });
    }
    // Ctrl+Enter (or Cmd+Enter) triggers generate
    document.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            generate();
        }
    });
    // Auto-connect to server on page load
    checkHealth();
});
)sa3web");

inline const std::string studio_js =
    std::string(R"sa3web("use strict";

// The native inference and training servers remain independent. Their browser
// views share this one shell: hiding the training iframe never reloads it.
const studio = {
  view: "studio", operation: "create", buffer: null, audioUrl: "", title: "",
  selection: [0, 1], cropMode: false, pointer: null, busy: false, peaks: null, frame: 0,
  activeModel: "",
};
const studioEl = (id) => document.getElementById(id);
const studioIcons = {
  upload: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 16V4m0 0-4 4m4-4 4 4M4 17v3h16v-3" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  crop: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 3v14a1 1 0 0 0 1 1h13M3 7h14a1 1 0 0 1 1 1v13" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  check: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m4 12 5 5L20 6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  play: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m8 5 11 7-11 7z" fill="currentColor"/></svg>',
  pause: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 5h3v14H7zm7 0h3v14h-3z" fill="currentColor"/></svg>',
  stop: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6h12v12H6z" fill="currentColor"/></svg>',
};

function studioShell() {
  const root = studioEl("studio-root");
  root.innerHTML = `<div class="studio-shell">
    <header class="studio-header">
      <h1><a href="https://github.com/betweentwomidnights/sa3.cpp" target="_blank" rel="noopener" style="color:var(--accent);text-decoration:none">SA3.CPP</a> <small>studio</small></h1>
      <nav aria-label="Studio views"><button type="button" data-studio-view="studio" aria-current="page">Studio</button><button type="button" data-studio-view="train">LoRA training</button><button type="button" data-studio-view="models">Models</button><button type="button" data-studio-view="settings">Settings</button></nav>
      <div class="status" id="studio-header-status"></div>
    </header>
    <main id="studio-view-studio" class="studio-view">
      <div class="studio-hero"><div class="studio-actions"><button type="button" class="primary" data-studio-operation="create">create</button><button type="button" data-studio-operation="continue" hidden disabled>continue</button><button type="button" data-studio-operation="transform" hidden disabled>transform</button></div></div>
      <section class="studio-panel" aria-label="Current take"><div class="studio-panel-title"><h2>Current take</h2><span id="studio-take-info">No audio yet</span></div><div class="studio-wave-wrap" id="studio-wave-wrap"><canvas id="studio-waveform" aria-label="Audio waveform; click to seek"></canvas><div id="studio-wave-empty">Create audio or drop a WAV here.</div><div class="studio-wave-actions"><button id="studio-upload-wave" type="button" aria-label="Upload WAV" title="Upload WAV">${studioIcons.upload}</button><button id="studio-crop" type="button" aria-label="Select crop" title="Select crop" disabled>${studioIcons.crop}</button></div><div id="studio-drop-hint" hidden>Drop WAV to open it</div><input id="studio-import-file" type="file" accept="audio/wav,.wav" hidden></div><div class="studio-take-tools"><div id="studio-audio-slot"></div><button id="studio-play" type="button" aria-label="Play" title="Play" disabled>${studioIcons.play}</button><button id="studio-stop" type="button" aria-label="Stop" title="Stop" disabled>${studioIcons.stop}</button><span id="studio-time" class="studio-time">0:00 / 0:00</span><span id="studio-selection-label" hidden>No selection</span><button id="studio-download" type="button" disabled>Download WAV</button><button id="studio-clear" type="button" disabled>Clear take</button></div><div id="studio-meta-slot"></div><div id="studio-progress-slot"></div><div id="studio-error-slot"></div></section>
      <div class=")sa3web") +
    std::string(R"sa3web(studio-history-head"><h2>Past takes <span id="studio-history-count"></span></h2><div id="studio-history-actions"></div></div><div id="studio-history-slot"></div>
    </main>
    <main id="studio-view-train" class="studio-view" hidden><iframe id="studio-training-frame" title="LoRA training workspace"></iframe></main>
    <main id="studio-view-models" class="studio-view" hidden><div class="studio-hero"><div><div class="studio-eyebrow">Model library</div><h2>Get the sound you need.</h2><p>Model weights live beside the runtime in your configured models folder.</p></div></div><div id="studio-model-cards" class="studio-model-grid"></div><section class="studio-panel" style="margin-top:16px"><h2 style="font-size:1rem;margin:0 0 12px;border:0">Download a model set</h2><div class="param-grid"><div><label for="studio-download-variant">Variant</label><select id="studio-download-variant"><option value="medium">Medium</option><option value="small-music">Small music</option><option value="small-sfx">Small SFX</option></select></div><div><label for="studio-download-tier">DiT tier</label><select id="studio-download-tier"><option value="f16">F16</option><option value="q4_k_m">Q4_K_M · compact</option><option value="f32">F32</option><option value="q5_k_m">Q5_K_M</option><option value="q8_0">Q8_0</option></select></div></div><p class="small-note" style="margin:12px 0">The set includes an F32 autoencoder, F16 text encoder, conditioner, and tokenizer. <a href="https://huggingface.co/thepatch" target="_blank" rel="noopener" style="color:var(--accent)">Review model licenses and cards</a>.</p><button id="studio-download-model" type="button" class="primary">Download model set</button><p id="studio-download-status" class="small-note" role="status" style="margin:10px 0 0"></p></section><section class="studio-panel" style="margin-top:16px"><h2 style="font-size:1rem;margin:0 0 8px;border:0">Decoder corrections</h2><p class="small-note">A selected correction runs with every generation on its compatible model. Creative LoRAs stay in the render dialogs.</p><div id="studio-decoder-list"></div><button id="studio-download-decoder" type="button">Download SAME-L correction</button><p id="studio-decoder-status" class="small-note" role="status"></p></section></main>
    <main id="studio-view-settings" class="studio-view studio-settings" hidden><div class="studio-hero"><div><div class="studio-eyebrow">Studio settings</div><h2>Runtime and output.</h2><p>Keep the model loaded and tune output processing here.</p></div></div><div id="studio-session-settings"></div><details><summary>Audio processing</summary><div id="studio-audio-settings"></div></details></main>
    <div id="studio-footer"></div>
  </div>
    <div id="studio-modal" class="studio-modal" hidden><div class="studio-dialog" role="dialog" aria-modal="true" aria-labelledby="studio-dialog-title"><div class="studio-dialog-head"><h2 id="studio-dialog-title">create</h2><button id="studio-dialog-close" type="button" aria-label="Close dialog">×</button></div><p id="studio-dialog-description" class="studio-dialog-description">Start a new take from a prompt.</p><div id="studio-dialog-prompt"></div><div id="studio-dialog-basic"></div><div class="studio-extra-controls"><div id="studio-tail-wrap"><span class="studio-field-label">Ending</span><div class="studio-segmented"><label><input type="radio" name="studio-tail" value="0"> Ends here</label><label><input type="radio" name="studio-tail" value="6"> Keeps going</label></div></div><div id="studio-loop-wrap"><label class="inline-label"><input id="studio-loop" type="checkbox"> Make a loop</label><div id="studio-loop-fields" class="studio-loop-options studio-conditional" hidden></div></div><div id="studio-noise-wrap" class="studio-conditional" hidden><label for="studio-noise">Transform strength <output id="studio-noise-value">0.85</output></label><input id="studio-noise" type="range" min="0" max="1" step="0.05" value="0.85"></div></div><div id="studio-dialog-loras"></div><det)sa3web") +
    std::string(R"sa3web(ails id="studio-advanced"><summary>Advanced</summary><div id="studio-dialog-cfg"></div><div id="studio-dialog-ds"></div><div id="studio-inpaint-wrap" hidden><label class="inline-label"><input id="studio-inpaint" type="checkbox"> Inpaint part of the selected audio</label><div class="param-grid studio-conditional" id="studio-inpaint-fields" hidden><div><label for="studio-inpaint-start">Start (seconds)</label><input id="studio-inpaint-start" type="number" min="0" step="0.01" value="0"></div><div><label for="studio-inpaint-end">End (seconds)</label><input id="studio-inpaint-end" type="number" min="0" step="0.01" value="0"></div></div></div></details><p id="studio-dialog-error" role="alert" hidden></p><div class="studio-dialog-actions"><button id="studio-dialog-cancel" type="button">Cancel</button><button id="studio-dialog-submit" type="button" class="primary">Create</button></div></div></div>`;

  const move = (selector, target) => { const node = document.querySelector(selector); if (node) studioEl(target).append(node); };
  move("#server-status", "studio-header-status"); move("#model-info", "studio-header-status");
  move("#top-bar", "studio-session-settings");
  // Keep the original prompt card so config loading and the existing event wiring work.
  const prompt = studioEl("prompt");
  if (prompt) studioEl("studio-dialog-prompt").append(prompt.closest(".card"));
  const negativePrompt = studioEl("negative-prompt")?.closest(".row.gapped");
  if (negativePrompt) studioEl("studio-dialog-cfg").prepend(negativePrompt);
  const basic = studioEl("duration")?.closest(".card");
  if (basic) studioEl("studio-dialog-basic").append(basic);
  const padding = studioEl("duration-padding")?.parentElement?.parentElement;
  if (padding) padding.hidden = true;
  // Keep the original inputs for presets and request serialization; place the useful controls in context.
  for (const [id, target] of [["cfg-section","studio-dialog-cfg"],["ds-section","studio-dialog-ds"],["chunk-section","studio-audio-settings"],["loud-section","studio-audio-settings"]]) {
    const section = studioEl(id); if (section) target && studioEl(target).append(section.closest(".card"));
  }
  const loraCard = studioEl("lora-section")?.closest(".card");
  if (loraCard) { loraCard.classList.add("studio-lora-card"); studioEl("studio-dialog-loras").append(loraCard); }
  const loopBpm = studioEl("loop-bpm")?.parentElement;
  const loopBars = studioEl("loop-bars")?.parentElement;
  if (loopBpm) studioEl("studio-loop-fields").append(loopBpm);
  if (loopBars) studioEl("studio-loop-fields").append(loopBars);
  move("#progress-wrap", "studio-progress-slot"); move("#result-section", "studio-audio-slot"); move("#seed-info", "studio-meta-slot"); move("#error-msg", "studio-error-slot");
  move("#past-songs", "studio-history-slot"); move("#past-count", "studio-history-count"); move("#clear-all-btn", "studio-history-actions");
  const credit = document.querySelector("#legacy-root .studio-credit"); if (credit) studioEl("studio-footer").append(credit);
}

function studioShowView(name) {
  studio.view = name;
  for (const view of ["studio", "train", "models", "settings"]) {
    studioEl(`studio-view-${view}`).hidden = view !== name;
    document.querySelector(`[data-studio-view="${view}"]`).setAttribute("aria-current", view === name ? "page" : "false");
  }
  if (name === "train") {
    const frame = studioEl("studio-training-frame");
    if (!frame.src) frame.src = "/training/?embedded=1";
  }
  if (name === "models") studioRefreshModels();
  location.hash = name === "studio" ? "" : name;
}

function studioDuration() { return studio.buffer?.duration || 0; }
function studioRange() { const d = studioDuration(); return [studio.selection[0]*d, studio.selection[1]*d]; }
function studioTime(sec) { const s = Math.max(0, sec); return `${Math.floor(s/60)}:${String(Math.floor(s%60)).padStart(2,"0")}.${String(Math.floor(s%1*100)).padStart(2,"0")}`; }
function studioUpdateTakeControls() {
  const has = !!stud)sa3web") +
    std::string(R"sa3web(io.buffer;
  for (const op of ["continue", "transform"]) {
    const button=document.querySelector(`[data-studio-operation="${op}"]`);
    button.hidden=!has;
    button.disabled=!has || studio.busy;
  }
  const crop=studioEl("studio-crop");
  crop.disabled=!has || studio.busy;
  crop.innerHTML=studio.cropMode?studioIcons.check:studioIcons.crop;
  crop.setAttribute("aria-label",studio.cropMode?"Apply crop":"Select crop");
  crop.title=studio.cropMode?"Apply crop":"Select crop";
  crop.classList.toggle("active",studio.cropMode);
  studioEl("studio-download").disabled = !has;
  studioEl("studio-clear").disabled = !has;
  studioEl("studio-play").disabled = !has;
  studioEl("studio-stop").disabled = !has;
  const paused=studioEl("result-audio").paused;
  studioEl("studio-play").innerHTML=paused?studioIcons.play:studioIcons.pause;
  studioEl("studio-play").setAttribute("aria-label",paused?"Play":"Pause");
  studioEl("studio-play").title=paused?"Play":"Pause";
  studioEl("studio-wave-empty").hidden = has;
  studioEl("studio-selection-label").hidden=!has || !studio.cropMode;
  studioEl("studio-waveform").setAttribute("aria-label",studio.cropMode?
    "Audio waveform; drag to select a crop range":"Audio waveform; click to seek");
  studioEl("studio-take-info").textContent = has ? `${studio.title || "Take"} · ${studioDuration().toFixed(1)}s` : "No audio yet";
  const [start,end] = studioRange();
  studioEl("studio-selection-label").textContent = has ? `${start.toFixed(2)}–${end.toFixed(2)}s selected` : "No selection";
  studioUpdateClock();
}

function studioUpdateClock() {
  studioEl("studio-time").textContent = studio.buffer ? `${studioTime(studioEl("result-audio").currentTime)} / ${studioTime(studioDuration())}` : "0:00 / 0:00";
}

function studioPeakBuckets(width) {
  if (!studio.buffer) return [];
  if (studio.peaks?.buffer === studio.buffer && studio.peaks.width === width) return studio.peaks.values;
  const channels = Array.from({length:Math.min(2,studio.buffer.numberOfChannels)},(_,i)=>studio.buffer.getChannelData(i));
  const values = new Float32Array(width*2), samples = studio.buffer.length;
  for(let x=0;x<width;x++) {
    const a=Math.floor(x/width*samples),b=Math.min(samples,Math.max(a+1,Math.floor((x+1)/width*samples)));
    const stride=Math.max(1,Math.floor((b-a)/64));let lo=1,hi=-1;
    for(let i=a;i<b;i+=stride)for(const ch of channels){const v=ch[i];if(v<lo)lo=v;if(v>hi)hi=v;}
    values[x*2]=hi<lo?0:lo;values[x*2+1]=hi<lo?0:hi;
  }
  studio.peaks={buffer:studio.buffer,width,values};return values;
}

function studioDrawWaveform() {
  const canvas = studioEl("studio-waveform"); if (!canvas) return;
  const rect = canvas.getBoundingClientRect(), scale = devicePixelRatio || 1;
  const width=Math.max(1,Math.round(rect.width*scale)),height=Math.max(1,Math.round(rect.height*scale));
  if(canvas.width!==width)canvas.width=width;if(canvas.height!==height)canvas.height=height;
  const ctx = canvas.getContext("2d"), w = canvas.width, h = canvas.height;
  ctx.fillStyle = "#100d0d"; ctx.fillRect(0,0,w,h);
  ctx.strokeStyle = "#382326"; ctx.beginPath(); ctx.moveTo(0,h/2); ctx.lineTo(w,h/2); ctx.stroke();
  if (!studio.buffer) return;
  const peaks=studioPeakBuckets(w),played=Math.min(w,Math.max(0,Math.floor(studioEl("result-audio").currentTime/studioDuration()*w)));
  ctx.lineWidth=Math.max(1,scale);
  for(const [a,b,color] of [[0,played,"#85262b"],[played,w,"#e23b42"]]){
    ctx.strokeStyle=color;ctx.beginPath();
    for(let x=a;x<b;x++){ctx.moveTo(x,h/2-peaks[x*2+1]*h*.44);ctx.lineTo(x,h/2-peaks[x*2]*h*.44);}
    ctx.stroke();
  }
  if(studio.cropMode){
    const left=studio.selection[0]*w,right=studio.selection[1]*w;
    ctx.fillStyle="rgba(0,0,0,.48)";ctx.fillRect(0,0,left,h);ctx.fillRect(right,0,w-right,h);
    ctx.strokeStyle="#f4eded";ctx.lineWidth=2*scale;
    for(const x of [left,right]){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke();ctx.fillStyle="#f4eded";ctx.fillRect(Math.max(0,x-4*scale),h/2-12*scale,8*)sa3web") +
    std::string(R"sa3web(scale,24*scale);}
  }
  if(Number.isFinite(played)){ctx.strokeStyle="#ffffff";ctx.lineWidth=1.5*scale;ctx.beginPath();ctx.moveTo(played,0);ctx.lineTo(played,h);ctx.stroke();}
}

function studioPlaybackFrame() {
  const audio=studioEl("result-audio"),end=studioRange()[1];
  if(studio.cropMode&&studio.selection[1]-studio.selection[0]<.995&&audio.currentTime>=end){audio.pause();audio.currentTime=end;}
  studioDrawWaveform();studioUpdateClock();
  if(!audio.paused)studio.frame=requestAnimationFrame(studioPlaybackFrame);
  else studio.frame=0;
}

async function studioLoadTake(entry) {
  const url=entry.audioUrl; if(!url) return;
  const data=await (await fetch(url)).arrayBuffer();
  const context=new (window.AudioContext||window.webkitAudioContext)();
  try { studio.buffer=await context.decodeAudioData(data.slice(0)); } finally { await context.close(); }
  studio.peaks=null;
  studio.audioUrl=url; studio.title=entry.prompt || entry.title || "Take"; studio.selection=[0,1];studio.cropMode=false;
  currentResult=entry;
  const audio=studioEl("result-audio"); audio.pause();audio.src=url; studioEl("result-section").style.display="block";
  studioUpdateTakeControls(); studioDrawWaveform(); studioShowView("studio");
}

function studioWav(buffer, start, end) {
  const a=Math.max(0,Math.floor(start*buffer.sampleRate)), b=Math.min(buffer.length,Math.ceil(end*buffer.sampleRate));
  const count=Math.max(0,b-a), channels=Math.min(2,buffer.numberOfChannels), bytes=new ArrayBuffer(44+count*channels*2), view=new DataView(bytes);
  const str=(off,s)=>{ for(let i=0;i<s.length;i++)view.setUint8(off+i,s.charCodeAt(i)); };
  str(0,"RIFF");view.setUint32(4,bytes.byteLength-8,true);str(8,"WAVE");str(12,"fmt ");view.setUint32(16,16,true);view.setUint16(20,1,true);view.setUint16(22,channels,true);view.setUint32(24,buffer.sampleRate,true);view.setUint32(28,buffer.sampleRate*channels*2,true);view.setUint16(32,channels*2,true);view.setUint16(34,16,true);str(36,"data");view.setUint32(40,count*channels*2,true);
  const data=Array.from({length:channels},(_,i)=>buffer.getChannelData(i)); let p=44;
  for(let i=a;i<b;i++)for(const ch of data){const v=Math.max(-1,Math.min(1,ch[i]));view.setInt16(p,v<0?v*32768:v*32767,true);p+=2;}
  return new Blob([bytes],{type:"audio/wav"});
}

function studioDataUrl(blob) {
  return new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result);reader.onerror=()=>reject(reader.error);reader.readAsDataURL(blob);});
}

async function studioCrop() {
  if(!studio.buffer) return;
  const [start,end]=studioRange(); if(end-start<.05 || end-start>=studioDuration()-.01)return;
  const blob=studioWav(studio.buffer,start,end), url=await studioDataUrl(blob);
  if (currentResult) pushPastSong(currentResult);
  currentResult={timestamp:Date.now(),seed:currentResult?.seed??-1,audioUrl:url,prompt:`${studio.title} · crop`,params:{...currentResult?.params}};
  await studioLoadTake(currentResult);
}

async function studioUploadSelection() {
  const [start,end]=studioRange(), blob=studioWav(studio.buffer,start,end);
  const form=new FormData(); form.append("file",blob,`studio-source-${Date.now()}.wav`);
  const response=await fetch(`${apiBase()}/init-audio/upload`,{method:"POST",body:form});
  const body=await response.json(); if(!response.ok||!body.success)throw new Error(body.error||`Upload failed (${response.status})`);
  return {path:body.path,duration:end-start};
}

function studioOpenOperation(operation) {
  if(studio.busy || (operation!=="create"&&!studio.buffer))return;
  studio.operation=operation;
  studioEl("studio-dialog-error").hidden=true;
  studioEl("studio-dialog-title").textContent=operation;
  studioEl("studio-dialog-submit").textContent=operation[0].toUpperCase()+operation.slice(1);
  const descriptions={create:"Start a new take from a prompt.",continue:"Extend the selected audio with a new passage.",transform:"Reimagine the selected audio while keeping its length."};
  studioEl("studio-dialog)sa3web") +
    std::string(R"sa3web(-description").textContent=descriptions[operation];
  studioEl("studio-loop-wrap").hidden=operation!=="create";
  studioEl("studio-noise-wrap").hidden=operation!=="transform";
  studioEl("studio-tail-wrap").hidden=operation==="transform";
  studioEl("studio-inpaint-wrap").hidden=operation!=="transform";
  studioEl("studio-inpaint").checked=false;studioEl("studio-inpaint-fields").hidden=true;
  const tail=localStorage.getItem(`sa3-tail-${operation}`) || (operation==="continue"?"6":"0");
  const radio=document.querySelector(`input[name="studio-tail"][value="${tail}"]`);if(radio)radio.checked=true;
  studioEl("studio-advanced").open=false;
  const duration=studioEl("duration-num");
  const label=document.querySelector('label[for="duration"]');
  if(label)label.textContent=operation==="continue"?"Add seconds":operation==="transform"?"Selected length (seconds)":"Duration (seconds)";
  duration.readOnly=operation==="transform";
  if(operation==="continue")duration.value="8";
  if(operation==="transform"){
    const length=studioRange()[1]-studioRange()[0];duration.value=length.toFixed(2);
    studioEl("studio-inpaint-start").value="0";studioEl("studio-inpaint-end").value=length.toFixed(2);
  }
  studioEl("studio-modal").hidden=false; studioEl("prompt").focus();
}

function studioCloseOperation() { studioEl("studio-modal").hidden=true; }

async function studioSubmit() {
  const button=studioEl("studio-dialog-submit"); button.disabled=true;
  try {
    const requested=Number(studioEl("duration-num").value);
    if(!Number.isFinite(requested)||requested<=0)throw new Error("Duration must be greater than zero.");
    studioEl("duration").value=String(requested);
    studioEl("steps").value=studioEl("steps-num").value;
    let overrides={duration:requested,init_path:"",inpaint_start:-1,inpaint_end:-1,
      duration_padding_sec:studio.operation==="transform"?0:Number(document.querySelector('input[name="studio-tail"]:checked')?.value||0)};
    if(studio.operation!=="transform")localStorage.setItem(`sa3-tail-${studio.operation}`,String(overrides.duration_padding_sec));
    if(studio.operation!=="create") {
      const source=await studioUploadSelection();
      overrides.init_path=source.path;
      if(studio.operation==="continue") { overrides.duration=source.duration+requested; overrides.inpaint_start=source.duration; overrides.inpaint_end=source.duration+requested; }
      else {
        overrides.duration=source.duration;overrides.init_noise_level=Number(studioEl("studio-noise").value);
        if(studioEl("studio-inpaint").checked){
          const start=Number(studioEl("studio-inpaint-start").value),end=Number(studioEl("studio-inpaint-end").value);
          if(!Number.isFinite(start)||!Number.isFinite(end)||start<0||end<=start||end>source.duration+.01)throw new Error("Inpaint range must fit inside the selected audio.");
          overrides.inpaint_start=start;overrides.inpaint_end=end;
        }
      }
    }
    studioSetBusy(true);
    if(studio.operation==="create"&&studioEl("studio-loop").checked)await generateLoop(overrides);
    else await generate(overrides);
    if(pollTimer)studioCloseOperation();
    else { studioDialogError(studioEl("error-msg").textContent||"Could not start generation.");studioSetBusy(false); }
  } catch(error) { studioDialogError(error.message||String(error)); studioSetBusy(false); }
  finally { button.disabled=false; }
}

function studioDialogError(message) { const error=studioEl("studio-dialog-error");error.textContent=message;error.hidden=false; }

function studioSetBusy(busy) {
  studio.busy=busy;
  document.querySelector('[data-studio-operation="create"]').disabled=busy;
  studioUpdateTakeControls();
}

async function studioRefreshModels() {
  const container=studioEl("studio-model-cards");
  try {
    const data=await apiGet("/models/catalog");
    studio.activeModel=data.models.find(model=>model.active)?.variant || "";
    container.innerHTML="";
    for(const model of data.models) {
      const c)sa3web") +
    std::string(R"sa3web(ard=document.createElement("div");card.className="studio-model-card";
      const title=document.createElement("strong");title.textContent=model.name;
      const summary=document.createElement("p");summary.textContent=model.description;
      const status=document.createElement("div");status.className="status";status.textContent=model.installed.length?`Installed: ${model.installed.join(", ")}`:"No complete set installed";
      card.append(title,summary,status);
      if(model.installed.length) { const select=document.createElement("select");select.setAttribute("aria-label",`${model.name} tier`);for(const tier of model.installed){const o=document.createElement("option");o.value=tier;o.textContent=tier;select.append(o);} if(model.active&&model.installed.includes(model.active_encoding))select.value=model.active_encoding;const button=document.createElement("button");const update=()=>{const selected=model.active&&model.active_encoding===select.value;button.textContent=selected?"Active":"Use model";button.disabled=selected;};button.style.marginTop="10px";select.onchange=update;update();button.onclick=async()=>{try{await apiPost("/models/select",{variant:model.variant,encoding:select.value});await checkHealth();studioRefreshModels();}catch(e){studioEl("studio-download-status").textContent=e.message;}};card.append(select,button); }
      container.append(card);
    }
    studioEl("studio-download-status").textContent=data.models_dir?`Models folder: ${data.models_dir}`:"";
    studioRenderDecoderLoras();
  }catch(error){container.textContent=`Model manager unavailable: ${error.message}`;}
}

function studioDecoderFamily(variant) { return variant==="medium"?"same-l":variant.startsWith("small-")?"same-s":""; }
function studioDecoderLoraFamily(lora) {
  const hint=`${lora.base_model||""} ${lora.name||""}`.toLowerCase();
  if(hint.includes("same-l")||hint.includes("squeakfix"))return "same-l";
  if(hint.includes("same-s")||hint.includes("declora"))return "same-s";
  return "";
}
function studioRenderDecoderLoras() {
  const list=studioEl("studio-decoder-list");if(!list)return;
  const family=studioDecoderFamily(studio.activeModel), key=`sa3-decoder-${family}`;
  const stored=family?localStorage.getItem(key)||"":"";
  const decoders=loraList.filter(l=>l.target==="decoder");
  const selected=decoders.some(l=>l.name===stored&&studioDecoderLoraFamily(l)===family)?stored:"";
  list.replaceChildren();
  const makeChoice=(name,label,detail,disabled=false)=>{
    const row=document.createElement("label");row.className="studio-decoder-choice";
    const radio=document.createElement("input");radio.type="radio";radio.name="studio-decoder";radio.value=name;radio.checked=selected===name;radio.disabled=disabled;
    radio.onchange=()=>{if(family)localStorage.setItem(key,name);};
    const title=document.createElement("span");title.textContent=label;
    const note=document.createElement("small");note.textContent=detail;
    row.append(radio,title,note);list.append(row);
  };
  makeChoice("","Off","No correction");
  for(const lora of decoders){
    const target=studioDecoderLoraFamily(lora), compatible=!!family&&target===family;
    makeChoice(lora.name,lora.name,target?`${target.toUpperCase()} · installed`:
      "Model family unknown",!compatible);
  }
  const download=studioEl("studio-download-decoder");
  download.disabled=decoders.some(l=>studioDecoderLoraFamily(l)==="same-l"&&l.name.toLowerCase().includes("squeakfix"));
  download.textContent=download.disabled?"SAME-L correction installed":"Download SAME-L correction";
  studioEl("studio-decoder-status").textContent=family?`Active model: ${studio.activeModel} (${family.toUpperCase()})`:"Select a model to enable a correction.";
}

async function studioStartDecoderDownload() {
  const status=studioEl("studio-decoder-status"),button=studioEl("studio-download-decoder");button.disabled=true;
  try{
    const job=await apiPost("/models/decoder/download",{});
    status.textContent="Preparing decoder c)sa3web") +
    std::string(R"sa3web(orrection…";
    const poll=async()=>{
      try{
        const s=await apiGet(`/models/download/${job.id}`);
        status.textContent=`${s.done}/${s.total} · ${s.message||s.status}`;
        if(s.status==="running")setTimeout(poll,1000);
        else{await loadLoras();studioRenderDecoderLoras();status.textContent=s.message||s.status;if(s.status!=="completed")button.disabled=false;}
      }catch(error){status.textContent=error.message;button.disabled=false;}
    };poll();
  }catch(error){status.textContent=error.message;button.disabled=false;}
}

window.studioDecoderAdapter=()=>{
  const family=studioDecoderFamily(studio.activeModel),name=family?localStorage.getItem(`sa3-decoder-${family}`):"";
  const entry=loraList.find(l=>l.name===name&&l.target==="decoder"&&studioDecoderLoraFamily(l)===family);
  return entry?[{name:entry.name,strength:1}]:[];
};
window.studioLorasReady=studioRenderDecoderLoras;

async function studioStartDownload() {
  const status=studioEl("studio-download-status"),button=studioEl("studio-download-model");button.disabled=true;
  try { const job=await apiPost("/models/download",{variant:studioEl("studio-download-variant").value,encoding:studioEl("studio-download-tier").value});
    status.textContent="Starting download…";
    const poll=async()=>{try{const s=await apiGet(`/models/download/${job.id}`);status.textContent=`${s.done}/${s.total} · ${s.message||s.status}`;if(s.status==="running")setTimeout(poll,1000);else{button.disabled=false;await studioRefreshModels();status.textContent=s.message||s.status;}}catch(e){status.textContent=e.message;button.disabled=false;}};poll();
  }catch(error){status.textContent=error.message;button.disabled=false;}
}

function studioWire() {
  for(const button of document.querySelectorAll("[data-studio-view]"))button.onclick=()=>studioShowView(button.dataset.studioView);
  for(const button of document.querySelectorAll("[data-studio-operation]"))button.onclick=()=>{
    try { studioOpenOperation(button.dataset.studioOperation); }
    catch(error) { console.error("Cannot open render dialog",error);showError(error.message||String(error)); }
  };
  studioEl("studio-dialog-close").onclick=studioCloseOperation;studioEl("studio-dialog-cancel").onclick=studioCloseOperation;
  studioEl("studio-modal").onclick=e=>{if(e.target===studioEl("studio-modal"))studioCloseOperation();};
  studioEl("studio-dialog-submit").onclick=studioSubmit;
  studioEl("studio-loop").onchange=()=>studioEl("studio-loop-fields").hidden=!studioEl("studio-loop").checked;
  studioEl("studio-noise").oninput=()=>studioEl("studio-noise-value").textContent=Number(studioEl("studio-noise").value).toFixed(2);
  studioEl("studio-inpaint").onchange=()=>studioEl("studio-inpaint-fields").hidden=!studioEl("studio-inpaint").checked;
  studioEl("studio-play").onclick=async()=>{
    const audio=studioEl("result-audio");if(!studio.buffer)return;
    if(!audio.paused){audio.pause();return;}
    const [start,end]=studio.cropMode?studioRange():[0,studioDuration()];
    if(audio.currentTime<start||audio.currentTime>=end-.02)audio.currentTime=start;
    try{await audio.play();}catch(error){showError(`Playback failed: ${error.message}`);}
  };
  studioEl("studio-stop").onclick=()=>{const audio=studioEl("result-audio");audio.pause();audio.currentTime=0;studioDrawWaveform();studioUpdateClock();};
  studioEl("studio-crop").onclick=async()=>{
    if(!studio.buffer)return;
    if(!studio.cropMode){studio.cropMode=true;studio.selection=[0,1];}
    else {
      if(studioRange()[1]-studioRange()[0]>=.05&&studio.selection[1]-studio.selection[0]<.995)
        await studioCrop();
      studio.cropMode=false;studio.selection=[0,1];
    }
    studioUpdateTakeControls();studioDrawWaveform();
  };
  studioEl("studio-download").onclick=()=>{if(!studio.audioUrl)return;const a=document.createElement("a");a.href=studio.audioUrl;a.download=`sa3-take-${Date.now()}.wav`;a.click();};
  studioEl("studio-clear").onclick=()=>{deleteCurrentSong();studioClearT)sa3web") +
    std::string(R"sa3web(ake();};
  const importFile=async file=>{
    if(!file)return;
    if(!/\.wav$/i.test(file.name)){showError("Choose a WAV file.");return;}
    try{
      const entry={audioUrl:await studioDataUrl(file),title:file.name,prompt:file.name,timestamp:Date.now(),seed:-1,params:{}};
      const previous=currentResult;
      await studioLoadTake(entry);showError("");
      if(previous)pushPastSong(previous);
    }catch(error){showError(`Cannot open WAV: ${error.message}`);}
  };
  studioEl("studio-upload-wave").onclick=()=>studioEl("studio-import-file").click();
  studioEl("studio-import-file").onchange=e=>{importFile(e.target.files?.[0]);e.target.value="";};
  const waveWrap=studioEl("studio-wave-wrap");let dragDepth=0;
  waveWrap.ondragenter=e=>{e.preventDefault();dragDepth++;waveWrap.classList.add("drag-over");studioEl("studio-drop-hint").hidden=false;};
  waveWrap.ondragover=e=>{e.preventDefault();e.dataTransfer.dropEffect="copy";};
  waveWrap.ondragleave=e=>{e.preventDefault();if(--dragDepth<=0){dragDepth=0;waveWrap.classList.remove("drag-over");studioEl("studio-drop-hint").hidden=true;}};
  waveWrap.ondrop=e=>{e.preventDefault();dragDepth=0;waveWrap.classList.remove("drag-over");studioEl("studio-drop-hint").hidden=true;importFile(e.dataTransfer.files?.[0]);};
  studioEl("studio-download-model").onclick=studioStartDownload;
  studioEl("studio-download-decoder").onclick=studioStartDecoderDownload;
  const canvas=studioEl("studio-waveform");
  const fraction=e=>{const r=canvas.getBoundingClientRect();return Math.max(0,Math.min(1,(e.clientX-r.left)/r.width));};
  canvas.onpointerdown=e=>{
    if(!studio.buffer||e.button!==0)return;
    const pos=fraction(e),edge=12/canvas.getBoundingClientRect().width;
    if(!studio.cropMode){studio.pointer={start:pos,moved:false,mode:"seek"};canvas.setPointerCapture(e.pointerId);return;}
    const nearStart=Math.abs(pos-studio.selection[0])<edge,nearEnd=Math.abs(pos-studio.selection[1])<edge;
    studio.pointer={start:pos,moved:false,mode:nearStart?"left":nearEnd?"right":"new"};
    canvas.setPointerCapture(e.pointerId);
  };
  canvas.onpointermove=e=>{
    if(!studio.pointer||studio.pointer.mode==="seek")return;
    const pos=fraction(e),p=studio.pointer;
    if(Math.abs(pos-p.start)<4/canvas.getBoundingClientRect().width&&!p.moved)return;
    p.moved=true;
    if(p.mode==="left")studio.selection[0]=Math.min(pos,studio.selection[1]-.002);
    else if(p.mode==="right")studio.selection[1]=Math.max(pos,studio.selection[0]+.002);
    else studio.selection=[Math.min(pos,p.start),Math.max(pos,p.start)];
    studioUpdateTakeControls();studioDrawWaveform();
  };
  canvas.onpointerup=e=>{
    if(!studio.pointer)return;
    if(studio.pointer.mode==="seek"||!studio.pointer.moved){studioEl("result-audio").currentTime=fraction(e)*studioDuration();}
    else if(studio.selection[1]-studio.selection[0]<.002)studio.selection=[0,1];
    studio.pointer=null;studioUpdateTakeControls();studioDrawWaveform();
  };
  canvas.onpointercancel=()=>{studio.pointer=null;};
  const audio=studioEl("result-audio");
  audio.addEventListener("play",()=>{studioUpdateTakeControls();if(!studio.frame)studio.frame=requestAnimationFrame(studioPlaybackFrame);});
  audio.addEventListener("pause",()=>{if(studio.frame)cancelAnimationFrame(studio.frame);studio.frame=0;studioUpdateTakeControls();studioDrawWaveform();});
  audio.addEventListener("timeupdate",()=>{if(audio.paused){studioUpdateClock();studioDrawWaveform();}});
  new ResizeObserver(studioDrawWaveform).observe(canvas);
  document.addEventListener("keydown",e=>{
    if(e.key!=="Escape")return;
    if(!studioEl("studio-modal").hidden)studioCloseOperation();
    else if(studio.cropMode){studio.cropMode=false;studio.selection=[0,1];studioUpdateTakeControls();studioDrawWaveform();}
  });
  window.addEventListener("hashchange",()=>{const view=location.hash.slice(1)||"studio";if(["studio","train","models","settings"].includes(view)&&view!==studio.view)studioShowView(view);});
  const init)sa3web") +
    std::string(R"sa3web(ial=location.hash.slice(1);if(["train","models","settings"].includes(initial))studioShowView(initial);
}

function studioClearTake() { const audio=studioEl("result-audio");audio.pause();if(studio.frame)cancelAnimationFrame(studio.frame);studio.frame=0;studio.buffer=null;studio.peaks=null;studio.audioUrl="";studio.title="";studio.selection=[0,1];studio.cropMode=false;studioEl("result-section").style.display="none";audio.removeAttribute("src");audio.load();studioUpdateTakeControls();studioDrawWaveform(); }
window.studioLoadTake=studioLoadTake;
window.studioClearTake=studioClearTake;
window.studioResultReady=entry=>studioLoadTake(entry).catch(e=>showError(`Cannot display waveform: ${e.message}`));
window.studioSetBusy=studioSetBusy;
document.addEventListener("DOMContentLoaded",()=>{
  try { studioShell();studioWire();studioUpdateTakeControls();studioDrawWaveform();studioRefreshModels(); }
  catch(error) { console.error("Studio initialization failed",error); studioEl("studio-header-status").textContent=`Studio initialization failed: ${error.message}`; }
});
)sa3web");

inline const std::string studio_css =
    std::string(R"sa3web(/* The studio shell keeps the original controls and authorship while giving the
   inference and training views one persistent workspace. */
body { max-width:none; padding:0; margin:0; }
#legacy-root { display:none; }
#studio-root { min-height:100vh; }
.studio-shell { max-width:1180px; margin:auto; padding:22px 24px 64px; }
.studio-shell button, .studio-modal button { text-transform:lowercase; }
.studio-header { display:flex; align-items:center; gap:18px; flex-wrap:wrap; border-bottom:1px solid var(--border); padding-bottom:16px; }
.studio-header h1 { margin:0; font-size:1.3rem; }
.studio-header nav { display:flex; flex-wrap:wrap; gap:4px; }
.studio-header nav button { border:0; background:transparent; color:var(--muted); padding:8px 12px; }
.studio-header nav button[aria-current="page"] { color:var(--text); background:#38171b; box-shadow:inset 0 -2px var(--accent); }
.studio-header .status { margin-left:auto; display:flex; gap:8px; color:var(--muted); font-size:.78rem; }
.studio-view { margin-top:24px; }
.studio-view[hidden], .studio-modal[hidden], .studio-conditional[hidden] { display:none!important; }
.studio-hero { display:flex; justify-content:flex-start; align-items:center; gap:20px; flex-wrap:wrap; margin-bottom:18px; }
.studio-hero h2 { margin:0 0 4px; border:0; padding:0; font-size:1.4rem; }
.studio-hero p { margin:0; color:var(--muted); }
.studio-eyebrow { color:var(--accent); font-size:.72rem; text-transform:uppercase; letter-spacing:.13em; font-weight:700; margin-bottom:6px; }
.studio-actions { display:flex; flex-wrap:wrap; gap:9px; }
.studio-actions button { min-width:104px; }
.studio-panel { background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:18px; }
.studio-panel + .studio-panel { margin-top:16px; }
.studio-panel-title { display:flex; justify-content:space-between; align-items:center; gap:10px; }
.studio-panel-title h2 { margin:0; border:0; padding:0; font-size:1rem; }
.studio-panel-title span { color:var(--muted); font-size:.78rem; }
.studio-wave-wrap { position:relative; margin-top:14px; background:#100d0d; border:1px solid var(--border); border-radius:9px; overflow:hidden; touch-action:none; }
.studio-wave-wrap.drag-over { border-color:var(--accent); box-shadow:0 0 0 2px #b6222b66; }
#studio-waveform { display:block; width:100%; height:188px; cursor:crosshair; }
#studio-wave-empty { position:absolute; inset:0; display:grid; place-items:center; color:var(--muted); text-align:center; padding:18px; pointer-events:none; }
#studio-wave-empty[hidden] { display:none; }
.studio-wave-actions { position:absolute; right:10px; top:10px; display:flex; gap:7px; z-index:2; }
.studio-wave-actions button { width:36px; height:36px; display:grid; place-items:center; padding:6px; background:#211818dd; border-color:#725055; }
.studio-wave-actions button.active { background:#8e2027; border-color:var(--accent); }
.studio-wave-actions svg { width:20px; height:20px; }
#studio-drop-hint { position:absolute; inset:0; display:grid; place-items:center; background:#140b0ddd; color:var(--text); font-weight:600; pointer-events:none; z-index:3; }
#studio-drop-hint[hidden] { display:none; }
.studio-take-tools { display:flex; align-items:center; flex-wrap:wrap; gap:9px; margin-top:12px; }
#studio-audio-slot { display:none; }
#studio-play, #studio-stop { width:40px; height:40px; display:grid; place-items:center; padding:8px; }
#studio-play svg, #studio-stop svg { width:19px; height:19px; }
.studio-take-tools .studio-time { color:var(--muted); font-size:.78rem; font-variant-numeric:tabular-nums; }
#studio-selection-label { margin-left:auto; color:var(--muted); font-size:.78rem; }
#result-section { display:none; }
#result-section #delete-current-btn { display:none; }
#studio-meta-slot { color:var(--muted); font-size:.75rem; margin-top:10px; }
#progress-wrap { margin-top:14px; }
#progress-bar { height:5px; }
#progress-label { margin-top:6px; }
#error-msg { min-height:1em; }
.studio-history-head { display:flex; ju)sa3web") +
    std::string(R"sa3web(stify-content:space-between; align-items:center; margin:24px 0 9px; gap:8px; }
.studio-history-head h2 { margin:0; border:0; padding:0; font-size:1rem; }
#past-songs { max-height:none; padding:4px 16px; }
.song-entry { flex-wrap:wrap; }
.song-entry audio { min-width:150px; flex:1; }
.studio-shell > .studio-credit { margin:24px 0 0; }
#studio-training-frame { width:100%; height:calc(100vh - 145px); min-height:580px; border:1px solid var(--border); border-radius:12px; background:var(--bg); }
.studio-model-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:14px; }
.studio-model-card { background:#211818; border:1px solid var(--border); border-radius:10px; padding:16px; }
.studio-model-card strong { display:block; margin-bottom:5px; }
.studio-model-card p { color:var(--muted); font-size:.78rem; margin-bottom:10px; }
.studio-model-card .status { color:var(--muted); font-size:.78rem; }
#studio-decoder-list { display:grid; gap:8px; margin:14px 0; }
.studio-decoder-choice { display:flex; align-items:center; gap:10px; padding:10px 12px; border:1px solid var(--border); border-radius:8px; cursor:pointer; }
.studio-decoder-choice input { width:auto; accent-color:var(--accent); }
.studio-decoder-choice span { flex:1; }
.studio-decoder-choice small { color:var(--muted); }
.studio-decoder-choice:has(input:checked) { border-color:var(--accent); background:#301619; }
.studio-decoder-choice:has(input:disabled) { opacity:.55; cursor:not-allowed; }
.studio-settings h2 { margin-top:18px; }
.studio-settings .card { margin-bottom:12px; }
.studio-settings details { background:var(--surface); border:1px solid var(--border); border-radius:10px; margin-top:12px; }
.studio-settings details summary { cursor:pointer; padding:13px 16px; color:var(--text); font-weight:600; }
.studio-settings details[open] summary { border-bottom:1px solid var(--border); }
.studio-settings details .card { border:0; background:transparent; margin:0; }
.studio-settings #top-bar { margin-top:14px; }
.studio-settings #top-bar > div:first-child { flex:1; }
.studio-settings #top-bar #server-status, .studio-settings #top-bar #model-info { display:none!important; }
.studio-modal { position:fixed; inset:0; z-index:30; background:rgba(0,0,0,.78); display:grid; place-items:center; padding:16px; }
.studio-dialog { width:min(100%,540px); max-height:min(90vh,860px); overflow-y:auto; background:#100d0d; border:2px solid var(--accent); border-radius:15px; padding:22px; box-shadow:0 18px 60px #000b; scrollbar-width:thin; scrollbar-color:#8e2027 #100d0d; }
.studio-dialog::-webkit-scrollbar { width:9px; }
.studio-dialog::-webkit-scrollbar-track { background:#100d0d; border-radius:8px; }
.studio-dialog::-webkit-scrollbar-thumb { background:#8e2027; border:2px solid #100d0d; border-radius:8px; }
.studio-dialog::-webkit-scrollbar-thumb:hover { background:var(--accent); }
.studio-dialog-head { display:flex; align-items:start; justify-content:space-between; gap:12px; }
.studio-dialog-head h2 { margin:0; padding:0; border:0; text-transform:lowercase; font-size:1.4rem; }
.studio-dialog-head button { border:0; background:transparent; font-size:1.4rem; padding:0 6px; }
.studio-dialog-description { color:var(--muted); font-size:.82rem; margin:8px 0 16px; }
.studio-dialog .card { border:0; background:transparent; margin:0; padding:0; }
.studio-dialog .card + .card { margin-top:12px; }
.studio-dialog .card .row.gapped { margin:0 0 10px; }
.studio-dialog textarea { min-height:75px; }
.studio-dialog .param-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
.studio-dialog #studio-dialog-basic input[type=range] { display:none; }
.studio-dialog #studio-dialog-basic .range-row input[type=number] { width:100%; }
.studio-dialog #studio-dialog-basic .range-row { display:block; }
.studio-dialog .collapse-body.collapsed { max-height:0; }
.studio-dialog-actions { display:flex; justify-content:flex-end; gap:8px; margin-top:18px; }
.studio-dialog .studio-loop-options { display:grid; )sa3web") +
    std::string(R"sa3web(grid-template-columns:1fr 1fr; gap:10px; margin-top:10px; }
.studio-dialog .studio-extra-controls { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:10px; }
.studio-dialog .studio-extra-controls > div { min-width:0; }
.studio-dialog .studio-field-label { display:block; color:var(--muted); font-size:.78rem; margin-bottom:5px; }
.studio-dialog input[type="range"] { accent-color:var(--accent); }
.studio-dialog #duration-num[readonly] { color:var(--muted); cursor:default; }
.studio-segmented { display:flex; gap:4px; background:#211818; padding:4px; border-radius:8px; }
.studio-segmented label { flex:1; padding:7px 9px; margin:0; border-radius:6px; text-align:center; cursor:pointer; color:var(--muted); }
.studio-segmented input { width:auto; accent-color:var(--accent); }
.studio-segmented label:has(input:checked) { background:#8e2027; color:white; }
.studio-dialog #studio-advanced { border-top:1px solid var(--border); margin-top:15px; padding-top:10px; }
.studio-dialog #studio-advanced summary { cursor:pointer; color:var(--muted); font-size:.85rem; }
.studio-dialog #studio-advanced .card { padding:12px 0 0; }
.studio-dialog #studio-inpaint-wrap { padding-top:12px; }
.studio-dialog #studio-inpaint-fields { margin-top:10px; }
.studio-dialog #studio-dialog-error { color:var(--red); font-size:.82rem; margin:12px 0 0; }
.studio-dialog #studio-dialog-error[hidden] { display:none; }
.studio-dialog .studio-lora-row { display:grid; grid-template-columns:minmax(80px,1fr) minmax(110px,2fr) 42px auto; align-items:center; gap:9px; margin:8px 0; }
.studio-dialog .studio-lora-row input { accent-color:var(--accent); padding:0; border:0; }
.studio-dialog .studio-lora-row output { color:var(--muted); font-size:.78rem; font-variant-numeric:tabular-nums; }
.studio-dialog .studio-lora-card { border-top:1px solid var(--border); padding-top:12px; margin-top:12px; }
.studio-dialog .studio-lora-card .collapse-toggle { color:var(--text); font-weight:600; }
.studio-dialog #gen-btn, .studio-dialog #loop-btn { display:none; }
@media(max-width:650px) {
  .studio-shell { padding:12px 14px 40px; }
  .studio-header { gap:7px; }
  .studio-header .status { margin-left:0; width:100%; }
  .studio-hero .studio-actions { width:100%; }
  .studio-hero .studio-actions button { flex:1; }
  #studio-waveform { height:142px; }
  .studio-dialog .studio-lora-row { grid-template-columns:1fr 1.6fr 36px auto; }
  .studio-dialog .param-grid, .studio-dialog .studio-extra-controls { grid-template-columns:1fr; }
  .song-entry audio { min-width:100%; }
}
)sa3web");

} // namespace embedded_web
