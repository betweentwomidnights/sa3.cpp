#pragma once
// embedded_web.h — web UI assets embedded in the binary at build time.
// AUTO-GENERATED from web/index.html and web/app.js. Do not edit by hand.
// Rebuild with: python3 tools/gen_embedded_web.py

namespace embedded_web {

inline const char* index_html = R"sa3web(
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>sa3.cpp – Web Interface</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --red: #f85149; --orange: #d29922;
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
    font-family: inherit; font-size: 0.88rem; background: #010409; color: var(--text);
    border: 1px solid var(--border); border-radius: var(--radius); padding: 6px 10px;
    width: 100%; outline: none; transition: border .15s;
  }
  input:focus, select:focus, textarea:focus { border-color: var(--accent); }
  textarea { resize: vertical; min-height: 48px; font-family: var(--mono); font-size: 0.82rem; }
  input[type=number] { font-family: var(--mono); font-size: 0.82rem; }
  select { cursor: pointer; }
  button {
    font-family: inherit; font-size: 0.85rem; cursor: pointer; border: 1px solid var(--border);
    border-radius: var(--radius); padding: 7px 16px; background: #21262d; color: var(--text);
    transition: background .15s, border-color .15s; white-space: nowrap;
  }
  button:hover { background: #30363d; }
  button.primary { background: #1f6feb; border-color: #1f6feb; color: #fff; font-weight: 600; }
  button.primary:hover { background: #388bfd; }
  button.primary:disabled { opacity: .5; cursor: not-allowed; }
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
  .range-row { display: flex; align-items: center; gap: 8px; }
  .range-row input[type=range] { flex: 1; accent-color: var(--accent); background: transparent; border: none; padding: 0; }
  .range-row input[type=number] { width: 70px; }
  #progress-wrap { margin-top: 10px; }
  #progress-bar { height: 6px; background: var(--accent); border-radius: 3px; width: 0%; transition: width .3s; }
  #progress-label { font-size: 0.82rem; color: var(--muted); margin-top: 4px; }
  #result-section { margin-top: 12px; }
  #result-section audio { width: 100%; }
  #seed-info { font-size: 0.8rem; color: var(--muted); margin-top: 4px; }
  .lora-tag { display: inline-flex; align-items: center; gap: 4px; background: #1c2333;
    border: 1px solid #2d3748; border-radius: 4px; padding: 3px 8px; font-size: 0.82rem; margin: 2px; }
  .lora-tag .lora-str { color: var(--muted); }
  .lora-tag button { padding: 0 4px; font-size: 1rem; line-height: 1; background: none; border: none; color: var(--muted); }
  .lora-tag button:hover { color: var(--red); }
  #active-loras { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 6px; min-height: 26px; }
  #error-msg { color: var(--red); font-size: 0.85rem; margin-top: 6px; min-height: 1.2em; }
  .small-note { font-size: 0.75rem; color: var(--muted); margin-top: 2px; }
  hr { border: none; border-top: 1px solid var(--border); margin: 12px 0; }
  @media (max-width: 600px) {
    .row { flex-direction: column; }
    .col { min-width: 100%; }
    .param-grid { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>

<h1><a href="https://github.com/betweentwomidnights/sa3.cpp" target="_blank" style="color:var(--accent);text-decoration:none">SA3.CPP</a> <small>Stable Audio 3 Web Interface</small></h1>

<!-- ─── Server Status & Presets ──────────────────────────────────────── -->
<div id="top-bar" class="card" style="padding:8px 16px;display:flex;align-items:center;gap:12px;flex-wrap:wrap">
  <div style="display:flex;align-items:center;gap:10px;flex:1;min-width:180px">
    <span id="server-status" style="display:none;font-weight:600;font-size:0.85rem"></span>
    <span id="model-info" style="display:none;font-size:0.82rem;color:var(--muted)"></span>
  </div>
  <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap">
    <button id="save-config-btn" class="small" title="Save current config">💾 Save</button>
    <button id="load-config-btn" class="small" title="Load config file">📂 Load</button>
    <span id="config-filename" style="font-size:0.78rem;color:var(--muted);max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap"></span>
    <input id="load-config-input" type="file" accept=".json" style="display:none">
  </div>
</div>

<!-- ─── Prompt ────────────────────────────────────────────────────────── -->
<div class="card">
  <div class="row gapped">
    <div class="col" style="flex:3">
      <label for="prompt">Prompt</label>
      <textarea id="prompt" rows="2" placeholder="e.g. upbeat funk groove with slap bass, bright horns, tight drums">upbeat funk groove with slap bass, bright horns, tight drums</textarea>
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
        <input id="duration" type="range" min="1" max="300" value="30" step="0.5">
        <input id="duration-num" type="number" min="1" max="300" value="30" step="0.5">
      </div>
    </div>
    <div>
      <label for="steps">Sampling Steps</label>
      <div class="range-row">
        <input id="steps" type="range" min="1" max="100" value="8" step="1">
        <input id="steps-num" type="number" min="1" max="100" value="8" step="1">
      </div>
    </div>
    <div>
      <label for="seed">Seed (<span style="color:var(--muted)">-1 = random</span>)</label>
      <input id="seed" type="number" value="-1">
    </div>
    <div>
      <label for="duration-padding">Duration Padding (s)</label>
      <div class="range-row">
        <input id="duration-padding" type="range" min="0" max="30" value="6" step="0.5">
        <input id="duration-padding-num" type="number" min="0" max="30" value="6" step="0.5">
      </div>
    </div>
  </div>
</div>

<!-- ─── Advanced: CFG ────────────────────────────────────────────────── -->
<h2>Advanced <span class="badge">collapsible</span></h2>

<div class="card">
  <button class="collapse-toggle" data-target="cfg-section" type="button">Classifier-Free Guidance</button>
  <div id="cfg-section" class="collapse-body">
    <div class="param-grid" style="margin-top:10px">
      <div><label for="cfg-scale">CFG Scale</label><input id="cfg-scale" type="number" value="1.0" step="0.1" min="0"></div>
      <div><label for="cfg-rescale">CFG Rescale</label><input id="cfg-rescale" type="number" value="0.0" step="0.05" min="0" max="1"></div>
      <div><label for="apg-scale">APG Scale</label><input id="apg-scale" type="number" value="1.0" step="0.05" min="0" max="1"></div>
      <div><label for="cfg-norm-threshold">CFG Norm Threshold</label><input id="cfg-norm-threshold" type="number" value="0.0" step="0.1" min="0"></div>
      <div><label for="cfg-interval-min">CFG Interval Min</label><input id="cfg-interval-min" type="number" value="0.0" step="0.05" min="0" max="1"></div>
      <div><label for="cfg-interval-max">CFG Interval Max</label><input id="cfg-interval-max" type="number" value="1.0" step="0.05" min="0" max="1"></div>
    </div>
  </div>
</div>

<div class="card">
  <button class="collapse-toggle" data-target="ds-section" type="button">Distribution Shift</button>
  <div id="ds-section" class="collapse-body">
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
      <div class="col"><label for="dsp1">p1</label><input id="dsp1" type="number" value="2000" step="any"></div>
      <div class="col"><label for="dsp2">p2</label><input id="dsp2" type="number" value="-6.2" step="any"></div>
      <div class="col"><label for="dsp3">p3</label><input id="dsp3" type="number" value="0" step="any"></div>
      <div class="col"><label for="dsp4">p4</label><input id="dsp4" type="number" value="2" step="any"></div>
    </div>
    <div class="small-note">Params auto-fill per-type defaults. Edit to override.</div>
  </div>
</div>

<div class="card">
  <button class="collapse-toggle" data-target="chunk-section" type="button">Chunked SAME Encode/Decode</button>
  <div id="chunk-section" class="collapse-body">
    <div class="param-grid" style="margin-top:10px">
      <div><label for="encode-chunk-size">Encode Chunk Size</label><input id="encode-chunk-size" type="number" value="0" min="0" step="1"></div>
      <div><label for="encode-overlap">Encode Overlap</label><input id="encode-overlap" type="number" value="32" min="0" step="1"></div>
      <div><label for="decode-chunk-size">Decode Chunk Size</label><input id="decode-chunk-size" type="number" value="0" min="0" step="1"></div>
      <div><label for="decode-overlap">Decode Overlap</label><input id="decode-overlap" type="number" value="32" min="0" step="1"></div>
    </div>
    <div class="small-note">0 = monolithic. Overlap must be &lt; chunk size when chunk size &gt; 0.</div>
  </div>
</div>

<div class="card">
  <button class="collapse-toggle" data-target="loud-section" type="button">Loudness / Output Processing</button>
  <div id="loud-section" class="collapse-body">
    <div class="param-grid" style="margin-top:10px">
      <div><label for="latent-rescale">Latent Rescale</label><input id="latent-rescale" type="number" value="1.0" step="0.05" min="0"></div>
      <div><label for="latent-shift">Latent Shift</label><input id="latent-shift" type="number" value="0.0" step="0.05"></div>
      <div><label for="latent-target-std">Latent Target Std <span class="small-note">(blank=off)</span></label><input id="latent-target-std" type="number" value="" step="0.05" min="0" placeholder="off"></div>
      <div><label for="latent-adapt-min">Adapt Min</label><input id="latent-adapt-min" type="number" value="0.9" step="0.05" min="0"></div>
      <div><label for="latent-adapt-max">Adapt Max</label><input id="latent-adapt-max" type="number" value="1.0" step="0.05" min="0"></div>
      <div><label for="peak-normalize-db">Peak Normalize dB <span class="small-note">(blank=off)</span></label><input id="peak-normalize-db" type="number" value="2.0" step="0.5" placeholder="off"></div>
      <div><label for="limiter-ceiling-db">Limiter Ceiling dB <span class="small-note">(blank=off)</span></label><input id="limiter-ceiling-db" type="number" value="-0.3" step="0.1" placeholder="off"></div>
      <div><label for="limiter-knee">Limiter Knee</label><input id="limiter-knee" type="number" value="0.8" step="0.05" min="0" max="1"></div>
    </div>
  </div>
</div>

<div class="card">
  <div class="row gapped">
    <div class="col"><label class="inline-label"><input id="keep-models" type="checkbox"> Keep Models Resident (faster next request, high VRAM)</label></div>
  </div>
</div>

<!-- ─── Init Audio / Inpaint ─────────────────────────────────────────── -->
<h2>Init Audio &amp; Inpainting</h2>
<div class="card">
  <div class="param-grid">
    <div style="grid-column:1/-1">
      <label for="init-path">Init Audio WAV Path (server-side)</label>
      <input id="init-path" type="text" placeholder="leave blank for text-to-music">
    </div>
    <div><label for="init-noise-level">Init Noise Level</label><input id="init-noise-level" type="number" value="0.85" step="0.05" min="0" max="1"></div>
    <div><label for="inpaint-start">Inpaint Start (s)</label><input id="inpaint-start" type="number" value="-1" step="0.5"></div>
    <div><label for="inpaint-end">Inpaint End (s)</label><input id="inpaint-end" type="number" value="-1" step="0.5"></div>
  </div>
  <div class="small-note">Set inpaint_start ≥ 0 to enable inpainting (requires init audio + local-cond DiT). -1 = disabled.</div>
</div>

<!-- ─── LoRAs ─────────────────────────────────────────────────────────── -->
<h2>LoRA Adapters</h2>
<div class="card">
  <div class="row gapped">
    <div class="col" style="flex:2">
      <label for="lora-select">Name</label>
      <div class="row" style="gap:6px">
        <select id="lora-select" style="flex:1"></select>
        <input id="lora-strength" type="number" value="1.0" step="0.05" min="0" style="width:70px" placeholder="str">
        <button id="lora-add-btn" class="small">Add</button>
      </div>
    </div>
  </div>
  <div id="active-loras"></div>
  <div class="small-note">Click Connect above to load available LoRAs from the server.</div>
</div>

<!-- ─── Generate Buttons ─────────────────────────────────────────────── -->
<h2>Generate</h2>
<div class="card">
  <div class="row gapped">
    <div class="col"><button id="gen-btn" class="primary" style="width:100%">🎵 Generate</button></div>
    <div class="col">
      <button id="loop-btn" style="width:100%">🔄 Generate Loop</button>
    </div>
  </div>
  <div class="row gapped" style="margin-top:8px">
    <div class="col" style="flex:0 0 100px"><label for="loop-bpm">BPM</label><input id="loop-bpm" type="number" value="120" min="20" max="300"></div>
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
    <audio id="result-audio" controls></audio>
    <div id="seed-info"></div>
  </div>
</div>

<script src="app.js"></script>
</body>
</html>

)sa3web";

inline const char* app_js = R"sa3web(
"use strict";
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
    return `http://${server.host}:${server.port}`;
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
    if (!r.ok)
        throw new Error(`HTTP ${r.status}: ${r.statusText}`);
    return r.json();
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
        loras: activeLoras,
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
    }
    catch {
        statusEl.textContent = "✗ Server unreachable";
        statusEl.className = "err";
        modelInfo.style.display = "none";
    }
}
// ─── Loras ──────────────────────────────────────────────────────────────────
async function loadLoras() {
    try {
        const r = await apiGet("/loras");
        loraList = r.loras;
        renderLoraDropdown();
    }
    catch {
        // server not connected yet
    }
}
function renderLoraDropdown() {
    const sel = $("#lora-select");
    sel.innerHTML = '<option value="">— select —</option>';
    for (const l of loraList) {
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
    const strength = num("#lora-strength");
    if (activeLoras.some((l) => l.name === name))
        return;
    activeLoras.push({ name, strength });
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
        const tag = document.createElement("span");
        tag.className = "lora-tag";
        tag.innerHTML = `${escapeHtml(l.name)} <span class="lora-str">(${l.strength.toFixed(2)})</span> <button class="small" data-name="${escapeHtml(l.name)}" title="Remove">&times;</button>`;
        tag.querySelector("button").addEventListener("click", () => removeLora(l.name));
        container.appendChild(tag);
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
// ─── Generate ───────────────────────────────────────────────────────────────
let pollTimer = null;
async function generate() {
    clearPolling();
    const body = readForm();
    const btn = $("#gen-btn");
    btn.disabled = true;
    btn.textContent = "Generating…";
    try {
        const r = await apiPost("/generate", body);
        startPolling(r.session_id);
    }
    catch (e) {
        showError(e instanceof Error ? e.message : "Request failed");
        btn.disabled = false;
        btn.textContent = "Generate";
    }
}
async function generateLoop() {
    clearPolling();
    const body = {
        ...readForm(),
        bpm: num("#loop-bpm"),
        bars: int("#loop-bars"),
    };
    const btn = $("#loop-btn");
    btn.disabled = true;
    btn.textContent = "Generating loop…";
    try {
        const r = await apiPost("/generate/loop", body);
        startPolling(r.session_id);
    }
    catch (e) {
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
            else if (r.status === "generating" || r.status === "encoding") {
                progressLabel.textContent = `${r.status} step ${r.step}/${r.total_steps} (${r.progress}%)`;
            }
            else if (r.status === "completed") {
                progressLabel.textContent = `completed (${r.progress}%)`;
                if (r.audio_data) {
                    resultAudio.src = `data:audio/wav;base64,${r.audio_data}`;
                    resultSection.style.display = "block";
                }
                const metaParts = [];
                if (r.meta?.seed != null)
                    metaParts.push(`Seed: ${r.meta.seed}`);
                if (r.meta?.loudness) {
                    const lm = r.meta.loudness;
                    if (lm.final_peak != null)
                        metaParts.push(`Peak: ${Number(lm.final_peak).toFixed(3)}`);
                    if (lm.decoded_peak != null)
                        metaParts.push(`Decoded: ${Number(lm.decoded_peak).toFixed(3)}`);
                }
                seedInfo.textContent = metaParts.join(" · ");
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
        limiter_ceiling_db: lcRaw.length > 0 ? num("#limiter-ceiling-db") : null,
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
            showError("");
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

)sa3web";

} // namespace embedded_web
