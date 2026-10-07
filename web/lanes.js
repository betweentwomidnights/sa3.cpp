// Automation lanes: draw per-LoRA envelopes over a take's waveform and generate with them.
// Talks to sa3-server: /health, /loras, /prompts, /generate (lora_lanes), /poll_status.
"use strict";

const $ = (s) => document.querySelector(s);
const COLORS = ["#e2a03b", "#4fb3d9", "#b07be0", "#6cc58b", "#e5677f", "#d6d65a", "#5f8ff0", "#e08a5c"];
const MAX_LANES = 8;
const LANE_H = 96;
const STORE_KEY = "sa3-lanes-editor-v1";

const state = {
  prompt: "keygen music, crunchy square leads, tight chip drums",
  negative: "",
  duration: 20, steps: 8, cfg: 1, seed: 1234, seedLock: false, mode: "velocity",
  lanes: [],
};
let loras = [];          // DiT adapters from /loras
let takes = [];          // {id, url, peaks, dur, request, seed, secs, lanes}
let current = null;      // the take on screen
let nextLaneId = 1, nextTakeId = 1;
let pollTimer = null;
let view = "spec";        // take display: "spec" | "wave"

// ─── persistence (per-viewer convenience only) ─────────────────────────────

function save() {
  try { localStorage.setItem(STORE_KEY, JSON.stringify(state)); } catch { /* storage unavailable */ }
}
function load() {
  try {
    const s = JSON.parse(localStorage.getItem(STORE_KEY) || "null");
    if (s && Array.isArray(s.lanes)) Object.assign(state, s);
  } catch { /* ignore */ }
  for (const l of state.lanes) nextLaneId = Math.max(nextLaneId, (l.id || 0) + 1);
}

// ─── envelope maths (mirrors the C++ sampler: sorted, linear, held flat past the ends) ──

function amountAt(points, t) {
  if (!points.length) return 0;
  if (t <= points[0][0]) return points[0][1];
  for (let k = 1; k < points.length; k++) {
    if (t <= points[k][0]) {
      const [t0, a0] = points[k - 1], [t1, a1] = points[k];
      return t1 > t0 ? a0 + (t - t0) / (t1 - t0) * (a1 - a0) : a1;
    }
  }
  return points[points.length - 1][1];
}
const sortPts = (l) => l.points.sort((a, b) => a[0] - b[0]);
const round = (v, q) => Math.round(v / q) * q;

// ─── api ────────────────────────────────────────────────────────────────────

async function apiGet(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return r.json();
}
async function apiPost(path, body) {
  const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const j = await r.json().catch(() => ({}));
  if (!r.ok || j.success === false) throw new Error(j.error || `${path}: HTTP ${r.status}`);
  return j;
}

async function connect() {
  const st = $("#status");
  try {
    const h = await apiGet("/health");
    st.textContent = "✓ connected"; st.className = "ok";
    $("#model").textContent = `${h.model} / ${h.encoding}`;
    const r = await apiGet("/loras");
    loras = (r.loras || []).filter((l) => (l.target || "dit") === "dit");
    if (!state.lanes.length && loras.length) { state.lanes.push(newLane()); save(); }
    renderLanes();
  } catch (e) {
    st.textContent = "✗ server unreachable"; st.className = "err";
  }
}

async function dicePrompt(loraName) {
  const q = loraName ? `?lora=${encodeURIComponent(loraName)}` : "";
  const r = await apiGet(`/prompts${q}`);
  const dice = (r.prompts && r.prompts.dice) || {};
  const pool = dice.instrumental && dice.instrumental.length ? dice.instrumental : [].concat(...Object.values(dice));
  return pool.length ? pool[Math.floor(Math.random() * pool.length)] : "";
}

// ─── form <-> state ─────────────────────────────────────────────────────────

function readForm() {
  state.prompt = $("#prompt").value;
  state.negative = $("#negative").value;
  state.duration = Math.max(1, +$("#duration").value || 20);
  state.steps = Math.max(1, Math.round(+$("#steps").value || 8));
  state.cfg = +$("#cfg").value || 1;
  state.seed = Math.max(0, Math.round(+$("#seed").value || 0));
  state.seedLock = $("#seed-lock").checked;
  state.mode = $("#mode").value;
}
function writeForm() {
  $("#prompt").value = state.prompt;
  $("#negative").value = state.negative;
  $("#duration").value = state.duration;
  $("#steps").value = state.steps;
  $("#cfg").value = state.cfg;
  $("#seed").value = state.seed;
  $("#seed-lock").checked = state.seedLock;
  $("#mode").value = state.mode;
}
const randomSeed = () => Math.floor(Math.random() * 2 ** 31);

// ─── lanes ──────────────────────────────────────────────────────────────────

function newLane() {
  const used = new Set(state.lanes.map((l) => l.color));
  const color = COLORS.find((c) => !used.has(c)) || COLORS[state.lanes.length % COLORS.length];
  const taken = new Set(state.lanes.map((l) => l.lora));
  const lora = (loras.find((l) => !taken.has(l.name)) || loras[0] || {}).name || "";
  const d = state.duration;
  return { id: nextLaneId++, color, lora, strength: 1, prompt: "", max: 1.5,
           points: [[0, 0], [d, 1]] };
}

const PRESETS = {
  "0": (d) => [[0, 0]],
  "1": (d) => [[0, 1]],
  "↗": (d) => [[0, 0], [d, 1]],
  "↘": (d) => [[0, 1], [d, 0]],
  "⊓": (d) => [[0, 0], [d * 0.25, 0], [d * 0.35, 1], [d * 0.65, 1], [d * 0.75, 0]],
  "½": (d) => [[0, 0], [d / 2 - 0.01, 0], [d / 2, 1]],
};

function renderLanes() {
  const host = $("#lanes");
  host.innerHTML = "";
  state.lanes.forEach((lane, idx) => {
    const row = document.createElement("div");
    row.className = "tl-row";
    const opts = loras.map((l) => `<option value="${esc(l.name)}" ${l.name === lane.lora ? "selected" : ""}>${esc(l.name)}</option>`).join("");
    const missing = lane.lora && !loras.some((l) => l.name === lane.lora) ? `<option selected>${esc(lane.lora)} (missing)</option>` : "";
    row.innerHTML = `
      <div class="tl-head lane-head">
        <div class="lane-head-top">
          <span class="swatch" style="background:${lane.color}"></span>
          <select data-k="lora" title="LoRA">${missing}${opts || "<option value=''>no LoRAs</option>"}</select>
          <button class="tiny" data-act="remove" title="remove lane">✕</button>
        </div>
        <div class="mini">
          <label style="margin:0">str</label><input type="number" data-k="strength" step="0.05" min="0" value="${lane.strength}">
          <label style="margin:0">max</label><input type="number" data-k="max" step="0.25" min="0.25" value="${lane.max}">
        </div>
        <textarea data-k="prompt" rows="1" placeholder="lane prompt (empty = base prompt)">${esc(lane.prompt)}</textarea>
        <div class="presets">
          <button class="tiny" data-act="dice" title="in-distribution prompt for this LoRA">🎲</button>
          ${Object.keys(PRESETS).map((p) => `<button class="tiny" data-preset="${p}" title="preset">${p}</button>`).join("")}
        </div>
      </div>
      <div class="tl-body"><canvas class="lane-canvas" data-h="${LANE_H}"></canvas><div class="tip"></div></div>`;
    host.appendChild(row);

    row.querySelectorAll("[data-k]").forEach((el) => {
      el.addEventListener("input", () => {
        const k = el.dataset.k;
        lane[k] = el.type === "number" ? Math.max(0, +el.value || 0) : el.value;
        if (k === "max") { lane.max = Math.max(0.25, lane.max); for (const p of lane.points) p[1] = Math.min(p[1], lane.max); }
        drawAll(); save();
      });
    });
    row.querySelector("[data-act=remove]").onclick = () => { state.lanes.splice(idx, 1); renderLanes(); save(); };
    row.querySelector("[data-act=dice]").onclick = async () => {
      try { lane.prompt = await dicePrompt(lane.lora); row.querySelector("textarea").value = lane.prompt; save(); }
      catch (e) { showError(e.message); }
    };
    row.querySelectorAll("[data-preset]").forEach((b) => {
      b.onclick = () => { lane.points = PRESETS[b.dataset.preset](state.duration).map((p) => [round(p[0], 0.01), p[1]]); drawAll(); save(); };
    });
    attachLaneEditor(row.querySelector("canvas"), row.querySelector(".tip"), lane);
  });
  $("#add-lane").disabled = state.lanes.length >= MAX_LANES;
  resize();
}
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// ─── geometry shared by every canvas ───────────────────────────────────────

const PAD = 6;                         // px inside each canvas, left/right
function xOf(cv, t) { return PAD + (t / state.duration) * (cv.clientWidth - 2 * PAD); }
function tOf(cv, x) { return Math.min(state.duration, Math.max(0, (x - PAD) / (cv.clientWidth - 2 * PAD) * state.duration)); }
const TOP = 8, BOT = 8;
function yOf(cv, a, max) { return TOP + (1 - a / max) * (cv.clientHeight - TOP - BOT); }
function aOf(cv, y, max) { return Math.min(max, Math.max(0, (1 - (y - TOP) / (cv.clientHeight - TOP - BOT)) * max)); }

function prep(cv) {
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
    cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
  }
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  return g;
}
function resize() {
  for (const cv of document.querySelectorAll("#timeline canvas")) cv.style.height = cv.dataset.h + "px";
  drawAll();
}

// ─── drawing ────────────────────────────────────────────────────────────────

function drawWave(g, cv, peaks, dur, alpha) {
  if (!peaks) return;
  const h = cv.clientHeight, mid = h / 2;
  g.fillStyle = `rgba(226,59,66,${alpha})`;
  const x0 = xOf(cv, 0), x1 = xOf(cv, Math.min(dur, state.duration));
  const n = peaks.length / 2;
  for (let x = Math.floor(x0); x < x1; x++) {
    const t = tOf(cv, x + 0.5);
    const i = Math.min(n - 1, Math.floor(t / dur * n));
    const lo = peaks[2 * i], hi = peaks[2 * i + 1];
    g.fillRect(x, mid - hi * mid * 0.95, 1, Math.max(1, (hi - lo) * mid * 0.95));
  }
}

function drawGrid(g, cv, withLabels) {
  const h = cv.clientHeight, d = state.duration;
  const step = d > 120 ? 10 : d > 40 ? 5 : d > 15 ? 2 : 1;
  g.strokeStyle = "#ffffff12"; g.fillStyle = "#b6a5a5"; g.font = "10px monospace"; g.lineWidth = 1;
  for (let t = 0; t <= d + 1e-6; t += step) {
    const x = Math.round(xOf(cv, t)) + 0.5;
    g.beginPath(); g.moveTo(x, 0); g.lineTo(x, h); g.stroke();
    if (withLabels && x + 26 < cv.clientWidth) g.fillText(`${t}s`, x + 2, 10);
  }
}

function drawPlayhead(g, cv) {
  const a = $("#audio");
  if (!current || !a.src) return;
  const x = Math.round(xOf(cv, a.currentTime)) + 0.5;
  g.beginPath(); g.moveTo(x, 0); g.lineTo(x, cv.clientHeight);
  g.strokeStyle = "#000000a0"; g.lineWidth = 3; g.stroke();
  g.strokeStyle = "#ffffff"; g.lineWidth = 1; g.stroke();
}

function envPath(g, cv, lane, max) {
  const w = cv.clientWidth;
  g.beginPath();
  for (let x = 0; x <= w; x += 2) {
    const y = yOf(cv, amountAt(lane.points, tOf(cv, x)), max);
    x ? g.lineTo(x, y) : g.moveTo(x, y);
  }
}

// Log-frequency spectrogram of a mono mix: SPEC_COLS frames x SPEC_ROWS bands, dB mapped onto a
// dark-red-yellow ramp, rendered once per take into an offscreen canvas and stretched on draw.
const SPEC_COLS = 1200, SPEC_ROWS = 160, NFFT = 2048;
function fftMag(re, im) {          // in-place radix-2, returns nothing; re/im length NFFT
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = a + len / 2;
        const tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
        const nr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = nr;
      }
    }
  }
}
function spectrogram(mono, sr) {
  const win = new Float32Array(NFFT).map((_, i) => 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (NFFT - 1)));
  const fmin = 30, fmax = Math.min(20000, sr / 2);
  // each row averages the FFT bins between two log-spaced edges (at least one bin)
  const edges = Array.from({ length: SPEC_ROWS + 1 }, (_, r) => fmin * Math.pow(fmax / fmin, r / SPEC_ROWS) * NFFT / sr);
  const db = new Float32Array(SPEC_COLS * SPEC_ROWS);
  const re = new Float32Array(NFFT), im = new Float32Array(NFFT);
  const hop = Math.max(1, (mono.length - NFFT) / (SPEC_COLS - 1));
  for (let c = 0; c < SPEC_COLS; c++) {
    const o = Math.floor(c * hop);
    for (let i = 0; i < NFFT; i++) { re[i] = (mono[o + i] || 0) * win[i]; im[i] = 0; }
    fftMag(re, im);
    for (let r = 0; r < SPEC_ROWS; r++) {
      const b0 = Math.floor(edges[r]), b1 = Math.max(b0 + 1, Math.floor(edges[r + 1]));
      let acc = 0;
      for (let b = b0; b < b1; b++) acc += re[b] * re[b] + im[b] * im[b];
      db[c * SPEC_ROWS + r] = 10 * Math.log10(acc / (b1 - b0) + 1e-10);
    }
  }
  let top = -Infinity;
  for (const v of db) if (v > top) top = v;
  const img = new ImageData(SPEC_COLS, SPEC_ROWS);
  for (let c = 0; c < SPEC_COLS; c++) for (let r = 0; r < SPEC_ROWS; r++) {
    const v = Math.pow(Math.min(1, Math.max(0, (db[c * SPEC_ROWS + r] - (top - 75)) / 75)), 1.7);   // 75 dB, darkened
    const o = ((SPEC_ROWS - 1 - r) * SPEC_COLS + c) * 4;
    img.data[o] = Math.min(255, v * 2.2 * 255);
    img.data[o + 1] = Math.max(0, Math.min(255, (v - 0.45) * 2.2 * 255));
    img.data[o + 2] = Math.max(0, Math.min(255, (v - 0.8) * 4 * 255));
    img.data[o + 3] = 255;
  }
  const off = document.createElement("canvas");
  off.width = SPEC_COLS; off.height = SPEC_ROWS;
  off.getContext("2d").putImageData(img, 0, 0);
  return off;
}

function drawWaveCanvas() {
  const cv = $("#wave"), g = prep(cv);
  if (current && view === "spec" && current.spec) {
    const x0 = xOf(cv, 0), x1 = xOf(cv, Math.min(current.dur, state.duration));
    g.imageSmoothingEnabled = true;
    g.drawImage(current.spec, 0, 0, Math.round(SPEC_COLS * Math.min(1, state.duration / current.dur)), SPEC_ROWS,
                x0, 0, x1 - x0, cv.clientHeight);
  }
  drawGrid(g, cv, true);
  if (current && view === "wave") drawWave(g, cv, current.peaks, current.dur, 0.85);
  if (!current) { g.fillStyle = "#b6a5a5"; g.font = "12px sans-serif"; g.fillText("Generate a take to see it here", 14, cv.clientHeight / 2); }
  // every lane's curve over the take, on a shared 0..max scale
  const lanes = current ? current.lanes : state.lanes;
  const max = Math.max(1, ...lanes.map((l) => l.max || 1));
  for (const l of lanes) {
    if (!l.points.length) continue;
    envPath(g, cv, l, max);
    g.strokeStyle = "#000000b0"; g.lineWidth = 4; g.stroke();     // outline so curves read over the spectrogram
    g.strokeStyle = l.color; g.lineWidth = 2; g.stroke();
  }
  drawPlayhead(g, cv);
}

function drawLane(cv, lane, hover) {
  const g = prep(cv), w = cv.clientWidth, h = cv.clientHeight;
  drawGrid(g, cv, false);
  if (current) drawWave(g, cv, current.peaks, current.dur, 0.22);
  // amount grid
  g.font = "10px monospace";
  for (const a of [0, 0.5, 1, 1.5, 2].filter((v) => v <= lane.max)) {
    const y = Math.round(yOf(cv, a, lane.max)) + 0.5;
    g.strokeStyle = a === 1 ? "#ffffff30" : "#ffffff14";
    g.beginPath(); g.moveTo(0, y); g.lineTo(w, y); g.stroke();
    g.fillStyle = "#b6a5a580"; g.fillText(a.toFixed(1), w - 24, y - 2);
  }
  if (!lane.points.length) { drawPlayhead(g, cv); return; }
  envPath(g, cv, lane, lane.max);
  g.lineTo(w, h); g.lineTo(0, h); g.closePath();
  g.fillStyle = lane.color + "26"; g.fill();
  envPath(g, cv, lane, lane.max);
  g.strokeStyle = lane.color; g.lineWidth = 2; g.stroke();
  lane.points.forEach((p, i) => {
    g.beginPath(); g.arc(xOf(cv, p[0]), yOf(cv, p[1], lane.max), i === hover ? 6 : 4, 0, Math.PI * 2);
    g.fillStyle = i === hover ? "#fff" : lane.color; g.fill();
  });
  drawPlayhead(g, cv);
}

const laneCanvases = new Map();   // canvas -> {lane, hover}
function drawAll() {
  drawWaveCanvas();
  for (const [cv, s] of laneCanvases) if (cv.isConnected) drawLane(cv, s.lane, s.hover); else laneCanvases.delete(cv);
  const a = $("#audio");
  $("#time").textContent = `${(a.currentTime || 0).toFixed(2)} / ${(current ? current.dur : 0).toFixed(2)}`;
}

// ─── lane editing ───────────────────────────────────────────────────────────

function attachLaneEditor(cv, tip, lane) {
  const s = { lane, hover: -1 };
  laneCanvases.set(cv, s);
  let drag = null;   // {kind: "point", i} | {kind: "draw", t0, t1, samples}
  const pos = (e) => { const r = cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
  const hit = (x, y) => lane.points.findIndex((p) => Math.hypot(xOf(cv, p[0]) - x, yOf(cv, p[1], lane.max) - y) < 8);
  const snap = (t, a, e) => e.ctrlKey || e.metaKey ? [round(t, 0.25), round(a, 0.05)] : [round(t, 0.01), round(a, 0.005)];
  const showTip = (x, y, t, a) => {
    tip.style.display = "block"; tip.textContent = `${t.toFixed(2)}s · ${a.toFixed(2)}`;
    tip.style.left = Math.min(x + 10, cv.clientWidth - 90) + "px"; tip.style.top = Math.max(0, y - 22) + "px";
  };

  cv.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    const [x, y] = pos(e);
    cv.setPointerCapture(e.pointerId);
    const i = hit(x, y);
    if (i >= 0) { drag = { kind: "point", i }; return; }
    const [t, a] = snap(tOf(cv, x), aOf(cv, y, lane.max), e);
    if (e.shiftKey) { drag = { kind: "draw", samples: [[t, a]] }; return; }
    lane.points.push([t, a]); sortPts(lane);
    drag = { kind: "point", i: lane.points.findIndex((p) => p[0] === t && p[1] === a) };
    s.hover = drag.i; drawAll();
  });

  cv.addEventListener("pointermove", (e) => {
    const [x, y] = pos(e);
    let [t, a] = snap(tOf(cv, x), aOf(cv, y, lane.max), e);
    if (drag && drag.kind === "point") {
      const P = lane.points, i = drag.i;
      const lo = i > 0 ? P[i - 1][0] : 0, hi = i < P.length - 1 ? P[i + 1][0] : state.duration;
      t = Math.min(hi, Math.max(lo, t));
      P[i] = [t, a]; s.hover = i;
      showTip(x, y, t, a); drawAll();
    } else if (drag && drag.kind === "draw") {
      drag.samples.push([t, a]);
      const ts = drag.samples.map((p) => p[0]);
      const t0 = Math.min(...ts), t1 = Math.max(...ts);
      // the stroke replaces whatever was under it, resampled at ~0.1 s
      const byT = [...drag.samples].sort((p, q) => p[0] - q[0]);
      const kept = lane.points.filter((p) => p[0] < t0 - 1e-6 || p[0] > t1 + 1e-6);
      const stroke = [];
      for (const p of byT) if (!stroke.length || p[0] - stroke[stroke.length - 1][0] >= 0.1) stroke.push([...p]);
      if (stroke.length && byT.length) stroke[stroke.length - 1] = [...byT[byT.length - 1]];
      lane.points = kept.concat(stroke); sortPts(lane);
      showTip(x, y, t, a); drawAll();
    } else {
      const i = hit(x, y);
      if (i !== s.hover) { s.hover = i; drawAll(); }
      cv.style.cursor = i >= 0 ? "grab" : "crosshair";
      showTip(x, y, i >= 0 ? lane.points[i][0] : t, i >= 0 ? lane.points[i][1] : amountAt(lane.points, t));
    }
  });

  const end = () => { if (drag) { drag = null; save(); } };
  cv.addEventListener("pointerup", end);
  cv.addEventListener("pointercancel", end);
  cv.addEventListener("pointerleave", () => { tip.style.display = "none"; if (!drag && s.hover >= 0) { s.hover = -1; drawAll(); } });
  const del = (e) => {
    const [x, y] = pos(e); const i = hit(x, y);
    if (i >= 0) { e.preventDefault(); lane.points.splice(i, 1); s.hover = -1; drawAll(); save(); }
  };
  cv.addEventListener("dblclick", del);
  cv.addEventListener("contextmenu", del);
}

// ─── generation ─────────────────────────────────────────────────────────────

function showError(msg) { $("#error").textContent = msg || ""; }

function buildRequest() {
  const lanes = state.lanes.filter((l) => l.lora && l.points.length);
  const body = {
    prompt: state.prompt,
    duration: state.duration,
    steps: state.steps,
    seed: state.seed,
    cfg_scale: state.cfg,
    keep_models: false,   // frugal: on 8 GB, DiT + lane adapters + decoder together oversubscribe VRAM
    lora_envelope_mode: state.mode,
    lora_lanes: lanes.map((l) => ({
      loras: [{ name: l.lora, strength: l.strength }],
      envelope: l.points.map((p) => [+p[0].toFixed(3), +p[1].toFixed(3)]),
      ...(l.prompt.trim() ? { prompt: l.prompt.trim() } : {}),
    })),
  };
  if (state.negative.trim()) body.negative_prompt = state.negative.trim();
  return body;
}

async function generate() {
  readForm();
  if (!state.seedLock) { state.seed = randomSeed(); writeForm(); }
  save();
  showError("");
  const body = buildRequest();
  const snapshot = JSON.parse(JSON.stringify(state));
  const btn = $("#gen");
  btn.disabled = true; btn.textContent = "Generating…";
  const t0 = performance.now();
  try {
    const r = await apiPost("/generate", body);
    poll(r.session_id, body, snapshot, t0);
  } catch (e) {
    showError(e.message); btn.disabled = false; btn.textContent = "Generate";
  }
}

function poll(id, body, snapshot, t0) {
  clearInterval(pollTimer);
  const bar = $("#progress"), lbl = $("#progress-label"), btn = $("#gen");
  const done = () => { clearInterval(pollTimer); btn.disabled = false; btn.textContent = "Generate"; };
  pollTimer = setInterval(async () => {
    try {
      const r = await apiGet(`/poll_status/${id}`);
      bar.style.width = `${r.progress || 0}%`;
      lbl.textContent = r.status === "completed" ? "done" : `${r.status} ${r.step ?? ""}/${r.total_steps ?? ""}`;
      if (r.status === "completed" && r.audio_data) {
        done();
        await addTake(r.audio_data, body, snapshot, (r.meta && r.meta.seed) ?? body.seed, (performance.now() - t0) / 1000);
      } else if (r.status === "failed" || r.status === "cancelled" || r.success === false) {
        done(); showError(r.error || r.status);
      }
    } catch (e) { done(); showError(e.message); }
  }, 400);
}

let audioCtx = null;
async function addTake(b64, request, snapshot, seed, secs) {
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const url = URL.createObjectURL(new Blob([bytes], { type: "audio/wav" }));
  audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
  const buf = await audioCtx.decodeAudioData(bytes.buffer.slice(0));
  // min/max peaks at 2000 columns over the whole take
  const n = 2000, ch = buf.getChannelData(0), ch2 = buf.numberOfChannels > 1 ? buf.getChannelData(1) : null;
  const peaks = new Float32Array(2 * n), per = Math.max(1, Math.floor(ch.length / n));
  for (let i = 0; i < n; i++) {
    let lo = 0, hi = 0;
    for (let j = i * per, e = Math.min(ch.length, j + per); j < e; j++) {
      const v = ch2 ? 0.5 * (ch[j] + ch2[j]) : ch[j];
      if (v < lo) lo = v; if (v > hi) hi = v;
    }
    peaks[2 * i] = lo; peaks[2 * i + 1] = hi;
  }
  const mono = ch2 ? ch.map((v, i) => 0.5 * (v + ch2[i])) : ch;
  const spec = spectrogram(mono, buf.sampleRate);
  const take = { id: nextTakeId++, url, peaks, spec, dur: buf.duration, request, seed, secs, lanes: snapshot.lanes, snapshot };
  takes.unshift(take);
  setCurrent(take, true);
}

function setCurrent(take, autoplay) {
  const a = $("#audio");
  const keepAt = current && a.src && !a.paused ? a.currentTime : null;
  current = take;
  a.src = take.url;
  if (keepAt != null) a.addEventListener("loadedmetadata", () => { a.currentTime = Math.min(keepAt, take.dur); }, { once: true });
  if (autoplay || keepAt != null) a.play().catch(() => {});
  const names = take.lanes.filter((l) => l.lora).map((l) => l.lora).join(" + ") || "base only";
  $("#take-info").textContent = `#${take.id} · seed ${take.seed} · ${names}`;
  renderTakes(); drawAll();
}

function renderTakes() {
  const host = $("#takes");
  if (!takes.length) return;
  host.innerHTML = "";
  for (const t of takes) {
    const div = document.createElement("div");
    div.className = "take" + (t === current ? " current" : "");
    const lanes = t.lanes.filter((l) => l.lora);
    const dots = lanes.map((l) => `<span class="swatch" style="background:${l.color};display:inline-block"></span>`).join(" ");
    div.innerHTML = `<b>#${t.id}</b> ${dots}
      <span class="lbl" title="${esc(t.request.prompt)}">${esc(lanes.map((l) => l.lora + (l.prompt ? "*" : "")).join(" + ") || "base")} — ${esc(t.request.prompt)}</span>
      <span class="meta">${t.request.lora_envelope_mode} · seed ${t.seed} · ${t.request.steps}st · ${t.secs.toFixed(1)}s</span>
      <button class="tiny" title="restore this take's settings into the editor">↺</button>`;
    div.onclick = () => setCurrent(t, true);
    div.querySelector("button").onclick = (e) => {
      e.stopPropagation();
      Object.assign(state, JSON.parse(JSON.stringify(t.snapshot)));
      state.seed = t.seed; state.seedLock = true;
      writeForm(); renderLanes(); save();
    };
    host.appendChild(div);
  }
}

// ─── transport ──────────────────────────────────────────────────────────────

function togglePlay() {
  const a = $("#audio");
  if (!a.src) return;
  a.paused ? a.play().catch(() => {}) : a.pause();
}
function tick() {
  const a = $("#audio");
  $("#play").textContent = a.paused ? "▶" : "❚❚";
  if (!a.paused) drawAll();
  requestAnimationFrame(tick);
}

// ─── wiring ─────────────────────────────────────────────────────────────────

function init() {
  load();
  if (!state.seed) state.seed = randomSeed();
  writeForm();
  for (const id of ["#prompt", "#negative", "#duration", "#steps", "#cfg", "#seed", "#seed-lock", "#mode"]) {
    $(id).addEventListener("input", () => { readForm(); drawAll(); save(); });
  }
  $("#gen").onclick = generate;
  $("#seed-dice").onclick = () => { state.seed = randomSeed(); writeForm(); save(); };
  $("#base-dice").onclick = async () => {
    try { state.prompt = await dicePrompt(""); writeForm(); save(); } catch (e) { showError(e.message); }
  };
  $("#add-lane").onclick = () => { if (state.lanes.length < MAX_LANES) { state.lanes.push(newLane()); renderLanes(); save(); } };
  $("#play").onclick = togglePlay;
  $("#view").onclick = () => { view = view === "spec" ? "wave" : "spec"; $("#view").textContent = view === "spec" ? "show waveform" : "show spectrogram"; drawAll(); };
  $("#stop").onclick = () => { const a = $("#audio"); a.pause(); a.currentTime = 0; drawAll(); };
  $("#wave").addEventListener("pointerdown", (e) => {
    if (!current) return;
    const cv = $("#wave"), r = cv.getBoundingClientRect();
    $("#audio").currentTime = Math.min(current.dur, tOf(cv, e.clientX - r.left));
    drawAll();
  });
  $("#audio").addEventListener("ended", drawAll);
  document.addEventListener("keydown", (e) => {
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName);
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); generate(); }
    else if (e.code === "Space" && !typing) { e.preventDefault(); togglePlay(); }
  });
  window.addEventListener("resize", resize);
  renderLanes();
  connect();
  requestAnimationFrame(tick);
}
init();
