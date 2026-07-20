#pragma once
// embedded_train_web.h — training web UI assets embedded in the binary at build time.
// AUTO-GENERATED from web/train.html and web/train.js. Do not edit by hand.
// Rebuild with: python3 tools/gen_embedded_train_web.py

namespace embedded_train_web {

inline const char* index_html = R"sa3trainweb(
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>sa3-train-web — LoRA Training</title>
  <style>
    :root { --bg:#0f1117; --panel:#171a23; --panel2:#1f2330; --fg:#e6e8ee; --muted:#8b93a7;
            --accent:#6ea8fe; --ok:#4ade80; --warn:#fbbf24; --bad:#f87171; --border:#2a2f3d; }
    * { box-sizing: border-box; }
    body { margin:0; font:14px/1.45 system-ui,Segoe UI,Roboto,sans-serif; background:var(--bg); color:var(--fg); }
    header { padding:10px 16px; border-bottom:1px solid var(--border); display:flex; align-items:center; gap:12px; }
    header h1 { font-size:16px; margin:0; font-weight:600; }
    header .spacer { flex:1; }
    #new-btn { background:var(--accent); color:#0b0e14; border:none; padding:7px 14px; border-radius:6px; cursor:pointer; font-weight:600; }
    #new-btn:disabled { opacity:.45; cursor:not-allowed; }
    .layout { display:flex; height:calc(100vh - 49px); }
    .sidebar { width:280px; border-right:1px solid var(--border); overflow-y:auto; background:var(--panel); }
    .sidebar h2 { font-size:11px; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); padding:12px 14px 6px; margin:0; }
    .run { padding:10px 14px; border-bottom:1px solid var(--border); cursor:pointer; }
    .run:hover { background:var(--panel2); }
    .run.active { background:var(--panel2); border-left:3px solid var(--accent); }
    .run .top { display:flex; justify-content:space-between; align-items:center; }
    .run .ds { font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:170px; }
    .badge { font-size:10px; padding:2px 7px; border-radius:10px; text-transform:uppercase; letter-spacing:.04em; }
    .b-running { background:rgba(110,168,254,.18); color:var(--accent); }
    .b-completed { background:rgba(74,222,128,.16); color:var(--ok); }
    .b-stopped { background:rgba(251,191,36,.16); color:var(--warn); }
    .b-failed { background:rgba(248,113,113,.16); color:var(--bad); }
    .b-queued { background:rgba(139,147,167,.16); color:var(--muted); }
    .run .meta { color:var(--muted); font-size:12px; margin-top:3px; }
    .run .bar { height:4px; background:var(--border); border-radius:3px; margin-top:6px; overflow:hidden; }
    .run .bar > i { display:block; height:100%; background:var(--accent); }
    .main { flex:1; overflow-y:auto; padding:18px; }
    .card { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:16px; margin-bottom:16px; }
    .card h3 { margin:0 0 12px; font-size:13px; color:var(--muted); text-transform:uppercase; letter-spacing:.06em; }
    .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; }
    label { display:block; font-size:12px; color:var(--muted); margin-bottom:4px; }
    input, select { width:100%; background:var(--panel2); border:1px solid var(--border); color:var(--fg);
                    padding:8px 10px; border-radius:6px; font-size:13px; }
    .row { display:flex; gap:10px; align-items:flex-end; flex-wrap:wrap; }
    .actions { display:flex; gap:10px; margin-top:14px; }
    button.sec { background:var(--panel2); color:var(--fg); border:1px solid var(--border); padding:8px 14px; border-radius:6px; cursor:pointer; }
    button.danger { background:rgba(248,113,113,.15); color:var(--bad); border:1px solid rgba(248,113,113,.35); padding:8px 14px; border-radius:6px; cursor:pointer; }
    button.primary { background:var(--accent); color:#0b0e14; border:none; padding:8px 16px; border-radius:6px; cursor:pointer; font-weight:600; }
    .stat { background:var(--panel2); border-radius:8px; padding:12px; }
    .stat .k { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.05em; }
    .stat .v { font-size:20px; font-weight:600; margin-top:3px; }
    #spark { width:100%; height:90px; background:var(--panel2); border-radius:8px; display:block; }
    #log { background:#0b0d12; border:1px solid var(--border); border-radius:8px; padding:10px; height:240px;
           overflow-y:auto; font:12px/1.4 ui-monospace,Menlo,Consolas,monospace; white-space:pre-wrap; color:#cdd3e0; }
    .art { display:flex; align-items:center; gap:10px; padding:8px 0; border-bottom:1px solid var(--border); }
    .art a { color:var(--accent); text-decoration:none; font-weight:600; }
    .art .sz { color:var(--muted); font-size:12px; }
    audio { width:100%; margin-top:6px; }
    .hidden { display:none; }
    .err { color:var(--bad); font-size:13px; margin-top:8px; }
  </style>
</head>
<body>
  <header>
    <h1>sa3-train-web</h1>
    <span id="health" class="badge b-queued">…</span>
    <div class="spacer"></div>
    <button id="new-btn">+ New training</button>
  </header>
  <div class="layout">
    <aside class="sidebar">
      <h2>Runs</h2>
      <div id="run-list"></div>
    </aside>
    <main class="main">
      <!-- Config form -->
      <section id="form-card" class="card hidden">
        <h3>New training run</h3>
        <div class="grid">
          <div><label>Dataset dir</label><input id="f-dataset" placeholder="/path/to/dataset" /></div>
          <div><label>Model</label><select id="f-model"><option>medium</option><option>small-music</option><option>small-sfx</option></select></div>
          <div><label>Adapter type</label><select id="f-adapter"><option>dora-rows</option><option>lora</option><option>dora-cols</option><option>bora</option><option>lora-xs</option><option>dora-rows-xs</option><option>dora-cols-xs</option><option>bora-xs</option></select></div>
          <div><label>Rank</label><input id="f-rank" type="number" value="16" /></div>
          <div><label>Alpha</label><input id="f-alpha" type="number" step="0.1" value="16" /></div>
          <div><label>Learning rate</label><input id="f-lr" type="number" step="1e-5" value="0.0001" /></div>
          <div><label>Max steps</label><input id="f-steps" type="number" value="10000" /></div>
          <div><label>Frames</label><input id="f-frames" type="number" value="512" /></div>
          <div><label>Batch size</label><input id="f-batch" type="number" value="1" /></div>
          <div><label>Checkpoint every</label><input id="f-ckpt" type="number" value="500" /></div>
          <div><label>Seed</label><input id="f-seed" type="number" value="42" /></div>
          <div><label>CFG dropout</label><input id="f-cfgdo" type="number" step="0.05" value="0.1" /></div>
          <div><label>Grad clip</label><input id="f-gradclip" type="number" step="0.1" value="1.0" /></div>
          <div><label>Output dir (optional)</label><input id="f-out" placeholder="train-runs/… (auto)" /></div>
        </div>
        <div class="row" style="margin-top:12px;">
          <label style="display:flex;align-items:center;gap:6px;color:var(--fg);"><input id="f-inpaint" type="checkbox" checked style="width:auto;" /> Inpainting loss</label>
        </div>
        <div class="actions">
          <button class="primary" id="start-btn">Start training</button>
          <button class="sec" id="cancel-btn">Cancel</button>
        </div>
        <div id="form-err" class="err"></div>
      </section>

      <!-- Detail / live view -->
      <section id="detail-card" class="card hidden">
        <h3 id="detail-title">Run</h3>
        <div class="grid" style="margin-bottom:14px;">
          <div class="stat"><div class="k">Status</div><div class="v" id="d-status">—</div></div>
          <div class="stat"><div class="k">Step</div><div class="v" id="d-step">0</div></div>
          <div class="stat"><div class="k">Loss</div><div class="v" id="d-loss">—</div></div>
          <div class="stat"><div class="k">LR</div><div class="v" id="d-lr">—</div></div>
          <div class="stat"><div class="k">Grad norm</div><div class="v" id="d-gn">—</div></div>
        </div>
        <div class="bar" style="height:6px;background:var(--border);border-radius:4px;overflow:hidden;margin-bottom:14px;">
          <i id="d-progress" style="display:block;height:100%;width:0;background:var(--accent);"></i>
        </div>
        <h3>Loss / LR</h3>
        <canvas id="spark"></canvas>
        <h3 style="margin-top:16px;">Log</h3>
        <div id="log"></div>
        <h3 style="margin-top:16px;">Artifacts</h3>
        <div id="artifacts"></div>
        <div class="actions">
          <button class="danger hidden" id="stop-btn">Stop training</button>
        </div>
      </section>

      <section id="empty-card" class="card">
        <h3>No run selected</h3>
        <p style="color:var(--muted);">Pick a run from the sidebar, or start a new training.</p>
      </section>
    </main>
  </div>
  <script src="train.js"></script>
</body>
</html>

)sa3trainweb";

inline const char* train_js = R"sa3trainweb(
"use strict";
// sa3-train-web frontend — vanilla JS, no build step (mirrors web/app.js).
// Talks to the companion's HTTP API: /api/train/*

const API = { host: location.hostname, port: location.port || 8016 };
const base = `http://${API.host}:${API.port}`;

const $ = (id) => document.getElementById(id);

let state = {
  runs: [],          // history list from /api/train/runs
  selectedId: null,  // currently viewed run
  pollTimer: null,
  logOffset: 0,
  metricsCache: [],  // [{step,lr,loss,grad_norm}] for sparkline
};

// ---- API helpers -----------------------------------------------------------
async function apiGet(path) {
  try {
    const res = await fetch(base + path);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.error("GET", path, "failed:", err);
    return null;
  }
}

async function apiPost(path, body) {
  try {
    const res = await fetch(base + path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: body ? JSON.stringify(body) : undefined,
    });
    const text = await res.text();
    let json = null;
    try { json = text ? JSON.parse(text) : null; } catch (_) { /* non-json */ }
    if (!res.ok) {
      const msg = (json && json.error) ? json.error : `HTTP ${res.status}`;
      throw new Error(msg);
    }
    return json;
  } catch (err) {
    console.error("POST", path, "failed:", err);
    throw err;
  }
}

// ---- health ----------------------------------------------------------------
async function refreshHealth() {
  const h = await apiGet("/api/health");
  const el = $("health");
  if (!h) { el.textContent = "offline"; el.className = "badge b-failed"; return; }
  el.textContent = "trainer ok";
  el.className = "badge b-completed";
  el.title = h.train_bin || "";
}

// ---- run list (sidebar) ----------------------------------------------------
function badgeClass(status) {
  return "badge b-" + (status || "queued");
}

function renderRunList() {
  const list = $("run-list");
  list.innerHTML = "";
  if (!state.runs.length) {
    list.innerHTML = '<div style="padding:12px 14px;color:var(--muted);">No runs yet.</div>';
    return;
  }
  for (const r of state.runs) {
    const div = document.createElement("div");
    div.className = "run" + (r.id === state.selectedId ? " active" : "");
    const pct = r.max_steps > 0 ? Math.min(100, Math.round((r.step || 0) / r.max_steps * 100)) : 0;
    div.innerHTML =
      `<div class="top"><span class="ds" title="${escapeHtml(r.dataset)}">${escapeHtml(r.dataset || r.id)}</span>` +
      `<span class="${badgeClass(r.status)}">${r.status}</span></div>` +
      `<div class="meta">${escapeHtml(r.model || "")} · ${escapeHtml(r.adapter_type || "")} · step ${r.step || 0}/${r.max_steps || 0}</div>` +
      `<div class="bar"><i style="width:${pct}%"></i></div>`;
    div.onclick = () => selectRun(r.id);
    list.appendChild(div);
  }
}

async function refreshRuns() {
  const runs = await apiGet("/api/train/runs");
  if (runs) {
    state.runs = runs;
    renderRunList();
    // keep "New training" disabled while a run is active
    const active = runs.find((r) => r.status === "running");
    $("new-btn").disabled = !!active;
  }
}

// ---- selecting a run -------------------------------------------------------
async function selectRun(id) {
  state.selectedId = id;
  state.logOffset = 0;
  state.metricsCache = [];
  renderRunList();
  $("empty-card").classList.add("hidden");
  $("form-card").classList.add("hidden");
  $("detail-card").classList.remove("hidden");
  await refreshDetail();
  startPolling();
}

function findSelected() {
  return state.runs.find((r) => r.id === state.selectedId) || null;
}

async function refreshDetail() {
  const r = findSelected();
  if (!r) return;
  $("detail-title").textContent = `${r.dataset || r.id} · ${r.model} · ${r.adapter_type}`;
  $("d-status").textContent = r.status;
  $("d-step").textContent = `${r.step || 0} / ${r.max_steps || 0}`;
  $("d-loss").textContent = (r.loss != null && r.loss !== 0) ? r.loss.toFixed(5) : "—";
  $("d-lr").textContent = r.lr ? r.lr.toExponential(2) : "—";
  $("d-gn").textContent = r.grad_norm ? r.grad_norm.toFixed(4) : "—";
  const pct = r.max_steps > 0 ? Math.min(100, Math.round((r.step || 0) / r.max_steps * 100)) : 0;
  $("d-progress").style.width = pct + "%";

  const isRunning = r.status === "running";
  $("stop-btn").classList.toggle("hidden", !isRunning);

  // metrics + log
  const metrics = await apiGet(`/api/train/metrics?run_id=${encodeURIComponent(r.id)}&limit=200`);
  if (metrics && Array.isArray(metrics)) {
    state.metricsCache = metrics.map((m) => ({
      step: m.update, lr: m.lr, loss: m.loss, grad_norm: m.grad_norm,
    }));
    drawSpark();
  }

  const log = await apiGet(`/api/train/log?run_id=${encodeURIComponent(r.id)}&offset=${state.logOffset}`);
  if (log) {
    if (log.log) {
      const el = $("log");
      el.textContent += log.log;
      el.scrollTop = el.scrollHeight;
    }
    if (typeof log.offset === "number") state.logOffset = log.offset;
  }

  // artifacts
  const arts = await apiGet(`/api/train/artifacts?run_id=${encodeURIComponent(r.id)}`);
  renderArtifacts(arts || []);
}

function renderArtifacts(arts) {
  const box = $("artifacts");
  box.innerHTML = "";
  if (!arts.length) {
    box.innerHTML = '<div style="color:var(--muted);">No artifacts yet.</div>';
    return;
  }
  for (const a of arts) {
    const div = document.createElement("div");
    div.className = "art";
    const dl = `${base}/api/train/download?run_id=${encodeURIComponent(state.selectedId)}&file=${encodeURIComponent(a.name)}`;
    const sz = a.size != null ? `(${(a.size / 1024 / 1024).toFixed(2)} MB)` : "";
    div.innerHTML = `<a href="${dl}" download>${escapeHtml(a.name)}</a><span class="sz">${sz}</span>`;
    box.appendChild(div);
    if (a.is_wav) {
      const audio = document.createElement("audio");
      audio.controls = true;
      audio.src = dl;
      audio.style.width = "100%";
      audio.style.marginTop = "6px";
      box.appendChild(audio);
    }
  }
}

// ---- sparkline (loss + lr, dual scale) -------------------------------------
function drawSpark() {
  const canvas = $("spark");
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr; canvas.height = h * dpr;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  const data = state.metricsCache;
  if (!data.length) return;
  const losses = data.map((d) => d.loss).filter((v) => isFinite(v));
  const lrs = data.map((d) => d.lr).filter((v) => isFinite(v) && v > 0);
  if (!losses.length) return;
  const lo = Math.min(...losses), hi = Math.max(...losses);
  const lrLo = lrs.length ? Math.min(...lrs) : 0, lrHi = lrs.length ? Math.max(...lrs) : 1;
  const n = data.length;
  const x = (i) => (n === 1 ? w / 2 : (i / (n - 1)) * w);
  const yLoss = (v) => h - 6 - ((hi === lo ? 0.5 : (v - lo) / (hi - lo)) * (h - 12));
  const yLr = (v) => h - 6 - ((lrHi === lrLo ? 0.5 : (Math.log(v) - Math.log(lrLo)) / (Math.log(lrHi) - Math.log(lrLo))) * (h - 12));

  // loss line
  ctx.strokeStyle = "#6ea8fe"; ctx.lineWidth = 2; ctx.beginPath();
  data.forEach((d, i) => { if (!isFinite(d.loss)) return; const px = x(i), py = yLoss(d.loss); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
  ctx.stroke();

  // lr line (log scale)
  if (lrs.length) {
    ctx.strokeStyle = "#fbbf24"; ctx.lineWidth = 1.5; ctx.beginPath();
    data.forEach((d, i) => { if (!(d.lr > 0)) return; const px = x(i), py = yLr(d.lr); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
    ctx.stroke();
  }
}

// ---- polling ---------------------------------------------------------------
function startPolling() {
  stopPolling();
  state.pollTimer = setInterval(async () => {
    await refreshRuns();          // updates sidebar + active flag
    if (state.selectedId) await refreshDetail();
  }, 1500);
}
function stopPolling() {
  if (state.pollTimer) { clearInterval(state.pollTimer); state.pollTimer = null; }
}

// ---- form ------------------------------------------------------------------
function openForm() {
  stopPolling();
  $("empty-card").classList.add("hidden");
  $("detail-card").classList.add("hidden");
  $("form-card").classList.remove("hidden");
  $("form-err").textContent = "";
}
function closeForm() {
  $("form-card").classList.add("hidden");
  $("empty-card").classList.remove("hidden");
}

async function startTraining() {
  const cfg = {
    dataset: $("f-dataset").value.trim(),
    model: $("f-model").value,
    adapter_type: $("f-adapter").value,
    rank: parseInt($("f-rank").value, 10) || 16,
    alpha: parseFloat($("f-alpha").value) || 16,
    learning_rate: parseFloat($("f-lr").value) || 1e-4,
    max_steps: parseInt($("f-steps").value, 10) || 10000,
    frames: parseInt($("f-frames").value, 10) || 512,
    batch_size: parseInt($("f-batch").value, 10) || 1,
    checkpoint_every: parseInt($("f-ckpt").value, 10) || 500,
    seed: parseInt($("f-seed").value, 10) || 42,
    cfg_dropout_prob: parseFloat($("f-cfgdo").value) || 0.1,
    grad_clip: parseFloat($("f-gradclip").value) || 1.0,
    inpainting: $("f-inpaint").checked,
    out: $("f-out").value.trim(),
  };
  if (!cfg.dataset) { $("form-err").textContent = "Dataset dir is required."; return; }
  try {
    const res = await apiPost("/api/train/start", cfg);
    if (res && res.run_id) {
      await refreshRuns();
      closeForm();
      await selectRun(res.run_id);
    }
  } catch (err) {
    $("form-err").textContent = err.message || "start failed";
  }
}

async function stopTraining() {
  try {
    await apiPost(`/api/train/stop?run_id=${encodeURIComponent(state.selectedId)}`);
  } catch (err) {
    console.error(err);
  }
}

// ---- utils -----------------------------------------------------------------
function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

// ---- wire up ---------------------------------------------------------------
$("new-btn").onclick = openForm;
$("cancel-btn").onclick = () => { closeForm(); $("empty-card").classList.remove("hidden"); };
$("start-btn").onclick = startTraining;
$("stop-btn").onclick = stopTraining;

// initial load
(async () => {
  await refreshHealth();
  await refreshRuns();
  setInterval(refreshHealth, 10000);
})();

)sa3trainweb";

} // namespace embedded_train_web
