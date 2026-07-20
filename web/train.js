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
