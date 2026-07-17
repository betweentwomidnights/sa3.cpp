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
// ─── Theme ─────────────────────────────────────────────────────────────────
function toggleTheme() {
    const root = document.documentElement;
    const current = root.dataset.theme || "dark";
    const next = current === "dark" ? "light" : "dark";
    root.dataset.theme = next;
    localStorage.setItem("sa3-theme", next);
    const btn = $("#theme-btn");
    btn.textContent = next === "dark" ? "☀️" : "🌙";
}
function loadTheme() {
    const saved = localStorage.getItem("sa3-theme");
    if (saved === "light" || saved === "dark") {
        document.documentElement.dataset.theme = saved;
        const btn = $("#theme-btn");
        btn.textContent = saved === "dark" ? "☀️" : "🌙";
    }
}
// ─── Past Songs ────────────────────────────────────────────────────────────
function pushPastSong(entry) {
    pastSongs.push(entry);
    localStorage.setItem("sa3-past-songs", JSON.stringify(pastSongs));
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
      <audio controls src="${s.audioUrl}"></audio>
      <span class="song-params">seed: ${s.seed}</span>
      <span class="song-actions">
        <button class="small load-params-btn" data-index="${i}" title="Load generation params">📋</button>
        <button class="small download-song-btn" data-index="${i}" title="Download WAV">⬇</button>
        <button class="small danger delete-song-btn" data-index="${i}" title="Delete">&times;</button>
      </span>`;
        container.appendChild(div);
    }
    for (const btn of container.querySelectorAll(".delete-song-btn")) {
        btn.addEventListener("click", () => {
            const idx = parseInt(btn.dataset.index || "0", 10);
            pastSongs.splice(idx, 1);
            localStorage.setItem("sa3-past-songs", JSON.stringify(pastSongs));
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
            if (!s || !s.params)
                return;
            loadParamsFromSnapshot(s.params);
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
    set("#peak-normalize-db", pndb != null && pndb !== false ? String(pndb) : "");
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
        activeLoras = loras.map((l) => ({ ...l }));
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
async function generate() {
    clearPolling();
    if (currentResult) {
        pushPastSong(currentResult);
        currentResult = null;
    }
    const body = readForm();
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
async function generateLoop() {
    clearPolling();
    if (currentResult) {
        pushPastSong(currentResult);
        currentResult = null;
    }
    const body = {
        ...readForm(),
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
                progressLabel.textContent = `${r.status} step ${r.step}/${r.total_steps} (${r.progress}%)`;
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
    loadTheme();
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
    $("#theme-btn").addEventListener("click", toggleTheme);
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
