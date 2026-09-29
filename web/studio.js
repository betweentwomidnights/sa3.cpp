"use strict";

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
      <div class="studio-history-head"><h2>Past takes <span id="studio-history-count"></span></h2><div id="studio-history-actions"></div></div><div id="studio-history-slot"></div>
    </main>
    <main id="studio-view-train" class="studio-view" hidden><iframe id="studio-training-frame" title="LoRA training workspace"></iframe></main>
    <main id="studio-view-models" class="studio-view" hidden><div class="studio-hero"><div><div class="studio-eyebrow">Model library</div><h2>Get the sound you need.</h2><p>Model weights live beside the runtime in your configured models folder.</p></div></div><div id="studio-model-cards" class="studio-model-grid"></div><section class="studio-panel" style="margin-top:16px"><h2 style="font-size:1rem;margin:0 0 12px;border:0">Download a model set</h2><div class="param-grid"><div><label for="studio-download-variant">Variant</label><select id="studio-download-variant"><option value="medium">Medium</option><option value="small-music">Small music</option><option value="small-sfx">Small SFX</option></select></div><div><label for="studio-download-tier">DiT tier</label><select id="studio-download-tier"><option value="f16">F16</option><option value="q4_k_m">Q4_K_M · compact</option><option value="f32">F32</option><option value="q5_k_m">Q5_K_M</option><option value="q8_0">Q8_0</option></select></div></div><p class="small-note" style="margin:12px 0">The set includes an F32 autoencoder, F16 text encoder, conditioner, and tokenizer. <a href="https://huggingface.co/thepatch" target="_blank" rel="noopener" style="color:var(--accent)">Review model licenses and cards</a>.</p><button id="studio-download-model" type="button" class="primary">Download model set</button><p id="studio-download-status" class="small-note" role="status" style="margin:10px 0 0"></p></section><section class="studio-panel" style="margin-top:16px"><h2 style="font-size:1rem;margin:0 0 8px;border:0">Decoder corrections</h2><p class="small-note">A selected correction runs with every generation on its compatible model. Creative LoRAs stay in the render dialogs.</p><div id="studio-decoder-list"></div><button id="studio-download-decoder" type="button">Download SAME-L correction</button><p id="studio-decoder-status" class="small-note" role="status"></p></section></main>
    <main id="studio-view-settings" class="studio-view studio-settings" hidden><div class="studio-hero"><div><div class="studio-eyebrow">Studio settings</div><h2>Runtime and output.</h2><p>Keep the model loaded and tune output processing here.</p></div></div><div id="studio-session-settings"></div><details><summary>Audio processing</summary><div id="studio-audio-settings"></div></details></main>
    <div id="studio-footer"></div>
  </div>
    <div id="studio-modal" class="studio-modal" hidden><div class="studio-dialog" role="dialog" aria-modal="true" aria-labelledby="studio-dialog-title"><div class="studio-dialog-head"><h2 id="studio-dialog-title">create</h2><button id="studio-dialog-close" type="button" aria-label="Close dialog">×</button></div><p id="studio-dialog-description" class="studio-dialog-description">Start a new take from a prompt.</p><div id="studio-dialog-prompt"></div><div id="studio-dialog-basic"></div><div class="studio-extra-controls"><div id="studio-tail-wrap"><span class="studio-field-label">Ending</span><div class="studio-segmented"><label><input type="radio" name="studio-tail" value="0"> Ends here</label><label><input type="radio" name="studio-tail" value="6"> Keeps going</label></div></div><div id="studio-loop-wrap"><label class="inline-label"><input id="studio-loop" type="checkbox"> Make a loop</label><div id="studio-loop-fields" class="studio-loop-options studio-conditional" hidden></div></div><div id="studio-noise-wrap" class="studio-conditional" hidden><label for="studio-noise">Transform strength <output id="studio-noise-value">0.85</output></label><input id="studio-noise" type="range" min="0" max="1" step="0.05" value="0.85"></div></div><div id="studio-dialog-loras"></div><details id="studio-advanced"><summary>Advanced</summary><div id="studio-dialog-cfg"></div><div id="studio-dialog-ds"></div><div id="studio-inpaint-wrap" hidden><label class="inline-label"><input id="studio-inpaint" type="checkbox"> Inpaint part of the selected audio</label><div class="param-grid studio-conditional" id="studio-inpaint-fields" hidden><div><label for="studio-inpaint-start">Start (seconds)</label><input id="studio-inpaint-start" type="number" min="0" step="0.01" value="0"></div><div><label for="studio-inpaint-end">End (seconds)</label><input id="studio-inpaint-end" type="number" min="0" step="0.01" value="0"></div></div></div></details><p id="studio-dialog-error" role="alert" hidden></p><div class="studio-dialog-actions"><button id="studio-dialog-cancel" type="button">Cancel</button><button id="studio-dialog-submit" type="button" class="primary">Create</button></div></div></div>`;

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
  if (name === "studio") loadLoras();
  location.hash = name === "studio" ? "" : name;
}

function studioDuration() { return studio.buffer?.duration || 0; }
function studioRange() { const d = studioDuration(); return [studio.selection[0]*d, studio.selection[1]*d]; }
function studioTime(sec) { const s = Math.max(0, sec); return `${Math.floor(s/60)}:${String(Math.floor(s%60)).padStart(2,"0")}.${String(Math.floor(s%1*100)).padStart(2,"0")}`; }
function studioUpdateTakeControls() {
  const has = !!studio.buffer;
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
    for(const x of [left,right]){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke();ctx.fillStyle="#f4eded";ctx.fillRect(Math.max(0,x-4*scale),h/2-12*scale,8*scale,24*scale);}
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
  studioEl("studio-dialog-description").textContent=descriptions[operation];
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
      const card=document.createElement("div");card.className="studio-model-card";
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
    status.textContent="Preparing decoder correction…";
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
  studioEl("studio-clear").onclick=()=>{deleteCurrentSong();studioClearTake();};
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
  window.addEventListener("message",event=>{if(event.origin===location.origin&&event.data?.type==="sa3-lora-registered")loadLoras();});
  const initial=location.hash.slice(1);if(["train","models","settings"].includes(initial))studioShowView(initial);
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
