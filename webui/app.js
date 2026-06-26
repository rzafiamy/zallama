const API = '';
let chatHistory = [];
let isStreaming = false;
let MODELS = {};            // name -> full model object from /api/models
let RUNNING = {};           // name -> running instance info from /api/ps
let pendingImage = null;    // {dataUrl, name} attached for the next chat turn

// One unified chat surface routes each modality to a separately-chosen model.
// The user picks a model per role in the header; a role is only usable when a
// matching model is actually loaded (running). `routes` holds the chosen name
// for each role (auto-filled from loaded models, user-overridable).
let routes = { chat: '', asr: '', tts: '' };
let routesInitialized = false;  // becomes true after first selector population

// Vision isn't a separate route — it's a capability of the selected chat model
// (a text model that ships an mmproj projector). The 📎 attach button and image
// turns key off this.
function chatHasVision() { return !!routes.chat && isVision(MODELS[routes.chat]); }

const MODALITY = {
  text:      { icon: '💬', label: 'Chat' },
  asr:       { icon: '🎙', label: 'Speech→Text' },
  tts:       { icon: '🔊', label: 'Text→Speech' },
  embedding: { icon: '🧬', label: 'Embedding' },
  rerank:    { icon: '🏷', label: 'Rerank' },
  image:     { icon: '🖼', label: 'Image' },
};

function modalityOf(m) {
  const mod = (m && m.modality) || 'text';
  return MODALITY[mod] ? mod : 'text';
}
// Vision = a text model that ships an mmproj projector artifact.
function isVision(m) {
  return modalityOf(m) === 'text' && !!(m && m.artifacts && m.artifacts.mmproj);
}
function isLoaded(name) { return !!RUNNING[name]; }

// Models eligible for each header selector. Only loaded models are listed so a
// selected route is always immediately usable.
// All registered models eligible for a role (loaded or not). The header is the
// primary picker now, so it must show everything; loaded state is shown per
// option and selecting an unloaded one offers to load it.
function modelsForRole(role) {
  return Object.keys(MODELS).filter(n => {
    const m = MODELS[n]; if (!m || !m.file_ok) return false;
    if (role === 'chat') return modalityOf(m) === 'text';
    return modalityOf(m) === role; // asr | tts
  });
}

// ── Theme ────────────────────────────────────────────────────────────────────
function applyTheme(t){ document.documentElement.setAttribute('data-theme', t); }
function toggleTheme(){
  const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
  applyTheme(next);
  try { localStorage.setItem('zallama-theme', next); } catch {}
}
(function initTheme(){
  let saved;
  try { saved = localStorage.getItem('zallama-theme'); } catch {}
  if (!saved) saved = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  applyTheme(saved);
})();

// ── Utilities ────────────────────────────────────────────────────────────────
async function apiFetch(path, opts = {}) {
  return fetch(API + path, opts);
}

function toast(msg, type = 'ok') {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className = type;
  el.style.display = 'block';
  clearTimeout(el._t);
  el._t = setTimeout(() => el.style.display = 'none', 3000);
}

// ── Model management drawer ──────────────────────────────────────────────────
function openDrawer() { document.body.classList.add('drawer-open'); refreshModels(); }
function closeDrawer() { document.body.classList.remove('drawer-open'); }

// ── Models ───────────────────────────────────────────────────────────────────
async function refreshModels() {
  try {
    const r = await apiFetch('/api/models');
    const data = await r.json();
    renderModels(data.models || []);
  } catch (e) {
    document.getElementById('model-list').innerHTML =
      '<div style="padding:20px;text-align:center;color:var(--error);font-size:12px">⚠ Cannot reach Zallama daemon</div>';
  }
}

function renderModels(models) {
  const el = document.getElementById('model-list');
  if (!models.length) {
    el.innerHTML = '<div style="padding:24px;text-align:center;color:var(--text-muted);font-size:12.5px">No models registered yet</div>';
    return;
  }
  // Cache full model objects so the header selectors and composer can look up a
  // model's modality / capabilities by name.
  MODELS = {};
  models.forEach(m => { MODELS[m.name] = m; });

  // Compact management rows: modality badge + name + load state + actions.
  // Sorted by modality then name so similar models group together.
  const order = { text: 0, asr: 1, tts: 2, embedding: 3, rerank: 4, image: 5 };
  const sorted = models.slice().sort((a, b) =>
    (order[modalityOf(a)] - order[modalityOf(b)]) || a.name.localeCompare(b.name));

  el.innerHTML = sorted.map(m => {
    const mod = modalityOf(m);
    const meta = MODALITY[mod] || MODALITY.text;
    const isTool = mod === 'embedding' || mod === 'rerank';
    const vram = m.running && RUNNING[m.name]?.vram_gb ? RUNNING[m.name].vram_gb + ' GB' : formatBytes(m.file_size);
    const safe = CSS.escape(m.name);
    // Primary action is the load toggle (always visible). Secondary actions —
    // Open (tools) and Remove — live in the ⋯ overflow menu to keep rows clean.
    return `
    <div class="model-row${m.running ? ' running' : ''}" id="card-${safe}">
      <div class="row-main">
        <div class="row-name">${m.name}${isVision(m) ? ' <span class="mod-badge vision">👁</span>' : ''}</div>
        <div class="row-sub"><span class="mod-badge ${mod}">${meta.icon} ${meta.label}</span><span class="row-size">${m.file_ok ? vram : '⚠ missing'}</span></div>
      </div>
      <div class="row-actions">
        <button class="load-toggle${m.running ? ' on' : ''}" role="switch" aria-checked="${m.running}"
          title="${m.file_ok ? (m.running ? 'Loaded — click to unload' : 'Click to load') : 'File missing'}"
          ${m.file_ok ? '' : 'disabled'}
          onclick="toggleLoad('${m.name}', ${m.running})"><span class="knob"></span></button>
        <div class="row-menu">
          <button class="icon-btn sm" title="More" aria-label="More actions" onclick="toggleRowMenu(event, '${safe}')">
            <svg viewBox="0 0 24 24" width="16" height="16" stroke-width="2" fill="none" stroke="currentColor"><circle cx="12" cy="5" r="1.4"/><circle cx="12" cy="12" r="1.4"/><circle cx="12" cy="19" r="1.4"/></svg>
          </button>
          <div class="row-menu-pop" id="menu-${safe}">
            ${isTool ? `<button ${m.running ? '' : 'disabled'} onclick="openTool('${m.name}','${mod}');closeRowMenus()">🔧 Open ${meta.label}</button>` : ''}
            <button class="danger" onclick="removeModel('${m.name}');closeRowMenus()">🗑 Remove from registry</button>
          </div>
        </div>
      </div>
    </div>
  `;
  }).join('');
}

// Toggle a model's loaded state from the row switch.
function toggleLoad(name, running) { running ? unloadModel(name) : loadModel(name); }

// Row overflow (⋯) menus: open one at a time, close on outside click.
function toggleRowMenu(ev, safe) {
  ev.stopPropagation();
  const pop = document.getElementById('menu-' + safe);
  const open = pop.classList.contains('open');
  closeRowMenus();
  if (!open) pop.classList.add('open');
}
function closeRowMenus() {
  document.querySelectorAll('.row-menu-pop.open').forEach(p => p.classList.remove('open'));
}
document.addEventListener('click', closeRowMenus);

async function loadModel(name) {
  // Optimistic feedback: dim the row's toggle while the load is in flight. The
  // real state is reconciled by refreshModels()/refreshRunning() below.
  const card = document.getElementById('card-' + CSS.escape(name));
  const toggle = card && card.querySelector('.load-toggle');
  if (toggle) toggle.classList.add('busy');
  toast(`⏳ Loading ${name}…`);
  const r = await apiFetch(`/api/models/${name}/load`, { method: 'POST' });
  const d = await r.json();
  if (r.ok) { toast(`✓ ${name} running on port ${d.port}`, 'ok'); }
  else { toast(`✗ ${d.detail}`, 'err'); }
  refreshModels(); refreshRunning();
}

async function unloadModel(name) {
  const r = await apiFetch(`/api/models/${name}/unload`, { method: 'POST' });
  if (r.ok) { toast(`Stopped ${name}`); }
  else { const d = await r.json(); toast(d.detail, 'err'); }
  // Drop any route that pointed at the now-stopped model.
  Object.keys(routes).forEach(role => { if (routes[role] === name) routes[role] = ''; });
  refreshModels(); refreshRunning();
}

async function removeModel(name) {
  if (!confirm(`Remove ${name} from registry?`)) return;
  const r = await apiFetch(`/api/models/${name}`, { method: 'DELETE' });
  if (r.ok) { toast(`Removed ${name}`); }
  else { const d = await r.json(); toast(d.detail, 'err'); }
  refreshModels(); refreshRunning();
}

// ── Running models, routes, VRAM meter ───────────────────────────────────────
async function refreshRunning() {
  try {
    const r = await apiFetch('/api/ps');
    const data = await r.json();
    RUNNING = {};
    (data.processes || []).forEach(p => { RUNNING[p.name] = p; });
    populateSelectors();
    renderVram(data.memory || {}, data.processes || []);
    syncComposer();
  } catch (e) { /* daemon unreachable — health dot handles the signal */ }
}

// Fill each header selector with loaded models of its role, preserving the
// current choice and auto-selecting the first available when none is set.
function populateSelectors() {
  const placeholder = { chat: 'Select chat model', asr: 'Select ASR model', tts: 'Select TTS model' };
  ['chat', 'asr', 'tts'].forEach(role => {
    const sel = document.getElementById('route-' + role);
    const avail = modelsForRole(role);
    // Drop a stale route if its model vanished from the registry.
    if (routes[role] && !avail.includes(routes[role])) routes[role] = '';
    // On first population (page load), auto-select a loaded model so running
    // models are immediately active. We DON'T re-auto-select on later polls,
    // otherwise clearing a route would silently snap back every refresh.
    if (!routesInitialized && !routes[role]) {
      const loaded = avail.filter(isLoaded);
      if (loaded.length) routes[role] = loaded[0];
    }
    sel.innerHTML = `<option value="">${avail.length ? placeholder[role] : 'No ' + role + ' model'}</option>` +
      avail.map(n => {
        const dot = isLoaded(n) ? '● ' : '○ ';
        return `<option value="${n}"${n === routes[role] ? ' selected' : ''}>${dot}${n}</option>`;
      }).join('');
    sel.classList.toggle('empty', !routes[role]);
  });
  routesInitialized = true;
}

// A route is only *usable* once its model is loaded. Selecting an unloaded model
// offers to load it; the route activates when the load succeeds.
async function onRouteChange(role) {
  const sel = document.getElementById('route-' + role);
  const name = sel.value;
  if (!name) { routes[role] = ''; syncComposer(); return; }
  if (!isLoaded(name)) {
    if (!confirm(`“${name}” isn't loaded yet. Load it now?`)) {
      sel.value = routes[role] || '';   // revert the dropdown
      return;
    }
    routes[role] = name;                 // optimistic; activates after load
    await loadModel(name);               // refreshes running state + selectors
    return;
  }
  routes[role] = name;
  syncComposer();
}

function renderVram(mem, procs) {
  // Prefer measured GPU VRAM; fall back to the estimated loaded footprint
  // (vram_used_gb is null on CPU-only backends).
  const used = (mem.vram_used_gb != null ? mem.vram_used_gb : mem.loaded_gb) ?? 0;
  const budget = mem.budget_gb ?? 0;
  const count = mem.loaded_count ?? procs.length;
  const fill = document.getElementById('vram-fill');
  const text = document.getElementById('vram-text');
  const label = mem.vram_used_gb != null ? 'VRAM' : 'RAM';
  if (budget > 0) {
    const pct = Math.min(100, (used / budget) * 100);
    fill.style.width = pct + '%';
    fill.className = 'vram-fill' + (pct > 90 ? ' full' : pct > 75 ? ' warn' : '');
    text.textContent = `${used.toFixed(1)}/${budget.toFixed(0)} GB · ${count}`;
    document.getElementById('vram-meter').title =
      `${label} ${used.toFixed(1)} GB of ${budget.toFixed(0)} GB budget · ${count} model(s) loaded`;
  } else {
    fill.style.width = used > 0 ? '45%' : '0';
    fill.className = 'vram-fill';
    text.textContent = used > 0 ? `${used.toFixed(1)} GB · ${count}` : `${count} loaded`;
    document.getElementById('vram-meter').title =
      `${label} ${used.toFixed(1)} GB · ${count} model(s) loaded (no budget configured)`;
  }
}

// Enable/disable composer controls based on which roles have a loaded model.
function syncComposer() {
  const input = document.getElementById('msg-input');
  const send = document.getElementById('send-btn');
  const attach = document.getElementById('attach-btn');
  const upload = document.getElementById('upload-audio-btn');
  const record = document.getElementById('record-btn');

  const hasChat = !!routes.chat;
  const hasVision = chatHasVision();
  const hasAsr = !!routes.asr;

  input.disabled = !hasChat;
  send.disabled = !hasChat;
  input.placeholder = hasChat
    ? 'Type a message… (Ctrl+Enter to send)'
    : 'Select a chat model to start chatting';

  setControl(attach, hasVision, 'Attach image', `Selected chat model has no vision support`);
  setControl(upload, hasAsr, 'Upload audio to transcribe', `No speech-to-text model loaded`);
  setControl(record, hasAsr, 'Record & transcribe', `No speech-to-text model loaded`);

  if (pendingImage && !hasVision) { pendingImage = null; renderImagePreview(); }
}

function setControl(btn, enabled, okTitle, offTitle) {
  if (!btn) return;
  btn.disabled = !enabled;
  btn.classList.toggle('disabled', !enabled);
  btn.title = enabled ? okTitle : offTitle;
}

function welcomeHTML() {
  return `
    <div id="welcome">
      <div class="welcome-logo-wrap">
        <span class="welcome-halo-ring"></span>
        <span class="welcome-halo-ring h2"></span>
        <span class="welcome-halo-ring h3"></span>
        <div class="welcome-logo">🦙</div>
      </div>
      <h2 class="welcome-title">Welcome to Zallama</h2>
      <p class="welcome-sub">${routes.chat ? 'Type a message below to start chatting.' : 'Select a chat model above to start chatting.'}</p>
      <div class="welcome-tip">OpenAI-compatible · Streaming · Multi-model</div>
    </div>`;
}

// Start a fresh conversation: clear the transcript & attachment, keep the
// selected models (routes) and return to the welcome screen.
function newChat() {
  chatHistory = [];
  pendingImage = null;
  replyTexts = [];
  renderImagePreview();
  document.getElementById('chat-area').innerHTML = welcomeHTML();
  const input = document.getElementById('msg-input');
  input.value = ''; input.style.height = 'auto';
  if (routes.chat) input.focus();
}

// /clear command: reset history but stay in the conversation view.
function clearChat() {
  chatHistory = [];
  pendingImage = null;
  replyTexts = [];
  renderImagePreview();
  document.getElementById('chat-area').innerHTML = '';
  appendSystem('History cleared.');
}

// ── Send (text chat, optionally with a vision attachment) ────────────────────
function sendMessage() {
  if (isStreaming) return;
  return sendChat();
}

async function sendChat() {
  if (!routes.chat) { toast('Load & select a chat model first', 'err'); return; }
  const input = document.getElementById('msg-input');
  const text = input.value.trim();
  if (!text && !pendingImage) return;
  if (text === '/clear') { clearChat(); input.value = ''; return; }

  // Vision: send a multimodal content array (text + image_url) when an image is
  // attached; otherwise a plain string. The backend forwards image_url blocks to
  // llama-server, which handles them via the model's mmproj projector.
  // A turn with an image goes to the vision model; plain text to the chat model.
  let userContent = text;
  let displayHtml = escapeHtml(text);
  // The chat model handles both text and images (vision is its own capability).
  const targetModel = routes.chat;
  if (pendingImage) {
    userContent = [
      ...(text ? [{ type: 'text', text }] : []),
      { type: 'image_url', image_url: { url: pendingImage.dataUrl } },
    ];
    displayHtml = `<img src="${pendingImage.dataUrl}" class="bubble-img" alt="attachment"/>` + (text ? '<br>' + escapeHtml(text) : '');
  }

  input.value = '';
  input.style.height = 'auto';
  chatHistory.push({ role: 'user', content: userContent });
  appendMsgHtml('user', displayHtml);
  pendingImage = null;
  renderImagePreview();

  isStreaming = true;
  document.getElementById('send-btn').disabled = true;

  const aId = appendMsg('assistant', '');
  const bubble = document.getElementById('bubble-' + aId);
  bubble.innerHTML = '<span class="cursor"></span>';

  try {
    const resp = await fetch(API + '/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: targetModel, messages: chatHistory, stream: true })
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let full = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, { stream: true });
      for (const line of chunk.split('\n')) {
        const l = line.replace(/^data: /, '').trim();
        if (!l || l === '[DONE]') continue;
        try {
          const j = JSON.parse(l);
          const delta = j.choices?.[0]?.delta?.content || '';
          if (delta) { full += delta; bubble.innerHTML = renderMarkdown(full) + '<span class="cursor"></span>'; }
        } catch {}
      }
      scrollBottom();
    }
    bubble.innerHTML = renderMarkdown(full) + replyActions(full);
    chatHistory.push({ role: 'assistant', content: full });
  } catch (e) {
    bubble.innerHTML = `<span style="color:var(--error)">Error: ${e.message}</span>`;
  }

  isStreaming = false;
  document.getElementById('send-btn').disabled = false;
  scrollBottom();
}

// Action chrome under a finished assistant reply. A 🔊 speak button appears only
// when a TTS model is loaded & routed; it synthesizes that reply on demand.
// Reply text is stashed in a side table and referenced by id, NOT inlined into
// the onclick string — inlining broke on any reply containing a quote (e.g.
// "I'm"), since encodeURIComponent doesn't escape single quotes and the stray '
// terminated the JS string early ("missing ) after argument list").
let replyTexts = [];
function replyActions(text) {
  if (!routes.tts) return '';
  const id = replyTexts.push(text) - 1;
  return `<div class="reply-actions"><button class="reply-btn" title="Speak this reply" onclick="speakText(${id}, this)">`
    + `<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"/></svg> Speak</button></div>`;
}

// Kokoro reads the input literally, so markdown punctuation, code blocks and
// blank lines come out as noise or odd pauses. Flatten the reply to clean,
// speakable prose before synthesis: drop code, unwrap emphasis/links/headings,
// strip list/table/quote markers, decode entities, normalize whitespace.
function cleanForTTS(md) {
  let t = md;
  t = t.replace(/\r\n?/g, '\n');
  // Fenced & indented code blocks — unspeakable, remove entirely.
  t = t.replace(/```[\s\S]*?```/g, ' ');
  t = t.replace(/~~~[\s\S]*?~~~/g, ' ');
  // Images: drop (alt text is rarely worth reading). Links: keep the label only.
  t = t.replace(/!\[[^\]]*\]\([^)]*\)/g, ' ');
  t = t.replace(/\[([^\]]+)\]\([^)]*\)/g, '$1');
  // Inline code → its contents, sans backticks.
  t = t.replace(/`([^`]+)`/g, '$1');
  // Bold / italic / strikethrough markers.
  t = t.replace(/(\*\*|__)(.*?)\1/g, '$2');
  t = t.replace(/(\*|_)(.*?)\1/g, '$2');
  t = t.replace(/~~(.*?)~~/g, '$1');
  // Headings (#), blockquotes (>), list bullets (-, *, +) and numbered markers.
  t = t.replace(/^\s{0,3}#{1,6}\s*/gm, '');
  t = t.replace(/^\s*>+\s?/gm, '');
  t = t.replace(/^\s*[-*+]\s+/gm, '');
  t = t.replace(/^\s*\d+[.)]\s+/gm, '');
  // Markdown tables → drop separator rows, turn cell pipes into pauses.
  // Strip the leading/trailing edge pipes per line first so we don't produce
  // a comma before the first cell or after the last.
  t = t.replace(/^\s*\|?[\s:|-]+\|?\s*$/gm, '');
  t = t.replace(/^[ \t]*\|/gm, '').replace(/\|[ \t]*$/gm, '');
  t = t.replace(/\s*\|\s*/g, ', ');
  // Horizontal rules.
  t = t.replace(/^\s*([-*_])\1{2,}\s*$/gm, '');
  // HTML tags & common entities.
  t = t.replace(/<[^>]+>/g, ' ');
  t = t.replace(/&nbsp;/g, ' ').replace(/&amp;/g, '&').replace(/&lt;/g, '<')
       .replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&#39;/g, "'");
  // Whitespace: blank lines → sentence breaks, single newlines → spaces,
  // collapse runs of spaces, tidy spaces left before punctuation.
  t = t.replace(/\n{2,}/g, '. ');
  t = t.replace(/\n+/g, ' ');
  t = t.replace(/[ \t]{2,}/g, ' ');
  t = t.replace(/\s+([,.!?;:])/g, '$1');
  t = t.replace(/,\s*\./g, '.');        // "cell," + line-break "." → single .
  t = t.replace(/([.!?])\s*,\s*/g, '$1 ');
  t = t.replace(/\.\s*\.\s*/g, '. ');   // avoid ".." from a line already ending in .
  t = t.replace(/(^|[.!?]\s*),\s*/g, '$1'); // drop a comma that starts a sentence
  return t.trim();
}

async function speakText(id, btn) {
  if (!routes.tts) { toast('No TTS model loaded', 'err'); return; }
  const raw = replyTexts[id];
  if (raw == null) return;
  const text = cleanForTTS(raw);
  if (!text) { toast('Nothing to speak', 'err'); return; }
  btn.disabled = true; btn.classList.add('busy');
  try {
    const r = await fetch(API + '/v1/audio/speech', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: routes.tts, input: text })
    });
    if (!r.ok) { throw new Error(await r.text()); }
    const url = URL.createObjectURL(await r.blob());
    const audio = new Audio(url); audio.play();
    audio.onended = () => URL.revokeObjectURL(url);
  } catch (e) {
    toast('Speak failed: ' + e.message, 'err');
  }
  btn.disabled = false; btn.classList.remove('busy');
}

let msgCount = 0;
function appendMsg(role, text) {
  const id = ++msgCount;
  const area = document.getElementById('chat-area');
  const div = document.createElement('div');
  div.className = 'message-row ' + role;
  div.innerHTML = `<div class="message-bubble ${role} ${role === 'assistant' ? 'prose' : ''}" id="bubble-${id}">${role === 'assistant' ? renderMarkdown(text) : escapeHtml(text)}</div>`;
  area.appendChild(div);
  scrollBottom();
  return id;
}

// Append a message whose body is already trusted HTML (e.g. an image, an audio
// player, or a formatted result). Used by the non-streaming modality handlers.
function appendMsgHtml(role, html) {
  const id = ++msgCount;
  const area = document.getElementById('chat-area');
  const div = document.createElement('div');
  div.className = 'message-row ' + role;
  div.innerHTML = `<div class="message-bubble ${role} ${role === 'assistant' ? 'prose' : ''}" id="bubble-${id}">${html}</div>`;
  area.appendChild(div);
  scrollBottom();
  return id;
}

// ── ASR: audio → transcript inserted into the chat input ─────────────────────
function pickAudio() { if (routes.asr) document.getElementById('audio-file-input').click(); }

async function transcribeFile(file) {
  if (!file) return;
  if (!routes.asr) { toast('No speech-to-text model loaded', 'err'); return; }
  const input = document.getElementById('msg-input');
  toast('🎙 Transcribing…');
  try {
    const fd = new FormData();
    fd.append('model', routes.asr);
    fd.append('file', file, file.name);
    const r = await fetch(API + '/v1/audio/transcriptions', { method: 'POST', body: fd });
    const ct = r.headers.get('content-type') || '';
    let text;
    if (ct.includes('application/json')) { const j = await r.json(); text = j.text ?? JSON.stringify(j); }
    else text = await r.text();
    if (!r.ok) throw new Error(text);
    // Insert the transcript into the composer so the user can edit then send.
    const t = (text || '').trim();
    input.value = input.value ? (input.value.replace(/\s*$/, '') + ' ' + t) : t;
    input.dispatchEvent(new Event('input'));
    input.focus();
    toast('✓ Transcribed', 'ok');
  } catch (e) {
    toast('Transcription failed: ' + e.message, 'err');
  }
}

let mediaRecorder = null, recChunks = [];
async function toggleRecord() {
  if (!routes.asr) { toast('No speech-to-text model loaded', 'err'); return; }
  const btn = document.getElementById('record-btn');
  if (mediaRecorder && mediaRecorder.state === 'recording') { mediaRecorder.stop(); return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    recChunks = [];
    mediaRecorder.ondataavailable = e => { if (e.data.size) recChunks.push(e.data); };
    mediaRecorder.onstop = () => {
      stream.getTracks().forEach(t => t.stop());
      btn.classList.remove('recording');
      const blob = new Blob(recChunks, { type: recChunks[0]?.type || 'audio/webm' });
      const ext = (blob.type.split('/')[1] || 'webm').split(';')[0];
      transcribeFile(new File([blob], `recording.${ext}`, { type: blob.type }));
    };
    mediaRecorder.start();
    btn.classList.add('recording');
    toast('🎙 Recording… click again to stop');
  } catch (e) {
    toast('Microphone unavailable: ' + e.message, 'err');
  }
}

// ── Embedding / Rerank tool panels (opened from a model card) ────────────────
function openTool(model, mod) {
  const panel = document.getElementById('tool-modal-panel');
  if (mod === 'embedding') {
    panel.innerHTML = `
      <h3>🧬 Embed — <span style="font-weight:500;color:var(--text-muted)">${model}</span></h3>
      <div class="form-group"><label>Text to embed</label>
        <textarea id="tool-embed-text" placeholder="Type or paste text…"></textarea></div>
      <div id="tool-embed-out"></div>
      <div class="modal-actions">
        <button class="btn large" onclick="closeTool()">Close</button>
        <button class="btn large primary" onclick="runEmbed('${model}')">Embed</button>
      </div>`;
  } else {
    panel.innerHTML = `
      <h3>🏷 Rerank — <span style="font-weight:500;color:var(--text-muted)">${model}</span></h3>
      <div class="form-group"><label>Query</label>
        <input type="text" id="tool-rr-query" placeholder="What are you looking for?"/></div>
      <div class="form-group"><label>Documents (one per line)</label>
        <textarea id="tool-rr-docs" style="min-height:90px" placeholder="First document&#10;Second document&#10;…"></textarea></div>
      <div id="tool-rr-out"></div>
      <div class="modal-actions">
        <button class="btn large" onclick="closeTool()">Close</button>
        <button class="btn large primary" onclick="runRerank('${model}')">Rerank</button>
      </div>`;
  }
  document.getElementById('tool-modal').classList.add('open');
}
function closeTool() { document.getElementById('tool-modal').classList.remove('open'); }

async function runEmbed(model) {
  const text = document.getElementById('tool-embed-text').value.trim();
  const out = document.getElementById('tool-embed-out');
  if (!text) { toast('Enter some text', 'err'); return; }
  out.innerHTML = '<div class="tool-loading">Embedding…</div>';
  try {
    const r = await fetch(API + '/v1/embeddings', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, input: text })
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || JSON.stringify(j));
    const vec = j.data?.[0]?.embedding || [];
    const preview = vec.slice(0, 16).map(x => x.toFixed(4)).join(', ');
    out.innerHTML = `<div class="result-head"><span class="dims-chip">${vec.length} dims</span></div>`
      + `<pre><code>[${escapeHtml(preview)}${vec.length > 16 ? ', …' : ''}]</code></pre>`;
  } catch (e) {
    out.innerHTML = `<div style="color:var(--error)">Embedding failed: ${escapeHtml(e.message)}</div>`;
  }
}

async function runRerank(model) {
  const query = document.getElementById('tool-rr-query').value.trim();
  const docs = document.getElementById('tool-rr-docs').value.split('\n').map(d => d.trim()).filter(Boolean);
  const out = document.getElementById('tool-rr-out');
  if (!query) { toast('Enter a query', 'err'); return; }
  if (docs.length < 2) { toast('Add at least 2 documents', 'err'); return; }
  out.innerHTML = '<div class="tool-loading">Ranking…</div>';
  try {
    const r = await fetch(API + '/v1/rerank', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, query, documents: docs })
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || JSON.stringify(j));
    const ranked = (j.results || []).slice().sort((a, b) => b.relevance_score - a.relevance_score);
    out.innerHTML = '<div class="rerank-list">' + ranked.map((res, i) => {
      const doc = docs[res.index] ?? '';
      const score = (res.relevance_score ?? 0).toFixed(3);
      return `<div class="rerank-row"><span class="rank">#${i + 1}</span><span class="rerank-doc">${escapeHtml(doc)}</span><span class="rerank-score">${score}</span></div>`;
    }).join('') + '</div>';
  } catch (e) {
    out.innerHTML = `<div style="color:var(--error)">Rerank failed: ${escapeHtml(e.message)}</div>`;
  }
}

// ── Vision image attachment ──────────────────────────────────────────────────
function pickImage() {
  if (!chatHasVision()) { toast('Selected chat model has no vision support', 'err'); return; }
  document.getElementById('image-file-input').click();
}
function attachImage(file) {
  if (!file) return;
  if (!chatHasVision()) { toast('Selected chat model has no vision support', 'err'); return; }
  const reader = new FileReader();
  reader.onload = () => { pendingImage = { dataUrl: reader.result, name: file.name }; renderImagePreview(); };
  reader.readAsDataURL(file);
}
function clearImage() { pendingImage = null; renderImagePreview(); }
function renderImagePreview() {
  const el = document.getElementById('image-preview');
  if (!pendingImage) { el.innerHTML = ''; el.style.display = 'none'; return; }
  el.style.display = 'flex';
  el.innerHTML = `<img src="${pendingImage.dataUrl}" alt="preview"/><span>${escapeHtml(pendingImage.name)}</span><button class="img-remove" onclick="clearImage()" title="Remove">✕</button>`;
}

function appendSystem(msg) {
  const area = document.getElementById('chat-area');
  const d = document.createElement('div');
  d.className = 'message-system';
  d.textContent = msg;
  area.appendChild(d);
  scrollBottom();
}

function scrollBottom() {
  const a = document.getElementById('chat-area');
  a.scrollTop = a.scrollHeight;
}

function escapeHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\n/g,'<br>');
}

function renderMarkdown(s) {
  // Lightweight markdown: fenced code, inline code, bold, then escape the rest.
  const codeBlocks = [];
  let t = s.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) => {
    codeBlocks.push(`<pre><code>${escapeHtml(code)}</code></pre>`);
    return ` ${codeBlocks.length - 1} `;
  });
  const inlineCodes = [];
  t = t.replace(/`([^`\n]+)`/g, (_, code) => {
    inlineCodes.push(`<code>${escapeHtml(code)}</code>`);
    return `${inlineCodes.length - 1}`;
  });
  t = escapeHtml(t);
  t = t.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  t = t.replace(/ (\d+) /g, (_, i) => codeBlocks[+i]);
  t = t.replace(/(\d+)/g, (_, i) => inlineCodes[+i]);
  return t;
}

// ── Add Modal ────────────────────────────────────────────────────────────────
function openAddModal() { document.getElementById('add-modal').classList.add('open'); }
function closeAddModal() { document.getElementById('add-modal').classList.remove('open'); }

async function addModel() {
  const name = document.getElementById('m-name').value.trim();
  const file = document.getElementById('m-file').value.trim();
  const desc = document.getElementById('m-desc').value.trim();
  if (!name || !file) { toast('Name and file are required', 'err'); return; }
  const r = await apiFetch('/api/models/add', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, file, description: desc })
  });
  const d = await r.json();
  if (r.ok) { toast(`✓ Model '${name}' added`); closeAddModal(); refreshModels(); }
  else { toast(`✗ ${d.detail}`, 'err'); }
}

// ── Pull Modal ───────────────────────────────────────────────────────────────
function openPullModal() {
  document.getElementById('pull-modal').classList.add('open');
  document.getElementById('p-presets').value = '';
  document.getElementById('p-name').value = '';
  document.getElementById('p-custom-group').style.display = 'none';
  document.getElementById('hf-search-input').value = '';
  document.getElementById('hf-search-results').style.display = 'none';
  document.getElementById('hf-search-results').innerHTML = '';
}

function closePullModal() {
  document.getElementById('pull-modal').classList.remove('open');
}

let lastRepoSearchQuery = '';

async function searchHF() {
  const query = document.getElementById('hf-search-input').value.trim();
  if (!query) return;
  const resEl = document.getElementById('hf-search-results');
  resEl.style.display = 'flex';
  resEl.innerHTML = '<div style="font-size:11px;color:var(--text-muted);text-align:center;padding:10px">Searching HuggingFace...</div>';

  try {
    const r = await apiFetch(`/api/models/search?q=${encodeURIComponent(query)}`);
    if (!r.ok) {
      resEl.innerHTML = `<div style="color:var(--error);font-size:11px">Failed to search: ${r.statusText}</div>`;
      return;
    }
    const data = await r.json();
    if (data.type === 'repo_list') {
      const results = data.results || [];
      if (!results.length) {
        resEl.innerHTML = '<div style="font-size:11px;color:var(--text-muted);text-align:center;padding:10px">No GGUF models found.</div>';
        return;
      }
      resEl.innerHTML = results.map(item => {
        let dls = item.downloads;
        let dls_str = dls > 1000000 ? (dls/1000000).toFixed(1)+'M' : (dls > 1000 ? (dls/1000).toFixed(0)+'k' : dls);
        return `
          <div class="search-item" onclick="selectHFRepo('${item.id}')" style="cursor:pointer;padding:6px 8px;border-radius:6px;transition:background .2s;display:flex;justify-content:space-between;align-items:center;font-size:11px">
            <span style="font-weight:600;color:var(--brand);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;margin-right:8px">${item.id}</span>
            <span style="color:var(--text-muted);flex-shrink:0">${dls_str} dls</span>
          </div>
        `;
      }).join('');
    } else if (data.type === 'file_list') {
      const files = data.files || [];
      const repo = data.repo;
      if (!files.length) {
        resEl.innerHTML = `<div style="font-size:11px;color:var(--text-muted);padding:10px">No .gguf files in this repo. <a href="#" onclick="goBackToRepoSearch()" style="color:var(--brand)">Back</a></div>`;
        return;
      }
      resEl.innerHTML = `
        <div style="font-size:11px;font-weight:600;margin-bottom:6px;display:flex;justify-content:space-between;align-items:center;padding:0 4px">
          <span style="color:var(--text-main);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;margin-right:8px">Files in ${repo}:</span>
          <a href="#" onclick="goBackToRepoSearch()" style="color:var(--brand);text-decoration:none;font-weight:500;flex-shrink:0">← Back</a>
        </div>
        <div style="display:flex;flex-direction:column;gap:4px">
          ${files.map(f => `
            <div class="search-item" onclick="selectHFFile('${repo}', '${f}')" style="cursor:pointer;padding:6px 8px;border-radius:6px;font-size:11px;word-break:break-all;color:var(--text-muted)">
              📄 ${f}
            </div>
          `).join('')}
        </div>
      `;
    }
  } catch (e) {
    resEl.innerHTML = `<div style="color:var(--error);font-size:11px;padding:10px">Error: ${e}</div>`;
  }
}

function goBackToRepoSearch() {
  document.getElementById('hf-search-input').value = lastRepoSearchQuery;
  searchHF();
}

async function selectHFRepo(repoId) {
  lastRepoSearchQuery = document.getElementById('hf-search-input').value.trim();
  document.getElementById('hf-search-input').value = repoId;
  searchHF();
}

function selectHFFile(repo, file) {
  document.getElementById('p-presets').value = 'custom';
  document.getElementById('p-custom-group').style.display = 'block';
  document.getElementById('p-name').value = `${repo}/${file}`;
  toast(`Selected ${file}`);
}

function applyPreset() {
  const sel = document.getElementById('p-presets');
  const val = sel.value;
  const customGroup = document.getElementById('p-custom-group');
  if (val === 'custom') {
    customGroup.style.display = 'block';
    document.getElementById('p-name').value = '';
  } else if (val) {
    customGroup.style.display = 'none';
    document.getElementById('p-name').value = val;
  } else {
    customGroup.style.display = 'none';
    document.getElementById('p-name').value = '';
  }
}

async function pullModel() {
  const model = document.getElementById('p-name').value.trim();
  if (!model) { toast('Please select or specify a model', 'err'); return; }
  toast(`⏳ Initiating pull for ${model}…`);
  const r = await apiFetch('/api/models/pull', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model })
  });
  const d = await r.json();
  if (r.ok) {
    toast(`✓ Pull started for ${model}`, 'ok');
    closePullModal();
    refreshDownloads();
  } else {
    toast(`✗ ${d.detail}`, 'err');
  }
}

let activeDownloadsCount = 0;

async function refreshDownloads() {
  try {
    const r = await apiFetch('/api/models/pull/status');
    if (!r.ok) return;
    const data = await r.json();
    const list = data.downloads || [];

    const active = list.filter(t => t.status === 'queued' || t.status === 'downloading');
    activeDownloadsCount = active.length;

    const el = document.getElementById('download-list');
    if (!list.length) {
      el.innerHTML = '';
      return;
    }

    el.innerHTML = list.map(t => {
      let speedStr = t.speed > 1024*1024
        ? (t.speed / 1024 / 1024).toFixed(1) + ' MB/s'
        : (t.speed / 1024).toFixed(0) + ' KB/s';
      let progressText = t.status === 'downloading'
        ? `${t.percent.toFixed(1)}% (${speedStr})`
        : t.status;

      let color = 'var(--brand)';
      if (t.status === 'completed') color = 'var(--success)';
      if (t.status === 'failed') color = 'var(--error)';
      if (t.status === 'queued') color = 'var(--warning)';

      return `
        <div style="background:var(--card-bg);border:1px solid var(--panel-border);border-radius:10px;padding:10px;font-size:11px">
          <div style="font-weight:600;margin-bottom:6px;display:flex;justify-content:space-between">
            <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1;margin-right:8px">⬇️ ${t.model}</span>
            <span style="color:${color};flex-shrink:0">${progressText}</span>
          </div>
          <div style="width:100%;height:5px;background:var(--code-bg);border-radius:3px;overflow:hidden">
            <div style="width:${t.percent}%;height:100%;background:${color};transition:width .2s;border-radius:3px"></div>
          </div>
          ${t.error ? `<div style="color:var(--error);font-size:10px;margin-top:4px">${t.error}</div>` : ''}
        </div>
      `;
    }).join('');

    if (list.some(t => t.status === 'completed')) {
      refreshModels();
    }
  } catch (e) {
    console.error(e);
  }
}

// ── Input auto-resize ────────────────────────────────────────────────────────
document.getElementById('msg-input').addEventListener('input', function() {
  this.style.height = 'auto';
  this.style.height = Math.min(this.scrollHeight, 200) + 'px';
});
document.getElementById('msg-input').addEventListener('keydown', e => {
  if (e.key === 'Enter' && e.ctrlKey) { e.preventDefault(); sendMessage(); }
});

// Close the drawer with Escape.
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeDrawer(); });

// ── Helpers ──────────────────────────────────────────────────────────────────
function formatBytes(b) {
  if (!b) return ''; if (b > 1e9) return (b/1e9).toFixed(1)+'GB'; return (b/1e6).toFixed(0)+'MB';
}

// ── Init ─────────────────────────────────────────────────────────────────────
// Load the registry first so MODELS is populated, then the running set so the
// header selectors and composer can resolve modalities.
(async function init() {
  await refreshModels();
  await refreshRunning();
  refreshDownloads();
})();
setInterval(async () => { await refreshModels(); refreshRunning(); }, 10000);
setInterval(refreshRunning, 4000);
setInterval(refreshDownloads, 2000);
