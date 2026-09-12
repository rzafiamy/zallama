// src/main.js — Zallama website interactivity (shared by index.html and docs.html)
import './style.css';

// ---------------------------------------------------------------------------
// 0. Version badge — injected from ../version.txt at build time (vite.config.js)
// ---------------------------------------------------------------------------
if (typeof __ZALLAMA_VERSION__ === 'string' && __ZALLAMA_VERSION__) {
  document.querySelectorAll('[data-version]').forEach((el) => {
    el.textContent = `v${__ZALLAMA_VERSION__}`;
  });
}

// ---------------------------------------------------------------------------
// 1. Scroll reveal
// ---------------------------------------------------------------------------
const revealObserver = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) entry.target.classList.add('active');
  });
}, { threshold: 0.05 });
document.querySelectorAll('.reveal').forEach((el) => revealObserver.observe(el));

// ---------------------------------------------------------------------------
// 2. Mobile navigation drawer
// ---------------------------------------------------------------------------
const mobileNavBtn = document.getElementById('mobile-nav-btn');
const mobileNav = document.getElementById('mobile-nav');
if (mobileNavBtn && mobileNav) {
  mobileNavBtn.addEventListener('click', () => {
    const open = mobileNav.classList.toggle('open');
    mobileNavBtn.setAttribute('aria-expanded', String(open));
  });
  mobileNav.querySelectorAll('a').forEach((a) => {
    a.addEventListener('click', () => {
      mobileNav.classList.remove('open');
      mobileNavBtn.setAttribute('aria-expanded', 'false');
    });
  });
}

// ---------------------------------------------------------------------------
// 3. Copy buttons on every code block
// ---------------------------------------------------------------------------
const COPY_ICON = `<svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"/></svg>`;
const CHECK_ICON = `<svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5"><path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7"/></svg>`;

function copyText(text, btn, label = 'Copy') {
  navigator.clipboard.writeText(text).then(() => {
    btn.classList.add('copied');
    btn.innerHTML = `${CHECK_ICON}<span>Copied</span>`;
    setTimeout(() => {
      btn.classList.remove('copied');
      btn.innerHTML = `${COPY_ICON}<span>${label}</span>`;
    }, 1800);
  }).catch((err) => console.error('copy failed', err));
}

document.querySelectorAll('.code-wrap').forEach((wrap) => {
  const pre = wrap.querySelector('pre');
  if (!pre) return;
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'copy-btn';
  btn.setAttribute('aria-label', 'Copy code');
  btn.innerHTML = `${COPY_ICON}<span>Copy</span>`;
  btn.addEventListener('click', () => {
    // Strip the "$ " prompt marker so the copied text pastes straight into a shell.
    const text = pre.innerText.replace(/^\$ /gm, '');
    copyText(text, btn);
  });
  wrap.appendChild(btn);
});

// ---------------------------------------------------------------------------
// 4. Docs sidebar scroll-spy (docs.html only)
// ---------------------------------------------------------------------------
const docsNav = document.querySelector('.docs-nav');
if (docsNav) {
  const links = Array.from(docsNav.querySelectorAll('a[href^="#"]'));
  const sections = links
    .map((a) => document.getElementById(a.getAttribute('href').slice(1)))
    .filter(Boolean);

  const setActive = (id) => {
    links.forEach((a) => a.classList.toggle('active', a.getAttribute('href') === `#${id}`));
  };

  const spy = new IntersectionObserver((entries) => {
    // Pick the top-most visible section
    const visible = entries
      .filter((e) => e.isIntersecting)
      .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
    if (visible.length) setActive(visible[0].target.id);
  }, { rootMargin: '-15% 0px -70% 0px', threshold: 0 });
  sections.forEach((s) => spy.observe(s));

  if (location.hash) setActive(location.hash.slice(1));
}

// ---------------------------------------------------------------------------
// 5. Terminal replay (index.html only)
// ---------------------------------------------------------------------------
const terminalBody = document.getElementById('terminal-body');
if (terminalBody) {
  const T = {
    serve: [
      { t: '$ zallama serve', k: 'cmd' },
      { t: '09:12:03 [INFO] zallama: 🦙 llama-server binary: ./bin/llama-server' },
      { t: '09:12:03 [INFO] zallama: 🎙️ parakeet-server binary: ./bin/parakeet-server' },
      { t: '09:12:03 [INFO] zallama: 🗣️ kokoro-server binary: ./bin/kokoro-server' },
      { t: '09:12:03 [INFO] zallama: 📒 registry: 6 model(s) registered' },
      { t: '09:12:03 [INFO] zallama: 📌 pre-warming pinned model tdt-0.6b-v3-q8_0 (parakeet-server :8100)', k: 'dim' },
      { t: '09:12:04 [INFO] zallama: 🚀 inference  → http://127.0.0.1:11435  (/v1/*)', k: 'ok' },
      { t: '09:12:04 [INFO] zallama: 🛠️ admin      → http://127.0.0.1:11436  (/api/*, /metrics)', k: 'ok' },
      { t: '09:12:04 [INFO] uvicorn: Uvicorn running (Press CTRL+C to quit)', k: 'dim' },
    ],
    pull: [
      { t: '$ zallama pull llama3.2:3b', k: 'cmd' },
      { t: '[downloader] preset llama3.2:3b → unsloth/Llama-3.2-3B-Instruct-GGUF' },
      { t: '[downloader] aria2c found — 8 parallel connections' },
      { t: '[aria2c] Llama-3.2-3B-Instruct-Q4_K_M.gguf  ████████████████████ 100%  2.02 GiB  61 MiB/s', k: 'ok' },
      { t: '[registry] registered "llama3.2:3b"  modality=text  backend=llama-server' },
      { t: '' },
      { t: '$ zallama pull parakeet:0.6b', k: 'cmd' },
      { t: '[downloader] preset parakeet:0.6b → mudler/parakeet-cpp-gguf' },
      { t: '[aria2c] parakeet-tdt-0.6b-v2-q8_0.gguf  ████████████████████ 100%  0.65 GiB', k: 'ok' },
      { t: '[registry] registered "parakeet:0.6b"  modality=asr  backend=parakeet-server' },
    ],
    set: [
      { t: '$ zallama set llama3.2:3b ctx_size=16384 n_gpu_layers=99 mem_gb=3.1', k: 'cmd' },
      { t: '[registry] llama3.2:3b' },
      { t: '  ctx_size      8192  →  16384' },
      { t: '  n_gpu_layers  99    →  99' },
      { t: '  mem_gb        —     →  3.1' },
      { t: '[registry] saved models/registry.yaml', k: 'ok' },
      { t: '[warn] llama3.2:3b is running with its old params — run `zallama reload llama3.2:3b` to apply.', k: 'warn' },
    ],
    run: [
      { t: '$ zallama run deepseek-r1:8b', k: 'cmd' },
      { t: '[manager] loading deepseek-r1:8b on :8101 … ready in 3.2s (5.5 GB)', k: 'dim' },
      { t: '' },
      { t: '>>> Why does a bigger ctx_size cost VRAM even before I send a long prompt?', k: 'user' },
      { t: '<think>\nThe KV cache is allocated up front for the full context window…\nper-token cost = layers × 2 × kv_heads × head_dim × bytes…\n</think>', k: 'think' },
      { t: 'Because llama.cpp reserves the KV cache for the whole context at load time. Every token slot costs memory across all layers, so the reservation grows linearly with ctx_size whether or not you fill it. Quantize it with cache_type_k=q8_0 to claw some back.', k: 'assist' },
      { t: '' },
      { t: '>>> /bye', k: 'user' },
    ],
    ps: [
      { t: '$ zallama ps', k: 'cmd' },
      { t: 'NAME                      PORT     MEM      UPTIME       LAST USED', k: 'hdr' },
      { t: '─────────────────────────────────────────────────────────────────────────', k: 'dim' },
      { t: 'deepseek-r1:8b            8101     5.5GB    4m12s        8s ago' },
      { t: 'tdt-0.6b-v3-q8_0  📌      8100     1.3GB    12m40s       1m ago' },
      { t: 'nomic-embed:v1.5          8102     0.7GB    2m03s        30s ago' },
      { t: '' },
      { t: 'Memory: 7.5GB / 12.0GB used  •  4.5GB free  •  3 loaded  •  1 pinned', k: 'ok' },
    ],
    calibrate: [
      { t: '$ zallama calibrate qwen3.5-4b-q4_k_m --probe --apply', k: 'cmd' },
      { t: '[calibrate] GPU: NVIDIA GeForce RTX 4090  24.0 GiB  •  margin 2.0 GiB (services group)' },
      { t: '[calibrate] trained context: 262144 — bisecting ctx_size between 4096 and 262144' },
      { t: '  load ctx_size=131072 … free 3.9 GiB   ✓ fits' },
      { t: '  load ctx_size=196608 … free 0.6 GiB   ✗ under margin' },
      { t: '  load ctx_size=163840 … free 2.3 GiB   ✓ fits' },
      { t: '  load ctx_size=180224 … free 1.4 GiB   ✗ under margin' },
      { t: '[calibrate] VRAM slope: 74.1 KiB/token  •  weights + artifacts: 4.6 GiB' },
      { t: '[calibrate] largest ctx_size leaving 2.0 GiB free: 163840  (measured mem_gb=16.4)', k: 'ok' },
      { t: '[registry] wrote ctx_size=163840 mem_gb=16.4 → qwen3.5-4b-q4_k_m', k: 'ok' },
    ],
    bench: [
      { t: '$ zallama bench qwen3.5-4b-q4_k_m --sweep reasoning=on,off --natural', k: 'cmd' },
      { t: '[bench] 2 point(s) × 3 runs (+1 warmup)  •  prompt 512  •  max-tokens 128' },
      { t: '' },
      { t: '┌──────────────┬────────┬─────┬────────┬─────────┬─────────┬─────────────┬─────────────┐', k: 'dim' },
      { t: '│ POINT        │ PROMPT │ GEN │ LOAD s │ VRAM GB │ TTFT ms │ PREFILL t/s │ DECODE t/s  │', k: 'hdr' },
      { t: '├──────────────┼────────┼─────┼────────┼─────────┼─────────┼─────────────┼─────────────┤', k: 'dim' },
      { t: '│ reasoning=on │    512 │ 611 │    1.5 │     4.8 │  118 ±2 │    8925 ±20 │  179.8 ±0.2 │' },
      { t: '│ reasoning=off│    512 │  94 │    1.5 │     4.8 │  116 ±3 │    8901 ±34 │  180.1 ±0.3 │' },
      { t: '└──────────────┴────────┴─────┴────────┴─────────┴─────────┴─────────────┴─────────────┘', k: 'dim' },
      { t: '' },
      { t: '[bench] decode is identical — thinking costs tokens, not speed (611 vs 94 generated)', k: 'ok' },
      { t: '[bench] restored original params for qwen3.5-4b-q4_k_m', k: 'dim' },
    ],
    monitor: [
      { t: '$ zallama monitor --once', k: 'cmd' },
      { t: ' GPU  RTX 4090   util ████░░░░░░  38%   VRAM ██████████████████░░ 22.1/24.0 GiB  61°C  212W' },
      { t: ' CPU              load ██░░░░░░░░  14%   RAM  ██░░░░░░░░░░░░░░░░░░  4.9/62.5 GiB' },
      { t: '' },
      { t: ' MODELS  3 loaded/6  •  declared 22.2/23.8 GB  •  measured 22.1 GiB', k: 'hdr' },
      { t: '   nex:n2.5-mini        busy  21.4 GiB   active 1   idle 2s    up 6s    llama-server' },
      { t: '   granite-embedding…   idle   0.7 GiB   active 0   idle 40s   up 2m    embedding-server' },
      { t: '' },
      { t: ' IN FLIGHT  1', k: 'hdr' },
      { t: '   #7  nex:n2.5-mini  chat/completions   elapsed 1.9s   TTFT 158ms   tokens 337' },
      { t: '' },
      { t: ' RECENT  since 42m: 61 req, 1 err, 48,120 in / 9,312 out tokens', k: 'hdr' },
      { t: '   08:00:01  chat/completions   prompt 13   out 18    TTFT 221ms   decode 183.9/s   ok', k: 'dim' },
      { t: '   07:59:57  chat/completions   prompt 25   out 400   TTFT 158ms   decode 194.8/s   ok', k: 'dim' },
    ],
    apikey: [
      { t: '$ zallama apikey --expires 90d', k: 'cmd' },
      { t: '[apikey] generated a 256-bit key, valid until 2026-12-11' },
      { t: '[apikey] wrote api_key_sha256 + expiry to ~/.zallama/config.yaml (chmod 600)' },
      { t: '' },
      { t: '  zk_9f3c…a71e  ← shown ONCE. Store it in your password manager.', k: 'warn' },
      { t: '' },
      { t: '[apikey] restart the daemon to apply:  systemctl restart zallama', k: 'ok' },
      { t: '' },
      { t: '$ curl http://localhost:11435/v1/models -H "Authorization: Bearer $KEY"', k: 'cmd' },
      { t: '{"object":"list","data":[{"id":"llama3.2:3b","object":"model"},…]}', k: 'dim' },
    ],
  };

  const KIND_CLASS = {
    cmd: 'text-sky-300 font-bold',
    user: 'text-emerald-300 font-bold mt-1',
    think: 'text-slate-500 italic whitespace-pre-line border-l-2 border-slate-700 pl-2 my-1',
    assist: 'text-slate-100 pl-3 border-l-2 border-sky-400 whitespace-pre-wrap',
    hdr: 'text-slate-400 font-bold',
    ok: 'text-emerald-300',
    warn: 'text-amber-300',
    dim: 'text-slate-500',
  };

  let timer = null;
  function replay(name) {
    if (timer) clearTimeout(timer);
    terminalBody.innerHTML = '';
    const steps = T[name] || [];
    let i = 0;
    const tick = () => {
      if (i >= steps.length) return;
      const s = steps[i++];
      const line = document.createElement('div');
      line.className = `whitespace-pre ${KIND_CLASS[s.k] || 'text-slate-200'}`;
      line.textContent = s.t === '' ? ' ' : s.t;
      terminalBody.appendChild(line);
      terminalBody.scrollTop = terminalBody.scrollHeight;
      const delay = s.k === 'cmd' ? 350 : s.k === 'think' || s.k === 'assist' ? 500 : 110;
      timer = setTimeout(tick, delay);
    };
    tick();
  }

  const buttons = Array.from(document.querySelectorAll('.term-cmd'));
  buttons.forEach((btn) => {
    btn.addEventListener('click', () => {
      buttons.forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      replay(btn.dataset.cmd);
    });
  });
  replay('serve');
}

// ---------------------------------------------------------------------------
// 6. Memory-aware eviction simulator (index.html only)
// ---------------------------------------------------------------------------
const memSlider = document.getElementById('mem-budget-slider');
if (memSlider) {
  const SPECS = {
    qwen:     { id: 'qwen',     name: 'qwen3.5-4b-q4_k_m', size: 4.8, backend: 'llama-server',     port: 8103 },
    llama:    { id: 'llama',    name: 'llama3.2:3b',        size: 2.6, backend: 'llama-server',     port: 8104 },
    deepseek: { id: 'deepseek', name: 'deepseek-r1:8b',     size: 5.5, backend: 'llama-server',     port: 8105 },
    embed:    { id: 'embed',    name: 'nomic-embed:v1.5',   size: 0.7, backend: 'embedding-server', port: 8102 },
    parakeet: { id: 'parakeet', name: 'tdt-0.6b-v3-q8_0',   size: 1.3, backend: 'parakeet-server',  port: 8100 },
    kokoro:   { id: 'kokoro',   name: 'kokoro:82m',         size: 0.4, backend: 'kokoro-server',    port: 8101 },
  };

  let budget = parseFloat(memSlider.value);
  let loaded = [];       // ordered LRU → MRU
  let evictions = 0;

  const $ = (id) => document.getElementById(id);
  const display = $('mem-budget-display');
  const bar = $('visualizer-progress-bar');
  const statUsage = $('stat-mem-usage');
  const statCount = $('stat-loaded-count');
  const statFree = $('stat-free-gb');
  const statEvict = $('stat-evictions');
  const cards = $('visualizer-cards-container');
  const logBox = $('eviction-logs-body');

  const used = () => loaded.reduce((s, m) => s + m.size, 0);

  function log(msg, kind = 'info') {
    const line = document.createElement('div');
    const ts = new Date().toLocaleTimeString([], { hour12: false });
    const cls = { evict: 'text-rose-300', load: 'text-emerald-300', warn: 'text-amber-300', info: 'text-slate-400' }[kind];
    line.className = cls;
    line.textContent = `${ts} ${msg}`;
    logBox.appendChild(line);
    logBox.scrollTop = logBox.scrollHeight;
  }

  function render() {
    const total = used();
    const pct = Math.min(100, Math.round((total / budget) * 100));
    statUsage.textContent = `${total.toFixed(1)} / ${budget.toFixed(1)} GB (${pct}%)`;
    statCount.textContent = String(loaded.length);
    statFree.textContent = `${Math.max(0, budget - total).toFixed(1)} GB`;
    statEvict.textContent = String(evictions);
    bar.style.width = `${pct}%`;
    bar.className = 'h-full rounded-full transition-all duration-500 ease-out ' + (
      total > budget ? 'bg-gradient-to-r from-rose-500 to-rose-600' :
      pct > 85 ? 'bg-gradient-to-r from-amber-400 to-rose-400' :
      'bg-gradient-to-r from-sky-500 to-sky-600'
    );

    cards.innerHTML = '';
    if (!loaded.length) {
      cards.innerHTML = `<div class="h-24 flex flex-col items-center justify-center border-2 border-dashed border-sky-100 rounded-2xl text-slate-400 text-xs font-semibold">
        <span class="text-2xl mb-1">🦙</span>No models loaded — click a "Load" button above.</div>`;
      return;
    }
    loaded.forEach((m, i) => {
      const pctOf = Math.round((m.size / budget) * 100);
      const el = document.createElement('div');
      el.className = 'flex items-center justify-between gap-3 p-3 bg-sky-50/60 border border-sky-100 rounded-2xl';
      el.innerHTML = `
        <div class="flex items-center gap-3 min-w-0">
          <div class="w-9 h-9 shrink-0 rounded-xl bg-white border border-sky-200 flex items-center justify-center font-mono text-[10px] font-bold text-sky-600">${m.port}</div>
          <div class="min-w-0">
            <div class="flex items-center gap-2">
              <span class="font-mono font-bold text-slate-800 text-xs truncate">${m.name}</span>
              ${m.pinned ? '<span class="text-[10px]" title="pinned">📌</span>' : ''}
              ${i === 0 && !m.pinned ? '<span class="text-[9px] font-bold bg-white border border-sky-200 text-slate-400 px-1.5 rounded-full">LRU</span>' : ''}
            </div>
            <div class="text-[11px] text-slate-500 font-medium">${m.backend} · <span class="text-sky-600 font-bold">${m.size.toFixed(1)} GB</span> (${pctOf}% of budget)</div>
          </div>
        </div>
        <div class="flex items-center gap-1.5 shrink-0">
          <button data-act="touch" data-id="${m.id}" class="px-2 py-1 rounded-lg text-[10px] font-bold bg-white border border-sky-200 text-slate-500 hover:text-sky-600 hover:border-sky-300 transition" title="Send a request (marks it most-recently-used)">use</button>
          <button data-act="pin" data-id="${m.id}" class="px-2 py-1 rounded-lg text-[10px] font-bold bg-white border border-sky-200 ${m.pinned ? 'text-sky-600 border-sky-300' : 'text-slate-500'} hover:text-sky-600 hover:border-sky-300 transition" title="Toggle pinned: true">${m.pinned ? 'unpin' : 'pin'}</button>
          <button data-act="unload" data-id="${m.id}" class="px-2 py-1 rounded-lg text-[10px] font-bold bg-white border border-sky-200 text-slate-500 hover:text-rose-500 hover:border-rose-300 transition">unload</button>
        </div>`;
      cards.appendChild(el);
    });
  }

  function evictToFit(incomingSize, reason) {
    while (used() + incomingSize > budget) {
      const victimIdx = loaded.findIndex((m) => !m.pinned);
      if (victimIdx === -1) return false;          // everything left is pinned
      const [victim] = loaded.splice(victimIdx, 1);
      evictions++;
      log(`[evict] ${victim.name} (${victim.size} GB, LRU) — ${reason}`, 'evict');
    }
    return true;
  }

  function load(id) {
    const spec = SPECS[id];
    const idx = loaded.findIndex((m) => m.id === id);
    if (idx > -1) {
      const [m] = loaded.splice(idx, 1);
      loaded.push(m);
      log(`[route] ${m.name} already loaded — served, now most-recently-used`, 'info');
      return render();
    }
    log(`[request] model=${spec.name} needs ${spec.size} GB (budget ${budget} GB, used ${used().toFixed(1)} GB)`, 'info');
    const fits = evictToFit(spec.size, `making room for ${spec.name}`);
    if (!fits) {
      log(`[warn] cap reached but every loaded model is pinned — admitting ${spec.name} OVER BUDGET rather than killing a warm pinned service`, 'warn');
    }
    loaded.push({ ...spec, pinned: false });
    log(`[start] ${spec.backend} :${spec.port} ← ${spec.name} … healthy`, 'load');
    render();
  }

  cards.addEventListener('click', (e) => {
    const btn = e.target.closest('button[data-act]');
    if (!btn) return;
    const idx = loaded.findIndex((m) => m.id === btn.dataset.id);
    if (idx === -1) return;
    const m = loaded[idx];
    if (btn.dataset.act === 'unload') {
      loaded.splice(idx, 1);
      log(`[unload] ${m.name} stopped — ${m.size} GB returned`, 'info');
    } else if (btn.dataset.act === 'pin') {
      m.pinned = !m.pinned;
      log(m.pinned
        ? `[registry] ${m.name}: pinned=true — exempt from idle sweep and eviction`
        : `[registry] ${m.name}: pinned=false — evictable again`, 'info');
    } else if (btn.dataset.act === 'touch') {
      loaded.splice(idx, 1);
      loaded.push(m);
      log(`[route] request served by ${m.name} — LRU position refreshed`, 'info');
    }
    render();
  });

  document.querySelectorAll('#visualizer-load-buttons button').forEach((b) => {
    b.addEventListener('click', () => load(b.dataset.model));
  });

  memSlider.addEventListener('input', (e) => {
    budget = parseFloat(e.target.value);
    display.textContent = `${budget} GB`;
    if (used() > budget) {
      log(`[config] mem_budget_gb lowered to ${budget} — sweeping LRU models`, 'warn');
      evictToFit(0, 'budget shrank');
    }
    render();
  });

  $('btn-reset-eviction').addEventListener('click', () => {
    loaded = [];
    evictions = 0;
    logBox.innerHTML = '<div class="text-slate-500">[manager] reset — load a model to begin</div>';
    render();
  });

  render();
}

// ---------------------------------------------------------------------------
// 7. API playground (index.html only)
// ---------------------------------------------------------------------------
const apiSnippet = document.getElementById('api-snippet');
if (apiSnippet) {
  const BASE = 'http://localhost:11435/v1';
  const S = {
    chat: {
      note: 'Streaming, tool calls and reasoning blocks pass straight through. Set the model\'s launch defaults with `zallama set`; anything in the request wins.',
      curl: `curl ${BASE}/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "qwen3.5-4b-q4_k_m",
    "messages": [{"role": "user", "content": "Tell me a joke."}],
    "stream": true
  }'`,
      python: `from openai import OpenAI

client = OpenAI(base_url="${BASE}", api_key="not-needed-on-localhost")

stream = client.chat.completions.create(
    model="qwen3.5-4b-q4_k_m",
    messages=[{"role": "user", "content": "Tell me a joke."}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)`,
      js: `import OpenAI from "openai";

const client = new OpenAI({ baseURL: "${BASE}", apiKey: "not-needed-on-localhost" });

const stream = await client.chat.completions.create({
  model: "qwen3.5-4b-q4_k_m",
  messages: [{ role: "user", content: "Tell me a joke." }],
  stream: true,
});
for await (const chunk of stream) process.stdout.write(chunk.choices[0]?.delta?.content ?? "");`,
    },
    vision: {
      note: 'The model must be registered with an `mmproj` artifact. Images go in as standard `image_url` content parts (data URI or URL).',
      curl: `curl ${BASE}/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "qwen2-vl-7b",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo..."}}
      ]
    }]
  }'`,
      python: `import base64
from openai import OpenAI

client = OpenAI(base_url="${BASE}", api_key="x")
img = base64.b64encode(open("chart.png", "rb").read()).decode()

r = client.chat.completions.create(
    model="qwen2-vl-7b",
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}},
    ]}],
)
print(r.choices[0].message.content)`,
      js: `import fs from "node:fs";
import OpenAI from "openai";

const client = new OpenAI({ baseURL: "${BASE}", apiKey: "x" });
const img = fs.readFileSync("chart.png").toString("base64");

const r = await client.chat.completions.create({
  model: "qwen2-vl-7b",
  messages: [{ role: "user", content: [
    { type: "text", text: "What is in this image?" },
    { type: "image_url", image_url: { url: \`data:image/png;base64,\${img}\` } },
  ]}],
});
console.log(r.choices[0].message.content);`,
    },
    asr: {
      note: 'Any input format is transcoded to WAV via ffmpeg. `response_format` accepts text, json or verbose_json (add timestamp_granularities[]=word for per-word timing).',
      curl: `curl ${BASE}/audio/transcriptions \\
  -F model=tdt-0.6b-v3-q8_0 \\
  -F file=@speech.mp3 \\
  -F response_format=text`,
      python: `from openai import OpenAI

client = OpenAI(base_url="${BASE}", api_key="x")

with open("speech.mp3", "rb") as f:
    text = client.audio.transcriptions.create(
        model="tdt-0.6b-v3-q8_0",
        file=f,
        response_format="text",
    )
print(text)`,
      js: `import fs from "node:fs";
import OpenAI from "openai";

const client = new OpenAI({ baseURL: "${BASE}", apiKey: "x" });

const text = await client.audio.transcriptions.create({
  model: "tdt-0.6b-v3-q8_0",
  file: fs.createReadStream("speech.mp3"),
  response_format: "text",
});
console.log(text);`,
    },
    tts: {
      note: 'Returns a WAV stream. Leave `voice` out and Zallama picks one matching the detected language of the text; set it explicitly (ff_siwis, af_heart, …) for a guaranteed result.',
      curl: `curl ${BASE}/audio/speech \\
  -H "Content-Type: application/json" \\
  -d '{"model":"kokoro:82m","input":"Bonjour, comment allez-vous ?","voice":"ff_siwis"}' \\
  -o speech.wav`,
      python: `from openai import OpenAI

client = OpenAI(base_url="${BASE}", api_key="x")

with client.audio.speech.with_streaming_response.create(
    model="kokoro:82m",
    input="Bonjour, comment allez-vous ?",
    voice="ff_siwis",
) as resp:
    resp.stream_to_file("speech.wav")`,
      js: `import fs from "node:fs";
import OpenAI from "openai";

const client = new OpenAI({ baseURL: "${BASE}", apiKey: "x" });

const wav = await client.audio.speech.create({
  model: "kokoro:82m",
  input: "Bonjour, comment allez-vous ?",
  voice: "ff_siwis",
});
fs.writeFileSync("speech.wav", Buffer.from(await wav.arrayBuffer()));`,
    },
    image: {
      note: 'Returns base64 PNG. steps / cfg_scale / sampler / negative_prompt fall back to the model\'s registry values when left out of the request.',
      curl: `curl ${BASE}/images/generations \\
  -H "Content-Type: application/json" \\
  -d '{"model":"flux:klein","prompt":"a lighthouse at dawn, cinematic","size":"1024x1024","response_format":"b64_json"}'`,
      python: `import base64
from openai import OpenAI

client = OpenAI(base_url="${BASE}", api_key="x")

r = client.images.generate(
    model="flux:klein",
    prompt="a lighthouse at dawn, cinematic",
    size="1024x1024",
    response_format="b64_json",
)
open("dawn.png", "wb").write(base64.b64decode(r.data[0].b64_json))`,
      js: `import fs from "node:fs";
import OpenAI from "openai";

const client = new OpenAI({ baseURL: "${BASE}", apiKey: "x" });

const r = await client.images.generate({
  model: "flux:klein",
  prompt: "a lighthouse at dawn, cinematic",
  size: "1024x1024",
  response_format: "b64_json",
});
fs.writeFileSync("dawn.png", Buffer.from(r.data[0].b64_json, "base64"));`,
    },
    embed: {
      note: 'Runs llama-server in --embedding mode via the embedding-server backend. `zallama pull nomic-embed:v1.5` registers one for you.',
      curl: `curl ${BASE}/embeddings \\
  -H "Content-Type: application/json" \\
  -d '{"model":"nomic-embed:v1.5","input":["Own your AI.","Rent your AI."]}'`,
      python: `from openai import OpenAI

client = OpenAI(base_url="${BASE}", api_key="x")

r = client.embeddings.create(
    model="nomic-embed:v1.5",
    input=["Own your AI.", "Rent your AI."],
)
print(len(r.data[0].embedding), "dims")`,
      js: `import OpenAI from "openai";

const client = new OpenAI({ baseURL: "${BASE}", apiKey: "x" });

const r = await client.embeddings.create({
  model: "nomic-embed:v1.5",
  input: ["Own your AI.", "Rent your AI."],
});
console.log(r.data[0].embedding.length, "dims");`,
    },
    rerank: {
      note: 'Cohere/Jina-style response: { results: [{ index, relevance_score, document? }] } sorted by score. Runs llama-server in --reranking mode.',
      curl: `curl ${BASE}/rerank \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "bge-reranker:v2-m3",
    "query": "How do I unload a model?",
    "documents": ["zallama unload <name> stops a model", "zallama pull fetches a model"],
    "top_n": 2,
    "return_documents": true
  }'`,
      python: `import requests

r = requests.post("${BASE}/rerank", json={
    "model": "bge-reranker:v2-m3",
    "query": "How do I unload a model?",
    "documents": ["zallama unload <name> stops a model", "zallama pull fetches a model"],
    "top_n": 2,
    "return_documents": True,
})
for hit in r.json()["results"]:
    print(f'{hit["relevance_score"]:.3f}  {hit["document"]}')`,
      js: `const r = await fetch("${BASE}/rerank", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    model: "bge-reranker:v2-m3",
    query: "How do I unload a model?",
    documents: ["zallama unload <name> stops a model", "zallama pull fetches a model"],
    top_n: 2,
    return_documents: true,
  }),
});
for (const hit of (await r.json()).results) console.log(hit.relevance_score.toFixed(3), hit.document);`,
    },
    zvec: {
      note: 'zvec embeds and searches through Zallama\'s own /v1/embeddings, so it only needs rag.embedding_model set. Add rerank_model to re-score candidates with a cross-encoder.',
      curl: `# create a collection and add documents (auto-embedded)
curl ${BASE}/zvec/collections -H "Content-Type: application/json" \\
  -d '{"name":"notes"}'
curl ${BASE}/zvec/notes/upsert -H "Content-Type: application/json" \\
  -d '{"documents":[{"id":"1","text":"zallama unload <name> stops a model"},{"id":"2","text":"zallama pull fetches a model"}]}'

# semantic search, optionally reranked
curl ${BASE}/zvec/notes/query -H "Content-Type: application/json" \\
  -d '{"query":"how to unload a model","top_k":3,"rerank_model":"bge-reranker:v2-m3"}'`,
      python: `import requests

B = "${BASE}/zvec"
requests.post(f"{B}/collections", json={"name": "notes"})
requests.post(f"{B}/notes/upsert", json={"documents": [
    {"id": "1", "text": "zallama unload <name> stops a model"},
    {"id": "2", "text": "zallama pull fetches a model"},
]})
hits = requests.post(f"{B}/notes/query", json={
    "query": "how to unload a model", "top_k": 3, "rerank_model": "bge-reranker:v2-m3",
}).json()
print(hits)`,
      js: `const B = "${BASE}/zvec";
const post = (p, body) => fetch(B + p, {
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
}).then((r) => r.json());

await post("/collections", { name: "notes" });
await post("/notes/upsert", { documents: [
  { id: "1", text: "zallama unload <name> stops a model" },
  { id: "2", text: "zallama pull fetches a model" },
]});
console.log(await post("/notes/query", { query: "how to unload a model", top_k: 3, rerank_model: "bge-reranker:v2-m3" }));`,
    },
  };

  let tab = 'chat';
  let lang = 'curl';
  const note = document.getElementById('api-note');
  const tabBtns = Array.from(document.querySelectorAll('#api-tabs .tab-btn'));
  const langBtns = Array.from(document.querySelectorAll('#api-langs button'));

  function renderSnippet() {
    apiSnippet.textContent = S[tab][lang];
    note.textContent = S[tab].note;
    tabBtns.forEach((b) => b.classList.toggle('active', b.dataset.tab === tab));
    langBtns.forEach((b) => {
      const on = b.dataset.lang === lang;
      b.className = on
        ? 'text-white border-b-2 border-sky-400 pb-1'
        : 'text-slate-400 hover:text-white transition pb-1 border-b-2 border-transparent';
    });
  }

  tabBtns.forEach((b) => b.addEventListener('click', () => { tab = b.dataset.tab; renderSnippet(); }));
  langBtns.forEach((b) => b.addEventListener('click', () => { lang = b.dataset.lang; renderSnippet(); }));

  const copyBtn = document.getElementById('btn-copy-snippet');
  copyBtn.addEventListener('click', () => copyText(S[tab][lang], copyBtn));

  renderSnippet();
}
