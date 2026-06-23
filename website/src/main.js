// src/main.js — Zallama Website Interactivity
import './style.css';

// ---------------------------------------------------------------------------
// 1. Clipboard Copy Helper
// ---------------------------------------------------------------------------
window.copyToClipboard = function(elementId, buttonId) {
  const element = document.getElementById(elementId);
  if (!element) return;
  
  const text = element.innerText || element.textContent;
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.getElementById(buttonId);
    if (!btn) return;
    
    // Save original innerHTML
    const originalHTML = btn.innerHTML;
    btn.classList.add('copied');
    btn.innerHTML = `
      <svg class="w-5 h-5 text-emerald-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
      </svg>
    `;
    
    setTimeout(() => {
      btn.classList.remove('copied');
      btn.innerHTML = originalHTML;
    }, 2000);
  }).catch(err => {
    console.error('Failed to copy text: ', err);
  });
};

// ---------------------------------------------------------------------------
// 2. Terminal Simulator
// ---------------------------------------------------------------------------
const terminalBody = document.getElementById('terminal-body');
const terminalCommands = {
  serve: [
    { text: '$ zallama serve', delay: 300, type: 'command' },
    { text: '18:00:03 [INFO] zallama: 🦙 llama-server binary: ./bin/llama-server', delay: 200 },
    { text: '18:00:03 [INFO] zallama: 🎙️ kokoro-server binary: ./bin/kokoro-server', delay: 100 },
    { text: '18:00:03 [INFO] zallama: ✅ Zallama ready — 4 model(s) registered', delay: 150 },
    { text: '18:00:03 [INFO] zallama: 🚀 Starting Zallama on http://127.0.0.1:11435', delay: 200 },
    { text: '18:00:03 [INFO] uvicorn: Uvicorn running on http://127.0.0.1:11435 (Press CTRL+C to quit)', delay: 100 },
    { text: '18:00:35 [INFO] zallama: [MANAGER] Idle sweep checks in progress...', delay: 600 }
  ],
  pull: [
    { text: '$ zallama pull kokoro:82m', delay: 300, type: 'command' },
    { text: '[DOWNLOADER] Pulling preset kokoro:82m from rzafiamy/kokoro.cpp-gguf', delay: 200 },
    { text: '[DOWNLOADER] Found aria2c binary. Launching 8-threaded download...', delay: 150 },
    { text: '[aria2c] Downloading: kokoro-v0.19-q8_0.bin', delay: 100 },
    { text: '[aria2c] Progress: [████████████████████] 100% (15.5 MB/s) - Done.', delay: 400 },
    { text: '[MANAGER] Verifying SHA256 checksum...', delay: 200 },
    { text: '18:01:45 [INFO] zallama: Model kokoro:82m successfully registered in registry.yaml.', delay: 200 }
  ],
  run: [
    { text: '$ zallama run llama3.2:3b', delay: 300, type: 'command' },
    { text: '[MANAGER] Pre-loading model Llama-3.2-3B-Instruct (2.0GB) on port 8100...', delay: 150 },
    { text: '[SERVER] llama-server listening on port 8100. Connection established.', delay: 300 },
    { text: '>>> User: Explain quantum physics in one sentence.', delay: 450, type: 'user' },
    { text: '<think>\nAnalyzing terms: quantum physics, atomic scale, wave-particle duality...\nFormulating simple analogy...\n</think>', delay: 600, type: 'thinking' },
    { text: 'Quantum physics is the study of how matter and light behave at the atomic and subatomic level, where particles can exist in multiple states at once and behave like waves.', delay: 300, type: 'assistant' }
  ],
  ps: [
    { text: '$ zallama ps', delay: 300, type: 'command' },
    { text: 'NAME                      PORT     MEM      UPTIME       LAST USED', delay: 100, type: 'header' },
    { text: '─────────────────────────────────────────────────────────────────────────', delay: 50 },
    { text: 'qwen3.5-4b-q4_k_m         8100     4.0GB    12m42s       12s ago', delay: 100 },
    { text: 'tdt-0.6b-v3-q8_0          8101     0.8GB    3m18s        1m ago', delay: 100 },
    { text: 'kokoro-v1.0.bin           8104     0.1GB    45s          1s ago', delay: 100 },
    { text: '\nMemory: 4.9GB / 12.0GB used  •  7.1GB free  •  3 loaded', delay: 150, type: 'highlight' }
  ],
  set: [
    { text: '$ zallama set qwen3.5-4b-q4_k_m reasoning=false ctx_size=8192', delay: 300, type: 'command' },
    { text: '[REGISTRY] Modifying registry params for qwen3.5-4b-q4_k_m...', delay: 200 },
    { text: ' - Set reasoning -> false', delay: 100 },
    { text: ' - Set ctx_size -> 8192', delay: 100 },
    { text: '[REGISTRY] registry.yaml saved successfully.', delay: 150 },
    { text: '18:02:11 [WARNING] qwen3.5-4b-q4_k_m is already running. Run "zallama reload qwen3.5-4b-q4_k_m" to apply parameter modifications.', delay: 200 }
  ]
};

let termTimeoutId = null;

function runTerminalSimulation(cmdName) {
  // Clear existing logs
  if (termTimeoutId) clearTimeout(termTimeoutId);
  terminalBody.innerHTML = '';
  
  const steps = terminalCommands[cmdName];
  if (!steps) return;
  
  let currentStep = 0;
  
  function printStep() {
    if (currentStep >= steps.length) return;
    const step = steps[currentStep];
    const lineElement = document.createElement('div');
    
    // Custom styling depending on the line type
    if (step.type === 'command') {
      lineElement.className = 'text-brand font-bold';
    } else if (step.type === 'user') {
      lineElement.className = 'text-emerald-500 font-bold mt-2';
    } else if (step.type === 'thinking') {
      lineElement.className = 'text-text-muted italic bg-panel p-2 rounded border border-panel-border my-1 whitespace-pre-line';
    } else if (step.type === 'assistant') {
      lineElement.className = 'text-text-main pl-4 border-l-2 border-brand mt-1';
    } else if (step.type === 'header') {
      lineElement.className = 'text-text-muted font-bold';
    } else if (step.type === 'highlight') {
      lineElement.className = 'text-brand font-bold';
    } else {
      lineElement.className = 'text-text-main';
    }
    
    lineElement.textContent = step.text;
    terminalBody.appendChild(lineElement);
    
    // Auto-scroll
    terminalBody.scrollTop = terminalBody.scrollHeight;
    
    currentStep++;
    termTimeoutId = setTimeout(printStep, step.delay);
  }
  
  printStep();
}

// Setup terminal button listeners
const termButtons = {
  serve: document.getElementById('term-cmd-serve'),
  pull: document.getElementById('term-cmd-pull'),
  run: document.getElementById('term-cmd-run'),
  ps: document.getElementById('term-cmd-ps'),
  set: document.getElementById('term-cmd-set')
};

Object.keys(termButtons).forEach(key => {
  const btn = termButtons[key];
  if (!btn) return;
  btn.addEventListener('click', () => {
    // Reset all buttons to inactive styling
    Object.values(termButtons).forEach(b => {
      b.className = 'w-full text-left px-2 py-1.5 rounded font-mono text-xs text-text-muted hover:bg-panel transition';
    });
    // Set clicked button to active
    btn.className = 'w-full text-left px-2 py-1.5 rounded font-mono text-xs text-brand bg-brand-light border border-brand/10 hover:bg-panel transition';
    
    runTerminalSimulation(key);
  });
});

// Run default simulation on mount
runTerminalSimulation('serve');

// ---------------------------------------------------------------------------
// 3. Memory Eviction Simulator
// ---------------------------------------------------------------------------
const memoryModels = {
  qwen: { id: 'qwen', name: 'Qwen 2.5 4B Coder', size: 4.0, file: 'Qwen2.5-Coder-4B-Q4_K_M.gguf', port: 8100 },
  llama: { id: 'llama', name: 'Llama 3.2 3B Instruct', size: 3.6, file: 'Llama-3.2-3B-Instruct-Q4_K_M.gguf', port: 8101 },
  deepseek: { id: 'deepseek', name: 'DeepSeek R1 8B', size: 5.5, file: 'DeepSeek-R1-Distill-Q4_K_M.gguf', port: 8102 },
  parakeet: { id: 'parakeet', name: 'Parakeet ASR 0.6B', size: 0.8, file: 'tdt-0.6b-v3-q8_0.gguf', port: 8103 },
  kokoro: { id: 'kokoro', name: 'Kokoro TTS 82M', size: 0.1, file: 'kokoro-v0.19-q8_0.bin', port: 8104 }
};

let memBudget = 12.0;
let loadedModels = [];

const memSlider = document.getElementById('mem-budget-slider');
const memDisplay = document.getElementById('mem-budget-display');
const progressBar = document.getElementById('visualizer-progress-bar');
const statMemUsage = document.getElementById('stat-mem-usage');
const statLoadedCount = document.getElementById('stat-loaded-count');
const statFreeGb = document.getElementById('stat-free-gb');
const cardsContainer = document.getElementById('visualizer-cards-container');
const evictionLogs = document.getElementById('eviction-logs-body');

// Eviction log utility
function logEviction(message, type = 'info') {
  const line = document.createElement('div');
  const timestamp = new Date().toLocaleTimeString();
  
  if (type === 'evict') {
    line.className = 'text-rose-500 font-semibold';
    line.textContent = `[${timestamp}] [EVICTION] ${message}`;
  } else if (type === 'load') {
    line.className = 'text-emerald-500';
    line.textContent = `[${timestamp}] [LOADER] ${message}`;
  } else {
    line.className = 'text-text-muted';
    line.textContent = `[${timestamp}] [MANAGER] ${message}`;
  }
  
  evictionLogs.appendChild(line);
  evictionLogs.scrollTop = evictionLogs.scrollHeight;
}

// Render the visualizer state
function renderVisualizer() {
  // Update texts
  const currentTotal = loadedModels.reduce((sum, m) => sum + m.size, 0);
  const percentage = Math.min(100, Math.round((currentTotal / memBudget) * 100));
  
  statMemUsage.textContent = `${currentTotal.toFixed(1)}GB / ${memBudget.toFixed(1)}GB (${percentage}%)`;
  statLoadedCount.textContent = loadedModels.length.toString();
  statFreeGb.textContent = `${Math.max(0, memBudget - currentTotal).toFixed(1)} GB`;
  
  // Progress bar width
  progressBar.style.width = `${percentage}%`;
  
  // Progress bar color warning threshold
  if (percentage > 90) {
    progressBar.className = 'h-full rounded-full bg-gradient-to-r from-red-600 to-rose-500 transition-all duration-500 ease-out';
  } else if (percentage > 70) {
    progressBar.className = 'h-full rounded-full bg-gradient-to-r from-amber-500 to-brand transition-all duration-500 ease-out';
  } else {
    progressBar.className = 'h-full rounded-full bg-gradient-to-r from-brand to-brand-hover transition-all duration-500 ease-out';
  }
  
  // Cards Container rendering
  cardsContainer.innerHTML = '';
  if (loadedModels.length === 0) {
    cardsContainer.innerHTML = `
      <div class="h-32 flex flex-col items-center justify-center border border-dashed border-panel-border rounded-xl text-text-muted">
        <span class="text-2xl mb-1">🦙</span>
        <span class="text-xs">No models active in daemon memory. Try clicking a "Load" button above.</span>
      </div>
    `;
    return;
  }
  
  loadedModels.forEach((model, index) => {
    const card = document.createElement('div');
    card.className = 'flex items-center justify-between p-4 bg-bg border border-panel-border rounded-xl hover:border-brand/40 transition duration-150 animate-float';
    card.style.animationDuration = `${6 + index}s`; // Offset animation cycles
    
    // Calculate memory percentage of the slot
    const cardPct = Math.round((model.size / memBudget) * 100);
    
    card.innerHTML = `
      <div class="flex items-center space-x-4">
        <div class="w-10 h-10 rounded-lg bg-brand-light border border-brand/20 flex items-center justify-center font-bold text-brand text-sm">
          ${model.port}
        </div>
        <div>
          <div class="flex items-center space-x-2">
            <h5 class="font-bold text-text-main text-sm">${model.name}</h5>
            <span class="text-[9px] font-mono bg-panel text-text-muted px-1 py-0.2 rounded border border-panel-border">${model.file}</span>
          </div>
          <p class="text-xs text-text-muted mt-0.5">Resident Space: <span class="text-brand font-bold">${model.size.toFixed(1)}GB</span> (${cardPct}% of total budget)</p>
        </div>
      </div>
      
      <div class="flex items-center space-x-3">
        <span class="flex h-2 w-2 relative">
          <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-brand opacity-75"></span>
          <span class="relative inline-flex rounded-full h-2 w-2 bg-brand"></span>
        </span>
        <button onclick="unloadModel('${model.id}')" class="px-2.5 py-1 rounded bg-panel hover:bg-rose-500/10 hover:text-rose-500 text-xs text-text-muted border border-panel-border transition">
          Unload
        </button>
      </div>
    `;
    cardsContainer.appendChild(card);
  });
}

// Unload manual action
window.unloadModel = function(modelId) {
  const modelIdx = loadedModels.findIndex(m => m.id === modelId);
  if (modelIdx > -1) {
    const model = loadedModels[modelIdx];
    loadedModels.splice(modelIdx, 1);
    logEviction(`Unloading model '${model.name}' to free up ${model.size}GB`, 'info');
    renderVisualizer();
  }
};

// Load model execution
function loadModel(modelId) {
  const spec = memoryModels[modelId];
  if (!spec) return;
  
  // Check if already loaded
  const existingIdx = loadedModels.findIndex(m => m.id === modelId);
  if (existingIdx > -1) {
    // Bring to end (Most Recently Used)
    const [existing] = loadedModels.splice(existingIdx, 1);
    loadedModels.push(existing);
    logEviction(`Model '${spec.name}' is already loaded. Updated LRU position.`, 'info');
    renderVisualizer();
    return;
  }
  
  logEviction(`Request received to load model '${spec.name}' (${spec.size}GB)...`, 'info');
  
  // Eviction Loop
  let currentUsage = loadedModels.reduce((sum, m) => sum + m.size, 0);
  while (currentUsage + spec.size > memBudget && loadedModels.length > 0) {
    // Evict oldest (MRU is last, LRU is first element)
    const evicted = loadedModels.shift();
    logEviction(`Evicting Least-Recently-Used (LRU) model '${evicted.name}' (${evicted.size}GB) to satisfy memory budget limit.`, 'evict');
    currentUsage = loadedModels.reduce((sum, m) => sum + m.size, 0);
  }
  
  if (currentUsage + spec.size > memBudget) {
    logEviction(`Error: Model size (${spec.size}GB) exceeds total memory budget allocation (${memBudget}GB). Increase slider budget.`, 'evict');
    return;
  }
  
  // Load new model
  loadedModels.push(spec);
  logEviction(`Successfully spawned runner process on port ${spec.port} for '${spec.name}'.`, 'load');
  renderVisualizer();
}

// Budget Slider Listener
memSlider.addEventListener('input', (e) => {
  memBudget = parseFloat(e.target.value);
  memDisplay.textContent = `${memBudget.toFixed(1)} GB`;
  
  // Check if we need to evict immediate models post budget shrink
  let currentUsage = loadedModels.reduce((sum, m) => sum + m.size, 0);
  while (currentUsage > memBudget && loadedModels.length > 0) {
    const evicted = loadedModels.shift();
    logEviction(`Evicting model '${evicted.name}' due to global RAM budget downsizing.`, 'evict');
    currentUsage = loadedModels.reduce((sum, m) => sum + m.size, 0);
  }
  
  renderVisualizer();
});

// Reset Button Listener
document.getElementById('btn-reset-eviction').addEventListener('click', () => {
  loadedModels = [];
  evictionLogs.innerHTML = '<span class="text-text-muted">[MANAGER] System resetting... Done.</span>';
  renderVisualizer();
});

// Setup click listeners for loader buttons
document.getElementById('btn-load-qwen').addEventListener('click', () => loadModel('qwen'));
document.getElementById('btn-load-llama').addEventListener('click', () => loadModel('llama'));
document.getElementById('btn-load-deepseek').addEventListener('click', () => loadModel('deepseek'));
document.getElementById('btn-load-parakeet').addEventListener('click', () => loadModel('parakeet'));
document.getElementById('btn-load-kokoro').addEventListener('click', () => loadModel('kokoro'));

// Initial visualizer setup
renderVisualizer();

// ---------------------------------------------------------------------------
// 4. API Code Snippets Playground
// ---------------------------------------------------------------------------
const codeSnippets = {
  chat: {
    curl: `curl http://localhost:11435/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "qwen3.5-4b-q4_k_m",
    "messages": [
      {"role": "user", "content": "Tell me a joke."}
    ],
    "stream": true
  }'`,
    python: `from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11435/v1",
    api_key="your-api-key-if-set"
)

response = client.chat.completions.create(
    model="qwen3.5-4b-q4_k_m",
    messages=[{"role": "user", "content": "Tell me a joke."}],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="", flush=True)`,
    js: `import OpenAI from 'openai';

const openai = new OpenAI({
  baseURL: 'http://localhost:11435/v1',
  apiKey: 'your-api-key-if-set'
});

const stream = await openai.chat.completions.create({
  model: 'qwen3.5-4b-q4_k_m',
  messages: [{ role: 'user', content: 'Tell me a joke.' }],
  stream: true,
});

for await (const chunk of stream) {
  process.stdout.write(chunk.choices[0]?.delta?.content || '');
}`
  },
  vision: {
    curl: `curl http://localhost:11435/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "qwen2-vl-7b",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Analyze this chart"},
          {
            "type": "image_url",
            "image_url": {
              "url": "data:image/png;base64,iVBORw0KGgo..."
            }
          }
        ]
      }
    ]
  }'`,
    python: `from openai import OpenAI
import base64

client = OpenAI(base_url="http://localhost:11435/v1")

# Base64 encode the chart image
with open("chart.png", "rb") as img_file:
    encoded = base64.b64encode(img_file.read()).decode("utf-8")

response = client.chat.completions.create(
    model="qwen2-vl-7b",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this chart"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}
            ]
        }
    ]
)
print(response.choices[0].message.content)`,
    js: `import OpenAI from 'openai';
import fs from 'fs';

const openai = new OpenAI({ baseURL: 'http://localhost:11435/v1' });
const imgB64 = fs.readFileSync('chart.png', { encoding: 'base64' });

const response = await openai.chat.completions.create({
  model: 'qwen2-vl-7b',
  messages: [
    {
      role: 'user',
      content: [
        { type: 'text', text: 'Analyze this chart' },
        { type: 'image_url', image_url: { url: \`data:image/png;base64,\${imgB64}\` } }
      ]
    }
  ]
});
console.log(response.choices[0].message.content);`
  },
  asr: {
    curl: `curl http://localhost:11435/v1/audio/transcriptions \\
  -F "model=tdt-0.6b-v3-q8_0" \\
  -F "file=@speech.mp3" \\
  -F "response_format=verbose_json" \\
  -F "timestamp_granularities[]=word"`,
    python: `from openai import OpenAI

client = OpenAI(base_url="http://localhost:11435/v1")

with open("speech.mp3", "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        model="tdt-0.6b-v3-q8_0",
        file=audio_file,
        response_format="verbose_json"
    )
print(transcript)`,
    js: `import OpenAI from 'openai';
import fs from 'fs';

const openai = new OpenAI({ baseURL: 'http://localhost:11435/v1' });

const transcript = await openai.audio.transcriptions.create({
  model: 'tdt-0.6b-v3-q8_0',
  file: fs.createReadStream('speech.mp3'),
  response_format: 'verbose_json'
});
console.log(transcript);`
  },
  tts: {
    curl: `curl http://localhost:11435/v1/audio/speech \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "kokoro:82m",
    "input": "Hello world, this is a local voice generated by Zallama.",
    "voice": "af_bella"
  }' \\
  --output speech.mp3`,
    python: `from openai import OpenAI

client = OpenAI(base_url="http://localhost:11435/v1")

response = client.audio.speech.create(
    model="kokoro:82m",
    voice="af_bella",
    input="Hello world, this is a local voice generated by Zallama."
)
response.write_to_file("speech.mp3")`,
    js: `import OpenAI from 'openai';
import fs from 'fs';

const openai = new OpenAI({ baseURL: 'http://localhost:11435/v1' });

const mp3 = await openai.audio.speech.create({
  model: 'kokoro:82m',
  voice: 'af_bella',
  input: 'Hello world, this is a local voice generated by Zallama.',
});

const buffer = Buffer.from(await mp3.arrayBuffer());
await fs.promises.writeFile('speech.mp3', buffer);`
  }
};

let activeTab = 'chat';
let activeLang = 'curl';

const tabButtons = {
  chat: document.getElementById('btn-tab-chat'),
  vision: document.getElementById('btn-tab-vision'),
  asr: document.getElementById('btn-tab-asr'),
  tts: document.getElementById('btn-tab-tts')
};

const langButtons = {
  curl: document.getElementById('lang-btn-curl'),
  python: document.getElementById('lang-btn-python'),
  js: document.getElementById('lang-btn-js')
};

const codeSnippetBox = document.getElementById('code-snippet-box');

// Render Code Snippet
function renderCodeSnippet() {
  const rawCode = codeSnippets[activeTab][activeLang];
  
  // Syntax highlight helpers (simple regex replacer for display colors)
  let highlighted = rawCode
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/(^|[^a-zA-Z0-9])(import|from|const|let|await|for|in|of|print|console|await|client|openai|import|with|as|for|return|def)([^a-zA-Z0-9]|$)/g, '$1<span class="text-brand font-semibold">$2</span>$3')
    .replace(/("[^"\\]*(?:\\.[^"\\]*)*"|'[^'\\]*(?:\\.[^'\\]*)*')/g, '<span class="text-emerald-500">$1</span>')
    .replace(/(#.*|\/\/.*)/g, '<span class="text-text-muted italic">$1</span>');
  
  codeSnippetBox.innerHTML = `<pre>${highlighted}</pre>`;
}

// Code copying
document.getElementById('btn-copy-code').addEventListener('click', () => {
  const code = codeSnippets[activeTab][activeLang];
  navigator.clipboard.writeText(code).then(() => {
    const btn = document.getElementById('btn-copy-code');
    const originalText = btn.innerHTML;
    btn.innerHTML = `
      <svg class="w-4 h-4 text-emerald-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
      </svg>
      <span class="text-emerald-500">Copied!</span>
    `;
    setTimeout(() => {
      btn.innerHTML = originalText;
    }, 2000);
  });
});

// Setup tab listeners
Object.keys(tabButtons).forEach(key => {
  const btn = tabButtons[key];
  if (!btn) return;
  btn.addEventListener('click', () => {
    Object.values(tabButtons).forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    activeTab = key;
    renderCodeSnippet();
  });
});

// Setup lang listeners
Object.keys(langButtons).forEach(key => {
  const btn = langButtons[key];
  if (!btn) return;
  btn.addEventListener('click', () => {
    Object.values(langButtons).forEach(b => {
      b.className = 'text-text-muted hover:text-text-main transition pb-1';
    });
    btn.className = 'text-text-main font-bold border-b-2 border-brand pb-1';
    activeLang = key;
    renderCodeSnippet();
  });
});

renderCodeSnippet();

// ---------------------------------------------------------------------------
// 5. Global Theme Toggle (adapting standard root attribute data-theme)
// ---------------------------------------------------------------------------
const themeToggleBtn = document.getElementById('theme-toggle-btn');
const themeSun = document.getElementById('theme-sun');
const themeMoon = document.getElementById('theme-moon');

// Set default moon/sun visual state based on system or local preference
if (document.documentElement.getAttribute('data-theme') === 'light') {
  themeSun.classList.remove('hidden');
  themeMoon.classList.add('hidden');
} else {
  themeSun.classList.add('hidden');
  themeMoon.classList.remove('hidden');
}

themeToggleBtn.addEventListener('click', () => {
  const currentTheme = document.documentElement.getAttribute('data-theme');
  
  if (currentTheme === 'light') {
    // Switch to dark theme
    document.documentElement.removeAttribute('data-theme');
    themeSun.classList.add('hidden');
    themeMoon.classList.remove('hidden');
  } else {
    // Switch to light theme
    document.documentElement.setAttribute('data-theme', 'light');
    themeSun.classList.remove('hidden');
    themeMoon.classList.add('hidden');
  }
});
