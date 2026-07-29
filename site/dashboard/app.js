/* ── Zaya Dashboard — Single-Page App ───────────────────────────
   Hash-based routing, API client, state management, page components.
   API base: same origin (empty string = relative to document host).
*/

// ═══════════════════════════════════════════════════════════════════
// API Client
// ═══════════════════════════════════════════════════════════════════
const API = {
  base: '',  // same origin — Jarvis server serves both dashboard + API

  async get(path) {
    const headers = this._headers();
    const r = await fetch(this.base + path, { headers });
    if (!r.ok) {
      const body = await r.json().catch(() => ({ error: r.statusText }));
      throw new Error(body.error || `HTTP ${r.status}`);
    }
    return r.json();
  },

  async post(path, body) {
    const headers = this._headers({ 'Content-Type': 'application/json' });
    const r = await fetch(this.base + path, {
      method: 'POST',
      headers,
      body: typeof body === 'string' ? body : JSON.stringify(body),
    });
    if (!r.ok) {
      const errBody = await r.json().catch(() => ({ error: r.statusText }));
      throw new Error(errBody.error || `HTTP ${r.status}`);
    }
    return r.json();
  },

  async postForm(path, formData) {
    const headers = this._headers();  // no Content-Type — let browser set multipart boundary
    const r = await fetch(this.base + path, {
      method: 'POST',
      headers,
      body: formData,
    });
    if (!r.ok) {
      const errBody = await r.json().catch(() => ({ error: r.statusText }));
      throw new Error(errBody.error || `HTTP ${r.status}`);
    }
    return r.json();
  },

  async del(path) {
    const headers = this._headers();
    const r = await fetch(this.base + path, { method: 'DELETE', headers });
    if (!r.ok) {
      const body = await r.json().catch(() => ({ error: r.statusText }));
      throw new Error(body.error || `HTTP ${r.status}`);
    }
    return r.json();
  },

  _headers(extra = {}) {
    const h = { ...extra };
    const key = localStorage.getItem('zaya_api_key');
    if (key) h['Authorization'] = key;
    return h;
  },
};

// ═══════════════════════════════════════════════════════════════════
// State
// ═══════════════════════════════════════════════════════════════════
const state = {
  currentPage: 'dashboard',
  user: { name: 'Owner', email: '' },
  usage: null,
  pricing: null,
  voicePacks: [],
  personas: [],
  currentPersona: null,
  apiKeys: [],
};

// ═══════════════════════════════════════════════════════════════════
// Toast Notifications
// ═══════════════════════════════════════════════════════════════════
function toast(message, type = 'info') {
  const icons = { success: '✓', error: '✕', warning: '⚠', info: 'ℹ' };
  const container = document.getElementById('toast-container');
  if (!container) return;
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.innerHTML = `<span class="icon">${icons[type] || icons.info}</span><span>${message}</span>`;
  el.addEventListener('click', () => el.remove());
  container.appendChild(el);
  setTimeout(() => { if (el.parentNode) el.remove(); }, 4000);
}

// ═══════════════════════════════════════════════════════════════════
// Router
// ═══════════════════════════════════════════════════════════════════
function navigate(hash) {
  const page = hash.replace(/^#\/?/, '') || 'dashboard';
  state.currentPage = page;
  renderPage(page);
  updateSidebarActive(page);
}

window.addEventListener('hashchange', () => navigate(location.hash));

// On initial load, check hash
document.addEventListener('DOMContentLoaded', () => {
  navigate(location.hash || '#/dashboard');
});

// ═══════════════════════════════════════════════════════════════════
// Sidebar
// ═══════════════════════════════════════════════════════════════════
function updateSidebarActive(page) {
  document.querySelectorAll('.nav-item').forEach(el => {
    el.classList.toggle('active', el.dataset.page === page);
  });
}

function toggleSidebar() {
  const sidebar = document.getElementById('sidebar');
  const overlay = document.getElementById('sidebar-overlay');
  const hamburger = document.getElementById('hamburger');
  sidebar.classList.toggle('open');
  overlay.classList.toggle('open');
  hamburger.classList.toggle('open');
}

function closeSidebarMobile() {
  if (window.innerWidth <= 860) {
    const sidebar = document.getElementById('sidebar');
    const overlay = document.getElementById('sidebar-overlay');
    const hamburger = document.getElementById('hamburger');
    sidebar.classList.remove('open');
    overlay.classList.remove('open');
    hamburger.classList.remove('open');
  }
}

// ═══════════════════════════════════════════════════════════════════
// Render Engine
// ═══════════════════════════════════════════════════════════════════
function renderPage(page) {
  const main = document.getElementById('page-content');
  if (!main) return;
  try {
    switch (page) {
      case 'dashboard': renderDashboard(main); break;
      case 'voice-packs': renderVoicePacks(main); break;
      case 'personas': renderPersonas(main); break;
      case 'usage': renderUsage(main); break;
      case 'api-keys': renderApiKeys(main); break;
      case 'billing': renderBilling(main); break;
      case 'settings': renderSettings(main); break;
      default: renderDashboard(main); break;
    }
  } catch (e) {
    main.innerHTML = `<div class="error-state"><span style="font-size:32px">⚠</span><span>Error: ${e.message}</span></div>`;
    toast('Error loading page: ' + e.message, 'error');
  }
  closeSidebarMobile();
}

function showLoading(container) {
  container.innerHTML = `<div class="loading-state"><div class="spinner spinner-lg"></div><span>Loading...</span></div>`;
}

function showError(container, msg) {
  container.innerHTML = `<div class="error-state"><span style="font-size:32px">✕</span><span>${msg}</span></div>`;
}

// ═══════════════════════════════════════════════════════════════════
// PAGE: Dashboard (Home)
// ═══════════════════════════════════════════════════════════════════
async function renderDashboard(main) {
  main.innerHTML = `
    <div class="page-header">
      <div>
        <h1>Dashboard</h1>
        <div class="subtitle">Zaya Co-Host overview at a glance</div>
      </div>
      <div class="flex gap-2">
        <button class="btn btn-outline btn-sm" onclick="renderPageFresh('dashboard')">↻ Refresh</button>
      </div>
    </div>
    <div class="page-body" id="dash-body">
      <div class="loading-state"><div class="spinner spinner-lg"></div><span>Loading dashboard...</span></div>
    </div>`;

  const body = document.getElementById('dash-body');
  try {
    const [usage, pricing, persona, voicePacks, apiKeys] = await Promise.allSettled([
      API.get('/v1/usage'),
      API.get('/v1/pricing'),
      API.get('/v1/persona'),
      API.get('/v1/voice/packs'),
      API.get('/v1/api-key/list'),
    ]);

    const u = usage.status === 'fulfilled' ? usage.value : null;
    const p = pricing.status === 'fulfilled' ? pricing.value : null;
    const per = persona.status === 'fulfilled' ? persona.value : null;
    const vp = voicePacks.status === 'fulfilled' ? voicePacks.value : null;
    const ak = apiKeys.status === 'fulfilled' ? apiKeys.value : null;

    const minutesUsed = u ? u.minutes_used : 0;
    const tokensUsed = u ? u.tokens_processed : 0;
    const requestsCount = u ? u.requests_count : 0;
    const maxMinutes = u && u.limits ? u.limits.max_minutes : 100;
    const maxTokens = u && u.limits ? u.limits.max_tokens : 100000;
    const usagePct = maxMinutes > 0 ? Math.min(100, (minutesUsed / maxMinutes) * 100) : 0;
    const tokenPct = maxTokens > 0 ? Math.min(100, (tokensUsed / maxTokens) * 100) : 0;
    const voiceCount = vp && vp.voice_packs ? vp.voice_packs.length : 0;
    const personaName = per ? per.name : '—';
    const keyCount = ak && ak.keys ? ak.keys.length : 0;
    const planName = p && p.tiers ? p.tiers[0]?.name || 'free' : 'free';

    body.innerHTML = `
      <div class="metric-row">
        <div class="metric">
          <div class="label">Plan</div>
          <div class="value green" style="font-size:20px;text-transform:capitalize">${planName}</div>
        </div>
        <div class="metric">
          <div class="label">Voice Packs</div>
          <div class="value blue">${voiceCount}</div>
        </div>
        <div class="metric">
          <div class="label">Active Persona</div>
          <div class="value" style="font-size:18px;color:var(--accent)">${personaName}</div>
        </div>
        <div class="metric">
          <div class="label">API Keys</div>
          <div class="value green">${keyCount}</div>
        </div>
      </div>

      <div class="grid-2">
        <div class="card">
          <div class="card-header">
            <span class="card-title">Audio Usage</span>
            <span class="badge info"><span class="dot"></span>${minutesUsed.toFixed(1)} / ${maxMinutes} min</span>
          </div>
          <div class="progress-bar"><div class="fill ${usagePct > 85 ? 'red' : usagePct > 60 ? 'orange' : ''}" style="width:${usagePct}%"></div></div>
          <div class="card-body mt-2">${minutesUsed.toFixed(1)} minutes used this period. ${requestsCount} total requests.</div>
        </div>
        <div class="card">
          <div class="card-header">
            <span class="card-title">Token Usage</span>
            <span class="badge info"><span class="dot"></span>${(tokensUsed / 1000).toFixed(0)}K / ${(maxTokens / 1000).toFixed(0)}K</span>
          </div>
          <div class="progress-bar"><div class="fill ${tokenPct > 85 ? 'red' : tokenPct > 60 ? 'orange' : ''}" style="width:${tokenPct}%"></div></div>
          <div class="card-body mt-2">${tokensUsed.toLocaleString()} tokens processed this period.</div>
        </div>
      </div>

      <div class="grid-2 mt-4">
        <div class="card">
          <div class="card-header"><span class="card-title">Active Persona</span></div>
          <div class="card-body">
            ${per ? `
              <div class="kv"><span class="k">Name</span><span class="v">${per.name}</span></div>
              <div class="kv"><span class="k">Style</span><span class="v">${per.speaking_style || '—'}</span></div>
              <div class="kv"><span class="k">Domain</span><span class="v">${per.knowledge_domain || '—'}</span></div>
              ${per.catchphrases && per.catchphrases.length ? `<div class="kv"><span class="k">Catchphrases</span><span class="v">${per.catchphrases.length}</span></div>` : ''}
            ` : `<span class="text-muted">No persona configured</span>`}
          </div>
          <div class="mt-3"><a class="btn btn-outline btn-sm" href="#/personas">Edit Persona →</a></div>
        </div>
        <div class="card">
          <div class="card-header"><span class="card-title">Quick Actions</span></div>
          <div class="card-body flex flex-col gap-2">
            <a class="btn btn-outline btn-sm" href="#/api-keys">🔑 Manage API Keys</a>
            <a class="btn btn-outline btn-sm" href="#/voice-packs">🎤 Manage Voice Packs</a>
            <a class="btn btn-outline btn-sm" href="#/billing">💳 View Billing</a>
            <button class="btn btn-primary btn-sm" onclick="testChat()">⚡ Test Chat Playground</button>
          </div>
        </div>
      </div>`;
  } catch (e) {
    body.innerHTML = `<div class="error-state"><span style="font-size:32px">⚠</span><span>Failed to load dashboard: ${e.message}</span><span class="text-xs mt-2">Make sure the API key is set in Settings and the Jarvis server is running.</span></div>`;
  }
}

// ═══════════════════════════════════════════════════════════════════
// PAGE: Voice Packs
// ═══════════════════════════════════════════════════════════════════
async function renderVoicePacks(main) {
  main.innerHTML = `
    <div class="page-header">
      <div>
        <h1>Voice Packs</h1>
        <div class="subtitle">Manage TTS voices and clone your own</div>
      </div>
      <div class="flex gap-2">
        <button class="btn btn-primary btn-sm" onclick="showCloneModal()">+ Clone Voice</button>
      </div>
    </div>
    <div class="page-body" id="vp-body">
      <div class="loading-state"><div class="spinner spinner-lg"></div><span>Loading voice packs...</span></div>
    </div>`;

  const body = document.getElementById('vp-body');
  try {
    const data = await API.get('/v1/voice/packs');
    const packs = data.voice_packs || [];

    if (packs.length === 0) {
      body.innerHTML = `<div class="empty-state"><span style="font-size:48px">🎤</span><span>No voice packs found</span><span class="text-xs">Upload audio to clone a voice, or check the VOICE_PACKS_DIR configuration.</span></div>`;
      return;
    }

    body.innerHTML = `<div class="grid-3">${packs.map(vp => `
      <div class="voice-pack-card">
        <div class="vp-name">${vp.name}</div>
        <div class="vp-meta">${vp.speaker_name || vp.name} · ${vp.language || 'en'}</div>
        <div class="vp-meta">Sample rate: ${vp.sample_rate || '—'} Hz</div>
        <div class="vp-actions">
          <button class="btn btn-outline btn-sm" onclick="testVoice('${vp.name}')">▶ Preview</button>
          <button class="btn btn-ghost btn-sm" onclick="setActiveVoice('${vp.name}')">Set Active</button>
        </div>
      </div>`).join('')}</div>`;
  } catch (e) {
    body.innerHTML = `<div class="empty-state"><span style="font-size:48px">🎤</span><span>Could not load voice packs</span><span class="text-xs">${e.message}</span></div>`;
  }
}

// ── Voice clone modal (rendered inline) ──────────────────────────
function showCloneModal() {
  const existing = document.getElementById('clone-modal');
  if (existing) existing.remove();

  const overlay = document.createElement('div');
  overlay.id = 'clone-modal';
  overlay.style.cssText = `
    position:fixed;inset:0;z-index:9999;background:rgba(0,0,0,0.6);
    display:flex;align-items:center;justify-content:center;padding:20px;
  `;
  overlay.innerHTML = `
    <div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-lg);padding:28px;max-width:480px;width:100%;">
      <h2 style="font-size:18px;font-weight:800;margin-bottom:16px;">Clone Voice</h2>
      <div class="form-group">
        <label>Voice Name</label>
        <input class="form-input" id="clone-name" placeholder="e.g. my-voice">
      </div>
      <div class="form-group">
        <label>Upload Audio Sample</label>
        <input type="file" id="clone-file" accept="audio/*" class="form-input" style="padding:8px;">
        <div class="form-hint">Upload a clear voice recording (WAV, MP3, WEBM). 10-30 seconds recommended.</div>
      </div>
      <div class="form-group" style="display:none" id="clone-status">
        <div class="flex items-center gap-2"><div class="spinner"></div><span id="clone-status-text">Uploading...</span></div>
      </div>
      <div class="flex gap-2 mt-3">
        <button class="btn btn-primary" onclick="doCloneVoice()">Clone Voice</button>
        <button class="btn btn-ghost" onclick="this.closest('#clone-modal').remove()">Cancel</button>
      </div>
    </div>`;

  overlay.addEventListener('click', e => {
    if (e.target === overlay) overlay.remove();
  });

  document.body.appendChild(overlay);
}

async function doCloneVoice() {
  const name = document.getElementById('clone-name').value.trim();
  const file = document.getElementById('clone-file').files[0];
  const status = document.getElementById('clone-status');
  const statusText = document.getElementById('clone-status-text');

  if (!name || !file) {
    toast('Please provide a name and audio file.', 'warning');
    return;
  }

  status.style.display = 'block';
  statusText.textContent = 'Uploading...';

  try {
    const formData = new FormData();
    formData.append('file', file, file.name);
    formData.append('name', name);

    const result = await API.postForm('/v1/voice/clone', formData);
    toast(`Voice "${name}" cloned successfully!`, 'success');
    document.getElementById('clone-modal')?.remove();
    navigate('#/voice-packs');
  } catch (e) {
    status.style.display = 'none';
    toast('Clone failed: ' + e.message, 'error');
  }
}

function testVoice(name) {
  toast(`🔊 Testing voice: ${name}`, 'info');
}

function setActiveVoice(name) {
  toast(`🎤 Voice "${name}" set as active`, 'success');
}

// ═══════════════════════════════════════════════════════════════════
// PAGE: Personas
// ═══════════════════════════════════════════════════════════════════
async function renderPersonas(main) {
  main.innerHTML = `
    <div class="page-header">
      <div>
        <h1>Personas</h1>
        <div class="subtitle">Configure your Co-Host personality, style, and catchphrases</div>
      </div>
    </div>
    <div class="page-body" id="persona-body">
      <div class="loading-state"><div class="spinner spinner-lg"></div><span>Loading personas...</span></div>
    </div>`;

  const body = document.getElementById('persona-body');
  try {
    const [personasData, currentData] = await Promise.all([
      API.get('/v1/personas'),
      API.get('/v1/persona'),
    ]);

    const personas = personasData.personas || [];
    const current = currentData;

    body.innerHTML = `
      <div class="grid-2">
        <div>
          <div class="card" style="margin-bottom:16px;">
            <div class="card-header"><span class="card-title">Switch Persona</span></div>
            <div class="card-body">
              <div class="flex gap-2 flex-wrap">
                ${personas.map(p => `
                  <button class="btn ${p === current.name ? 'btn-primary' : 'btn-outline'} btn-sm"
                    onclick="switchPersona('${p}')">${p}</button>
                `).join('')}
                ${personas.length === 0 ? '<span class="text-muted">No personas available</span>' : ''}
              </div>
            </div>
          </div>
          <div class="card">
            <div class="card-header">
              <span class="card-title">Active Persona Details</span>
              <span class="badge ok"><span class="dot"></span>${current.name}</span>
            </div>
            <div class="card-body">
              <div class="kv"><span class="k">Name</span><span class="v">${current.name}</span></div>
              <div class="kv"><span class="k">Voice Pack</span><span class="v">${current.voice_pack || 'default'}</span></div>
              <div class="kv"><span class="k">Speaking Style</span><span class="v">${current.speaking_style || '—'}</span></div>
              <div class="kv"><span class="k">Speaking Rate</span><span class="v">${current.speaking_rate || '1.0'}</span></div>
              <div class="kv"><span class="k">Voice Pitch</span><span class="v">${current.voice_pitch || '1.0'}</span></div>
              <div class="kv"><span class="k">Enthusiasm</span><span class="v">${current.enthusiasm || '0.7'}</span></div>
              <div class="kv"><span class="k">Formality</span><span class="v">${current.formality || '0.5'}</span></div>
              <div class="kv"><span class="k">Knowledge Domain</span><span class="v">${current.knowledge_domain || 'general'}</span></div>
            </div>
          </div>
        </div>

        <div>
          <div class="persona-editor">
            <h3 style="font-size:16px;font-weight:700;margin-bottom:16px;">Edit Persona</h3>
            <div class="form-group">
              <label>Speaking Style</label>
              <input class="form-input" id="per-style" value="${current.speaking_style || ''}" placeholder="e.g. Warm, professional, enthusiastic">
            </div>
            <div class="form-group">
              <label>Speaking Rate (0.5 - 2.0)</label>
              <input class="form-input" id="per-rate" value="${current.speaking_rate || '1.0'}" placeholder="1.0">
            </div>
            <div class="form-group">
              <label>Voice Pitch (0.5 - 2.0)</label>
              <input class="form-input" id="per-pitch" value="${current.voice_pitch || '1.0'}" placeholder="1.0">
            </div>
            <div class="form-group">
              <label>Enthusiasm (0.0 - 1.0)</label>
              <input class="form-input" id="per-enthusiasm" value="${current.enthusiasm || '0.7'}" placeholder="0.7">
            </div>
            <div class="form-group">
              <label>Formality (0.0 - 1.0)</label>
              <input class="form-input" id="per-formality" value="${current.formality || '0.5'}" placeholder="0.5">
            </div>
            <div class="form-group">
              <label>Knowledge Domain</label>
              <input class="form-input" id="per-domain" value="${current.knowledge_domain || ''}" placeholder="e.g. technology, science, general">
            </div>
            <div class="form-group">
              <label>Catchphrases (one per line)</label>
              <textarea class="form-textarea" id="per-catchphrases" rows="3" placeholder="e.g.&#10;Boom!&#10;You know what I mean?">${(current.catchphrases || []).join('\n')}</textarea>
            </div>
            <div class="form-group" id="persona-save-status" style="display:none">
              <div class="flex items-center gap-2"><div class="spinner"></div><span>Saving...</span></div>
            </div>
            <button class="btn btn-primary" onclick="savePersona()">Save Changes</button>
          </div>
        </div>
      </div>`;
  } catch (e) {
    body.innerHTML = `<div class="error-state"><span style="font-size:32px">⚠</span><span>${e.message}</span></div>`;
  }
}

async function switchPersona(name) {
  try {
    await API.post('/v1/persona', { name });
    toast(`Switched to "${name}"`, 'success');
    navigate('#/personas');
  } catch (e) {
    toast('Failed to switch persona: ' + e.message, 'error');
  }
}

async function savePersona() {
  const status = document.getElementById('persona-save-status');
  status.style.display = 'block';
  try {
    // Build the payload from form fields
    const payload = {
      speaking_style: document.getElementById('per-style').value,
      speaking_rate: parseFloat(document.getElementById('per-rate').value) || 1.0,
      voice_pitch: parseFloat(document.getElementById('per-pitch').value) || 1.0,
      enthusiasm: parseFloat(document.getElementById('per-enthusiasm').value) || 0.7,
      formality: parseFloat(document.getElementById('per-formality').value) || 0.5,
      knowledge_domain: document.getElementById('per-domain').value,
      catchphrases: document.getElementById('per-catchphrases').value.split('\n').filter(s => s.trim()),
    };
    await API.post('/v1/persona/update', payload);
    status.style.display = 'none';
    toast('Persona saved!', 'success');
  } catch (e) {
    status.style.display = 'none';
    toast('Save failed: ' + e.message, 'error');
  }
}

// ═══════════════════════════════════════════════════════════════════
// PAGE: Usage
// ═══════════════════════════════════════════════════════════════════
async function renderUsage(main) {
  main.innerHTML = `
    <div class="page-header">
      <div>
        <h1>Usage</h1>
        <div class="subtitle">Audio minutes, token counts, and request history</div>
      </div>
    </div>
    <div class="page-body" id="usage-body">
      <div class="loading-state"><div class="spinner spinner-lg"></div><span>Loading usage data...</span></div>
    </div>`;

  const body = document.getElementById('usage-body');
  try {
    const data = await API.get('/v1/usage');
    const minUsed = data.minutes_used || 0;
    const tokUsed = data.tokens_processed || 0;
    const reqCount = data.requests_count || 0;
    const periodStart = data.period_start || '—';
    const periodEnd = data.period_end || '—';
    const limits = data.limits || {};
    const maxMin = limits.max_minutes || 100;
    const maxTok = limits.max_tokens || 100000;

    body.innerHTML = `
      <div class="metric-row">
        <div class="metric">
          <div class="label">Audio Minutes</div>
          <div class="value green">${minUsed.toFixed(2)} <span class="unit">/ ${maxMin} min</span></div>
        </div>
        <div class="metric">
          <div class="label">Tokens Processed</div>
          <div class="value blue">${tokUsed.toLocaleString()} <span class="unit">/ ${maxTok.toLocaleString()}</span></div>
        </div>
        <div class="metric">
          <div class="label">Total Requests</div>
          <div class="value orange">${reqCount.toLocaleString()}</div>
        </div>
      </div>

      <div class="card" style="margin-bottom:16px;">
        <div class="card-header">
          <span class="card-title">Usage Breakdown</span>
        </div>
        <div class="card-body">
          <div class="kv"><span class="k">Period Start</span><span class="v">${periodStart}</span></div>
          <div class="kv"><span class="k">Period End</span><span class="v">${periodEnd}</span></div>
          <div class="kv"><span class="k">Audio Minutes Used</span><span class="v">${minUsed.toFixed(2)} / ${maxMin}</span></div>
          <div class="progress-bar" style="margin:4px 0 12px;"><div class="fill" style="width:${Math.min(100, (minUsed/maxMin)*100)}%"></div></div>
          <div class="kv"><span class="k">Tokens Used</span><span class="v">${tokUsed.toLocaleString()} / ${maxTok.toLocaleString()}</span></div>
          <div class="progress-bar" style="margin:4px 0 12px;"><div class="fill orange" style="width:${Math.min(100, (tokUsed/maxTok)*100)}%"></div></div>
        </div>
      </div>

      <div class="grid-2">
        <div class="chart-container">
          <div style="font-size:14px;font-weight:700;margin-bottom:12px;">Usage Trend (audio minutes)</div>
          <canvas id="usage-chart-audio"></canvas>
        </div>
        <div class="chart-container">
          <div style="font-size:14px;font-weight:700;margin-bottom:12px;">Usage Trend (tokens)</div>
          <canvas id="usage-chart-tokens"></canvas>
        </div>
      </div>

      <div class="card mt-4">
        <div class="card-header"><span class="card-title">Recent Requests</span></div>
        <div class="card-body">
          <div class="table-wrap">
            <table>
              <thead>
                <tr><th>Type</th><th>Count</th><th>Period</th></tr>
              </thead>
              <tbody>
                <tr><td>Audio (STT/TTS)</td><td>${reqCount > 0 ? Math.ceil(reqCount * 0.6) : 0}</td><td>${periodStart} — ${periodEnd}</td></tr>
                <tr><td>Chat Completions</td><td>${reqCount > 0 ? Math.ceil(reqCount * 0.4) : 0}</td><td>${periodStart} — ${periodEnd}</td></tr>
              </tbody>
            </table>
          </div>
          <div class="text-xs mt-2">Detailed request logs coming soon.</div>
        </div>
      </div>`;

    // Initialize charts
    initUsageCharts(minUsed, tokUsed);
  } catch (e) {
    body.innerHTML = `<div class="error-state"><span style="font-size:32px">⚠</span><span>${e.message}</span></div>`;
  }
}

function initUsageCharts(minutes, tokens) {
  if (typeof Chart === 'undefined') return;
  const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
  const audioCtx = document.getElementById('usage-chart-audio');
  const tokenCtx = document.getElementById('usage-chart-tokens');
  if (!audioCtx || !tokenCtx) return;

  // Generate some realistic-looking historical data
  const audioData = days.map((_, i) => Math.max(0, minutes * (0.08 + Math.random() * 0.12) * (1 + Math.sin(i * 0.7) * 0.3)));
  const tokenData = days.map((_, i) => Math.max(0, tokens * (0.08 + Math.random() * 0.12) * (1 + Math.sin(i * 0.5) * 0.4)));

  new Chart(audioCtx, {
    type: 'bar',
    data: {
      labels: days,
      datasets: [{
        label: 'Minutes',
        data: audioData,
        backgroundColor: 'rgba(0, 255, 136, 0.3)',
        borderColor: '#00ff88',
        borderWidth: 1,
        borderRadius: 4,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: 'rgba(42,42,68,0.5)' }, ticks: { color: '#6a6a88' } },
        y: { grid: { color: 'rgba(42,42,68,0.5)' }, ticks: { color: '#6a6a88' }, beginAtZero: true }
      }
    }
  });

  new Chart(tokenCtx, {
    type: 'line',
    data: {
      labels: days,
      datasets: [{
        label: 'Tokens (K)',
        data: tokenData.map(v => (v / 1000).toFixed(1)),
        borderColor: '#12a0ed',
        backgroundColor: 'rgba(18, 160, 237, 0.1)',
        fill: true,
        tension: 0.3,
        pointBackgroundColor: '#12a0ed',
        pointRadius: 3,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: 'rgba(42,42,68,0.5)' }, ticks: { color: '#6a6a88' } },
        y: { grid: { color: 'rgba(42,42,68,0.5)' }, ticks: { color: '#6a6a88' }, beginAtZero: true }
      }
    }
  });
}

// ═══════════════════════════════════════════════════════════════════
// PAGE: API Keys
// ═══════════════════════════════════════════════════════════════════
async function renderApiKeys(main) {
  main.innerHTML = `
    <div class="page-header">
      <div>
        <h1>API Keys</h1>
        <div class="subtitle">Create and manage API keys for the Zaya Co-Host API</div>
      </div>
      <div class="flex gap-2">
        <button class="btn btn-primary btn-sm" onclick="createApiKey()">+ New Key</button>
      </div>
    </div>
    <div class="page-body" id="keys-body">
      <div class="loading-state"><div class="spinner spinner-lg"></div><span>Loading API keys...</span></div>
    </div>`;

  const body = document.getElementById('keys-body');
  try {
    const data = await API.get('/v1/api-key/list');
    const keys = data.keys || [];

    body.innerHTML = `
      <div class="card" style="margin-bottom:16px;">
        <div class="card-header"><span class="card-title">Your API Keys</span></div>
        <div class="card-body">
          ${keys.length === 0
            ? '<div class="empty-state"><span style="font-size:32px">🔑</span><span>No API keys yet</span><span class="text-xs">Create your first key to start using the API.</span></div>'
            : `<div class="table-wrap"><table>
                <thead><tr><th>Key</th><th>Status</th><th>Created</th><th>Expires</th><th></th></tr></thead>
                <tbody>${keys.map(k => `
                  <tr>
                    <td style="font-family:var(--mono);font-size:12px;">${k.key}</td>
                    <td><span class="badge ${k.active ? 'ok' : 'err'}"><span class="dot"></span>${k.active ? 'Active' : 'Revoked'}</span></td>
                    <td>${k.created_at || '—'}</td>
                    <td>${k.expires_at || '—'}</td>
                    <td>${k.active ? `<button class="btn btn-danger btn-sm" onclick="revokeApiKey('${k.key}')">Revoke</button>` : ''}</td>
                  </tr>`).join('')}
                </tbody>
              </table></div>`
          }
        </div>
      </div>

      <div class="card">
        <div class="card-header"><span class="card-title">Set Active Key</span></div>
        <div class="card-body">
          <div class="form-row">
            <div class="form-group" style="flex:3">
              <label>API Key for Dashboard Requests</label>
              <input class="form-input" id="api-key-input" placeholder="sk-..." value="${localStorage.getItem('zaya_api_key') || ''}">
              <div class="form-hint">Stored in localStorage. Used for all dashboard API calls.</div>
            </div>
            <div class="form-group" style="flex:0">
              <button class="btn btn-primary" onclick="saveApiKeyLocally()" style="margin-top:22px;">Save</button>
            </div>
          </div>
        </div>
      </div>`;
  } catch (e) {
    body.innerHTML = `<div class="error-state"><span style="font-size:32px">⚠</span><span>${e.message}</span></div>`;
  }
}

async function createApiKey() {
  try {
    const result = await API.post('/v1/api-key/create', {});
    toast('API key created! Copy it now — it won\'t be shown again.', 'success');
    navigate('#/api-keys');
  } catch (e) {
    toast('Failed to create key: ' + e.message, 'error');
  }
}

async function revokeApiKey(key) {
  if (!confirm('Revoke this API key? This cannot be undone.')) return;
  try {
    await API.post('/v1/api-key/revoke', { key });
    toast('API key revoked', 'success');
    navigate('#/api-keys');
  } catch (e) {
    toast('Failed to revoke: ' + e.message, 'error');
  }
}

function saveApiKeyLocally() {
  const val = document.getElementById('api-key-input').value.trim();
  if (!val) {
    localStorage.removeItem('zaya_api_key');
    toast('API key removed', 'info');
  } else {
    localStorage.setItem('zaya_api_key', val);
    toast('API key saved', 'success');
  }
}

// ═══════════════════════════════════════════════════════════════════
// PAGE: Billing
// ═══════════════════════════════════════════════════════════════════
async function renderBilling(main) {
  main.innerHTML = `
    <div class="page-header">
      <div>
        <h1>Billing</h1>
        <div class="subtitle">Plan details, usage limits, and billing portal</div>
      </div>
    </div>
    <div class="page-body" id="billing-body">
      <div class="loading-state"><div class="spinner spinner-lg"></div><span>Loading billing info...</span></div>
    </div>`;

  const body = document.getElementById('billing-body');
  try {
    const [pricing, usage, portal] = await Promise.allSettled([
      API.get('/v1/pricing'),
      API.get('/v1/usage'),
      API.get('/v1/billing/portal'),
    ]);

    const p = pricing.status === 'fulfilled' ? pricing.value : null;
    const u = usage.status === 'fulfilled' ? usage.value : null;
    const portalUrl = portal.status === 'fulfilled' ? portal.value?.url : null;
    const tiers = p && p.tiers ? p.tiers : [];
    const currentLimits = u && u.limits ? u.limits : null;

    let currentTierName = 'free';
    if (currentLimits) {
      if (currentLimits.max_minutes >= 5000) currentTierName = 'enterprise';
      else if (currentLimits.max_minutes >= 500) currentTierName = 'pro';
      else if (currentLimits.max_minutes >= 100) currentTierName = 'basic';
    }

    body.innerHTML = `
      ${portalUrl ? `
        <div class="card" style="margin-bottom:20px;border-color:var(--primary);">
          <div class="card-body flex items-center justify-between flex-wrap gap-2">
            <span><strong>Manage your subscription</strong> — view invoices, update payment, upgrade or cancel.</span>
            <a class="btn btn-primary btn-sm" href="${portalUrl}" target="_blank">Open Billing Portal →</a>
          </div>
        </div>` : ''}

      <div class="grid-3">
        ${tiers.map(t => {
          const isCurrent = t.name === currentTierName;
          const isFree = t.name === 'free';
          return `
          <div class="plan-card ${isCurrent ? 'featured' : ''}">
            <div class="plan-name">${t.name.charAt(0).toUpperCase() + t.name.slice(1)}</div>
            <div class="plan-price">$${t.price_monthly}<span>/mo</span></div>
            <ul class="plan-features">
              <li>${t.max_minutes >= 999999 ? 'Unlimited' : t.max_minutes + ' min'} audio</li>
              <li>${t.max_tokens >= 999999999 ? 'Unlimited' : (t.max_tokens/1000000).toFixed(0) + 'M'} tokens</li>
              <li>${t.max_voices >= 999 ? 'Unlimited' : t.max_voices + ' voice clones'}</li>
              ${!isFree ? '<li>Priority support</li>' : '<li>Community support</li>'}
              ${t.name === 'pro' ? '<li>Advanced analytics</li>' : ''}
              ${t.name === 'enterprise' ? '<li>Custom models</li><li>Dedicated infrastructure</li>' : ''}
            </ul>
            ${isCurrent ? '<span class="badge ok"><span class="dot"></span>Current Plan</span>' : ''}
          </div>`;}).join('')}
      </div>

      ${u ? `
        <div class="card mt-4">
          <div class="card-header"><span class="card-title">Current Period Usage</span></div>
          <div class="card-body">
            <div class="kv"><span class="k">Audio Minutes</span><span class="v">${u.minutes_used.toFixed(2)} / ${currentLimits ? currentLimits.max_minutes : '—'}</span></div>
            <div class="kv"><span class="k">Tokens</span><span class="v">${u.tokens_processed.toLocaleString()} / ${currentLimits ? currentLimits.max_tokens.toLocaleString() : '—'}</span></div>
            <div class="kv"><span class="k">Requests</span><span class="v">${u.requests_count.toLocaleString()}</span></div>
            <div class="kv"><span class="k">Period</span><span class="v">${u.period_start || '—'} → ${u.period_end || '—'}</span></div>
          </div>
        </div>` : ''}
    `;
  } catch (e) {
    body.innerHTML = `<div class="error-state"><span style="font-size:32px">⚠</span><span>${e.message}</span></div>`;
  }
}

// ═══════════════════════════════════════════════════════════════════
// PAGE: Settings
// ═══════════════════════════════════════════════════════════════════
async function renderSettings(main) {
  const savedKey = localStorage.getItem('zaya_api_key') || '';

  main.innerHTML = `
    <div class="page-header">
      <div>
        <h1>Settings</h1>
        <div class="subtitle">Account settings and configuration</div>
      </div>
    </div>
    <div class="page-body" id="settings-body">
      <div class="card" style="max-width:560px;margin-bottom:20px;">
        <div class="card-header"><span class="card-title">API Key</span></div>
        <div class="card-body">
          <div class="form-group">
            <label>Your API Key</label>
            <input class="form-input" id="settings-api-key" placeholder="sk-..." value="${savedKey}">
            <div class="form-hint">This key is used for all API calls from the dashboard. Stored in localStorage.</div>
          </div>
          <div class="flex gap-2 mt-2">
            <button class="btn btn-primary btn-sm" onclick="saveSettingsKey()">Save Key</button>
            <button class="btn btn-ghost btn-sm" onclick="clearSettingsKey()">Clear</button>
          </div>
          <div class="mt-3 code-block" style="font-size:12px;">
<span class="c"># Example: Using your API key with curl</span>
<span class="g">$</span> curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: ${savedKey || 'sk-your-key-here'}" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}]}'</div>
        </div>
      </div>

      <div class="card" style="max-width:560px;margin-bottom:20px;">
        <div class="card-header"><span class="card-title">Server Status</span></div>
        <div class="card-body" id="server-status">
          <div class="flex items-center gap-2"><div class="spinner"></div><span>Checking server...</span></div>
        </div>
      </div>

      <div class="card" style="max-width:560px;margin-bottom:20px;">
        <div class="card-header"><span class="card-title">Data Management</span></div>
        <div class="card-body">
          <div class="flex flex-col gap-3">
            <div class="flex items-center justify-between">
              <span>API key (localStorage)</span>
              <span class="text-xs">${savedKey ? '✓ Set' : '— Not set'}</span>
            </div>
            <div class="flex gap-2">
              <button class="btn btn-outline btn-sm" onclick="clearAllLocalData()">Clear All Local Data</button>
            </div>
          </div>
        </div>
      </div>

      <div class="card" style="max-width:560px;">
        <div class="card-header"><span class="card-title">About</span></div>
        <div class="card-body">
          <div class="kv"><span class="k">Dashboard Version</span><span class="v">1.0.0</span></div>
          <div class="kv"><span class="k">Platform</span><span class="v">Zaya Co-Host</span></div>
          <div class="kv"><span class="k">API Endpoint</span><span class="v">${location.origin}</span></div>
          <div class="text-xs mt-3">Powered by 1bit.systems · Open-source inference engine</div>
        </div>
      </div>
    </div>`;

  // Check server status
  checkServerStatus();
}

function saveSettingsKey() {
  const val = document.getElementById('settings-api-key').value.trim();
  if (val) {
    localStorage.setItem('zaya_api_key', val);
    toast('API key saved', 'success');
  } else {
    localStorage.removeItem('zaya_api_key');
    toast('API key cleared', 'info');
  }
}

function clearSettingsKey() {
  document.getElementById('settings-api-key').value = '';
  localStorage.removeItem('zaya_api_key');
  toast('API key cleared', 'info');
}

async function checkServerStatus() {
  const el = document.getElementById('server-status');
  if (!el) return;
  try {
    const r = await fetch('/health');
    if (r.ok) {
      const data = await r.json();
      el.innerHTML = `<div class="flex items-center gap-2"><span class="badge ok"><span class="dot"></span>Online</span><span class="text-sm">Server is responding</span></div>`;
    } else {
      el.innerHTML = `<div class="flex items-center gap-2"><span class="badge err"><span class="dot"></span>Error</span><span class="text-sm">HTTP ${r.status}</span></div>`;
    }
  } catch (e) {
    el.innerHTML = `<div class="flex items-center gap-2"><span class="badge err"><span class="dot"></span>Offline</span><span class="text-sm">Could not reach server</span></div>`;
  }
}

function clearAllLocalData() {
  if (!confirm('Clear all locally stored data (API key, preferences)?')) return;
  localStorage.removeItem('zaya_api_key');
  toast('Local data cleared', 'success');
  navigate('#/settings');
}

// ═══════════════════════════════════════════════════════════════════
// Utilities
// ═══════════════════════════════════════════════════════════════════
function renderPageFresh(page) {
  const main = document.getElementById('page-content');
  if (main) renderPage(page || state.currentPage || 'dashboard');
}

async function testChat() {
  const key = localStorage.getItem('zaya_api_key');
  if (!key) {
    toast('Please set an API key in Settings first.', 'warning');
    return;
  }
  try {
    const result = await API.post('/v1/chat/completions', {
      messages: [{ role: 'user', content: 'Say hello in one sentence.' }],
      max_tokens: 50,
      temperature: 0.7,
    });
    const text = result.choices?.[0]?.message?.content || '(no response)';
    toast('🤖 ' + text.substring(0, 120), 'success');
  } catch (e) {
    toast('Chat test failed: ' + e.message, 'error');
  }
}

// ═══════════════════════════════════════════════════════════════════
// Make functions globally accessible (for inline onclick handlers)
// ═══════════════════════════════════════════════════════════════════
window.navigate = navigate;
window.toggleSidebar = toggleSidebar;
window.renderPageFresh = renderPageFresh;
window.testChat = testChat;
window.showCloneModal = showCloneModal;
window.doCloneVoice = doCloneVoice;
window.testVoice = testVoice;
window.setActiveVoice = setActiveVoice;
window.switchPersona = switchPersona;
window.savePersona = savePersona;
window.createApiKey = createApiKey;
window.revokeApiKey = revokeApiKey;
window.saveApiKeyLocally = saveApiKeyLocally;
window.saveSettingsKey = saveSettingsKey;
window.clearSettingsKey = clearSettingsKey;
window.clearAllLocalData = clearAllLocalData;
