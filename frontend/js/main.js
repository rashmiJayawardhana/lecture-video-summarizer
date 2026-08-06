/* ═══════════════════════════════════════════════════════════
   INTEGRA — Main Application Script
   ═══════════════════════════════════════════════════════════ */

// Base URL for the backend API, resolved in priority order:
//   1. ?api=<url> query param - lets a Vercel-deployed (static, build-time-baked)
//      frontend point at a fresh Colab tunnel URL each demo session without a
//      rebuild. Persisted to localStorage so it survives a page refresh.
//   2. VITE_API_URL (baked in at build time via frontend/.env) - for local dev
//      against a Colab backend without needing the query param every time.
//   3. '' (relative paths) - the Vite dev proxy (vite.config.js) forwards these
//      to http://localhost:8000 for local dev against a local backend.
const API_BASE = (() => {
  const fromQuery = new URLSearchParams(window.location.search).get('api');
  if (fromQuery) {
    localStorage.setItem('integra-api-base', fromQuery);
    return fromQuery;
  }
  return localStorage.getItem('integra-api-base') || import.meta.env.VITE_API_URL || '';
})();

const STATUS_POLL_INTERVAL_MS = 2000;

// Tracks the most recent job so tab switches can redraw canvas-based charts
// (their width depends on clientWidth, which is 0 while their tab panel is
// hidden, so a chart drawn once while off-screen would otherwise stay blank).
let currentJobId = null;

// ── Theme Toggle ──
function initTheme() {
  const toggle = document.getElementById('theme-toggle');
  const sunIcon = document.getElementById('theme-icon-sun');
  const moonIcon = document.getElementById('theme-icon-moon');
  if (!toggle) return;

  // Check saved preference or system preference
  const saved = localStorage.getItem('integra-theme');
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const theme = saved || (prefersDark ? 'dark' : 'light');
  applyTheme(theme);

  toggle.addEventListener('click', () => {
    const current = document.documentElement.getAttribute('data-theme');
    const next = current === 'light' ? 'dark' : 'light';
    applyTheme(next);
    localStorage.setItem('integra-theme', next);
  });

  function applyTheme(theme) {
    if (theme === 'light') {
      document.documentElement.setAttribute('data-theme', 'light');
      if (sunIcon) sunIcon.style.display = 'none';
      if (moonIcon) moonIcon.style.display = 'block';
    } else {
      document.documentElement.removeAttribute('data-theme');
      if (sunIcon) sunIcon.style.display = 'block';
      if (moonIcon) moonIcon.style.display = 'none';
    }
  }
}

// ── Particle Background ──
function initParticles() {
  const canvas = document.getElementById('particles-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let particles = [];
  const PARTICLE_COUNT = 60;
  const MAX_DIST = 120;

  function resize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  }
  resize();
  window.addEventListener('resize', resize);

  function getParticleColor() {
    const style = getComputedStyle(document.documentElement);
    return {
      r: parseInt(style.getPropertyValue('--particle-r')) || 59,
      g: parseInt(style.getPropertyValue('--particle-g')) || 130,
      b: parseInt(style.getPropertyValue('--particle-b')) || 246,
    };
  }

  class Particle {
    constructor() { this.reset(); }
    reset() {
      this.x = Math.random() * canvas.width;
      this.y = Math.random() * canvas.height;
      this.vx = (Math.random() - 0.5) * 0.4;
      this.vy = (Math.random() - 0.5) * 0.4;
      this.r = Math.random() * 2 + 0.5;
      this.alpha = Math.random() * 0.5 + 0.1;
    }
    update() {
      this.x += this.vx;
      this.y += this.vy;
      if (this.x < 0 || this.x > canvas.width) this.vx *= -1;
      if (this.y < 0 || this.y > canvas.height) this.vy *= -1;
    }
    draw(color) {
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${color.r}, ${color.g}, ${color.b}, ${this.alpha})`;
      ctx.fill();
    }
  }

  for (let i = 0; i < PARTICLE_COUNT; i++) particles.push(new Particle());

  function animate() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const color = getParticleColor();
    particles.forEach(p => { p.update(); p.draw(color); });
    // Draw connections
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < MAX_DIST) {
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = `rgba(${color.r}, ${color.g}, ${color.b}, ${0.08 * (1 - dist / MAX_DIST)})`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }
    }
    requestAnimationFrame(animate);
  }
  animate();
}

// ── Navigation ──
function initNav() {
  const nav = document.getElementById('main-nav');
  const links = document.querySelectorAll('.nav-link');
  const sections = document.querySelectorAll('.section');

  window.addEventListener('scroll', () => {
    nav?.classList.toggle('scrolled', window.scrollY > 50);

    let current = '';
    sections.forEach(sec => {
      const top = sec.offsetTop - 120;
      if (window.scrollY >= top) current = sec.getAttribute('id');
    });
    links.forEach(link => {
      link.classList.remove('active');
      if (link.getAttribute('href') === `#${current}`) link.classList.add('active');
    });
  });

  links.forEach(link => {
    link.addEventListener('click', e => {
      e.preventDefault();
      const target = document.querySelector(link.getAttribute('href'));
      target?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  });
}

// ── Scroll Reveal ──
function initReveal() {
  const targets = document.querySelectorAll(
    '.section-header, .stat-card, .stage, .metric-card, .team-card, .config-panel, .log-panel, .drop-zone'
  );
  targets.forEach(el => el.classList.add('reveal'));

  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry, i) => {
      if (entry.isIntersecting) {
        setTimeout(() => entry.target.classList.add('visible'), i * 80);
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.1, rootMargin: '0px 0px -40px 0px' });

  targets.forEach(el => observer.observe(el));
}

// ── File Upload ──
function initUpload() {
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  const filePreview = document.getElementById('file-preview');
  const previewVideo = document.getElementById('preview-video');
  const previewName = document.getElementById('preview-name');
  const previewSize = document.getElementById('preview-size');
  const previewType = document.getElementById('preview-type');
  const removeBtn = document.getElementById('remove-file-btn');
  const startBtn = document.getElementById('start-pipeline-btn');

  if (!dropZone) return;

  let selectedFile = null;

  dropZone.addEventListener('click', () => fileInput.click());

  ['dragenter', 'dragover'].forEach(evt =>
    dropZone.addEventListener(evt, e => { e.preventDefault(); dropZone.classList.add('drag-over'); })
  );
  ['dragleave', 'drop'].forEach(evt =>
    dropZone.addEventListener(evt, e => { e.preventDefault(); dropZone.classList.remove('drag-over'); })
  );

  dropZone.addEventListener('drop', e => {
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('video/')) handleFile(file);
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) handleFile(fileInput.files[0]);
  });

  function handleFile(file) {
    selectedFile = file;
    previewName.textContent = file.name;
    previewSize.textContent = formatBytes(file.size);
    previewType.textContent = file.type.split('/')[1]?.toUpperCase() || 'VIDEO';

    const url = URL.createObjectURL(file);
    previewVideo.src = url;
    previewVideo.load();

    dropZone.style.display = 'none';
    filePreview.style.display = 'flex';
    startBtn.disabled = false;
    addLog('File selected: ' + file.name, 'info');
  }

  removeBtn?.addEventListener('click', () => {
    selectedFile = null;
    previewVideo.src = '';
    fileInput.value = '';
    dropZone.style.display = '';
    filePreview.style.display = 'none';
    startBtn.disabled = true;
    addLog('File removed.', 'info');
  });

  startBtn?.addEventListener('click', () => {
    if (!selectedFile) return;
    startBtn.disabled = true;
    runRealPipeline(selectedFile);
  });
}

function formatBytes(bytes) {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

// ── Pipeline Configuration Sliders ──
function initSliders() {
  const sliders = [
    { id: 'config-w1', valId: 'config-w1-val' },
    { id: 'config-w2', valId: 'config-w2-val' },
    { id: 'config-w3', valId: 'config-w3-val' },
  ];

  sliders.forEach(({ id, valId }) => {
    const slider = document.getElementById(id);
    const valEl = document.getElementById(valId);
    if (!slider || !valEl) return;
    slider.addEventListener('input', () => {
      valEl.textContent = (parseInt(slider.value) / 100).toFixed(2);
    });
  });

  const toggleBtn = document.getElementById('toggle-advanced-btn');
  const advancedSettings = document.getElementById('advanced-settings');
  if (toggleBtn && advancedSettings) {
    toggleBtn.addEventListener('click', () => {
      const isHidden = advancedSettings.style.display === 'none';
      advancedSettings.style.display = isHidden ? 'block' : 'none';
      toggleBtn.classList.toggle('active', isHidden);
    });
  }
}

// ── Results Tabs ──
function initTabs() {
  const btns = document.querySelectorAll('.tab-btn');
  const panels = document.querySelectorAll('.tab-panel');

  btns.forEach(btn => {
    btn.addEventListener('click', () => {
      btns.forEach(b => b.classList.remove('active'));
      panels.forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      const target = document.getElementById(btn.dataset.tab);
      target?.classList.add('active');

      // Canvas charts size themselves from clientWidth, which is 0 while
      // their panel is hidden - redraw with real dimensions now that this
      // tab is actually visible.
      if (!currentJobId) return;
      if (btn.dataset.tab === 'tab-m1') renderModule1(currentJobId);
      if (btn.dataset.tab === 'tab-fusion') renderFusion(currentJobId);
    });
  });
}

// ── Log Output ──
function addLog(msg, type = 'info') {
  const logEl = document.getElementById('log-output');
  if (!logEl) return;
  const line = document.createElement('p');
  line.className = `log-line log-${type}`;
  const time = new Date().toLocaleTimeString('en-GB', { hour12: false });
  line.textContent = `[${time}] ${msg}`;
  logEl.appendChild(line);
  logEl.scrollTop = logEl.scrollHeight;
}

function initLog() {
  const clearBtn = document.getElementById('clear-log-btn');
  clearBtn?.addEventListener('click', () => {
    const logEl = document.getElementById('log-output');
    if (logEl) logEl.innerHTML = '<p class="log-line log-info">Log cleared.</p>';
  });
}

// ── Pipeline (Real Backend) ──
const MODULE_STAGE_IDS = { module1: 1, module2: 2, module3: 3, module4: 4 };
const MODULE_NAMES = {
  module1: 'Keyframe Detection (ResNet-50 + BiLSTM)',
  module2: 'Content Summarization (Whisper + BERT)',
  module3: 'Visual Understanding (ViT + OCR)',
  module4: 'Video Synthesis (Fusion + MoviePy)',
};

function setStageUI(moduleKey, moduleStatus) {
  const id = MODULE_STAGE_IDS[moduleKey];
  const stageEl = document.getElementById(`stage-${id}`);
  const progressFill = document.getElementById(`progress-${id}`);
  const pctEl = document.getElementById(`pct-${id}`);

  const pct = moduleStatus === 'completed' ? 100 : moduleStatus === 'running' ? 50 : 0;
  const domStatus = moduleStatus === 'completed' ? 'done'
    : moduleStatus === 'failed' ? 'error'
    : moduleStatus === 'running' ? 'running'
    : 'waiting';

  stageEl?.setAttribute('data-status', domStatus);
  if (progressFill) progressFill.style.width = pct + '%';
  if (pctEl) pctEl.textContent = pct + '%';
}

async function runRealPipeline(file) {
  const statusBadge = document.getElementById('status-indicator');
  const statusText = statusBadge?.querySelector('.status-text');
  const startBtn = document.getElementById('start-pipeline-btn');

  statusBadge?.classList.remove('status-idle', 'status-done', 'status-error');
  statusBadge?.classList.add('status-running');
  if (statusText) statusText.textContent = 'Uploading';

  addLog('═══ Pipeline started ═══', 'success');
  addLog('Uploading ' + file.name + '…', 'info');

  let jobId;
  try {
    const formData = new FormData();
    formData.append('file', file);

    const uploadRes = await fetch(`${API_BASE}/api/upload`, {
      method: 'POST',
      body: formData,
      headers: { 'ngrok-skip-browser-warning': 'true' },
    });
    if (!uploadRes.ok) throw new Error(`Upload failed: HTTP ${uploadRes.status}`);
    const uploadData = await uploadRes.json();
    jobId = uploadData.job_id;
    currentJobId = jobId;
    addLog(`Upload accepted. Job ID: ${jobId}`, 'success');
  } catch (err) {
    addLog(`Upload error: ${err.message}`, 'error');
    setPipelineError(err.message);
    return;
  }

  if (statusText) statusText.textContent = 'Running';
  const seenModuleStatus = {};

  const poll = setInterval(async () => {
    let status;
    try {
      const res = await fetch(`${API_BASE}/api/jobs/${jobId}/status`, {
        headers: { 'ngrok-skip-browser-warning': 'true' },
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      status = await res.json();
    } catch (err) {
      addLog(`Status check failed: ${err.message}`, 'error');
      return; // transient - keep polling
    }

    for (const moduleKey of Object.keys(MODULE_STAGE_IDS)) {
      const moduleStatus = status[moduleKey];
      if (moduleStatus && moduleStatus !== seenModuleStatus[moduleKey]) {
        seenModuleStatus[moduleKey] = moduleStatus;
        setStageUI(moduleKey, moduleStatus);
        addLog(`▸ ${MODULE_NAMES[moduleKey]}: ${moduleStatus}`, moduleStatus === 'failed' ? 'error' : 'info');
      }
    }

    if (status.status === 'completed') {
      clearInterval(poll);
      addLog('═══ Pipeline completed successfully ═══', 'success');
      statusBadge?.classList.remove('status-running');
      statusBadge?.classList.add('status-done');
      if (startBtn) startBtn.disabled = false;

      if (status.module4 === 'skipped') {
        // A partial run (e.g. ENABLED_MODULES=module2 for a module-by-module
        // demo) - no fused video/JSON exists, so don't try to fetch one, but
        // still render real results for whichever module(s) did complete.
        if (statusText) statusText.textContent = 'Done (partial run)';
        addLog('Module 4 skipped - this was a partial run of only the enabled module(s).', 'info');
        await renderAvailableModuleResults(jobId);
      } else {
        if (statusText) statusText.textContent = 'Done';
        await populateRealResults(jobId);
        const videoTabBtn = document.querySelector('.tab-btn[data-tab="tab-video"]');
        if (videoTabBtn) videoTabBtn.click();
      }
    } else if (status.status === 'failed') {
      clearInterval(poll);
      addLog(`Pipeline failed: ${status.error || 'unknown error'}`, 'error');
      setPipelineError(status.error);
    }
  }, STATUS_POLL_INTERVAL_MS);

  function setPipelineError(message) {
    statusBadge?.classList.remove('status-running');
    statusBadge?.classList.add('status-error');
    if (statusText) statusText.textContent = 'Error';
    if (startBtn) startBtn.disabled = false;
  }
}

// ── Populate Real Results ──
function formatDuration(totalSeconds) {
  const s = Math.round(totalSeconds || 0);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str ?? '';
  return div.innerHTML;
}

// Fetches a single module's real output JSON for this job. Returns null
// (not throwing) if that module hasn't produced output yet, so callers can
// show an honest "not available" state instead of crashing.
async function fetchModuleJSON(jobId, moduleName) {
  try {
    const res = await fetch(`${API_BASE}/api/jobs/${jobId}/module/${moduleName}`, {
      headers: { 'ngrok-skip-browser-warning': 'true' },
    });
    if (!res.ok) return null;
    return await res.json();
  } catch (err) {
    return null;
  }
}

async function populateRealResults(jobId) {
  const video = document.getElementById('video-output');
  if (video) {
    // A native <video><source src="..."> request can't carry custom headers,
    // so it can't send ngrok-skip-browser-warning and gets stuck on ngrok's
    // free-tier interstitial page. Fetch the bytes ourselves (where we CAN
    // set the header) and hand the browser a local blob URL instead.
    try {
      const videoRes = await fetch(`${API_BASE}/api/jobs/${jobId}/download-video`, {
        headers: { 'ngrok-skip-browser-warning': 'true' },
      });
      if (!videoRes.ok) throw new Error(`HTTP ${videoRes.status}`);
      const blob = await videoRes.blob();
      const blobUrl = URL.createObjectURL(blob);
      const source = video.querySelector('source') || video;
      source.src = blobUrl;
      video.load();
    } catch (err) {
      addLog(`Video load failed: ${err.message}`, 'error');
    }
  }

  // metric-rouge/precision/recall/wer/f1 are research-evaluation metrics computed
  // against held-out human annotations (see scripts/evaluate_trained_model.py) -
  // they have no real-time equivalent for an arbitrary user-uploaded video with
  // no ground truth, so they're marked as not applicable rather than faked.
  const notApplicable = ['metric-rouge', 'metric-precision', 'metric-recall', 'metric-wer', 'metric-f1'];
  notApplicable.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.textContent = 'N/A';
  });

  try {
    const res = await fetch(`${API_BASE}/api/jobs/${jobId}/download-json`, {
      headers: { 'ngrok-skip-browser-warning': 'true' },
    });
    if (res.ok) {
      const finalData = await res.json();
      const durationEl = document.getElementById('metric-duration');
      if (durationEl) durationEl.textContent = formatDuration(finalData.selected_duration_seconds);
      addLog(
        `Selected ${finalData.segments_selected} of ${finalData.total_segments_considered} segments ` +
        `(${formatDuration(finalData.selected_duration_seconds)} total).`,
        'info'
      );
    }
  } catch (err) {
    addLog(`Could not load final results JSON: ${err.message}`, 'error');
  }

  await renderAvailableModuleResults(jobId);
}

// Renders whichever modules have actually produced output for this job -
// each tab independently fetches its own module's real JSON and shows an
// honest "not available" message if that module hasn't run/finished
// (e.g. a partial ENABLED_MODULES demo run, or one module still in progress).
async function renderAvailableModuleResults(jobId) {
  await Promise.all([
    renderModule1(jobId),
    renderModule2(jobId),
    renderModule3(jobId),
    renderFusion(jobId),
  ]);
}

function getChartColors() {
  const isLight = document.documentElement.getAttribute('data-theme') === 'light';
  return {
    text: isLight ? 'rgba(15,23,42,0.5)' : 'rgba(148,163,184,0.5)',
    barBase: isLight ? [37, 99, 235] : [59, 130, 246],
    bg: isLight ? 'rgba(0,0,0,0.03)' : 'rgba(0,0,0,0.2)',
  };
}

async function renderModule1(jobId) {
  const canvas = document.getElementById('heatmap-canvas');
  const tableContainer = document.getElementById('m1-table-container');
  const data = await fetchModuleJSON(jobId, 'module1'); // [{segment_id, timestamp_start, timestamp_end, score_V}, ...]

  if (!data || !data.length) {
    if (tableContainer) tableContainer.innerHTML = '<p class="placeholder-text">Module 1 output not available for this job.</p>';
    if (canvas) canvas.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height);
    return;
  }

  const segments = [...data].sort((a, b) => a.timestamp_start - b.timestamp_start);

  if (canvas) {
    const ctx = canvas.getContext('2d');
    canvas.width = canvas.parentElement.clientWidth;
    const w = canvas.width, h = canvas.height;
    const barW = (w - 40) / segments.length;
    const colors = getChartColors();

    ctx.clearRect(0, 0, w, h);
    segments.forEach((seg, i) => {
      const score = Math.max(0, Math.min(1, seg.score_V ?? 0));
      const barH = score * (h - 40);
      const x = 20 + i * barW, y = h - 20 - barH;
      const r = colors.barBase[0] + Math.round(score * 80);
      const g = colors.barBase[1] - Math.round(score * 40);
      const b = colors.barBase[2] - Math.round(score * 100);
      ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.8)`;
      ctx.beginPath();
      ctx.roundRect(x + 1, y, Math.max(barW - 2, 1), barH, [3, 3, 0, 0]);
      ctx.fill();
    });

    ctx.fillStyle = colors.text;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillText('score_V', 20, 14);
    ctx.fillText('Segments →', w - 80, h - 4);
  }

  if (tableContainer) {
    const rows = segments.slice(0, 20);
    tableContainer.innerHTML = `
      <table style="width:100%;border-collapse:collapse;font-size:0.75rem;font-family:var(--font-mono);">
        <tr style="color:var(--text-muted);border-bottom:1px solid var(--glass-border);">
          <th style="padding:0.5rem;text-align:left;">Segment</th>
          <th style="padding:0.5rem;text-align:left;">Time</th>
          <th style="padding:0.5rem;text-align:right;">score_V</th>
        </tr>
        ${rows.map(seg => `<tr style="border-bottom:1px solid var(--glass-border);color:var(--text-secondary);">
            <td style="padding:0.4rem 0.5rem;">${escapeHtml(seg.segment_id)}</td>
            <td style="padding:0.4rem 0.5rem;">${formatDuration(seg.timestamp_start)} – ${formatDuration(seg.timestamp_end)}</td>
            <td style="padding:0.4rem 0.5rem;text-align:right;color:var(--accent-blue);font-weight:600;">${(seg.score_V ?? 0).toFixed(3)}</td>
          </tr>`).join('')}
      </table>
      ${segments.length > rows.length ? `<p class="placeholder-text" style="margin-top:0.5rem;">Showing first ${rows.length} of ${segments.length} segments.</p>` : ''}
    `;
  }
}

async function renderModule2(jobId) {
  const container = document.getElementById('transcript-container');
  if (!container) return;
  const data = await fetchModuleJSON(jobId, 'module2'); // [{sentence, timestamp_start, timestamp_end, is_important, importance_ratio_T}, ...]

  if (!data || !data.length) {
    container.innerHTML = '<p class="placeholder-text">Module 2 output not available for this job.</p>';
    return;
  }

  const sentences = [...data].sort((a, b) => a.timestamp_start - b.timestamp_start);

  container.innerHTML = sentences.map(s => `
    <p style="padding:0.4rem 0.6rem;margin:0.2rem 0;border-radius:6px;border-left:3px solid ${s.is_important ? 'var(--accent-blue)' : 'transparent'};background:${s.is_important ? 'rgba(59,130,246,0.06)' : 'transparent'};cursor:default;" title="importance_ratio_T: ${s.importance_ratio_T}">
      <span style="font-weight:${s.is_important ? '600' : '400'};color:${s.is_important ? 'var(--text-primary)' : 'var(--text-secondary)'};">${escapeHtml(s.sentence)}</span>
      <span style="font-family:var(--font-mono);font-size:0.7rem;color:${s.is_important ? 'var(--accent-blue)' : 'var(--text-muted)'};margin-left:0.5rem;">${Number(s.importance_ratio_T ?? 0).toFixed(2)}</span>
    </p>
  `).join('');
}

async function renderModule3(jobId) {
  const grid = document.getElementById('slides-grid');
  if (!grid) return;
  const data = await fetchModuleJSON(jobId, 'module3'); // [{frame_time, label, ocr_text, frame_path, semantic_analysis:{visual_topic,...}}, ...]

  if (!data || !data.length) {
    grid.innerHTML = '<p class="placeholder-text">Module 3 output not available for this job.</p>';
    return;
  }

  const colorVars = { Critical: '--accent-blue', Important: '--accent-violet', Skip: '--text-muted' };
  const bgColors = { Critical: 'rgba(59,130,246,0.08)', Important: 'rgba(139,92,246,0.08)', Skip: 'rgba(100,116,139,0.05)' };
  const sorted = [...data].sort((a, b) => a.frame_time - b.frame_time);

  grid.innerHTML = sorted.map((item, i) => {
    const label = item.label || 'Unknown';
    const bg = bgColors[label] || bgColors.Skip;
    const colorVar = colorVars[label] || colorVars.Skip;
    const caption = (item.semantic_analysis && item.semantic_analysis.visual_topic) ||
      (item.ocr_text ? item.ocr_text.slice(0, 60) : '');
    return `
    <div style="background:${bg};border:1px solid var(--glass-border);border-radius:10px;padding:1rem;text-align:center;">
      <div id="m3-thumb-${i}" style="width:100%;aspect-ratio:16/9;background:var(--bg-tertiary);border-radius:6px;margin-bottom:0.6rem;overflow:hidden;display:flex;align-items:center;justify-content:center;font-size:0.7rem;color:var(--text-muted);font-family:var(--font-mono);">${formatDuration(item.frame_time)}</div>
      <span style="font-size:0.7rem;font-weight:600;font-family:var(--font-mono);color:var(${colorVar});padding:0.15rem 0.5rem;border-radius:99px;background:${bg};">${escapeHtml(label)}</span>
      ${caption ? `<p style="font-size:0.7rem;color:var(--text-muted);margin-top:0.4rem;word-break:break-word;">${escapeHtml(caption)}</p>` : ''}
    </div>`;
  }).join('');

  // Real thumbnails, loaded separately as blob URLs: a plain <img src> can't
  // send the ngrok-skip-browser-warning header, so we fetch each frame
  // ourselves (where we can) and swap it in once it's ready. Falls back to
  // the timestamp label already in place if a given frame can't be loaded.
  sorted.forEach(async (item, i) => {
    if (!item.frame_path) return;
    const thumbEl = document.getElementById(`m3-thumb-${i}`);
    if (!thumbEl) return;
    try {
      const res = await fetch(`${API_BASE}/api/jobs/${jobId}/frame?path=${encodeURIComponent(item.frame_path)}`, {
        headers: { 'ngrok-skip-browser-warning': 'true' },
      });
      if (!res.ok) return;
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      thumbEl.innerHTML = `<img src="${url}" alt="Slide at ${formatDuration(item.frame_time)}" style="width:100%;height:100%;object-fit:cover;" />`;
    } catch (err) {
      // leave the timestamp placeholder in place on failure
    }
  });
}

async function renderFusion(jobId) {
  const canvas = document.getElementById('fusion-canvas');
  const tableContainer = document.getElementById('fusion-table-container');
  const data = await fetchModuleJSON(jobId, 'module4'); // module4_final_output.json

  if (!data || !data.selected_segments || !data.selected_segments.length) {
    if (tableContainer) tableContainer.innerHTML = '<p class="placeholder-text">Fusion output not available for this job.</p>';
    if (canvas) canvas.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height);
    return;
  }

  const segments = [...data.selected_segments].sort((a, b) => a.timestamp_start - b.timestamp_start);
  const weights = data.fusion_weights || {};

  if (canvas) {
    const ctx = canvas.getContext('2d');
    canvas.width = canvas.parentElement.clientWidth;
    const w = canvas.width, h = canvas.height;
    const barGroupW = (w - 60) / segments.length;
    const colors = getChartColors();
    const barColors = ['rgba(59,130,246,0.7)', 'rgba(139,92,246,0.7)', 'rgba(6,182,212,0.7)'];

    ctx.clearRect(0, 0, w, h);

    segments.forEach((seg, i) => {
      const subW = (barGroupW - 4) / 3;
      [seg.score_V, seg.score_T, seg.score_L].forEach((score, j) => {
        const barH = Math.max(0, Math.min(1, score ?? 0)) * (h - 60);
        const x = 30 + i * barGroupW + j * subW;
        const y = h - 30 - barH;
        ctx.fillStyle = barColors[j];
        ctx.beginPath();
        ctx.roundRect(x + 0.5, y, Math.max(subW - 1, 1), barH, [2, 2, 0, 0]);
        ctx.fill();
      });
    });

    // Legend
    const legendY = 14;
    const items = [['V (Visual)', barColors[0]], ['T (Text)', barColors[1]], ['L (Slide)', barColors[2]]];
    let lx = 30;
    items.forEach(([label, color]) => {
      ctx.fillStyle = color;
      ctx.fillRect(lx, legendY - 7, 10, 10);
      ctx.fillStyle = colors.text;
      ctx.font = '9px "JetBrains Mono", monospace';
      ctx.fillText(label, lx + 14, legendY + 2);
      lx += ctx.measureText(label).width + 28;
    });
  }

  if (tableContainer) {
    const rows = segments.slice(0, 20);
    tableContainer.innerHTML = `
      <p style="font-size:0.75rem;color:var(--text-muted);margin-bottom:0.5rem;">
        Weights: w₁=${weights.w1_visual ?? '—'} w₂=${weights.w2_text ?? '—'} w₃=${weights.w3_slide ?? '—'} ·
        ${data.segments_selected ?? segments.length}/${data.total_segments_considered ?? '—'} segments selected ·
        ${formatDuration(data.selected_duration_seconds)} total
      </p>
      <table style="width:100%;border-collapse:collapse;font-size:0.75rem;font-family:var(--font-mono);">
        <tr style="color:var(--text-muted);border-bottom:1px solid var(--glass-border);">
          <th style="padding:0.5rem;text-align:left;">Segment</th>
          <th style="padding:0.5rem;text-align:right;">V</th>
          <th style="padding:0.5rem;text-align:right;">T</th>
          <th style="padding:0.5rem;text-align:right;">L</th>
          <th style="padding:0.5rem;text-align:right;">Fused S</th>
        </tr>
        ${rows.map(seg => `<tr style="border-bottom:1px solid var(--glass-border);color:var(--text-secondary);">
            <td style="padding:0.4rem 0.5rem;">${escapeHtml(seg.segment_id)}</td>
            <td style="padding:0.4rem 0.5rem;text-align:right;">${(seg.score_V ?? 0).toFixed(2)}</td>
            <td style="padding:0.4rem 0.5rem;text-align:right;">${(seg.score_T ?? 0).toFixed(2)}</td>
            <td style="padding:0.4rem 0.5rem;text-align:right;">${(seg.score_L ?? 0).toFixed(2)}</td>
            <td style="padding:0.4rem 0.5rem;text-align:right;color:var(--accent-blue);font-weight:600;">${(seg.fused_score ?? 0).toFixed(3)}</td>
          </tr>`).join('')}
      </table>
      ${segments.length > rows.length ? `<p class="placeholder-text" style="margin-top:0.5rem;">Showing first ${rows.length} of ${segments.length} selected segments.</p>` : ''}
    `;
  }
}

// ── Initialize Everything ──
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initParticles();
  initNav();
  initReveal();
  initUpload();
  initSliders();
  initTabs();
  initLog();
});
