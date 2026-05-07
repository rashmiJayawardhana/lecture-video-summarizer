/* ═══════════════════════════════════════════════════════════
   INTEGRA — Main Application Script
   ═══════════════════════════════════════════════════════════ */

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
    runPipelineDemo();
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

// ── Pipeline Demo (Simulated) ──
function runPipelineDemo() {
  const statusBadge = document.getElementById('status-indicator');
  const statusText = statusBadge?.querySelector('.status-text');

  statusBadge?.classList.remove('status-idle', 'status-done', 'status-error');
  statusBadge?.classList.add('status-running');
  if (statusText) statusText.textContent = 'Running';

  addLog('═══ Pipeline started ═══', 'success');
  addLog('Configuration loaded. Target duration: ' +
    (document.getElementById('config-duration')?.value || 600) + 's', 'info');

  const stages = [
    { id: 1, name: 'Keyframe Detection', model: 'ResNet-50 + BiLSTM', duration: 3000, messages: [
      'Extracting frames at 1 FPS…',
      'Running ResNet-50 feature extraction…',
      'BiLSTM temporal scoring complete.',
      'Generated importance scores for 360 segments.'
    ]},
    { id: 2, name: 'Content Summarization', model: 'Whisper + BERT', duration: 3500, messages: [
      'Transcribing audio with Whisper large-v3…',
      'Segmenting transcript into sentences…',
      'Running BERT sentence classification…',
      'Identified 47 important sentences.'
    ]},
    { id: 3, name: 'Visual Understanding', model: 'ViT + TrOCR', duration: 2500, messages: [
      'Detecting slide transitions…',
      'Classifying slides with ViT-base…',
      'Extracting text with TrOCR…',
      '18 Critical, 12 Important, 6 Skip slides.'
    ]},
    { id: 4, name: 'Video Synthesis', model: 'Fusion + MoviePy', duration: 4000, messages: [
      'Computing fused scores: S = w₁·V + w₂·T + w₃·L',
      'Selecting top segments above threshold…',
      'Rendering Pipeline A — Highlight reel…',
      'Rendering Pipeline B — AI slideshow…',
      'Output videos generated successfully!'
    ]},
  ];

  let stageIndex = 0;

  function runStage() {
    if (stageIndex >= stages.length) {
      addLog('═══ Pipeline completed successfully ═══', 'success');
      statusBadge?.classList.remove('status-running');
      statusBadge?.classList.add('status-done');
      if (statusText) statusText.textContent = 'Done';
      populateDemoResults();
      document.getElementById('start-pipeline-btn').disabled = false;
      
      // Auto-switch to Videos tab for user-friendly experience
      const videoTabBtn = document.querySelector('.tab-btn[data-tab="tab-video"]');
      if (videoTabBtn) videoTabBtn.click();
      
      return;
    }

    const stage = stages[stageIndex];
    const stageEl = document.getElementById(`stage-${stage.id}`);
    const progressFill = document.getElementById(`progress-${stage.id}`);
    const pctEl = document.getElementById(`pct-${stage.id}`);

    stageEl?.setAttribute('data-status', 'running');
    addLog(`▸ Stage ${stage.id}: ${stage.name} (${stage.model})`, 'info');

    let progress = 0;
    const interval = stage.duration / 100;
    let msgIdx = 0;

    const progressTimer = setInterval(() => {
      progress += 1;
      if (progressFill) progressFill.style.width = progress + '%';
      if (pctEl) pctEl.textContent = progress + '%';

      if (msgIdx < stage.messages.length && progress >= ((msgIdx + 1) / stage.messages.length) * 100) {
        addLog('  ' + stage.messages[msgIdx], 'info');
        msgIdx++;
      }

      if (progress >= 100) {
        clearInterval(progressTimer);
        stageEl?.setAttribute('data-status', 'done');
        addLog(`[Done] Stage ${stage.id} complete.`, 'success');
        stageIndex++;
        setTimeout(runStage, 400);
      }
    }, interval);
  }

  runStage();
}

// ── Populate Demo Results ──
function populateDemoResults() {
  const metrics = {
    'metric-rouge': '0.438',
    'metric-precision': '0.812',
    'metric-recall': '0.779',
    'metric-wer': '11.2%',
    'metric-f1': '0.743',
    'metric-duration': '9:47'
  };
  Object.entries(metrics).forEach(([id, val]) => {
    const el = document.getElementById(id);
    if (el) {
      el.textContent = val;
      el.style.animation = 'fadeIn 0.5s ease';
    }
  });

  drawHeatmap();
  populateTranscript();
  populateSlideGrid();
  drawFusionChart();
}

function getChartColors() {
  const isLight = document.documentElement.getAttribute('data-theme') === 'light';
  return {
    text: isLight ? 'rgba(15,23,42,0.5)' : 'rgba(148,163,184,0.5)',
    barBase: isLight ? [37, 99, 235] : [59, 130, 246],
    bg: isLight ? 'rgba(0,0,0,0.03)' : 'rgba(0,0,0,0.2)',
  };
}

function drawHeatmap() {
  const canvas = document.getElementById('heatmap-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  canvas.width = canvas.parentElement.clientWidth;
  const w = canvas.width, h = canvas.height;
  const segments = 36;
  const barW = (w - 40) / segments;
  const colors = getChartColors();

  ctx.clearRect(0, 0, w, h);

  for (let i = 0; i < segments; i++) {
    const score = Math.random() * 0.6 + 0.2;
    const barH = score * (h - 40);
    const x = 20 + i * barW, y = h - 20 - barH;
    const r = colors.barBase[0] + Math.round(score * 80);
    const g = colors.barBase[1] - Math.round(score * 40);
    const b = colors.barBase[2] - Math.round(score * 100);
    ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.8)`;
    ctx.beginPath();
    ctx.roundRect(x + 1, y, barW - 2, barH, [3, 3, 0, 0]);
    ctx.fill();
  }

  ctx.fillStyle = colors.text;
  ctx.font = '10px "JetBrains Mono", monospace';
  ctx.fillText('score_V', 20, 14);
  ctx.fillText('Segments →', w - 80, h - 4);

  const tableContainer = document.getElementById('m1-table-container');
  if (tableContainer) {
    tableContainer.innerHTML = `
      <table style="width:100%;border-collapse:collapse;font-size:0.75rem;font-family:var(--font-mono);">
        <tr style="color:var(--text-muted);border-bottom:1px solid var(--glass-border);">
          <th style="padding:0.5rem;text-align:left;">Segment</th>
          <th style="padding:0.5rem;text-align:left;">Time</th>
          <th style="padding:0.5rem;text-align:right;">score_V</th>
        </tr>
        ${Array.from({length: 8}, (_, i) => {
          const score = (Math.random() * 0.5 + 0.45).toFixed(3);
          return `<tr style="border-bottom:1px solid var(--glass-border);color:var(--text-secondary);">
            <td style="padding:0.4rem 0.5rem;">seg_${String(i+1).padStart(3,'0')}</td>
            <td style="padding:0.4rem 0.5rem;">${i*10}:00 – ${i*10+10}:00</td>
            <td style="padding:0.4rem 0.5rem;text-align:right;color:var(--accent-blue);font-weight:600;">${score}</td>
          </tr>`;
        }).join('')}
      </table>`;
  }
}

function populateTranscript() {
  const container = document.getElementById('transcript-container');
  if (!container) return;
  const sentences = [
    { text: 'A binary search tree is a data structure that maintains sorted order for efficient lookup.', important: true, ratio: 0.89 },
    { text: 'Today we will cover the fundamentals of tree-based structures.', important: false, ratio: 0.31 },
    { text: 'The time complexity for search in a balanced BST is O(log n).', important: true, ratio: 0.92 },
    { text: 'Let me show you this on the whiteboard here.', important: false, ratio: 0.12 },
    { text: 'AVL trees perform rotations to maintain balance after insertions and deletions.', important: true, ratio: 0.85 },
    { text: 'This is an important concept that comes up frequently in interviews.', important: false, ratio: 0.28 },
    { text: 'Red-black trees provide guaranteed O(log n) worst-case performance.', important: true, ratio: 0.91 },
    { text: 'The rotation operations preserve the binary search tree property.', important: true, ratio: 0.78 },
  ];

  container.innerHTML = sentences.map(s => `
    <p style="padding:0.4rem 0.6rem;margin:0.2rem 0;border-radius:6px;border-left:3px solid ${s.important ? 'var(--accent-blue)' : 'transparent'};background:${s.important ? 'rgba(59,130,246,0.06)' : 'transparent'};cursor:default;" title="importance_ratio_T: ${s.ratio}">
      <span style="font-weight:${s.important ? '600' : '400'};color:${s.important ? 'var(--text-primary)' : 'var(--text-secondary)'};">${s.text}</span>
      <span style="font-family:var(--font-mono);font-size:0.7rem;color:${s.important ? 'var(--accent-blue)' : 'var(--text-muted)'};margin-left:0.5rem;">${s.ratio.toFixed(2)}</span>
    </p>
  `).join('');
}

function populateSlideGrid() {
  const grid = document.getElementById('slides-grid');
  if (!grid) return;
  const labels = ['Critical', 'Important', 'Critical', 'Skip', 'Important', 'Critical', 'Important', 'Skip', 'Critical'];
  const colors = { Critical: '--accent-blue', Important: '--accent-violet', Skip: '--text-muted' };
  const bgColors = { Critical: 'rgba(59,130,246,0.08)', Important: 'rgba(139,92,246,0.08)', Skip: 'rgba(100,116,139,0.05)' };

  grid.innerHTML = labels.map((label, i) => `
    <div style="background:${bgColors[label]};border:1px solid var(--glass-border);border-radius:10px;padding:1rem;text-align:center;">
      <div style="width:100%;aspect-ratio:16/9;background:var(--bg-tertiary);border-radius:6px;margin-bottom:0.6rem;display:flex;align-items:center;justify-content:center;font-size:0.7rem;color:var(--text-muted);font-family:var(--font-mono);">Slide ${i + 1}</div>
      <span style="font-size:0.7rem;font-weight:600;font-family:var(--font-mono);color:var(${colors[label]});padding:0.15rem 0.5rem;border-radius:99px;background:${bgColors[label]};">${label}</span>
    </div>
  `).join('');
}

function drawFusionChart() {
  const canvas = document.getElementById('fusion-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  canvas.width = canvas.parentElement.clientWidth;
  const w = canvas.width, h = canvas.height;
  const segments = 20;
  const barGroupW = (w - 60) / segments;
  const colors = getChartColors();

  ctx.clearRect(0, 0, w, h);

  // Threshold line
  const threshold = 0.3;
  const thresholdY = h - 30 - threshold * (h - 60);
  ctx.setLineDash([4, 4]);
  ctx.strokeStyle = 'rgba(244,63,94,0.4)';
  ctx.beginPath();
  ctx.moveTo(30, thresholdY);
  ctx.lineTo(w - 20, thresholdY);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = 'rgba(244,63,94,0.6)';
  ctx.font = '9px "JetBrains Mono", monospace';
  ctx.fillText('θ = 0.30', w - 65, thresholdY - 4);

  const barColors = [
    'rgba(59,130,246,0.7)',
    'rgba(139,92,246,0.7)',
    'rgba(6,182,212,0.7)',
  ];

  for (let i = 0; i < segments; i++) {
    const subW = (barGroupW - 4) / 3;
    for (let j = 0; j < 3; j++) {
      const score = Math.random() * 0.7 + 0.15;
      const barH = score * (h - 60);
      const x = 30 + i * barGroupW + j * subW;
      const y = h - 30 - barH;
      ctx.fillStyle = barColors[j];
      ctx.beginPath();
      ctx.roundRect(x + 0.5, y, subW - 1, barH, [2, 2, 0, 0]);
      ctx.fill();
    }
  }

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
