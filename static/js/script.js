/* ============================================================
   SentimentIQ v2.0 – Main JavaScript
   ============================================================ */

'use strict';

// ── Toast system ─────────────────────────────────────────────
const Toast = (() => {
  let container = null;
  function getContainer() {
    if (!container) {
      container = document.createElement('div');
      container.className = 'toast-container';
      document.body.appendChild(container);
    }
    return container;
  }
  function show(msg, type = 'info', duration = 3800) {
    const icons = { success: '✅', error: '❌', info: 'ℹ️', warning: '⚠️' };
    const t = document.createElement('div');
    t.className = `toast ${type}`;
    t.innerHTML = `<span>${icons[type] || ''}</span><span>${msg}</span>`;
    getContainer().appendChild(t);
    setTimeout(() => { t.style.opacity = '0'; t.style.transition = 'opacity 0.3s'; setTimeout(() => t.remove(), 300); }, duration);
  }
  return { success: m => show(m, 'success'), error: m => show(m, 'error'),
           info: m => show(m, 'info'), warning: m => show(m, 'warning') };
})();

// ── Loading overlay ───────────────────────────────────────────
const Loading = (() => {
  let overlay = null;
  function getOverlay() {
    if (!overlay) {
      overlay = document.createElement('div');
      overlay.className = 'loading-overlay';
      overlay.innerHTML = '<div class="spinner"></div><div>Processing…</div>';
      document.body.appendChild(overlay);
    }
    return overlay;
  }
  return {
    show(msg = 'Processing…') {
      const o = getOverlay();
      o.querySelector('div:last-child').textContent = msg;
      o.classList.add('show');
    },
    hide() { getOverlay().classList.remove('show'); }
  };
})();

// ── Chart defaults ────────────────────────────────────────────
if (typeof Chart !== 'undefined') {
  Chart.defaults.color = '#94a3b8';
  Chart.defaults.borderColor = 'rgba(255,255,255,0.07)';
  Chart.defaults.font.family = 'Inter, system-ui, sans-serif';
  Chart.defaults.plugins.legend.labels.usePointStyle = true;
  Chart.defaults.plugins.legend.labels.padding = 16;
  Chart.defaults.animation.duration = 600;
}

// ── Sentiment Analyzer Page ───────────────────────────────────
function initAnalyzer() {
  const form      = document.getElementById('analyzeForm');
  const textArea  = document.getElementById('textInput');
  const resultDiv = document.getElementById('resultSection');
  const clearBtn  = document.getElementById('clearBtn');
  const charCount = document.getElementById('charCount');

  if (!form) return;

  textArea.addEventListener('input', () => {
    const n = textArea.value.length;
    if (charCount) charCount.textContent = n + ' chars';
  });

  clearBtn && clearBtn.addEventListener('click', () => {
    textArea.value = '';
    resultDiv.innerHTML = '';
    if (charCount) charCount.textContent = '0 chars';
  });

  form.addEventListener('submit', async e => {
    e.preventDefault();
    const text = textArea.value.trim();
    if (!text) { Toast.warning('Please enter some text.'); return; }

    const submitBtn = form.querySelector('[type=submit]');
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span class="spinner"></span> Analyzing…';

    try {
      const res  = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      const data = await res.json();
      if (!res.ok || data.error) { Toast.error(data.error || 'Prediction failed.'); return; }
      renderResult(data, resultDiv);
      Toast.success('Analysis complete!');
    } catch (err) {
      Toast.error('Network error. Please try again.');
    } finally {
      submitBtn.disabled = false;
      submitBtn.innerHTML = '🔍 Analyze Sentiment';
    }
  });
}

function renderResult(data, container) {
  const sentColors = { Positive: '#22c55e', Negative: '#ef4444', Neutral: '#f59e0b' };
  const sentEmojis = { Positive: '😊', Negative: '😞', Neutral: '😐' };
  const color = sentColors[data.sentiment] || '#6366f1';
  const emoji = sentEmojis[data.sentiment] || '🤔';
  const probs = data.probabilities || {};
  const conf  = Math.round((data.confidence || 0) * 100);

  // Probability bars
  const probHTML = ['Positive', 'Negative', 'Neutral'].map(cls => {
    const pct = Math.round((probs[cls] || 0) * 100);
    const cls2 = cls.toLowerCase();
    return `
      <div class="prob-row">
        <span class="prob-label">${cls}</span>
        <div class="progress-bar-wrap flex-1">
          <div class="progress-bar-fill ${cls2}" style="width:${pct}%"></div>
        </div>
        <span class="prob-pct">${pct}%</span>
      </div>`;
  }).join('');

  // Feature words
  let featHTML = '';
  if (data.top_features && data.top_features.length) {
    const tags = data.top_features.map(f => {
      const cls = f.sign === 'positive' ? 'pos' : 'neg';
      const arrow = f.sign === 'positive' ? '▲' : '▼';
      return `<span class="feature-tag ${cls}">${arrow} ${f.word}</span>`;
    }).join('');
    featHTML = `
      <div class="mb-2">
        <div class="form-label">Key Contributing Words</div>
        <div class="feature-tags">${tags}</div>
      </div>`;
  }

  // Cleaned text
  const cleanedHTML = data.cleaned_text ? `
    <div class="mb-2">
      <div class="form-label">Cleaned Text</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:var(--text-muted);background:var(--bg-surface);padding:10px 14px;border-radius:var(--radius-md);word-break:break-all;">${escHtml(data.cleaned_text)}</div>
    </div>` : '';

  container.innerHTML = `
    <div class="result-card mt-2">
      <div style="display:flex;align-items:flex-start;gap:20px;margin-bottom:20px;">
        <div style="font-size:3rem;">${emoji}</div>
        <div>
          <div class="result-label">Sentiment Prediction</div>
          <div class="result-sentiment" style="color:${color}">${data.sentiment}</div>
          <div class="result-confidence">
            <span class="badge badge-${data.sentiment.toLowerCase()}">${data.confidence_level || ''} Confidence</span>
            &nbsp; ${conf}% confident
          </div>
        </div>
        <div style="margin-left:auto;">
          <canvas id="probChart" width="130" height="130"></canvas>
        </div>
      </div>

      <div class="mb-2">
        <div class="form-label">Class Probabilities</div>
        ${probHTML}
      </div>

      ${featHTML}
      ${cleanedHTML}

      <div style="background:var(--bg-surface);border-radius:var(--radius-md);padding:12px 16px;font-size:0.85rem;color:var(--text-secondary);">
        💡 ${escHtml(data.explanation || '')}
      </div>
    </div>`;

  // Doughnut chart
  const ctx = document.getElementById('probChart');
  if (ctx) {
    new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: ['Positive', 'Negative', 'Neutral'],
        datasets: [{ data: [
          Math.round((probs.Positive || 0) * 100),
          Math.round((probs.Negative || 0) * 100),
          Math.round((probs.Neutral  || 0) * 100),
        ], backgroundColor: ['#22c55e','#ef4444','#f59e0b'], borderWidth: 0, hoverOffset: 4 }]
      },
      options: {
        cutout: '65%',
        plugins: { legend: { display: false }, tooltip: { callbacks: {
          label: c => ` ${c.label}: ${c.raw}%`
        }}},
        animation: { duration: 700 }
      }
    });
  }

  container.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ── Batch Page ────────────────────────────────────────────────
function initBatch() {
  const uploadZone = document.getElementById('uploadZone');
  const fileInput  = document.getElementById('csvFile');
  const form       = document.getElementById('batchForm');
  const resultDiv  = document.getElementById('batchResult');
  const colSelect  = document.getElementById('textColumn');
  const uploadBtn  = document.getElementById('uploadBtn');

  if (!uploadZone) return;

  uploadZone.addEventListener('click', () => fileInput.click());
  uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
  uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
  uploadZone.addEventListener('drop', e => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    if (e.dataTransfer.files[0]) { fileInput.files = e.dataTransfer.files; onFileSelected(); }
  });
  fileInput.addEventListener('change', onFileSelected);

  function onFileSelected() {
    const f = fileInput.files[0];
    if (!f) return;
    if (!f.name.endsWith('.csv')) { Toast.error('Please upload a .csv file.'); return; }
    uploadZone.querySelector('.upload-zone-icon').textContent = '📄';
    uploadZone.querySelector('p').textContent = f.name + ' (' + (f.size/1024).toFixed(1) + ' KB)';
    // Parse columns
    const reader = new FileReader();
    reader.onload = e => {
      const lines = e.target.result.split('\n');
      if (!lines[0]) return;
      const cols = lines[0].split(',').map(c => c.trim().replace(/^"|"$/g,''));
      if (colSelect) {
        colSelect.innerHTML = cols.map(c => `<option value="${escHtml(c)}">${escHtml(c)}</option>`).join('');
        // Auto-select text column
        const textCols = cols.filter(c => /text|review|comment|content|message/i.test(c));
        if (textCols.length) colSelect.value = textCols[0];
      }
      if (uploadBtn) uploadBtn.disabled = false;
    };
    reader.readAsText(f.slice(0, 4096)); // read header only
  }

  form && form.addEventListener('submit', async e => {
    e.preventDefault();
    if (!fileInput.files[0]) { Toast.warning('Please select a CSV file.'); return; }
    Loading.show('Running batch sentiment analysis…');

    const fd = new FormData();
    fd.append('file', fileInput.files[0]);
    fd.append('text_column', colSelect ? colSelect.value : 'text');

    try {
      const res  = await fetch('/batch-predict', { method: 'POST', body: fd });
      const data = await res.json();
      Loading.hide();
      if (!res.ok || data.error) { Toast.error(data.error || 'Batch prediction failed.'); return; }
      renderBatchResult(data, resultDiv);
      Toast.success(`Analyzed ${data.total} records!`);
    } catch (err) {
      Loading.hide();
      Toast.error('Network error during batch analysis.');
    }
  });
}

function renderBatchResult(data, container) {
  const counts   = data.counts || {};
  const total    = data.total  || 0;
  const preview  = data.preview || [];
  const dlFile   = data.download_file;

  const pct = s => total ? Math.round((counts[s]||0)/total*100) : 0;

  const rows = preview.map((r, i) => {
    const cls = r.sentiment.toLowerCase();
    return `<tr>
      <td>${i+1}</td>
      <td style="max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${escHtml(r.text)}">${escHtml(r.text)}</td>
      <td><span class="badge badge-${cls}">${r.sentiment}</span></td>
      <td>${r.confidence}</td>
    </tr>`;
  }).join('');

  container.innerHTML = `
    <div class="card mt-3">
      <div class="stats-grid" style="margin-bottom:20px;">
        <div class="stat-card info">  <div class="stat-label">Total Records</div><div class="stat-value">${total}</div><div class="stat-icon">📊</div></div>
        <div class="stat-card positive"><div class="stat-label">Positive</div><div class="stat-value" style="color:var(--positive)">${counts.Positive||0}</div><div class="stat-sub">${pct('Positive')}%</div><div class="stat-icon">😊</div></div>
        <div class="stat-card negative"><div class="stat-label">Negative</div><div class="stat-value" style="color:var(--negative)">${counts.Negative||0}</div><div class="stat-sub">${pct('Negative')}%</div><div class="stat-icon">😞</div></div>
        <div class="stat-card neutral"> <div class="stat-label">Neutral</div> <div class="stat-value" style="color:var(--neutral)">${counts.Neutral||0}</div><div class="stat-sub">${pct('Neutral')}%</div><div class="stat-icon">😐</div></div>
      </div>

      <div class="grid-2 mb-3">
        <div><canvas id="batchPieChart" height="200"></canvas></div>
        <div><canvas id="batchBarChart" height="200"></canvas></div>
      </div>

      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
        <div class="card-title"><span>📋</span> Results Preview (first ${preview.length})</div>
        ${dlFile ? `<a href="/download-results/${dlFile}" class="btn btn-success btn-sm">⬇️ Download Full CSV</a>` : ''}
      </div>

      <div class="table-wrap">
        <table>
          <thead><tr><th>#</th><th>Text</th><th>Sentiment</th><th>Confidence</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    </div>`;

  // Charts
  const pieCtx = document.getElementById('batchPieChart');
  if (pieCtx) {
    new Chart(pieCtx, {
      type: 'pie',
      data: {
        labels: ['Positive','Negative','Neutral'],
        datasets: [{ data: [counts.Positive||0, counts.Negative||0, counts.Neutral||0],
          backgroundColor: ['#22c55e','#ef4444','#f59e0b'], borderWidth: 2,
          borderColor: 'var(--bg-card)' }]
      },
      options: { plugins: { title: { display: true, text: 'Sentiment Distribution' } } }
    });
  }
  const barCtx = document.getElementById('batchBarChart');
  if (barCtx) {
    new Chart(barCtx, {
      type: 'bar',
      data: {
        labels: ['Positive','Negative','Neutral'],
        datasets: [{ label: 'Count', data: [counts.Positive||0, counts.Negative||0, counts.Neutral||0],
          backgroundColor: ['#22c55e','#ef4444','#f59e0b'], borderRadius: 6 }]
      },
      options: { plugins: { legend: { display: false }, title: { display: true, text: 'Count by Class' } },
        scales: { y: { beginAtZero: true } } }
    });
  }

  container.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ── Utilities ─────────────────────────────────────────────────
function escHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

function formatPct(val) {
  if (val == null) return '—';
  return (parseFloat(val) * 100).toFixed(1) + '%';
}

// ── Auto-init on DOM ready ────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initAnalyzer();
  initBatch();

  // Theme toggle
  const themeBtn = document.getElementById('themeToggle');
  if (themeBtn) {
    themeBtn.addEventListener('click', () => {
      const html = document.documentElement;
      const current = html.getAttribute('data-theme');
      html.setAttribute('data-theme', current === 'dark' ? 'light' : 'dark');
    });
  }
});
