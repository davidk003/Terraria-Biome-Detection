'use strict';

const BIOME_COLORS = {
  Corruption:  '#6a0dad',
  Crimson:     '#c84b31',
  Desert:      '#c8a45a',
  Dungeon:     '#2244cc',
  Forest:      '#5cb85c',
  Hallow:      '#cc77ff',
  Hell:        '#ff5500',
  Jungle:      '#2d7a2d',
  Mushroom:    '#4488cc',
  Ocean:       '#1a7acc',
  Snow:        '#88aacc',
  Space:       '#445566',
  Underground: '#886644',
};

// DOM refs
const dropZone            = document.getElementById('drop-zone');
const fileInput           = document.getElementById('file-input');
const uploadSection       = document.getElementById('upload-section');
const loadingOverlay      = document.getElementById('loading-overlay');
const errorBox            = document.getElementById('error-box');
const errorText           = document.getElementById('error-text');
const resultsSection      = document.getElementById('results-section');
const previewImg          = document.getElementById('preview-img');
const topPredictionCard   = document.getElementById('top-prediction-card');
const topBiomeName        = document.getElementById('top-biome-name');
const topConfidenceBar    = document.getElementById('top-confidence-bar');
const topProbabilityText  = document.getElementById('top-probability-text');
const lowConfidenceWarn   = document.getElementById('low-confidence-warning');
const chartContainer      = document.getElementById('probability-chart');
const resetBtn            = document.getElementById('reset-btn');

// ── Video demo ────────────────────────────────────────────────────────────────

const analyzeDemoBtn  = document.getElementById('analyze-demo-btn');
const videoLoading    = document.getElementById('video-loading');
const videoLoadingTxt = document.getElementById('video-loading-text');
const videoErrorBox   = document.getElementById('video-error-box');
const videoErrorTxt   = document.getElementById('video-error-text');
const videoResults    = document.getElementById('video-results');
const frameGrid       = document.getElementById('frame-grid');

analyzeDemoBtn.addEventListener('click', async () => {
  analyzeDemoBtn.disabled = true;
  videoErrorBox.classList.add('hidden');
  videoResults.classList.add('hidden');
  videoLoadingTxt.textContent = 'Downloading & analyzing video…';
  videoLoading.classList.remove('hidden');

  try {
    const response = await fetch('/predict-demo');
    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || `Server error (${response.status})`);
    }
    const data = await response.json();
    renderVideoResults(data.frames);
  } catch (err) {
    videoErrorTxt.textContent = err.message || 'Video analysis failed.';
    videoErrorBox.classList.remove('hidden');
    analyzeDemoBtn.disabled = false;
  } finally {
    videoLoading.classList.add('hidden');
  }
});

function renderVideoResults(frames) {
  frameGrid.innerHTML = '';

  frames.forEach((frame, idx) => {
    const color = BIOME_COLORS[frame.top_prediction] || '#5cb85c';
    const pct = (frame.top_probability * 100).toFixed(1);

    const card = document.createElement('div');
    card.className = 'frame-card';
    card.innerHTML = `
      <div class="frame-thumb-wrap">
        <img class="frame-thumb" src="${frame.thumbnail}" alt="Frame at ${frame.timestamp_str}" loading="lazy">
        <span class="frame-time-badge">${frame.timestamp_str}</span>
      </div>
      <div class="frame-info">
        <span class="frame-biome" style="color:${color}">${frame.top_prediction}</span>
        <span class="frame-conf">${pct}% confidence</span>
        <div class="frame-bar-track">
          <div class="frame-bar-fill" style="background:${color}"></div>
        </div>
      </div>
    `;
    frameGrid.appendChild(card);

    // Staggered bar animation
    const fill = card.querySelector('.frame-bar-fill');
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        setTimeout(() => { fill.style.width = `${pct}%`; }, idx * 40);
      });
    });
  });

  videoResults.classList.remove('hidden');
  analyzeDemoBtn.textContent = '▶  Re-analyze';
  analyzeDemoBtn.disabled = false;
}

// ── Drag & drop ───────────────────────────────────────────────────────────────

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-active');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('drag-active');
});

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-active');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    fileInput.click();
  }
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

// ── File handling ─────────────────────────────────────────────────────────────

function handleFile(file) {
  if (!file.type.startsWith('image/')) {
    showError('Please upload a valid image file (JPEG, PNG, WebP, etc.).');
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showError('File is too large. Maximum size is 10 MB.');
    return;
  }

  hideError();

  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewImg.onload = () => URL.revokeObjectURL(url);

  predict(file);
}

// ── Prediction ────────────────────────────────────────────────────────────────

async function predict(file) {
  showLoading(true);

  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch('/predict', { method: 'POST', body: formData });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || `Server error (${response.status})`);
    }

    const data = await response.json();
    renderResults(data);
  } catch (err) {
    showError(err.message || 'Prediction failed. Is the server running?');
  } finally {
    showLoading(false);
  }
}

// ── Render results ────────────────────────────────────────────────────────────

function renderResults(data) {
  uploadSection.classList.add('hidden');
  resultsSection.classList.remove('hidden');

  const color = BIOME_COLORS[data.top_prediction] || '#5cb85c';
  const pct = (data.top_probability * 100).toFixed(1);

  topBiomeName.textContent = data.top_prediction;
  topProbabilityText.textContent = `${pct}%`;
  topPredictionCard.style.borderLeftColor = color;

  // Animate confidence bar: reset to 0 first, then transition to final value
  topConfidenceBar.style.transition = 'none';
  topConfidenceBar.style.width = '0%';
  topConfidenceBar.style.background = color;
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      topConfidenceBar.style.transition = '';
      topConfidenceBar.style.width = `${pct}%`;
    });
  });

  lowConfidenceWarn.classList.toggle('hidden', !data.low_confidence);

  // Bar chart
  chartContainer.innerHTML = '';
  data.predictions.forEach((pred, idx) => {
    const barPct = (pred.probability * 100).toFixed(2);
    const isTop = idx === 0;
    const barColor = BIOME_COLORS[pred.biome] || '#5cb85c';

    const row = document.createElement('div');
    row.className = 'bar-row' + (isTop ? ' bar-row--top' : '');

    const label = document.createElement('span');
    label.className = 'bar-label';
    label.textContent = pred.biome;

    const track = document.createElement('div');
    track.className = 'bar-track';

    const fill = document.createElement('div');
    fill.className = 'bar-fill';
    fill.style.background = barColor;
    fill.style.width = '0%';
    track.appendChild(fill);

    const value = document.createElement('span');
    value.className = 'bar-value';
    value.textContent = `${barPct}%`;

    row.appendChild(label);
    row.appendChild(track);
    row.appendChild(value);
    chartContainer.appendChild(row);

    // Staggered animation
    const delay = idx * 30;
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        setTimeout(() => { fill.style.width = `${barPct}%`; }, delay);
      });
    });
  });
}

// ── Reset ─────────────────────────────────────────────────────────────────────

function resetUI() {
  fileInput.value = '';
  previewImg.src = '';
  chartContainer.innerHTML = '';
  topBiomeName.textContent = '—';
  topProbabilityText.textContent = '—';
  topConfidenceBar.style.width = '0%';
  lowConfidenceWarn.classList.add('hidden');
  resultsSection.classList.add('hidden');
  uploadSection.classList.remove('hidden');
  hideError();
}

resetBtn.addEventListener('click', resetUI);

// ── Utilities ─────────────────────────────────────────────────────────────────

function showLoading(visible) {
  loadingOverlay.classList.toggle('hidden', !visible);
}

function showError(msg) {
  errorText.textContent = msg;
  errorBox.classList.remove('hidden');
}

function hideError() {
  errorBox.classList.add('hidden');
}
