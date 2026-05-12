/* =====================================================================
   BOTANICAL ATLAS — dashboard.js
   Live prediction (debounced fetch), batch upload, Plotly charts.
   No external services — all calls go to internal Flask routes.
   ===================================================================== */

// ---- Plotly theme -----------------------------------------------------
const THEME = {
  ink:        '#f3ead2',
  ink2:       '#a8b0a3',
  hairline:   '#3e6655',
  bg:         'rgba(0,0,0,0)',
  leaf:       '#84a98c',
  moss:       '#a7c4ae',
  saffron:    '#e9c46a',
  turmeric:   '#d99549',
  terracotta: '#cc6b4a',
  sky:        '#6f9bb3',
  berry:      '#a44a5e',
};

const PLOTLY_FONT = { family: 'Manrope, sans-serif', size: 12, color: THEME.ink };
const PLOTLY_AXIS = {
  gridcolor:  THEME.hairline,
  zerolinecolor: THEME.hairline,
  linecolor:  THEME.hairline,
  tickcolor:  THEME.hairline,
  tickfont:   { color: THEME.ink2, size: 11, family: 'JetBrains Mono, monospace' },
  titlefont:  { color: THEME.ink2, size: 12 },
};
const PLOTLY_BASE_LAYOUT = {
  paper_bgcolor: THEME.bg,
  plot_bgcolor:  THEME.bg,
  font:          PLOTLY_FONT,
  margin:        { l: 60, r: 20, t: 24, b: 50 },
  xaxis:         { ...PLOTLY_AXIS },
  yaxis:         { ...PLOTLY_AXIS },
  hoverlabel:    { bgcolor: '#1b362d', font: { color: THEME.ink, family: 'JetBrains Mono' }, bordercolor: THEME.hairline },
};
const PLOTLY_CONFIG = { displayModeBar: false, responsive: true };

// ---- Helpers ----------------------------------------------------------
function debounce(fn, ms = 280) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

function fmt(n, d = 2) {
  if (typeof n !== 'number' || isNaN(n)) return '—';
  return n.toFixed(d);
}

// =====================================================================
// LIVE PREDICT
// =====================================================================
function initLivePredict() {
  const features = window.FEATURES || ['N','P','K','temperature','humidity','ph','rainfall'];

  // Wire dial pairs (range <-> number) to stay in sync
  features.forEach(f => {
    const r = document.getElementById('rng-' + f);
    const n = document.getElementById('num-' + f);
    if (!r || !n) return;
    r.addEventListener('input', () => { n.value = r.value; maybeRun(); });
    n.addEventListener('input', () => { r.value = n.value; maybeRun(); });
  });

  // Manual run button
  document.getElementById('run-now')?.addEventListener('click', () => runPredict(true));

  // Randomise
  document.getElementById('randomize')?.addEventListener('click', () => {
    const ranges = {
      N:[10,140], P:[10,145], K:[10,205],
      temperature:[12,40], humidity:[20,98], ph:[4.5,9], rainfall:[40,280]
    };
    features.forEach(f => {
      const [lo, hi] = ranges[f] || [0, 100];
      const v = (Math.random() * (hi - lo) + lo);
      const r = document.getElementById('rng-' + f);
      const n = document.getElementById('num-' + f);
      const step = parseFloat(r.step) || 1;
      const val = step >= 1 ? Math.round(v) : v.toFixed(2);
      r.value = val; n.value = val;
    });
    runPredict(true);
  });

  // Feedback handlers
  document.getElementById('fb-yes')?.addEventListener('click', () => sendFeedback(true));
  document.getElementById('fb-no') ?.addEventListener('click', () => sendFeedback(false));

  // Initial render
  drawEmptyCharts();
  runPredict(true);
}

const debouncedRun = debounce(() => runPredict(false), 320);
function maybeRun() {
  const auto = document.getElementById('auto-run')?.checked;
  if (auto) debouncedRun();
}

let _lastInputs = null;
let _lastResult = null;

async function runPredict(showSpinner) {
  const features = window.FEATURES || ['N','P','K','temperature','humidity','ph','rainfall'];
  const payload = { n_runs: 6 };
  features.forEach(f => {
    const n = document.getElementById('num-' + f);
    payload[f] = parseFloat(n.value);
  });
  _lastInputs = payload;

  const sp = document.getElementById('spinner');
  if (showSpinner && sp) sp.hidden = false;

  try {
    const resp = await fetch('/predict/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await resp.json();
    if (!data.ok) throw new Error(data.error || 'Prediction failed');
    _lastResult = data.result;
    renderResult(data.result, payload);
  } catch (e) {
    console.error(e);
    const note = document.getElementById('r-note');
    if (note) note.textContent = 'Error: ' + e.message;
  } finally {
    if (sp) sp.hidden = true;
  }
}

function renderResult(r, payload) {
  document.getElementById('r-crop').textContent = r.label;
  document.getElementById('r-conf').textContent = fmt(r.confidence, 1);

  const ring = document.getElementById('ring-fg');
  if (ring) {
    const C = 2 * Math.PI * 52; // circumference
    const v = (r.confidence / 100) * C;
    ring.setAttribute('stroke-dasharray', `${v} ${C - v}`);
  }

  const dossier = r.dossier || {};
  document.getElementById('r-family').textContent = dossier.family || '—';
  document.getElementById('r-season').textContent = dossier.season || '—';
  document.getElementById('r-note').textContent   = dossier.note   || '';

  // Reveal feedback row
  const fb = document.getElementById('feedback-row');
  if (fb) fb.hidden = false;

  // Top-5 horizontal bar
  drawTop5(r.top5);

  // Radar of standardised inputs
  drawRadar(r.scaled, r.features);

  // Full distribution
  drawDistribution(r.all_probs, r.label);
}

function drawEmptyCharts() {
  Plotly.newPlot('chart-top5', [{ type:'bar', x:[], y:[], orientation:'h' }],
    { ...PLOTLY_BASE_LAYOUT, height: 300 }, PLOTLY_CONFIG);
  Plotly.newPlot('chart-radar', [{ type:'scatterpolar', r:[], theta:[] }],
    { ...PLOTLY_BASE_LAYOUT, polar: { bgcolor: THEME.bg } }, PLOTLY_CONFIG);
  Plotly.newPlot('chart-dist', [{ type:'bar', x:[], y:[] }],
    { ...PLOTLY_BASE_LAYOUT }, PLOTLY_CONFIG);
}

function drawTop5(top5) {
  const labels = top5.map(t => t.label);
  const probs  = top5.map(t => t.prob);
  const colors = labels.map((_, i) => i === 0 ? THEME.saffron : THEME.leaf);
  const trace = {
    type: 'bar',
    orientation: 'h',
    x: probs.slice().reverse(),
    y: labels.slice().reverse(),
    marker: { color: colors.slice().reverse(),
              line: { color: THEME.hairline, width: 1 } },
    text: probs.slice().reverse().map(v => fmt(v, 1) + '%'),
    textposition: 'outside',
    textfont: { color: THEME.ink, family: 'JetBrains Mono', size: 11 },
    hovertemplate: '<b>%{y}</b><br>%{x:.2f}%<extra></extra>',
  };
  const layout = {
    ...PLOTLY_BASE_LAYOUT,
    height: 320,
    margin: { l: 110, r: 60, t: 16, b: 36 },
    xaxis: { ...PLOTLY_AXIS, range: [0, Math.max(110, Math.max(...probs) * 1.2)],
             ticksuffix: '%', title: { text: 'softmax %', font: { color: THEME.ink2, size: 11 } } },
    yaxis: { ...PLOTLY_AXIS, automargin: true,
             tickfont: { ...PLOTLY_AXIS.tickfont, family: 'Manrope', size: 13, color: THEME.ink } },
  };
  Plotly.react('chart-top5', [trace], layout, PLOTLY_CONFIG);
}

function drawRadar(scaled, features) {
  // Shorten long labels so they don't get clipped on the radar chart
  const LABEL_MAP = {
    temperature: 'Temp',
    humidity:    'Hum',
    rainfall:    'Rain',
  };
  const labels = features.map(f => LABEL_MAP[f] || f.toUpperCase());
  const r = scaled.map(v => v + 4);  // typical standardised features fall within [-3, +3]
  const trace = {
    type: 'scatterpolar',
    r: [...r, r[0]],
    theta: [...labels, labels[0]],
    fill: 'toself',
    line: { color: THEME.saffron, width: 2 },
    fillcolor: 'rgba(233,196,106,.18)',
    marker: { color: THEME.saffron, size: 6 },
    hovertemplate: '<b>%{theta}</b><br>z = %{customdata:.2f}<extra></extra>',
    customdata: [...scaled, scaled[0]],
  };
  const layout = {
    ...PLOTLY_BASE_LAYOUT,
    height: 320,
    margin: { l: 30, r: 30, t: 16, b: 16 },
    polar: {
      bgcolor: 'rgba(0,0,0,0)',
      radialaxis: { visible: true, range: [0, 8], showline: false,
                    gridcolor: THEME.hairline, tickfont: { size: 9, color: THEME.ink2 } },
      angularaxis: { gridcolor: THEME.hairline,
                     tickfont: { size: 11, color: THEME.ink, family: 'JetBrains Mono' } },
    },
    showlegend: false,
  };
  Plotly.react('chart-radar', [trace], layout, PLOTLY_CONFIG);
}

function drawDistribution(allProbs, winningLabel) {
  const labels = allProbs.map(t => t.label);
  const probs  = allProbs.map(t => t.prob);
  const colors = labels.map(l =>
    l === winningLabel ? THEME.saffron :
    probs[labels.indexOf(l)] > 5 ? THEME.leaf :
    THEME.hairline);
  const trace = {
    type: 'bar',
    x: labels,
    y: probs,
    marker: { color: colors, line: { color: THEME.hairline, width: 1 } },
    hovertemplate: '<b>%{x}</b><br>%{y:.2f}%<extra></extra>',
  };
  const layout = {
    ...PLOTLY_BASE_LAYOUT,
    height: 380,
    margin: { l: 50, r: 16, t: 16, b: 90 },
    xaxis: { ...PLOTLY_AXIS, tickangle: -40,
             tickfont: { ...PLOTLY_AXIS.tickfont, family: 'JetBrains Mono', size: 10 } },
    yaxis: { ...PLOTLY_AXIS, ticksuffix: '%',
             title: { text: 'probability', font: { color: THEME.ink2, size: 11 } } },
  };
  Plotly.react('chart-dist', [trace], layout, PLOTLY_CONFIG);
}

async function sendFeedback(agreed) {
  if (!_lastResult) return;
  const correct = document.getElementById('fb-correct')?.value || '';
  const status  = document.getElementById('fb-status');
  const payload = {
    ..._lastInputs,
    predicted:  _lastResult.label,
    confidence: _lastResult.confidence,
    agreed,
    user_label: correct,
    comment:    '',
  };
  try {
    await fetch('/feedback/submit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (status) {
      status.textContent = agreed ? '✓ Recorded — thanks for confirming.'
                                  : '✓ Recorded — we\'ll learn from this.';
      setTimeout(() => { status.textContent = ''; }, 3500);
    }
  } catch (e) {
    if (status) status.textContent = 'Could not save feedback';
  }
}

// =====================================================================
// BATCH UPLOAD
// =====================================================================
let _batchData = null;
let _batchPage = 0;
const _pageSize = 25;

function initBatchUpload() {
  const dz   = document.getElementById('dropzone');
  const file = document.getElementById('file');
  const browse = document.getElementById('browse');

  dz.addEventListener('click', (e) => {
    if (e.target === browse) return;
    file.click();
  });
  browse?.addEventListener('click', (e) => { e.stopPropagation(); file.click(); });

  file.addEventListener('change', () => {
    if (file.files[0]) handleFile(file.files[0]);
  });
  dz.addEventListener('dragover', (e) => { e.preventDefault(); dz.classList.add('over'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('over'));
  dz.addEventListener('drop', (e) => {
    e.preventDefault();
    dz.classList.remove('over');
    if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
  });

  // Pager
  document.getElementById('prev')?.addEventListener('click', () => {
    if (_batchPage > 0) { _batchPage--; renderTable(); }
  });
  document.getElementById('next')?.addEventListener('click', () => {
    const total = (_batchData?.rows?.length || 0);
    if ((_batchPage + 1) * _pageSize < total) { _batchPage++; renderTable(); }
  });
  document.getElementById('rowsearch')?.addEventListener('input', () => { _batchPage = 0; renderTable(); });
  document.getElementById('rowfilter')?.addEventListener('change', () => { _batchPage = 0; renderTable(); });
}

async function handleFile(f) {
  const status = document.getElementById('upload-status');
  status.hidden = false;
  status.className = 'upload-status info';
  status.innerHTML = `<span class="spinner" style="display:inline-block;border-color:rgba(111,155,179,.3);border-top-color:${THEME.sky};vertical-align:middle;margin-right:8px"></span> Reading <b>${f.name}</b> (${(f.size/1024).toFixed(1)} KB)…`;

  const fd = new FormData();
  fd.append('file', f);

  try {
    const resp = await fetch('/upload', { method: 'POST', body: fd });
    const data = await resp.json();
    if (!data.ok) throw new Error(data.error || 'Upload failed');

    // Build status line — include auto-feedback summary if present
    const fb = data.result.auto_feedback_summary;
    let fbNote = '';
    if (fb && fb.total > 0) {
      const pct = (fb.agreed / fb.total * 100).toFixed(1);
      fbNote = ` &nbsp;·&nbsp; Auto-feedback: <b>${fb.agreed}</b> agreed, <b>${fb.disagreed}</b> flagged `
             + `<span style="opacity:.6">(${pct}% pass rate)</span>`;
    }
    status.className = 'upload-status ok';
    status.innerHTML = `✓ Predicted <b>${data.result.n.toLocaleString()}</b> rows from <b>${f.name}</b>.${fbNote}`;
    showBatchResults(data.result);
  } catch (e) {
    status.className = 'upload-status err';
    status.textContent = '✗ ' + e.message;
  }
}

function showBatchResults(res) {
  _batchData = res;
  _batchPage = 0;
  document.getElementById('batch-results').hidden = false;
  document.getElementById('download-link').href = res.download;

  // Metric row & extra cards
  const mrow  = document.getElementById('metrics-row');
  const cmCard = document.getElementById('cm-card');
  const mcCard = document.getElementById('metrics-card');

  if (res.has_labels) {
    mrow.hidden = false; cmCard.hidden = false; mcCard.hidden = false;
    mrow.innerHTML = `
      <div class="mrow-cell"><div class="mrow-num">${fmt(res.accuracy,2)}<small>%</small></div><div class="mrow-lbl">accuracy</div></div>
      <div class="mrow-cell"><div class="mrow-num">${fmt(res.f1,3)}</div>           <div class="mrow-lbl">f1 weighted</div></div>
      <div class="mrow-cell"><div class="mrow-num">${fmt(res.precision,3)}</div>    <div class="mrow-lbl">precision</div></div>
      <div class="mrow-cell"><div class="mrow-num">${fmt(res.recall,3)}</div>       <div class="mrow-lbl">recall</div></div>
      <div class="mrow-cell"><div class="mrow-num">${res.n.toLocaleString()}</div>  <div class="mrow-lbl">rows scored</div></div>
    `;

    // Confusion matrix: unchanged, always uses the full class list from the model
    drawConfusion(res.confusion, res.classes);

    // F1 & Precision per class: only show crops present in this upload
    // (either as a true label or as a prediction — catches misclassified rows too)
    const presentClasses = new Set(res.rows.map(r => r.predicted));
    res.rows.forEach(r => r.true && presentClasses.add(r.true));
    const activeIdx     = res.classes.map((c, i) => presentClasses.has(c) ? i : -1).filter(i => i >= 0);
    const activeClasses = activeIdx.map(i => res.classes[i]);
    const activeF1      = activeIdx.map(i => res.f1_per[i]);
    const activePrec    = activeIdx.map(i => res.prec_per[i]);
    drawPerClass(activeF1, activePrec, activeClasses);

  } else {
    mrow.hidden = true; cmCard.hidden = true; mcCard.hidden = true;
  }

  drawCropMix(res.rows, res.classes);
  drawConfDist(res.rows);
  renderTable();

  document.getElementById('batch-results').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function drawCropMix(rows, classes) {
  const counts = {};
  classes.forEach(c => counts[c] = 0);
  rows.forEach(r => { counts[r.predicted] = (counts[r.predicted] || 0) + 1; });
  const sorted = Object.entries(counts).sort((a,b) => b[1] - a[1]);
  const labels = sorted.map(s => s[0]);
  const values = sorted.map(s => s[1]);
  const total  = values.reduce((a,b)=>a+b,0) || 1;
  const trace = {
    type: 'bar',
    x: values,
    y: labels,
    orientation: 'h',
    marker: { color: values.map((_, i) => i < 3 ? THEME.saffron : THEME.leaf),
              line: { color: THEME.hairline, width: 1 } },
    text: values.map(v => `${v} · ${(v/total*100).toFixed(1)}%`),
    textposition: 'outside',
    textfont: { color: THEME.ink2, family: 'JetBrains Mono', size: 11 },
    hovertemplate: '<b>%{y}</b><br>%{x} rows<extra></extra>',
  };
  const layout = {
    ...PLOTLY_BASE_LAYOUT,
    height: 380,
    margin: { l: 110, r: 80, t: 12, b: 36 },
    xaxis: { ...PLOTLY_AXIS, title: { text: 'rows', font: { color: THEME.ink2, size: 11 } } },
    yaxis: { ...PLOTLY_AXIS, automargin: true,
             tickfont: { ...PLOTLY_AXIS.tickfont, family: 'Manrope', size: 12, color: THEME.ink } },
  };
  Plotly.react('chart-mix', [trace], layout, PLOTLY_CONFIG);
}

function drawConfDist(rows) {
  const confs = rows.map(r => r.confidence);
  const trace = {
    type: 'histogram',
    x: confs,
    nbinsx: 25,
    marker: { color: THEME.leaf, line: { color: THEME.hairline, width: 1 } },
    hovertemplate: '%{x:.0f}–%{x:.0f}%<br>%{y} rows<extra></extra>',
  };
  const layout = {
    ...PLOTLY_BASE_LAYOUT,
    height: 380,
    margin: { l: 50, r: 16, t: 12, b: 36 },
    xaxis: { ...PLOTLY_AXIS, ticksuffix: '%',
             title: { text: 'confidence', font: { color: THEME.ink2, size: 11 } } },
    yaxis: { ...PLOTLY_AXIS, title: { text: 'rows', font: { color: THEME.ink2, size: 11 } } },
    bargap: 0.04,
  };
  Plotly.react('chart-conf', [trace], layout, PLOTLY_CONFIG);
}

function drawConfusion(cm, classes) {
  // Custom colorscale matching the theme
  const trace = {
    type: 'heatmap',
    z: cm,
    x: classes,
    y: classes,
    colorscale: [
      [0,    'rgba(27, 54, 45, 0.0)'],
      [0.05, 'rgba(132, 169, 140, 0.18)'],
      [0.4,  'rgba(132, 169, 140, 0.6)'],
      [0.7,  'rgba(233, 196, 106, 0.8)'],
      [1,    'rgba(233, 196, 106, 1)'],
    ],
    showscale: true,
    colorbar: { tickfont: { color: THEME.ink2, family: 'JetBrains Mono', size: 10 },
                outlinecolor: THEME.hairline, thickness: 12, len: 0.7 },
    hovertemplate: 'true: <b>%{y}</b><br>pred: <b>%{x}</b><br>count: %{z}<extra></extra>',
  };
  // Add count annotations only for nonzero cells to avoid clutter
  const annotations = [];
  for (let i = 0; i < classes.length; i++) {
    for (let j = 0; j < classes.length; j++) {
      const v = cm[i][j];
      if (v > 0) {
        annotations.push({
          x: classes[j], y: classes[i], text: String(v),
          showarrow: false,
          font: { color: i === j ? THEME.bg : THEME.ink, size: 10, family: 'JetBrains Mono' },
        });
      }
    }
  }
  const layout = {
    ...PLOTLY_BASE_LAYOUT,
    height: 540,
    margin: { l: 110, r: 30, t: 12, b: 110 },
    xaxis: { ...PLOTLY_AXIS, tickangle: -40, side: 'bottom',
             tickfont: { ...PLOTLY_AXIS.tickfont, size: 10 }, automargin: true },
    yaxis: { ...PLOTLY_AXIS, autorange: 'reversed',
             tickfont: { ...PLOTLY_AXIS.tickfont, size: 10 }, automargin: true },
    annotations,
  };
  Plotly.react('chart-cm', [trace], layout, PLOTLY_CONFIG);
}

function drawPerClass(f1, prec, classes) {
  // Guard: nothing to render
  if (!classes || classes.length === 0) return;

  const trace1 = {
    type: 'bar', name: 'F1',
    x: classes,
    y: f1.map(v => (typeof v === 'number' && !isNaN(v)) ? v : 0),
    marker: { color: THEME.leaf, line: { color: THEME.hairline, width: 1 } },
    text: f1.map(v => (typeof v === 'number' && !isNaN(v)) ? v.toFixed(2) : '0.00'),
    textposition: 'outside',
    textfont: { color: THEME.ink, family: 'JetBrains Mono', size: 10 },
    cliponaxis: false,
    hovertemplate: '<b>%{x}</b><br>F1: %{y:.3f}<extra></extra>',
  };
  const trace2 = {
    type: 'bar', name: 'Precision',
    x: classes,
    y: prec.map(v => (typeof v === 'number' && !isNaN(v)) ? v : 0),
    marker: { color: THEME.saffron, line: { color: THEME.hairline, width: 1 } },
    text: prec.map(v => (typeof v === 'number' && !isNaN(v)) ? v.toFixed(2) : '0.00'),
    textposition: 'outside',
    textfont: { color: THEME.ink, family: 'JetBrains Mono', size: 10 },
    cliponaxis: false,
    hovertemplate: '<b>%{x}</b><br>Precision: %{y:.3f}<extra></extra>',
  };
  // Dynamic height: taller when there are many classes
  const h = Math.max(420, classes.length * 28 + 160);
  const layout = {
    ...PLOTLY_BASE_LAYOUT,
    height: h,
    barmode: 'group',
    bargap: 0.25,
    bargroupgap: 0.08,
    margin: { l: 50, r: 20, t: 40, b: classes.length > 6 ? 110 : 70 },
    xaxis: {
      ...PLOTLY_AXIS,
      tickangle: classes.length > 6 ? -40 : 0,
      tickfont: { ...PLOTLY_AXIS.tickfont, family: 'JetBrains Mono', size: classes.length > 10 ? 9 : 11 },
      automargin: true,
    },
    yaxis: {
      ...PLOTLY_AXIS,
      range: [0, 1.18],
      title: { text: 'score', font: { color: THEME.ink2, size: 11 } },
    },
    legend: {
      font: { color: THEME.ink2, family: 'JetBrains Mono', size: 11 },
      bgcolor: 'rgba(0,0,0,0)',
      orientation: 'h',
      x: 0, y: 1.08,
    },
  };
  Plotly.react('chart-perclass', [trace1, trace2], layout, PLOTLY_CONFIG);
}

function renderTable() {
  if (!_batchData) return;
  const rows = _batchData.rows;
  const features = _batchData.features;
  const hasLabels = _batchData.has_labels;

  const search   = (document.getElementById('rowsearch')?.value || '').toLowerCase();
  const filter   = document.getElementById('rowfilter')?.value || 'all';

  let filtered = rows;
  if (filter === 'correct')   filtered = filtered.filter(r => r.correct);
  if (filter === 'incorrect') filtered = filtered.filter(r => r.correct === false);
  if (search) {
    filtered = filtered.filter(r =>
      r.predicted.toLowerCase().includes(search) ||
      (r.true && r.true.toLowerCase().includes(search))
    );
  }

  const head = document.getElementById('rows-head');
  const body = document.getElementById('rows-body');

  // Header
  let h = '<th>#</th>';
  features.forEach(f => h += `<th>${f}</th>`);
  h += '<th>Predicted</th><th>Conf.</th>';
  if (hasLabels) h += '<th>True</th><th>✓</th>';
  head.innerHTML = h;

  // Body (paged)
  const start = _batchPage * _pageSize;
  const end   = Math.min(start + _pageSize, filtered.length);
  let b = '';
  for (let i = start; i < end; i++) {
    const r = filtered[i];
    let row = `<tr><td class="mono small">${r.row}</td>`;
    features.forEach(f => row += `<td>${typeof r[f]==='number' ? fmt(r[f],2) : r[f]}</td>`);
    row += `<td><span class="crop-chip slim">${r.predicted}</span></td>`;
    row += `<td>${fmt(r.confidence, 1)}<small>%</small></td>`;
    if (hasLabels) {
      row += `<td><span class="crop-chip slim ${r.correct ? '' : 'alt'}">${r.true}</span></td>`;
      row += `<td>${r.correct ? '<span class="dot-yes">●</span>' : '<span class="dot-no">●</span>'}</td>`;
    }
    row += '</tr>';
    b += row;
  }
  body.innerHTML = b || `<tr><td colspan="20" class="empty" style="padding:24px">No rows match.</td></tr>`;

  document.getElementById('pageinfo').textContent =
    filtered.length === 0
      ? '0 / 0'
      : `${start + 1}–${end} of ${filtered.length}${filtered.length !== rows.length ? ` (filtered from ${rows.length})` : ''}`;
}