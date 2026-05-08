// CrowdVision v4 — Frontend Logic

const NUM_ZONES = 6;
const ZN = (i) => `Zone-${String.fromCharCode(65 + i)}`;
let zoneFiles = new Array(NUM_ZONES).fill(null);
let contextData = { zones: [] };
let sessionId = Date.now().toString();

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
  setupZoneGrid();
  checkHealth();
  loadSamples();
});

function showPage(pageId, navBtn) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.getElementById(`page-${pageId}`).classList.add('active');
  
  if (navBtn) {
    document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
    navBtn.classList.add('active');
  }
}

function setupZoneGrid() {
  const grid = document.getElementById('zoneGrid');
  grid.innerHTML = Array.from({length: NUM_ZONES}, (_, i) => `
    <div class="zone-slot" id="zs${i}">
      <div class="zone-name">${ZN(i)}</div>
      <div style="margin-bottom:8px; opacity:0.5">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="24" height="24"><path d="M23 19a2 2 0 01-2 2H3a2 2 0 01-2-2V8a2 2 0 012-2h4l2-3h6l2 3h4a2 2 0 012 2z"/><circle cx="12" cy="13" r="4"/></svg>
      </div>
      <div style="font-size:12px; color:var(--text-muted)">Drop image or click</div>
      <input type="file" accept="image/*" onchange="handleFileSelect(${i}, this.files[0])">
    </div>
  `).join('');

  // Setup drag & drop
  for (let i = 0; i < NUM_ZONES; i++) {
    const el = document.getElementById(`zs${i}`);
    el.ondragover = e => { e.preventDefault(); el.style.borderColor = 'var(--accent-blue)'; };
    el.ondragleave = () => el.style.borderColor = '';
    el.ondrop = e => {
      e.preventDefault();
      el.style.borderColor = '';
      if (e.dataTransfer.files[0]) handleFileSelect(i, e.dataTransfer.files[0]);
    };
  }
}

async function checkHealth() {
  const st = document.getElementById('sysStatus');
  const mg = document.getElementById('modelGrid');
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    
    st.innerHTML = `<div class="status-dot online"></div><span>System Online</span>`;
    
    // Model grid
    let mh = '';
    if (data.models.density) mh += `<div class="model-item"><span>Density Analysis</span> <span class="badge" style="color:var(--emerald);border-color:var(--emerald)">READY</span></div>`;
    if (data.models.anomaly) mh += `<div class="model-item"><span>Behavioral Anomaly</span> <span class="badge" style="color:var(--emerald);border-color:var(--emerald)">READY</span></div>`;
    if (data.openai) mh += `<div class="model-item"><span>Semantic Engine</span> <span class="badge" style="color:var(--emerald);border-color:var(--emerald)">READY</span></div>`;
    else mh += `<div class="model-item"><span>Semantic Engine</span> <span class="badge" style="color:var(--rose);border-color:var(--rose)">OFFLINE</span></div>`;
    
    mg.innerHTML = mh || 'No models loaded.';
  } catch (e) {
    st.innerHTML = `<div class="status-dot offline"></div><span>API Offline</span>`;
    mg.innerHTML = 'Cannot connect to backend server.';
  }
}

async function loadSamples() {
  try {
    const res = await fetch('/api/samples');
    const data = await res.json();
    const all = [...data.density, ...data.anomaly_normal, ...data.anomaly_abnormal];
    document.getElementById('sampleRow').innerHTML = all.map(u => 
      `<img class="sample-img" src="${u}" onclick="fillNextSample('${u}')">`
    ).join('');
  } catch (e) {
    document.getElementById('sampleRow').innerHTML = '<span style="font-size:12px;color:var(--text-muted)">No samples found.</span>';
  }
}

function handleFileSelect(i, file) {
  if (!file) return;
  zoneFiles[i] = file;
  const el = document.getElementById(`zs${i}`);
  const url = URL.createObjectURL(file);
  
  el.innerHTML = `
    <div class="zone-name">${ZN(i)}</div>
    <img src="${url}">
    <div style="font-size:11px; margin-top:8px; color:var(--emerald)">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="12" height="12" style="vertical-align:middle;margin-right:4px"><polyline points="20 6 9 17 4 12"/></svg>
      Feed Active
    </div>
    <input type="file" accept="image/*" onchange="handleFileSelect(${i}, this.files[0])">
  `;
  el.classList.add('filled');
  updateSetupCount();
}

async function fillNextSample(url) {
  const emptyIdx = zoneFiles.findIndex(f => f === null);
  if (emptyIdx < 0) return;
  const res = await fetch(url);
  const blob = await res.blob();
  handleFileSelect(emptyIdx, new File([blob], url.split('/').pop(), { type: blob.type }));
}

function updateSetupCount() {
  const count = zoneFiles.filter(f => f !== null).length;
  document.getElementById('zoneCount').textContent = `${count} / ${NUM_ZONES}`;
  document.getElementById('analyzeBtn').disabled = count === 0;
  document.getElementById('dZones').textContent = count;
}

// ── ANALYSIS LOGIC ───────────────────────────────────────────────────

async function startAnalysis() {
  const overlay = document.getElementById('procOverlay');
  const bar = document.getElementById('procFill');
  const msg = document.getElementById('procMsg');
  
  overlay.style.display = 'flex';
  contextData = { zones: [] };
  
  const filled = zoneFiles.map((f, i) => f ? { idx: i, file: f } : null).filter(Boolean);
  const total = filled.length;
  let totalPeople = 0;
  let maxRisk = 0;
  let latencies = [];

  // 1. Process Zones
  for (let j = 0; j < total; j++) {
    const { idx, file } = filled[j];
    msg.textContent = `Analyzing ${ZN(idx)} (${j + 1}/${total})...`;
    bar.style.width = `${((j) / total) * 60}%`;
    
    try {
      const fd = new FormData();
      fd.append('file', file);
      fd.append('zone_name', ZN(idx));
      
      const res = await fetch('/api/analyze', { method: 'POST', body: fd });
      const data = await res.json();
      contextData.zones.push(data);
      
      totalPeople += data.crowd_count || 0;
      if (data.risk_score > maxRisk) maxRisk = data.risk_score;
      if (data.density && data.density.latency_ms) latencies.push(data.density.latency_ms);
      
    } catch (e) {
      contextData.zones.push({ zone: ZN(idx), error: e.message });
    }
    bar.style.width = `${((j + 1) / total) * 60}%`;
  }

  // Update Dash stats
  document.getElementById('dTotalPeople').textContent = Math.round(totalPeople);
  document.getElementById('dThreatLevel').textContent = 
    maxRisk > 0.75 ? 'CRITICAL' : maxRisk > 0.55 ? 'HIGH' : maxRisk > 0.35 ? 'ELEVATED' : 'NORMAL';
  if (latencies.length) {
    document.getElementById('dLatency').textContent = `${Math.round(latencies.reduce((a,b)=>a+b,0)/latencies.length)}ms`;
  }

  // 2. Dispatch & Report
  msg.textContent = 'Generating AI dispatch decisions & safety report...';
  bar.style.width = '80%';
  
  try {
    const fd = new FormData();
    fd.append('data', JSON.stringify({ zones: contextData.zones }));
    const combo = await (await fetch('/api/dispatch_and_report', { method: 'POST', body: fd })).json();
    contextData.dispatch = combo.dispatch || {};
    contextData.report = combo.report || {};
  } catch (e) {
    contextData.dispatch = { error: e.message };
    contextData.report = { error: e.message };
  }

  bar.style.width = '100%';
  msg.textContent = 'Analysis complete!';
  
  setTimeout(() => {
    overlay.style.display = 'none';
    document.getElementById('navResults').style.display = 'flex';
    document.getElementById('navDispatch').style.display = 'flex';
    renderResults();
    renderDispatch();
    showPage('results', document.getElementById('navResults'));
  }, 500);
}

// ── RENDER RESULTS ───────────────────────────────────────────────────

function renderResults() {
  // Tabs
  const tabs = document.getElementById('zoneTabs');
  tabs.innerHTML = contextData.zones.map((z, i) => 
    `<button class="z-tab ${i === 0 ? 'active' : ''}" onclick="showZoneDetails(${i}, this)">${z.zone}</button>`
  ).join('');
  
  if (contextData.zones.length > 0) {
    showZoneDetails(0, tabs.firstChild);
  }

  // Risk Matrix
  const grid = document.getElementById('riskGrid');
  grid.innerHTML = contextData.zones.map(z => {
    if (z.error) return `<div class="risk-item"><b>${z.zone}</b><p style="color:var(--rose)">Error: ${z.error}</p></div>`;
    const lv = z.risk_level || 'LOW';
    return `
      <div class="risk-item">
        <div class="risk-level-badge risk-${lv}">${lv}</div>
        <div style="font-weight:600;font-size:16px;margin-bottom:8px">${z.zone}</div>
        <div style="display:flex;justify-content:space-between;font-size:13px;color:var(--text-muted);margin-bottom:4px">
          <span>Risk Score</span>
          <span style="color:var(--text-main);font-family:var(--font-mono)">${(z.risk_score * 100).toFixed(1)}%</span>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:13px;color:var(--text-muted)">
          <span>Crowd Count</span>
          <span style="color:var(--text-main);font-family:var(--font-mono)">${Math.round(z.crowd_count)}</span>
        </div>
      </div>
    `;
  }).join('');
}

function showZoneDetails(idx, btn) {
  document.querySelectorAll('.z-tab').forEach(b => b.classList.remove('active'));
  if (btn) btn.classList.add('active');
  
  const z = contextData.zones[idx];
  const detail = document.getElementById('zoneDetail');
  
  if (!z || z.error) {
    detail.innerHTML = `<div style="color:var(--rose)">Error analyzing zone: ${z?.error || 'Unknown error'}</div>`;
    return;
  }

  const ai = z.ai_analysis || {};
  const hyb = z.anomaly || {};
  const isAnom = hyb.is_anomaly;

  let html = `
    <div class="result-metrics">
      <div class="rm-card">
        <div class="rm-icon color-blue"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 00-3-3.87"/><path d="M16 3.13a4 4 0 010 7.75"/></svg></div>
        <div class="rm-info"><div class="rm-val">${Math.round(z.crowd_count)}</div><div class="rm-lbl">People Estimated</div></div>
      </div>
      <div class="rm-card" style="border-color:${isAnom ? 'var(--rose)' : 'var(--border-light)'}">
        <div class="rm-icon ${isAnom ? 'color-rose' : 'color-emerald'}">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
        </div>
        <div class="rm-info">
          <div class="rm-val" style="color:${isAnom ? 'var(--rose)' : 'var(--text-main)'}">${(hyb.hybrid_score * 100).toFixed(1)}%</div>
          <div class="rm-lbl">Hybrid Anomaly Score</div>
        </div>
      </div>
      <div class="rm-card">
        <div class="rm-icon color-amber"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="9" y1="21" x2="9" y2="9"/></svg></div>
        <div class="rm-info"><div class="rm-val" style="font-size:18px">${z.density_level}</div><div class="rm-lbl">Density Level</div></div>
      </div>
      <div class="rm-card">
        <div class="rm-icon color-emerald"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg></div>
        <div class="rm-info"><div class="rm-val" style="font-size:18px">${(z.density?.latency_ms + hyb?.latency_ms || 15).toFixed(0)}ms</div><div class="rm-lbl">Local ML Latency</div></div>
      </div>
    </div>
    
    <div class="result-images">
      <div class="img-container">
        <div class="img-lbl">Raw Camera Feed</div>
        <img class="res-img" src="data:image/png;base64,${z.original_b64}">
      </div>
      <div class="img-container">
        <div class="img-lbl">Density Analysis Overlay</div>
        <img class="res-img" src="data:image/png;base64,${z.density?.overlay_b64 || z.density?.heatmap_b64}">
      </div>
    </div>
    
    <div class="ai-insight">
      <h3>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="18" height="18"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
        Semantic Analysis Insights
      </h3>
      <div class="insight-row">
        ${ai.area_description ? `<div><span style="color:var(--text-muted)">Area:</span> ${ai.area_description}</div>` : ''}
        ${ai.crowd_description ? `<div><span style="color:var(--text-muted)">Crowd:</span> ${ai.crowd_description}</div>` : ''}
        ${ai.flow_pattern ? `<div><span style="color:var(--text-muted)">Flow:</span> ${ai.flow_pattern}</div>` : ''}
        ${ai.anomaly_details && ai.anomaly_details !== 'none' ? `<div style="margin-top:8px;color:${isAnom?'var(--rose)':'var(--text-main)'}"><b>Detection:</b> ${ai.anomaly_details}</div>` : ''}
        ${ai.recommended_action && ai.recommended_action !== 'none' ? `<div style="margin-top:4px;color:var(--amber)"><b>Recommended:</b> ${ai.recommended_action}</div>` : ''}
      </div>
    </div>
  `;
  
  detail.innerHTML = html;
}

// ── DISPATCH & REPORT ────────────────────────────────────────────────

function renderDispatch() {
  const dp = contextData.dispatch || {};
  let h = '';
  
  if (dp.error) {
    document.getElementById('dispatchContent').innerHTML = `<div class="color-rose">Error: ${dp.error}</div>`;
    return;
  }
  
  if (dp.overall_threat_level) {
    const t = dp.overall_threat_level;
    const cl = t==='CRITICAL'||t==='SEVERE' ? 'risk-CRITICAL' : t==='HIGH' ? 'risk-HIGH' : t==='ELEVATED' ? 'risk-ELEVATED' : 'risk-LOW';
    h += `<div style="margin-bottom:16px">Threat Level: <span class="risk-level-badge ${cl}" style="margin-left:8px;margin-bottom:0">${t}</span></div>`;
  }
  
  if (dp.commander_summary) {
    h += `<p style="font-size:15px;margin-bottom:20px;color:var(--text-main)">${dp.commander_summary}</p>`;
  }
  
  if (dp.deployments && dp.deployments.length > 0) {
    h += `<h3 style="font-size:14px;color:var(--text-muted);margin-bottom:12px;text-transform:uppercase">Unit Deployments</h3>`;
    dp.deployments.forEach(d => {
      const u = d.urgency || 'MEDIUM';
      const ucol = u==='IMMEDIATE' ? 'var(--rose)' : u==='HIGH' ? 'var(--amber)' : 'var(--emerald)';
      h += `
        <div class="deploy-item">
          <div style="display:flex;justify-content:space-between;margin-bottom:6px">
            <b style="color:var(--accent-blue)">${d.zone}</b>
            <span style="color:${ucol};font-size:12px;font-weight:700">${u}</span>
          </div>
          <div>${d.units}x ${d.unit_type.toUpperCase()} — ${d.action}</div>
        </div>
      `;
    });
  } else {
    h += `<div class="deploy-item">No deployments required at this time.</div>`;
  }
  
  if (dp.escalation_triggers && dp.escalation_triggers.length > 0) {
    h += `<div style="margin-top:16px;color:var(--amber);font-size:13px"><b>Watch For:</b> ${dp.escalation_triggers.join(', ')}</div>`;
  }
  
  document.getElementById('dispatchContent').innerHTML = h;
  
  // Render Report
  const rpt = contextData.report || {};
  const rb = document.getElementById('reportContent');
  if (rpt.error) {
    rb.innerHTML = `<div class="color-rose">Error: ${rpt.error}</div>`;
  } else if (rpt.report) {
    let txt = rpt.report;
    txt = txt.replace(/\n/g, '<br>');
    txt = txt.replace(/\*\*(.*?)\*\*/g, '<b>$1</b>');
    txt = txt.replace(/^# (.*)/gm, '<h1>$1</h1>');
    txt = txt.replace(/^## (.*)/gm, '<h2>$1</h2>');
    txt = txt.replace(/^### (.*)/gm, '<h3>$1</h3>');
    rb.innerHTML = txt;
  }
}

async function downloadPDF() {
  const fd = new FormData();
  fd.append('data', JSON.stringify({ ...contextData, report: contextData.report?.report || '' }));
  const r = await fetch('/api/report/pdf', { method: 'POST', body: fd });
  const b = await r.blob();
  const a = document.createElement('a');
  a.href = URL.createObjectURL(b);
  a.download = 'CrowdVision_Safety_Report.pdf';
  a.click();
}

async function downloadJSON() {
  const fd = new FormData();
  fd.append('data', JSON.stringify(contextData));
  const r = await fetch('/api/report/json', { method: 'POST', body: fd });
  const b = await r.blob();
  const a = document.createElement('a');
  a.href = URL.createObjectURL(b);
  a.download = 'CrowdVision_Analysis.json';
  a.click();
}

// ── CHAT ─────────────────────────────────────────────────────────────

async function sendChat() {
  const inp = document.getElementById('chatInput');
  const txt = inp.value.trim();
  if (!txt) return;
  inp.value = '';
  
  const box = document.getElementById('chatBox');
  box.innerHTML += `<div class="chat-msg user">${txt}</div>`;
  box.scrollTop = box.scrollHeight;
  
  try {
    const fd = new FormData();
    fd.append('message', txt);
    fd.append('context', JSON.stringify(contextData));
    fd.append('session_id', sessionId);
    
    const res = await fetch('/api/chat', { method: 'POST', body: fd });
    const data = await res.json();
    
    box.innerHTML += `<div class="chat-msg assistant">${data.reply.replace(/\n/g, '<br>')}</div>`;
  } catch (e) {
    box.innerHTML += `<div class="chat-msg assistant color-rose">Error connecting to AI.</div>`;
  }
  box.scrollTop = box.scrollHeight;
}

// ── RESET ────────────────────────────────────────────────────────────

function resetApp() {
  zoneFiles = new Array(NUM_ZONES).fill(null);
  contextData = { zones: [] };
  sessionId = Date.now().toString();
  
  document.getElementById('navResults').style.display = 'none';
  document.getElementById('navDispatch').style.display = 'none';
  
  setupZoneGrid();
  updateSetupCount();
  
  showPage('analyze', document.querySelector('[data-page="analyze"]'));
}
