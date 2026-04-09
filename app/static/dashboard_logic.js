requestAnimationFrame(()=>{
  document.body.classList.add('ready');
  document.querySelectorAll('.p').forEach((el, idx)=>{
    el.style.transitionDelay = (idx * 45) + 'ms';
  });
});
const wCtx=document.getElementById('wfc').getContext('2d');
const wD={labels:[],datasets:[{data:[],borderColor:'#ffffff',borderWidth:1.5,pointRadius:0,tension:.2,fill:false}]};
const wC=new Chart(wCtx,{type:'line',data:wD,options:{responsive:true,maintainAspectRatio:false,animation:{duration:0},scales:{x:{display:false},y:{min:-1.5,max:1.5,grid:{color:'#1a1a1a'},ticks:{color:'#666666',font:{size:9}}}},plugins:{legend:{display:false}}}});
let si=0,fc=0,actCost=0,tgtCost=0;
setInterval(()=>{document.getElementById('fps').textContent=fc+'fps';fc=0},1000);
setInterval(()=>{
  if(actCost < tgtCost) {
    actCost += Math.ceil((tgtCost - actCost) * 0.1);
    document.getElementById('scs').textContent = '₹' + actCost.toLocaleString();
  }
}, 50);

let aCtx, wO, nN, gN;
let audioEnabled = false;
function togAud() {
  const b = document.getElementById('aud');
  if(!aCtx) {
    aCtx = new (window.AudioContext || window.webkitAudioContext)();
    wO = aCtx.createOscillator(); wO.type = 'sine'; wO.frequency.value = 50;
    const bS = aCtx.createBufferSource(); const bF = aCtx.createBuffer(1, aCtx.sampleRate * 2, aCtx.sampleRate);
    const cD = bF.getChannelData(0); for(let i=0; i<bF.length; i++) cD[i] = Math.random() * 2 - 1;
    bS.buffer = bF; bS.loop = true;
    nN = aCtx.createGain(); nN.gain.value = 0; bS.connect(nN); nN.connect(aCtx.destination); bS.start();
    gN = aCtx.createGain(); gN.gain.value = 0.1; wO.connect(gN); gN.connect(aCtx.destination); wO.start();
    audioEnabled = true;
    b.innerText = '🔊 Audio: ON'; b.classList.add('on');
  } else {
    if(aCtx.state === 'running') {
      aCtx.suspend();
      audioEnabled = false;
      if(window.speechSynthesis) window.speechSynthesis.cancel();
      b.innerText = '🔊 Audio: OFF'; b.classList.remove('on');
    }
    else {
      aCtx.resume();
      audioEnabled = true;
      b.innerText = '🔊 Audio: ON'; b.classList.add('on');
    }
  }
}
function tS() {
  if(!audioEnabled || !aCtx || aCtx.state !== 'running') return;
  const o = aCtx.createOscillator(); o.type = 'square'; o.frequency.setValueAtTime(150, aCtx.currentTime); o.frequency.exponentialRampToValueAtTime(0.01, aCtx.currentTime + 0.1);
  const g = aCtx.createGain(); g.gain.setValueAtTime(0.3, aCtx.currentTime); g.gain.exponentialRampToValueAtTime(0.01, aCtx.currentTime + 0.1);
  o.connect(g); g.connect(aCtx.destination); o.start(); o.stop(aCtx.currentTime + 0.1);
}


// ITIC chart
const iCtx=document.getElementById('ic').getContext('2d');
const iC=new Chart(iCtx,{type:'scatter',data:{datasets:[
  {label:'Upper',data:[{x:.001,y:200},{x:.003,y:140},{x:.5,y:120},{x:10,y:110}],borderColor:'rgba(0,230,138,.25)',borderWidth:1,pointRadius:0,showLine:true,fill:false},
  {label:'Lower',data:[{x:.001,y:70},{x:.02,y:80},{x:.5,y:80},{x:10,y:87}],borderColor:'rgba(0,230,138,.25)',borderWidth:1,pointRadius:0,showLine:true,fill:false},
  {label:'Events',data:[],backgroundColor:[],pointRadius:5}
]},options:{responsive:true,maintainAspectRatio:false,animation:{duration:0},scales:{x:{type:'logarithmic',min:.001,max:100,grid:{color:'#1a2420'},ticks:{color:'#4a6a5a',font:{size:8}}},y:{min:0,max:220,grid:{color:'#1a2420'},ticks:{color:'#4a6a5a',font:{size:8}}}},plugins:{legend:{display:false}}}});

let _fc='normal';
let _hfAttrSig = null;
let _hfLedgerSig = null;
let _hfLastAttrData = null;
let _hfLastLedgerData = null;
let _demoTimers = [];
const ws=new WebSocket('ws://'+location.host+'/ws');
function animateFlow(zone) {
  const zones = ["feeder_a","motor_room","panel_b","production_floor"];

  zones.forEach((z,i)=>{
    setTimeout(()=>{
      let el = document.getElementById("z-"+z);
      if(el){
        el.classList.add("al");
        setTimeout(()=>el.classList.remove("al"), 1000);
      }
    }, i*400);
  });
}
ws.onmessage=(e)=>{
  fc++;
  const d=JSON.parse(e.data);
  
  // Waveform
  for(const s of d.samples){wD.labels.push(si++);wD.datasets[0].data.push(s);if(wD.labels.length>400){wD.labels.shift();wD.datasets[0].data.shift()}}
  wD.datasets[0].borderColor=d.classification.fault_class==='normal'?'#ffffff':'#ff4d4d';
  document.getElementById('wfc').style.filter = `drop-shadow(0 0 6px ${wD.datasets[0].borderColor})`;
  wC.update();
  
  // Metrics
  uM('mt',d.metrics.thd_percent.toFixed(1)+'%',d.metrics.thd_percent>8?'crit':d.metrics.thd_percent>5?'warn':'ok');
  document.getElementById('mt-f').style.width = Math.min(100, (d.metrics.thd_percent/15)*100) + '%';
  document.getElementById('mt-f').style.background = d.metrics.thd_percent>8?'var(--red)':'var(--ac)';
  uM('mr',d.metrics.rms_voltage.toFixed(0)+'V',d.metrics.rms_voltage<180||d.metrics.rms_voltage>280?'crit':d.metrics.rms_voltage<200||d.metrics.rms_voltage>260?'warn':'ok');
  document.getElementById('mr-f').style.width = Math.min(100, (d.metrics.rms_voltage/300)*100) + '%';
  document.getElementById('mr-f').style.background = (d.metrics.rms_voltage<180||d.metrics.rms_voltage>280)?'var(--red)':'var(--ac)';
  uM('mp',d.metrics.power_factor.toFixed(3),d.metrics.power_factor<.8?'crit':d.metrics.power_factor<.9?'warn':'ok');
  document.getElementById('mp-f').style.width = Math.min(100, d.metrics.power_factor*100) + '%';
  document.getElementById('mp-f').style.background = d.metrics.power_factor<.8?'var(--red)':'var(--ac)';
  uM('mf',d.metrics.frequency_hz.toFixed(1),Math.abs(d.metrics.frequency_hz-50)>2?'crit':Math.abs(d.metrics.frequency_hz-50)>.5?'warn':'ok');
  document.getElementById('mf-f').style.width = Math.min(100, (d.metrics.frequency_hz/60)*100) + '%';
  document.getElementById('mf-f').style.background = Math.abs(d.metrics.frequency_hz-50)>2?'var(--red)':'var(--ac)';
  
  // Severity
  let sv=0;
  if(d.metrics.thd_percent>8)sv+=25;if(d.metrics.rms_voltage<180||d.metrics.rms_voltage>280)sv+=30;
  if(d.metrics.power_factor<.85)sv+=15;if(Math.abs(d.metrics.frequency_hz-50)>1)sv+=20;
  if(d.incident)sv=d.incident.score||sv;
  sv=Math.min(sv,100);
  const se=document.getElementById('sv');se.textContent=sv;
  se.style.color=sv>60?'var(--red)':sv>30?'var(--amb)':'var(--ac)';
  if(sv>60) document.body.classList.add('shake'); else document.body.classList.remove('shake');
  if(nN) nN.gain.value = d.metrics.thd_percent / 200;
  if(d.incident && (sv>60 || d.incident.fault_class==='transient')) tS();
  
  // CCTV Interaction logic
  if(d.incident && d.incident.fault_class === 'sag') { document.getElementById('cc-rot').style.animationDuration = '12s'; setTimeout(()=>document.getElementById('cc-rot').style.animationDuration='0.5s', 12000); }
  if(d.incident && d.incident.fault_class === 'interruption') { document.getElementById('cc-rot').style.animationPlayState = 'paused'; setTimeout(()=>document.getElementById('cc-rot').style.animationPlayState='running', 10000); }
  if(sv > 50 || d.metrics.thd_percent > 6) { document.getElementById('cc-stat').style.opacity = Math.min(0.8, sv/100 + 0.2); } else { document.getElementById('cc-stat').style.opacity = 0; }

  // Agent Logic
  if(d.incident && d.incident.severity === 'critical' && !aLoc && !aAct) trigAov(d.incident.zone, d.incident.fault_class);
  
  // Standards
  if(d.standards){
    document.getElementById('ieee').textContent='IEEE 1159: '+d.standards.ieee;
    const it=document.getElementById('itic');it.textContent='ITIC: '+d.standards.itic;
    it.style.color=d.standards.itic==='outside_tolerance'?'var(--red)':'var(--tx2)';
  }
  document.getElementById('sensor').innerHTML='Sensor: '+(d.sensor.status==='valid'?'<span style="color:var(--ac)">✓ Valid</span>':'<span style="color:var(--red)">⚠ '+d.sensor.status+'</span>');
  
  // Plant map
  document.querySelectorAll('.zn').forEach(z=>z.classList.remove('al'));
  if(d.incident){
  animateFlow(d.incident.zone);}
  
  // Narration
  if(d.incident)document.getElementById('narr').innerHTML=d.incident.narration.replace(/\\n/g,'<br>')+'<span class="cur"></span>';
  
  // Trend + Pattern
  const tr=document.getElementById('trend');
  if(d.trend){tr.textContent='📈 '+d.trend;tr.style.display='block'}else{tr.style.display='none'}
  const ip=document.getElementById('ipat');
  if(d.incident_pattern){ip.textContent='⚠ '+d.incident_pattern;ip.style.display='block'}else{ip.style.display='none'}
  
  // Causal
  if(d.correlator)document.getElementById('causal').innerHTML='<b>'+d.correlator.pattern+'</b><br>'+d.correlator.explanation;
  
  // Scalogram
  if(d.classification.scalogram_b64&&d.classification.fault_class!=='normal'){
    const im=document.getElementById('scimg');im.src='data:image/png;base64,'+d.classification.scalogram_b64;im.style.display='block';
  }

  // Waveform image CV
  if(d.classification.waveform_cv){
    const cv=d.classification.waveform_cv;
    document.getElementById('wcv').innerHTML='<b>'+cv.label.replace(/_/g,' ')+'</b> ('+Math.round((cv.confidence||0)*100)+'%)<br>'+cv.explanation;
  }

  // CCTV vision
  if(d.cctv){
    const cam=document.getElementById('camimg');
    cam.src=(d.cctv.frame_kind==='webcam_jpg'?'data:image/jpeg;base64,':'data:image/svg+xml;base64,')+d.cctv.frame_b64;
    const anoms=(d.cctv.anomalies||[]).length?d.cctv.anomalies.map(a=>a.replace(/_/g,' ')).join(', '):'clear scene';
    const desc=document.getElementById('camdesc');
    desc.innerHTML='<b>'+d.cctv.camera_id+'</b> | '+d.cctv.zone_name+'<br>'+d.cctv.summary+'<br>Anomalies: '+anoms+'<br>Confidence: '+Math.round((d.cctv.confidence||0)*100)+'%';
    desc.style.color=(d.cctv.anomalies||[]).length?'#ffd6d6':'var(--tx2)';
    desc.style.fontWeight=(d.cctv.anomalies||[]).length?'600':'400';
  }
  
  // SHAP
  if(d.classification.shap&&Object.keys(d.classification.shap).length>0&&d.classification.fault_class!=='normal'){
    const c=document.getElementById('shap');let h='';
    const ent=Object.entries(d.classification.shap).sort((a,b)=>b[1]-a[1]);
    const mx=Math.max(...ent.map(e=>e[1]),.01);
    for(const[n,v] of ent)h+='<div class="sb"><span class="nm">'+n+'</span><div class="br"><div class="fl" style="width:'+(v/mx*100)+'%"></div></div></div>';
    c.innerHTML=h;
  }
  
  // Equipment
  if(d.equipment){let h='';for(const eq of d.equipment){
    const c=eq.health>80?'var(--ac)':eq.health>50?'var(--amb)':'var(--red)';
    h+='<div class="eq"><span>'+eq.name+'</span><div class="eqb"><div class="eqf" style="width:'+eq.health+'%;background:'+c+'"></div></div><span style="font-family:IBM Plex Mono;font-size:10px;color:'+c+'">'+eq.health.toFixed(0)+'%</span></div>';
  }document.getElementById('eqlist').innerHTML=h}
  renderDevices(d.devices);
  renderTwin(d.digital_twin);
  renderFleet(d.fleet);
  
  // Sustainability
  if(d.summary){
    document.getElementById('sa').textContent=d.summary.aging_hrs.toFixed(3);
    document.getElementById('sco').textContent=d.summary.co2_kg.toFixed(4);
    tgtCost = Math.round(d.summary.cost_inr);
  }
  if(d.executive_summary){
    document.getElementById('exFault').textContent=d.executive_summary.fault||'Normal';
    document.getElementById('exConf').textContent=d.executive_summary.confidence_label||'0% confidence';
    document.getElementById('exZone').textContent=d.executive_summary.zone||'Unknown Zone';
    document.getElementById('exSource').textContent=d.executive_summary.likely_source||'Awaiting attribution';
    document.getElementById('exSourceLbl').textContent=d.executive_summary.source_label||'Likely contributor: Unknown';
    document.getElementById('exSeverity').textContent=d.executive_summary.severity||'Nominal';
    document.getElementById('exSeverityScore').textContent='Score '+(d.executive_summary.severity_score||0);
    document.getElementById('exCost').textContent='₹'+Math.round(d.executive_summary.cost_inr||0).toLocaleString();
    document.getElementById('exAction').textContent=d.executive_summary.recommended_action||'Continue monitoring';
  }
  
  // Incidents
  if(d.incident){const lg=document.getElementById('ilog');const t=new Date().toLocaleTimeString();
    const bc=d.incident.severity==='critical'?'bd-c':d.incident.severity==='medium'?'bd-m':'bd-l';
    lg.innerHTML='<div class="inc sv-'+d.incident.severity+'"><div class="tp"><span class="bd '+bc+'">'+d.incident.severity+'</span><span style="font-size:9px;color:var(--tx2)">'+t+'</span></div><div style="font-size:11px;font-weight:500">'+d.incident.fault_class.replace(/_/g,' ')+'</div><div class="cs">'+d.incident.cause+'</div><div class="ac">→ '+d.incident.action+'</div></div>'+lg.innerHTML;
  }
  
  // Propagation
  if(d.propagation&&d.propagation.nodes){renderPG(d.propagation);document.getElementById('pn').textContent=d.propagation.narrative||''}
  
  // ITIC scatter
  if(d.itic_scatter&&d.itic_scatter.length>0){
    iC.data.datasets[2].data=d.itic_scatter.map(h=>({x:Math.max(.001,(h.duration||1)/50),y:h.magnitude||100}));
    iC.data.datasets[2].backgroundColor=d.itic_scatter.map(h=>h.status==='outside_tolerance'?'#ff4d6a':'#00e68a');
    iC.update();
  }
  
  _fc=d.classification.fault_class;

  const ds=document.getElementById('demoStatus');
  if(d.demo_mode && d.demo_mode.active && ds){
    ds.classList.add('active');
    ds.innerHTML='<b>Demo Mode:</b> Step '+d.demo_mode.step+'/'+d.demo_mode.total_steps+' - '+d.demo_mode.label+'<br><span style="color:var(--tx2)">Pitch this as: detect → explain → quantify impact → recommend action.</span>';
  } else if (ds) {
    ds.classList.remove('active');
    ds.innerHTML='<b>Demo Mode:</b> Ready. Launch a guided Feeder A harmonic incident storyline for judges, then use Reset Demo to replay it cleanly.';
  }

  // ── Harmonic Forensics live render ──
  if(d.attribution){
    const attrForUi = (d.attribution.sources && d.attribution.sources.length) ? d.attribution : (_hfLastAttrData || d.attribution);
    const attrSig = JSON.stringify(attrForUi);
    if (attrSig !== _hfAttrSig) {
      renderHfSources(attrForUi);
      _hfAttrSig = attrSig;
      if (attrForUi.sources && attrForUi.sources.length) _hfLastAttrData = JSON.parse(attrSig);
    }
  }
  if(d.ledger){
    const ledgerForUi = (d.ledger && Object.keys(d.ledger).length) ? d.ledger : (_hfLastLedgerData || d.ledger);
    const ledgerSig = JSON.stringify({zone:d.zone, ledger:ledgerForUi});
    if (ledgerSig !== _hfLedgerSig) {
      renderHfLedger(ledgerForUi, d.zone);
      _hfLedgerSig = ledgerSig;
      if (ledgerForUi && Object.keys(ledgerForUi).length) _hfLastLedgerData = JSON.parse(JSON.stringify(ledgerForUi));
    }
  }
};

// ── Harmonic Forensics JS helpers ─────────────────────────────
function hfBarColor(conf){
  if(conf > 0.65) return 'rgba(255,87,87,.85)';
  if(conf > 0.40) return 'rgba(239,183,40,.85)';
  return 'rgba(180,180,180,.6)';
}

function renderHfSources(attr) {
  const root = document.getElementById('hf-sources');
  const budgetFill = document.getElementById('hf-budget-fill');
  const budgetPct = document.getElementById('hf-budget-pct');
  if (!root) return;
  const sources = attr.sources || [];
  if (!sources.length) {
    if (root.innerHTML.trim()) return;
    root.innerHTML = '<div class="hf-card"><div class="hf-top"><span class="hf-dev">Monitoring in progress</span><span class="hf-conf">0%</span></div><div class="hf-bar-wrap"><div class="hf-bar-fill" style="width:0%;background:rgba(180,180,180,.6)"></div></div></div>';
  } else {
    let h = '';
    for (const s of sources) {
      const pct = Math.round(s.confidence * 100);
      const col = hfBarColor(s.confidence);
      h += '<div class="hf-card">';
      h += '<div class="hf-top"><span class="hf-dev">' + s.device + '</span>';
      h += '<span class="hf-conf">' + pct + '% match</span></div>';
      h += '<div class="hf-bar-wrap"><div class="hf-bar-fill" style="width:' + pct + '%;background:' + col + '"></div></div>';
      h += '<div class="hf-conf" style="margin-top:6px">Likely contributor based on harmonic fingerprint similarity.</div>';
      h += '</div>';
    }
    root.innerHTML = h;
  }
  const bu = attr.thd_budget_used || 0;
  const buPct = Math.min(100, bu);
  budgetFill.style.width = buPct + '%';
  budgetFill.style.background = bu > 100 ? 'var(--red)' : bu > 75 ? 'rgba(239,183,40,.9)' : 'rgba(0,230,138,.8)';
  if (bu > 80) budgetFill.classList.add('over'); else budgetFill.classList.remove('over');
  budgetPct.textContent = Math.round(bu) + '%';
}

function renderHfLedger(ledger, zone) {
  const root = document.getElementById('hf-ledger');
  if (!root) return;
  const zoneData = ledger || {};
  const rows = Object.entries(zoneData)
    .sort(([, a], [, b]) => (b.violations - a.violations) || ((b.budget_pct || 0) - (a.budget_pct || 0)) || ((b.penalty_score || 0) - (a.penalty_score || 0)))
    .slice(0, 4);
  if (!rows.length) {
    if (root.innerHTML.trim()) return;
    root.innerHTML = '<div class="hf-ledger-row" style="font-weight:600;font-size:9px;color:var(--tx2);text-transform:uppercase;letter-spacing:.8px"><span>Device</span><span>Budget%</span><span>Violations</span><span>Penalty</span></div><div class="hf-ledger-row"><span style="font-size:10px">Awaiting live data</span><span class="hf-pen">0.0</span><span class="hf-viol">0</span><span class="hf-pen">0.00</span></div>';
    return;
  }
  let h = '<div class="hf-ledger-row" style="font-weight:600;font-size:9px;color:var(--tx2);text-transform:uppercase;letter-spacing:.8px">';
  h += '<span>Device</span><span>Budget%</span><span>Violations</span><span>Penalty</span></div>';
  for (const [dev, info] of rows) {
    const vc = info.violations > 0 ? 'var(--red)' : 'var(--ac)';
    h += '<div class="hf-ledger-row">';
    h += '<span style="font-size:10px">' + dev + '</span>';
    h += '<span class="hf-pen">' + (info.budget_pct || 0).toFixed(1) + '</span>';
    h += '<span class="hf-viol" style="color:' + vc + '">' + info.violations + '</span>';
    h += '<span class="hf-pen">' + (info.penalty_score || 0).toFixed(2) + '</span>';
    h += '</div>';
  }
  root.innerHTML = h;
}

async function runHfSim() {
  const btn = document.getElementById('hf-sim-btn');
  const result = document.getElementById('hf-sim-result');
  btn.textContent = '⏳ Running...';
  btn.disabled = true;
  try {
    const r = await (await fetch('/api/simulate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        source_type: document.getElementById('hf-src-type').value,
        capacity_mw: parseFloat(document.getElementById('hf-cap-mw').value || '10'),
        feeder: document.getElementById('hf-feeder').value,
      })
    })).json();
    if (r.error) { result.textContent = 'Error: ' + r.error; result.className = 'hf-result'; return; }
    const ok = !r.exceeds_ieee519;
    result.className = 'hf-result ' + (ok ? 'ok' : 'bad');
    result.innerHTML =
      '<b>' + (ok ? '✅ APPROVE' : '❌ REJECT') + '</b> — ' + r.recommendation + '<br><br>' +
      'Baseline THD: <b>' + r.baseline_thd + '%</b> + Injection: <b>+' + r.thd_injection + '%</b> → Predicted: <b>' + r.predicted_thd + '%</b><br>' +
      'Aging delta: <b>' + r.aging_factor_delta + 'x</b> | ' +
      'Resonance risk: <b>' + (r.resonance_risk ? '⚠ YES' : 'No') + '</b> | ' +
      'Annual penalty est.: <b>₹' + (r.annual_penalty_inr || 0).toLocaleString() + '</b>';
  } catch(e) {
    result.textContent = 'Simulation failed: ' + e.message;
    result.className = 'hf-result';
  } finally {
    btn.textContent = '⚡ Run Impact Assessment';
    btn.disabled = false;
  }
}

function clearDemoTimers(){
  _demoTimers.forEach(t=>clearTimeout(t));
  _demoTimers = [];
}

async function startJudgeDemo(){
  clearDemoTimers();
  const btn = document.getElementById('demoBtn');
  const ds = document.getElementById('demoStatus');
  btn.disabled = true;
  ds.classList.add('active');
  ds.innerHTML = '<b>Demo Mode:</b> Starting scripted Feeder A harmonic storyline...';
  try{
    await fetch('/api/demo/start',{method:'POST'});
    document.getElementById('hf-src-type').value = 'Variable Frequency Drive';
    document.getElementById('hf-cap-mw').value = '14';
    document.getElementById('hf-feeder').value = 'feeder_a';
    document.getElementById('waction').value = 'isolate';
    document.getElementById('wzone').value = 'feeder_a';
    document.getElementById('wamt').value = '35';

    _demoTimers.push(setTimeout(()=>{
      ds.innerHTML = '<b>Demo Mode:</b> Harmonic distortion is now being demonstrated on Feeder A.';
    }, 1500));
    _demoTimers.push(setTimeout(()=>{
      ds.innerHTML = '<b>Demo Mode:</b> Vision AI is explaining the live waveform distortion.';
      trigVision();
    }, 8500));
    _demoTimers.push(setTimeout(()=>{
      ds.innerHTML = '<b>Demo Mode:</b> Running operator what-if response for feeder isolation.';
      runWhatIf();
    }, 12500));
    _demoTimers.push(setTimeout(()=>{
      ds.innerHTML = '<b>Demo Mode:</b> Running renewable impact simulation to show filter recommendation.';
      runHfSim();
    }, 15500));
    _demoTimers.push(setTimeout(()=>{
      btn.disabled = false;
      ds.innerHTML = '<b>Demo Mode:</b> Sequence complete. Use the current dashboard state to explain business impact and recommended mitigation.';
    }, 19000));
  }catch(e){
    btn.disabled = false;
    ds.classList.remove('active');
    ds.textContent = 'Demo launch failed: ' + e.message;
  }
}

async function resetJudgeDemo(){
  clearDemoTimers();
  const btn = document.getElementById('demoBtn');
  const ds = document.getElementById('demoStatus');
  try{
    await fetch('/api/demo/reset',{method:'POST'});
    btn.disabled = false;
    ds.classList.remove('active');
    ds.innerHTML = '<b>Demo Mode:</b> Reset complete. Launch a guided Feeder A harmonic incident storyline for judges, then use Reset Demo to replay it cleanly.';
  }catch(e){
    ds.textContent = 'Demo reset failed: ' + e.message;
  }
}

async function downloadHfPdf() {
  const btn = document.getElementById('hf-pdf-btn');
  const status = document.getElementById('hf-pdf-status');
  btn.disabled = true;
  status.textContent = '⏳ Generating...';
  try {
    const r = await (await fetch('/api/report_pdf')).json();
    if (!r.pdf_b64) { status.textContent = 'No data yet.'; return; }
    const bytes = atob(r.pdf_b64);
    const arr = new Uint8Array(bytes.length);
    for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i);
    const blob = new Blob([arr], {type: 'application/pdf'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = r.filename || 'compliance_report.pdf'; a.click();
    URL.revokeObjectURL(url);
    status.textContent = '✓ Downloaded';
    setTimeout(() => status.textContent = '', 3000);
  } catch(e) {
    status.textContent = 'Error: ' + e.message;
  } finally {
    btn.disabled = false;
  }
}

function renderPG(p){
  const svg=document.getElementById('pg');const W=svg.clientWidth||580,H=250;let h='';
  const nm={};p.nodes.forEach(n=>{nm[n.id]={x:n.x*(W/600),y:n.y*(H/380)+18}});
  h+='<defs><filter id="pgGlow" x="-60%" y="-60%" width="220%" height="220%"><feGaussianBlur stdDeviation="6" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter><linearGradient id="edgeGrad" x1="0" x2="1"><stop offset="0%" stop-color="rgba(255,255,255,.08)"/><stop offset="50%" stop-color="rgba(255,255,255,.38)"/><stop offset="100%" stop-color="rgba(255,255,255,.05)"/></linearGradient><radialGradient id="floorGlow" cx="50%" cy="30%" r="70%"><stop offset="0%" stop-color="rgba(255,255,255,.08)"/><stop offset="100%" stop-color="rgba(255,255,255,0)"/></radialGradient></defs>';
  h+='<rect x="0" y="0" width="'+W+'" height="'+H+'" rx="16" fill="url(#floorGlow)" opacity=".9"/>';
  h+='<ellipse cx="'+(W/2)+'" cy="'+(H-26)+'" rx="'+(W*0.34)+'" ry="24" fill="rgba(255,255,255,.035)"/>';
  p.edges.forEach((e,i)=>{const s=nm[e.source],t=nm[e.target];if(!s||!t)return;
    const mx=(s.x+t.x)/2, my=(s.y+t.y)/2-18;
    const active=!!e.active;
    const stroke=active?'rgba(255,255,255,.92)':'rgba(255,255,255,.12)';
    const glow=active?'rgba(255,255,255,.22)':'rgba(255,255,255,.05)';
    h+='<path d="M '+s.x+' '+s.y+' Q '+mx+' '+my+' '+t.x+' '+t.y+'" fill="none" stroke="'+glow+'" stroke-width="'+(active?8:3)+'" opacity="'+(active?.55:.2)+'"/>';
    h+='<path d="M '+s.x+' '+s.y+' Q '+mx+' '+my+' '+t.x+' '+t.y+'" fill="none" stroke="'+stroke+'" stroke-width="'+(active?2.4:1.1)+'" stroke-linecap="round" opacity="'+(active?.95:.45)+'"/>';
    if(active){
      h+='<circle r="3.2" fill="#fff" filter="url(#pgGlow)"><animateMotion dur="'+(1.6+i*0.18)+'s" repeatCount="indefinite" path="M '+s.x+' '+s.y+' Q '+mx+' '+my+' '+t.x+' '+t.y+'"/></circle>';
    }
  });
  p.nodes.forEach((n,i)=>{const pos=nm[n.id];if(!pos)return;
    const c={critical:'#ffffff',warning:'#d9d9d9',normal:'#9d9d9d',isolated:'#555'}[n.status]||'#d9d9d9';
    const halo={critical:'rgba(255,255,255,.28)',warning:'rgba(255,255,255,.18)',normal:'rgba(255,255,255,.1)',isolated:'rgba(255,255,255,.04)'}[n.status]||'rgba(255,255,255,.1)';
    const fill={critical:'rgba(255,255,255,.16)',warning:'rgba(255,255,255,.11)',normal:'rgba(255,255,255,.06)',isolated:'rgba(255,255,255,.03)'}[n.status]||'rgba(255,255,255,.06)';
    const r=n.status==='critical'?20:n.status==='warning'?16:12;
    const depth=6+r*.22;
    h+='<ellipse cx="'+pos.x+'" cy="'+(pos.y+depth+12)+'" rx="'+(r*1.2)+'" ry="'+(r*.38)+'" fill="rgba(0,0,0,.45)"/>';
    h+='<circle cx="'+pos.x+'" cy="'+(pos.y+depth*.35)+'" r="'+(r+6)+'" fill="'+halo+'" opacity=".55"/>';
    h+='<circle cx="'+pos.x+'" cy="'+pos.y+'" r="'+r+'" fill="'+fill+'" stroke="'+c+'" stroke-width="1.6" filter="url(#pgGlow)"/>';
    h+='<circle cx="'+(pos.x-r*.28)+'" cy="'+(pos.y-r*.3)+'" r="'+(r*.34)+'" fill="rgba(255,255,255,.28)"/>';
    if(n.risk>.08)h+='<text x="'+pos.x+'" y="'+(pos.y+4)+'" text-anchor="middle" fill="#fff" font-size="10" font-weight="700" font-family="IBM Plex Mono">'+(n.risk*100).toFixed(0)+'%</text>';
    h+='<text x="'+pos.x+'" y="'+(pos.y+r+18)+'" text-anchor="middle" fill="#bdbdbd" font-size="8.5" letter-spacing=".06em" font-family="Outfit">'+n.name+'</text>';
  });
  svg.innerHTML=h}

function renderTwin(items){
  const root=document.getElementById('dtwin');if(!items||!items.length){root.innerHTML='';return;}
  root.innerHTML=''; // Reset
  for(const item of items){
    const div = document.createElement('div'); div.className = 'dt';
    const tp = document.createElement('div'); tp.className = 'tp';
    const s1 = document.createElement('span'); s1.textContent = item.name;
    const s2 = document.createElement('span'); s2.textContent = item.profile.replace(/_/g,' ');
    tp.appendChild(s1); tp.appendChild(s2);
    
    const meta1 = document.createElement('div'); meta1.className = 'meta';
    meta1.textContent = `Concern: ${item.concern}`;
    const meta2 = document.createElement('div'); meta2.className = 'meta';
    meta2.textContent = `Checks: ${item.recommended_checks.slice(0,2).join(', ')}`;
    
    const bar = document.createElement('div'); bar.className = 'bar';
    const fill = document.createElement('div'); fill.className = 'fill';
    fill.style.width = Math.min(100, item.risk_driver) + '%';
    bar.appendChild(fill);
    
    const meta3 = document.createElement('div'); meta3.className = 'meta';
    meta3.textContent = `Risk driver ${item.risk_driver.toFixed(0)} | Health ${item.health.toFixed(0)}%`;
    
    div.appendChild(tp); div.appendChild(meta1); div.appendChild(meta2); div.appendChild(bar); div.appendChild(meta3);
    root.appendChild(div);
  }
}

function renderFleet(fleet){
  const root=document.getElementById('fleet');if(!fleet||!fleet.sites){root.innerHTML='';return;}
  let h='';
  for(const site of fleet.sites){
    h+='<div class="flc"><div class="nm">'+site.site+'</div><div class="tx">Risk: '+site.risk_score.toFixed(0)+'<br>Worst asset: '+site.worst_asset+'<br>Top fault: '+site.top_fault.replace(/_/g,' ')+'<br>Cost: ₹'+Math.round(site.cost_inr).toLocaleString()+'<br>CO₂: '+site.co2_kg.toFixed(4)+' kg</div></div>';
  }
  root.innerHTML=h;
}

function renderDevices(devices){
  const root=document.getElementById('devs');if(!root)return;
  if(!devices||!devices.length){root.innerHTML='<div class="dev"><div class="nm">No devices</div><div class="tx">Device telemetry unavailable.</div></div>';return;}
  let h='';
  for(const d of devices){
    const battery = d.battery_percent==null?'N/A':d.battery_percent.toFixed(0)+'%';
    const charging = d.charging==null?'unknown':(d.charging?'charging':'battery');
    const voltage = d.voltage_v==null?'N/A':d.voltage_v+' V';
    const current = d.current_a==null?'N/A':d.current_a+' A';
    const power = d.power_w==null?'N/A':d.power_w+' W';
    const cpu = d.cpu_percent==null?'N/A':d.cpu_percent.toFixed(0)+'%';
    const mem = d.memory_percent==null?'N/A':d.memory_percent.toFixed(0)+'%';
    h+=`<div class="dev"><div class="nm">${d.name}</div><div class="tx">IP: ${d.local_ip}<br>Battery: ${battery} (${charging})<br>Voltage: ${voltage} | Current: ${current}<br>Power: ${power}<br>CPU: ${cpu} | RAM: ${mem}<br>Phone URL: ${d.dashboard_url}</div></div>`;
  }
  root.innerHTML=h; // Safe: Internal static templates with trusted data
}

function uM(id,v,s){const e=document.getElementById(id);e.querySelector('.v').textContent=v;e.className='m '+s}
async function inj(t,z){await fetch('/api/inject',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({type:t,zone:z,duration:3})})}
async function togNoise(){const r=await(await fetch('/api/noise',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({})})).json();const b=document.getElementById('nbtn');b.textContent='Noise: '+(r.enabled?'ON':'OFF');b.classList.toggle('on',r.enabled)}
async function chgN(v){await fetch('/api/noise',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({snr:v})})}
async function brk(z){const r=await(await fetch('/api/breaker',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({zone:z})})).json();document.querySelectorAll('.zn').forEach(el=>{const zid=el.id.replace('z-','');if(r.isolated.includes(zid))el.classList.add('iso');else el.classList.remove('iso')});}
async function togWebcam(){
  const btn=document.getElementById('wcamBtn');
  const enable=!btn.classList.contains('on');
  const zone=document.getElementById('camzone').value;
  const r=await(await fetch('/api/webcam',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({enable,zone})})).json();
  btn.textContent='Webcam: '+(r.enabled?'ON':'OFF');
  btn.classList.toggle('on', r.enabled);
}
async function runWhatIf(){
  const action=document.getElementById('waction').value;
  const zone=document.getElementById('wzone').value;
  const amount=parseFloat(document.getElementById('wamt').value||'0');
  const el=document.getElementById('wires');
  el.textContent='⏳ Running simulation...';
  try{
    const r=await(await fetch('/api/whatif',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action,zone,amount})})).json();
    if(r.error){el.textContent='Error: '+r.error;return;}
    const sevColor = r.predicted_severity > 60 ? '#ff4d6a' : r.predicted_severity > 30 ? '#ffd166' : '#00e68a';
    el.innerHTML =
      `<b style="color:${sevColor}">Predicted Severity: ${r.predicted_severity}/100</b><br>` +
      `THD → ${r.predicted_thd}% | PF → ${r.predicted_pf} | Aging ${r.predicted_aging_factor}x<br>` +
      `Cost Impact: ₹${Math.round(r.predicted_cost_inr).toLocaleString()}<br>` +
      `<span style="color:var(--tx2)">${(r.notes||[]).join(' ')}</span><br>` +
      `<b>Recommendation:</b> ${r.recommendation}<br>` +
      `<span style="color:var(--tx2);font-size:10px">${r.propagation?.narrative||''}</span>`;
    // Update propagation map with what-if scenario
    if(r.propagation && r.propagation.nodes) renderPG(r.propagation);
  }catch(e){el.textContent='Simulation failed: '+e.message;}
}
async function exportReport(){
  const r=await(await fetch('/api/report')).json();
  const checks=(r.recommended_checks||[]).slice(0,4).join(', ');
  document.getElementById('rpt').textContent='Incident: '+r.incident_summary.fault_class.replace(/_/g,' ')+' in '+r.incident_summary.zone+'\nSeverity: '+r.incident_summary.severity+'\nRoot cause: '+r.probable_root_cause+'\nStandards: IEEE '+r.standards_violated.ieee_1159+' | ITIC '+r.standards_violated.itic+'\nChecks: '+checks+'\nImpact: '+r.estimated_impact.cost_inr+', '+r.estimated_impact.aging+', '+r.estimated_impact.co2;
}
let lastInc = null;
setInterval(async () => {
  try{ const r=await(await fetch('/api/ticker',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({})})).json();
  if(r.insight&&r.insight!==lastInc) { document.getElementById('tkrt').innerText = '>>> AI INSIGHT: ' + r.insight; lastInc=r.insight; } }catch(e){}
}, 20000);
async function askQ(){
  const i=document.getElementById('qi');
  const q=i.value.trim();
  if(!q)return;
  i.value='';
  const narr = document.getElementById('narr');
  narr.textContent='Thinking...';
  const r=await(await fetch('/api/ask', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({q:q})
  })).json();
  
  // Safe rendering: escape Q and use textContent for A
  narr.innerHTML = '';
  const qEl = document.createElement('b');
  qEl.textContent = 'Q: ' + q;
  const aEl = document.createElement('div');
  aEl.style.whiteSpace = 'pre-wrap';
  aEl.style.marginTop = '8px';
  aEl.textContent = r.a;
  narr.appendChild(qEl);
  narr.appendChild(aEl);
  const cur = document.createElement('span');
  cur.className = 'cur';
  narr.appendChild(cur);
}
async function trigVision(){
  const vr=document.getElementById('vres');
  vr.style.display='block';
  vr.textContent='🔍 Analyzing waveform...';
  try{
    const r=await(await fetch('/api/vision',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({fault_class:_fc})})).json();
    vr.innerHTML = ''; // Clear previous
    if(r.headline){
      const h = document.createElement('b'); h.textContent = r.headline;
      vr.appendChild(h); vr.appendChild(document.createElement('br'));
      vr.appendChild(document.createElement('br'));
    }
    if(r.analysis){
      const a = document.createElement('div'); a.textContent = r.analysis;
      vr.appendChild(a); vr.appendChild(document.createElement('br'));
      vr.appendChild(document.createElement('br'));
    }
    if(r.observations && r.observations.length){
      const obs = document.createElement('div');
      obs.textContent = r.observations.join('\n');
      obs.style.whiteSpace = 'pre-wrap';
      vr.appendChild(obs); vr.appendChild(document.createElement('br'));
      vr.appendChild(document.createElement('br'));
    }
    if(r.recommendation){
      const rec = document.createElement('div');
      const rb = document.createElement('b'); rb.textContent = 'Recommended next step: ';
      rec.appendChild(rb); rec.appendChild(document.createTextNode(r.recommendation));
      vr.appendChild(rec); vr.appendChild(document.createElement('br'));
      vr.appendChild(document.createElement('br'));
    }
    if(typeof r.confidence === 'number'){
      const conf = document.createElement('span');
      conf.style.color = 'var(--tx2)';
      conf.textContent = `Confidence: ${Math.round(r.confidence*100)}% | Live API-backed waveform context`;
      vr.appendChild(conf);
    }
  }catch(e){
    vr.textContent='Vision analysis failed: '+e.message;
  }
}

let aAct = false, aLoc = false, aTim = null, aCt = 10, aZn = '';
let vE = window.speechSynthesis;
function sysSpeak(txt) {
  if(!audioEnabled || !vE)return;
  const u = new SpeechSynthesisUtterance(txt);
  u.rate = 0.95; u.pitch = 0.8;
  const vs = vE.getVoices();
  u.voice = vs.find(v => v.name.includes('Google US English') || v.lang.includes('en-US')) || vs[0];
  vE.speak(u);
}

let isH = false;
function trigHack() {
  if(isH) return; isH = true;
  document.body.classList.add('hack-mode');
  document.getElementById('cc-hack').style.opacity = 1;
  sysSpeak("CRITICAL ALERT. UNAUTHORIZED NETWORK PENETRATION. GRID LOGIC COMPROMISED.");
  
  const ints = document.querySelectorAll('.v');
  const oT = Array.from(ints).map(el=>el.innerText);
  
  const scI = setInterval(() => {
    ints.forEach(el => el.innerText = Math.random().toString(36).substring(2, 6).toUpperCase());
    ints.forEach(el => el.classList.add('glitch'));
  }, 100);

  setTimeout(() => {
    clearInterval(scI);
    ints.forEach((el,i) => { el.innerText = oT[i]; el.classList.remove('glitch'); });
    document.body.classList.remove('hack-mode');
    document.getElementById('cc-hack').style.opacity = 0;
    isH = false;
    trigAov('network_router', 'MALICIOUS_PAYLOAD');
  }, 6000);
}

function trigAov(z, fc) {
  aAct = true; aZn = z; aCt = 10;
  sysSpeak(`Warning. Critical ${fc.replace(/_/g,' ')} detected. Cascading failure imminent. Isolating ${z.replace(/_/g,' ')}.`);
  document.getElementById('a-num').textContent = aCt;
  const txtEl = document.getElementById('a-txt');
  txtEl.innerHTML = ''; // Clear
  const bTag = document.createElement('b');
  bTag.textContent = fc.replace(/_/g,' ').toUpperCase();
  txtEl.appendChild(document.createTextNode('Critical Threat: '));
  txtEl.appendChild(bTag);
  txtEl.appendChild(document.createTextNode(' detected.'));
  txtEl.appendChild(document.createElement('br'));
  txtEl.appendChild(document.createTextNode(`Isolating ${z.replace(/_/g,' ').toUpperCase()} to prevent hardware cascade.`));
  
  document.getElementById('aov').classList.add('act');
  aTim = setInterval(() => {
    aCt--; document.getElementById('a-num').textContent = aCt;
    if(aCt <= 0) exeAov();
  }, 1000);
}
function abtAov() { clearInterval(aTim); clrAov(); }
function exeAov() { clearInterval(aTim); brk(aZn); clrAov(); }
function clrAov() { document.getElementById('aov').classList.remove('act'); aAct = false; aLoc = true; setTimeout(()=>aLoc=false, 20000); }
