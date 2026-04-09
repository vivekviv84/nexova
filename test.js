
const wCtx=document.getElementById('wfc').getContext('2d');
const wD={labels:[],datasets:[{data:[],borderColor:'#00e68a',borderWidth:1.5,pointRadius:0,tension:.2,fill:false}]};
const wC=new Chart(wCtx,{type:'line',data:wD,options:{responsive:true,maintainAspectRatio:false,animation:{duration:0},scales:{x:{display:false},y:{min:-1.5,max:1.5,grid:{color:'#1a2420'},ticks:{color:'#4a6a5a',font:{size:9}}}},plugins:{legend:{display:false}}}});

let aCtx, wHum, wNoise;
let wBuffer = [];
function initAudio(){
   if(aCtx) return;
   aCtx = new (window.AudioContext || window.webkitAudioContext)();
   wHum = aCtx.createOscillator(); wHum.type = 'sine'; wHum.frequency.value = 50; wHum.start();
   const humGain = aCtx.createGain(); humGain.gain.value = 0.05; wHum.connect(humGain); humGain.connect(aCtx.destination);
   
   const bufSize = aCtx.sampleRate * 2;
   const buf = aCtx.createBuffer(1, bufSize, aCtx.sampleRate);
   const output = buf.getChannelData(0);
   for(let i=0; i<bufSize; i++) output[i] = Math.random()*2-1;
   wNoise = aCtx.createBufferSource(); wNoise.buffer = buf; wNoise.loop = true; wNoise.start();
   const noiseGain = aCtx.createGain(); noiseGain.gain.value = 0;
   wNoise.connect(noiseGain); noiseGain.connect(aCtx.destination);
   
   window.audioObj = { humGain, noiseGain };
   document.getElementById('audBtn').textContent = '🔈 Audio: ON';
   document.getElementById('audBtn').classList.add('on');
}
function playSnap(){
   if(!aCtx)return;
   const o = aCtx.createOscillator(); o.type = 'sawtooth';
   const g = aCtx.createGain(); o.connect(g); g.connect(aCtx.destination);
   o.frequency.setValueAtTime(100, aCtx.currentTime); o.frequency.exponentialRampToValueAtTime(0.01, aCtx.currentTime + 0.1);
   g.gain.setValueAtTime(0.3, aCtx.currentTime); g.gain.exponentialRampToValueAtTime(0.01, aCtx.currentTime + 0.1);
   o.start(); o.stop(aCtx.currentTime + 0.1);
}

let si=0,fc=0,tgtCost=0,curCost=0;
setInterval(()=>{document.getElementById('fps').textContent=fc+'fps';fc=0},1000);

// ITIC chart
const iCtx=document.getElementById('ic').getContext('2d');
const iC=new Chart(iCtx,{type:'scatter',data:{datasets:[
  {label:'Upper',data:[{x:.001,y:200},{x:.003,y:140},{x:.5,y:120},{x:10,y:110}],borderColor:'rgba(0,230,138,.25)',borderWidth:1,pointRadius:0,showLine:true,fill:false},
  {label:'Lower',data:[{x:.001,y:70},{x:.02,y:80},{x:.5,y:80},{x:10,y:87}],borderColor:'rgba(0,230,138,.25)',borderWidth:1,pointRadius:0,showLine:true,fill:false},
  {label:'Events',data:[],backgroundColor:[],pointRadius:5}
]},options:{responsive:true,maintainAspectRatio:false,animation:{duration:0},scales:{x:{type:'logarithmic',min:.001,max:100,grid:{color:'#1a2420'},ticks:{color:'#4a6a5a',font:{size:8}}},y:{min:0,max:220,grid:{color:'#1a2420'},ticks:{color:'#4a6a5a',font:{size:8}}}},plugins:{legend:{display:false}}}});

let _fc='normal';
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
  
  if(d.news_ticker) document.getElementById('live_tkr').textContent = d.news_ticker;
  
  // Audio Map
  if(window.audioObj) {
     window.audioObj.humGain.gain.value = (d.metrics.rms_voltage / 280) * 0.12;
     window.audioObj.noiseGain.gain.value = Math.min((d.metrics.thd_percent || 0) / 20, 0.4);
     if(d.classification.fault_class === 'transient') playSnap();
  }
  
  // Waveform
  for(const s of d.samples){wD.labels.push(si++);wD.datasets[0].data.push(s);if(wD.labels.length>400){wD.labels.shift();wD.datasets[0].data.shift()}}
  wD.datasets[0].borderColor=d.classification.fault_class==='normal'?'#00e68a':'#ff4d6a';
  wC.update();
  
  if(d.incident && d.incident.severity === 'critical') {
      document.body.classList.add('shake-critical');
      setTimeout(()=>document.body.classList.remove('shake-critical'), 500);
  }
  
  // Metrics
  uM('mt',d.metrics.thd_percent.toFixed(1)+'%',d.metrics.thd_percent>8?'crit':d.metrics.thd_percent>5?'warn':'ok');
  uM('mr',d.metrics.rms_voltage.toFixed(0)+'V',d.metrics.rms_voltage<180||d.metrics.rms_voltage>280?'crit':d.metrics.rms_voltage<200||d.metrics.rms_voltage>260?'warn':'ok');
  uM('mp',d.metrics.power_factor.toFixed(3),d.metrics.power_factor<.8?'crit':d.metrics.power_factor<.9?'warn':'ok');
  uM('mf',d.metrics.frequency_hz.toFixed(1),Math.abs(d.metrics.frequency_hz-50)>2?'crit':Math.abs(d.metrics.frequency_hz-50)>.5?'warn':'ok');
  
  // Severity
  let sv=0;
  if(d.metrics.thd_percent>8)sv+=25;if(d.metrics.rms_voltage<180||d.metrics.rms_voltage>280)sv+=30;
  if(d.metrics.power_factor<.85)sv+=15;if(Math.abs(d.metrics.frequency_hz-50)>1)sv+=20;
  if(d.incident)sv=d.incident.score||sv;
  sv=Math.min(sv,100);
  const se=document.getElementById('sv');se.textContent=sv;
  se.style.color=sv>60?'var(--red)':sv>30?'var(--amb)':'var(--ac)';
  
  // Standards
  if(d.standards){
    document.getElementById('ieee').textContent='IEEE 1159: '+d.standards.ieee;
    const it=document.getElementById('itic');it.textContent='ITIC: '+d.standards.itic;
    it.style.color=d.standards.itic==='outside_tolerance'?'var(--red)':'var(--tx2)';
  }
  document.getElementById('sensor').innerHTML='Sensor: '+(d.sensor.status==='valid'?'<span style="color:var(--ac)">✓ Valid</span>':'<span style="color:var(--red)">⚠ '+d.sensor.status+'</span>');
  
  // Plant map
  document.querySelectorAll('.zn').forEach(z=>{
      z.classList.remove('al'); 
      if(d.isolated_nodes && d.isolated_nodes.includes(z.id.replace('z-',''))) {
          z.classList.add('iso');
      } else {
          z.classList.remove('iso');
      }
  });
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
  
  // Sustainability
  if(d.summary){
    document.getElementById('sa').textContent=d.summary.aging_hrs.toFixed(3);
    document.getElementById('sco').textContent=d.summary.co2_kg.toFixed(4);
    tgtCost = d.summary.cost_inr;
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
};

function renderPG(p){
  const svg=document.getElementById('pg');const W=svg.clientWidth||580,H=250;let h='';
  const nm={};p.nodes.forEach(n=>{nm[n.id]={x:n.x*(W/600),y:n.y*(H/380)+15}});
  p.edges.forEach(e=>{const s=nm[e.source],t=nm[e.target];if(!s||!t)return;
    h+='<line x1="'+s.x+'" y1="'+s.y+'" x2="'+t.x+'" y2="'+t.y+'" stroke="'+(e.active?'#ff4d6a':'#243530')+'" stroke-width="'+(e.active?2:1)+'" opacity="'+(e.active?.7:.3)+'"/>'});
  p.nodes.forEach(n=>{const pos=nm[n.id];if(!pos)return;
    const c={isolated:'#557766',critical:'#ff4d6a',warning:'#ffb833',normal:'#00e68a'}[n.status]||'#00e68a';
    const f={isolated:'transparent',critical:'rgba(255,77,106,.15)',warning:'rgba(255,184,51,.1)',normal:'rgba(0,230,138,.06)'}[n.status]||'rgba(0,230,138,.06)';
    const r=n.status==='critical'?18:n.status==='warning'?14:10;
    h+='<circle cx="'+pos.x+'" cy="'+pos.y+'" r="'+r+'" fill="'+f+'" stroke="'+c+'" stroke-width="1.5" style="cursor:pointer" onclick="tgtBreaker(\''+n.id+'\')"/>';
    h+='<text x="'+pos.x+'" y="'+(pos.y+r+12)+'" text-anchor="middle" fill="#7d9a8b" font-size="8" font-family="Outfit">'+n.name+'</text>';
    if(n.risk>.1 && n.status !== 'isolated')h+='<text x="'+pos.x+'" y="'+(pos.y+3)+'" text-anchor="middle" fill="'+c+'" font-size="9" font-weight="600" font-family="IBM Plex Mono">'+(n.risk*100).toFixed(0)+'%</text>'});
  svg.innerHTML=h}

setInterval(()=>{
    if(curCost < tgtCost) {
        curCost += (tgtCost - curCost) * 0.1 + 1;
        if(curCost > tgtCost) curCost = tgtCost;
        document.getElementById('scs').textContent='₹'+Math.floor(curCost).toLocaleString();
    }
}, 30);

function uM(id,v,s){const e=document.getElementById(id);e.querySelector('.v').textContent=v;e.className='m '+s}
async function inj(t,z){await fetch('/api/inject',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({type:t,zone:z,duration:3})})}
async function simLoad(key, val){ const body = {}; body[key] = parseFloat(val); await fetch('/api/simulate_load',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}); }
async function tgtBreaker(id){ await fetch('/api/breaker_trip',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({node:id})}); }
async function togNoise(){const r=await(await fetch('/api/noise',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({})})).json();const b=document.getElementById('nbtn');b.textContent='Noise: '+(r.enabled?'ON':'OFF');b.classList.toggle('on',r.enabled)}

let reco=null;
function toggleVoice(){
    if(!('webkitSpeechRecognition' in window)) return alert('Voice not supported');
    if(!reco) {
        reco = new webkitSpeechRecognition();
        reco.onstart = () => document.getElementById('vcBtn').style.background = '#ff4d6a';
        reco.onend = () => document.getElementById('vcBtn').style.background = 'var(--s2)';
        reco.onresult = (e) => { 
           document.getElementById('qi').value = e.results[0][0].transcript; askQ(true); 
        };
    }
    reco.start();
}

async function askQ(speak=false){
  const i=document.getElementById('qi');const q=i.value.trim();if(!q)return;
  i.value='';document.getElementById('narr').innerHTML='Thinking...<span class="cur"></span>';
  const res=await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({q:q})});
  const r=await res.json();
  document.getElementById('narr').innerHTML='<b>Q: '+q+'</b><br><br>'+r.a+'<span class="cur"></span>';
  if(speak && 'speechSynthesis' in window) {
     speechSynthesis.cancel();
     const ut = new SpeechSynthesisUtterance(r.a);
     ut.rate = 1.05; ut.pitch = 0.9;
     speechSynthesis.speak(ut);
  }
}
async function trigVision(){const vr=document.getElementById('vres');vr.style.display='block';vr.textContent='🔍 Analyzing waveform...';const r=await(await fetch('/api/vision',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({fault_class:_fc})})).json();vr.textContent='';const t=r.analysis;for(let i=0;i<t.length;i++){await new Promise(r=>setTimeout(r,12));vr.textContent=t.substring(0,i+1)}}
