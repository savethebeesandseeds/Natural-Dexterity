import { createFlightView } from '/flight-view.js';
import { createCharts } from '/charts.js';
import { createHinfInspector } from '/hinf-inspector.js';
import { updateDesign } from '/design.js';
import { renderEvaluation, evaluationUnavailable } from '/validation.js';
import { HINF_MODE, landingSupport, observationHold, pidIntegralNotice } from '/control-evidence.js';
import { updateParameterEstimate } from '/parameters.js';

const el = id => document.getElementById(id);
const put = (id, text) => { const node=el(id); if(node) node.textContent=text; };
const finite = value => typeof value==='number' && Number.isFinite(value);
const number = (value, digits=2) => finite(value) ? (Math.abs(value)<0.5*10**-digits ? 0 : value).toFixed(digits) : '—';
const vector = (value, digits=2) => Array.isArray(value) ? value.map(item=>number(item,digits)).join(', ') : '—';
const validVector = (value, length=3) => Array.isArray(value) && value.length===length && value.every(finite);
const valueText = value => Array.isArray(value) ? value.map(item=>valueText(item)).join(', ') : typeof value==='number' ? finite(value) ? String(Number(value.toPrecision(10))) : '—' : typeof value==='object' && value!==null ? JSON.stringify(value) : value===undefined ? '—' : String(value);
const magnitude = value => validVector(value) ? Math.hypot(...value) : NaN;
const view = createFlightView(el('scene'));
const charts = createCharts(el('plots-grid'), el('hover-time'));
const controllerInspector=createHinfInspector();
let state=null, runId=null, generation=0, resetting=false, actionPending=false;
let fatalBackendError=false;
let lastStateAt=0, lastHistoryAt=0, historyPayload=null, modelMetadata=null;
let evaluationPending=false, modelPending=false;
let historyPollTimer=null, historyBusy=false;

const solutionFields = [
  ['position_error','Position error · world m'],
  ['velocity_error','Velocity error · world m/s'],
  ['position_integral','PID position error integral · world m·s'],
  ['desired_acceleration','Desired acceleration · world m/s²'],
  ['desired_force_world','Desired force · world N'],
  ['attitude_error_body','Attitude error · body sine-like vector, dimensionless'],
  ['desired_torque_body','Desired torque · body N·m'],
  ['projected_thrust_n','Projected thrust · N'],
  ['commanded_thrust_n','Commanded thrust · N'],
  ['actual_thrust_n','Applied rotor thrust · N'],
  ['force_body','Applied force · body N'],
  ['force_world','Applied force + drag · world N'],
  ['torque_body','Applied torque · body N·m'],
  ['torque_world','Applied torque + damping · world N·m'],
  ['drag_force_world','Drag force · world N'],
  ['damping_torque_world','Angular damping torque · world N·m']
];
const solutionCells = new Map();
for(const [key,label] of solutionFields) {
  const row=document.createElement('tr'), name=document.createElement('th'), cell=document.createElement('td');
  name.scope='row'; name.textContent=label; cell.textContent='—'; row.append(name,cell); el('solution-values').append(row); solutionCells.set(key,cell);
}

async function request(url, options={}) {
  const controller=new AbortController();
  const timeoutMs=5000;
  const timeout=setTimeout(()=>controller.abort(),timeoutMs);
  try {
    const response=await fetch(url,{...options,cache:'no-store',signal:controller.signal});
    let data;
    try { data=await response.json(); } catch { throw new Error(`Invalid JSON from ${url}.`); }
    if(!response.ok) {
      const error=new Error(data?.error || `Request failed (${response.status}).`); error.status=response.status; error.responseData=data; throw error;
    }
    return data;
  } finally { clearTimeout(timeout); }
}

function refreshButtons() {
  const offline=!lastStateAt || performance.now()-lastStateAt>2500;
  for(const id of ['play-button','reset-button','gust-button']) el(id).disabled=actionPending || (offline && !fatalBackendError) || (fatalBackendError && id!=='reset-button') || (id==='gust-button' && (state?.completed || state?.crashed || !state?.running));
  el('controller-mode').disabled=actionPending || (offline && !fatalBackendError) || ![HINF_MODE,'pid_baseline'].includes(state?.controller_mode);
}

function clearHistory(message='Loading the new run…') {
  historyPayload=null; lastHistoryAt=0;
  charts.clear(); view.setTrail([]); put('history-status',message);
}

function receiveState(data) {
  if(!data || !finite(data.time) || !validVector(data.position) || !validVector(data.velocity) || !validVector(data.target) || !validVector(data.quaternion,4)) throw new Error('Incomplete simulation state.');
  if(fatalBackendError) { fatalBackendError=false; el('request-error').hidden=true; }
  const nextRunId=String(data.run_id ?? 'legacy');
  const changedRun=runId!==null && (runId!==nextRunId || (state && data.time < state.time-0.05));
  if(changedRun) {
    generation++;
    clearHistory('Run changed · loading simulator history…');
    // Model and report provenance may change when the native server restarts.
    loadModel(); loadEvaluation();
  }
  runId=nextRunId; state=data; lastStateAt=performance.now();
  el('connection-dot').className='status-dot live'; put('connection-label','Simulator connected');
  put('state-age',data.completed || data.crashed ? 'Final state' : data.running ? 'Live simulated state' : 'Paused state');
  const phase=String(data.phase||'ready').replace(/[_-]+/g,' ');
  put('phase',data.crashed ? 'Flight crashed' : data.completed ? 'Mission complete' : phase);
  put('flight-label',data.crashed ? 'Flight stopped · crash detected' : data.completed ? 'Mission complete' : data.running ? `Running · ${phase}` : `Paused · ${phase}`);
  put('run-label',`Run ${data.run_id ?? '—'}`);
  put('elapsed',number(data.time,1)); put('duration',number(data.mission?.duration ?? 70,1));
  el('mission-progress').max=data.mission?.duration ?? 70; el('mission-progress').value=data.time;
  put('mission-name',data.mission?.name || 'Takeoff → hover → waypoints → gust recovery → return → land');
  put('altitude',number(data.position[2]));
  put('error',number(Math.hypot(...data.target.map((value,index)=>value-data.position[index]))));
  put('speed',number(magnitude(data.velocity)));
  for(const [label,values] of [['position',data.position],['reference',data.target],['velocity',data.velocity],['attitude',data.attitude_deg],['wind',data.wind]]) for(let index=0;index<3;index++) put(`${label}-${index}`,number(values?.[index]));
  const solution=data.solution;
  const available=solution?.available===true;
  put('solution-time',available ? `t = ${number(solution.time,2)} s` : 'No applied step');
  put('applied-force',available ? vector(solution.force_world) : '—');
  put('applied-torque',available ? vector(solution.torque_body,3) : '—');
  put('thrust',available ? `${number(solution.commanded_thrust_n)} → ${number(solution.actual_thrust_n)}` : '—');
  put('solution-status',!available ? 'Waiting for the first control step.' : !solution.active ? 'Controller inactive; actual forces are still reported.' : observationHold(data.hinf) ? 'Prepared trim observation interval; dynamic feedback is inactive.' : landingSupport(data.hinf) ? 'PID landing support is active.' : data.hinf?.applied_controller===HINF_MODE ? 'Dynamic H∞ output feedback; design certificate and operating scope are shown separately.' : data.controller_mode==='pid_baseline' ? 'PID baseline: position feedback → attitude target → actuator commands.' : 'Applied controller has not been identified.');
  for(const [key,cell] of solutionCells) {
    const value=solution?.[key];
    const notice=key==='position_integral' ? pidIntegralNotice(data) : null;
    cell.textContent=!available ? '—' : notice || (value===null ? 'Not computed by this controller' : Array.isArray(value) ? vector(value,4) : number(value,4));
  }
  put('quaternion-value',vector(data.quaternion,4)); put('body-rates-value',vector(data.body_rates,4));
  put('actuators-value',vector(['collective','cyclic_long','cyclic_lat','tail'].map(key=>data.actuators?.[key]),4));
  put('input-radians',vector(data.input_rad,4)); put('pitch-radians',vector(data.actuator_pitch_rad,4));
  put('inflow-value',vector(data.inflow_m_s,4)); put('flap-value',vector(data.flap_rad,4));
  updateParameterEstimate(data,modelMetadata);
  const finished=data.completed || data.crashed;
  put('play-label',finished ? 'Replay' : data.running ? 'Pause' : 'Play'); put('play-icon',!finished && data.running ? 'Ⅱ' : '▶');
  el('play-button').setAttribute('aria-label',finished ? 'Replay mission' : data.running ? 'Pause simulation' : 'Play simulation');
  put('gpu-name',data.gpu?.available ? `GPU access: ${data.gpu.name || 'available'} · rendering: browser WebGL` : 'GPU device access unavailable · rendering: browser WebGL');
  view.update(data); controllerInspector.update(data); refreshButtons();
  if(modelMetadata) {
    put('model-status',`${modelMetadata.name || 'Native model'} · Simulated plant mass ${number(data.mass_kg,3)} kg · Simulated plant inertia [${vector(data.inertia_body_kg_m2,4)}] kg·m². ${modelMetadata.telemetry?.timing || ''}`);
  }
}

async function pollState() {
  const started=performance.now(), requestGeneration=generation;
  try {
    if(!actionPending) {
      const data=await request('/api/state');
      if(requestGeneration===generation && !actionPending) receiveState(data);
    }
  } catch(error) {
    if(requestGeneration===generation && error.status===503 && error.responseData?.error) {
      fatalBackendError=true;
      put('request-error',`Simulation failure: ${error.responseData.error}`); el('request-error').hidden=false;
      refreshButtons();
    }
    if(!lastStateAt || performance.now()-lastStateAt>2500) {
      el('connection-dot').className='status-dot offline'; put('connection-label','Awaiting simulator · retrying');
      put('state-age','Stale data'); put('flight-label','Snapshot is stale · last known pose'); refreshButtons();
    }
  } finally { setTimeout(pollState,Math.max(40,100-(performance.now()-started))); }
}

async function pollHistory() {
  clearTimeout(historyPollTimer);
  if(historyBusy) return;
  historyBusy=true;
  const requestGeneration=generation;
  try {
    if(!actionPending) {
      const data=await request('/api/telemetry');
      if(requestGeneration!==generation || actionPending || runId===null || String(data.run_id)!==runId) return;
      if(!Array.isArray(data.samples)) throw new Error('Invalid history response.');
      const samples=data.samples.filter(sample=>finite(sample.time) && validVector(sample.position) && validVector(sample.target));
      if(samples.some((sample,index)=>index>0 && sample.time<samples[index-1].time)) throw new Error('Telemetry samples are not in time order.');
      // A same-ID process restart can arrive between state and telemetry reads.
      // Reject history from the future of the currently observed run.
      if(samples.length && state && samples[samples.length-1].time>state.time+1.0) return;
      historyPayload={...data,samples}; lastHistoryAt=performance.now();
      charts.update(samples,state?.mission?.duration || 70);
      view.setTrail(samples.map(sample=>sample.position));
      const duration=samples.length ? `${number(samples[samples.length-1].time,1)} s` : '0 s';
      put('history-status',`${data.history_complete ? 'Full run history' : 'Partial run history'} · ${samples.length} samples · ${number(1/(data.sample_period_seconds||0.1),0)} Hz · ${duration}${data.finished ? ' · finished' : ''}`);
    }
  } catch(error) {
    if(requestGeneration===generation) put('history-status',historyPayload ? 'History unavailable · displaying last received samples' : `History unavailable · ${error.message}`);
  } finally { historyBusy=false; historyPollTimer=setTimeout(pollHistory,500); }
}

async function control(action) {
  if(actionPending) return;
  actionPending=true;
  generation++;
  const newRun=['reset','hinf','pid'].includes(action);
  if(newRun) { resetting=true; clearHistory('Restarting mission…'); }
  refreshButtons(); el('request-error').hidden=true;
  try {
    const data=await request('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action})});
    resetting=false; receiveState(data);
    if(newRun) { clearTimeout(historyPollTimer); pollHistory(); }
  } catch(error) {
    put('request-error',error.name==='AbortError' ? 'The simulator did not respond in time. The action may have been applied; check the live state before retrying.' : error.message);
    el('request-error').hidden=false;
  } finally { actionPending=false; resetting=false; if(state) el('controller-mode').value=state.controller_mode===HINF_MODE ? 'hinf' : state.controller_mode==='pid_baseline' ? 'pid' : ''; refreshButtons(); }
}

function addParameter(container, label, value, unit='') {
  const row=document.createElement('tr'), th=document.createElement('th'), td=document.createElement('td');
  th.scope='row'; th.textContent=label; td.textContent=`${valueText(value)}${unit ? ` ${unit}` : ''}`; row.append(th,td); container.append(row);
}
function parameterGroup(container,title) {
  const row=document.createElement('tr'), th=document.createElement('th'); row.className='group-row'; th.colSpan=2; th.textContent=title; row.append(th); container.append(row);
}
function parameterObject(container,object,prefix='') {
  for(const [key,value] of Object.entries(object || {})) {
    const label=`${prefix}${key==='handover_settling' ? 'predicted handover settling constraints' : key==='terminal_settling' ? 'predicted terminal settling constraints' : key.replace(/_/g,' ')}`;
    if(value && typeof value==='object' && !Array.isArray(value)) parameterObject(container,value,`${label} / `);
    else addParameter(container,label,value);
  }
}

async function loadModel() {
  if(modelPending) return;
  modelPending=true;
  try {
    const data=await request('/api/model'); modelMetadata=data;
    updateDesign(data); controllerInspector.setModel(data); charts.setModel(data);
    put('model-json',JSON.stringify(data,null,2));
    put('controller-explanation',[data.hinf?.method,data.hinf?.certificate_scope,data.controller?.architecture].filter(Boolean).join(' ') || 'No minimum-entropy H∞ metadata has been reported.');
    const equations=el('equations'); equations.replaceChildren();
    const equationList=value=>Array.isArray(value) ? value : Object.entries(value || {}).map(([name,expression])=>({name:name.replace(/_/g,' '),expression:valueText(expression)}));
    for(const equation of [...equationList(data.equations),...equationList(data.plant?.equations)]) {
      const card=document.createElement('div'); card.className='equation';
      const name=document.createElement('strong'), formula=document.createElement('code'), note=document.createElement('p');
      name.textContent=equation.name; formula.textContent=equation.expression; note.textContent=equation.note || '';
      card.append(name,formula,note); equations.append(card);
    }
    const parameters=el('model-parameters'); parameters.replaceChildren();
    parameterGroup(parameters,'Physics');
    const definitions=[
      ['nominal_mass_kg','Nominal mass','kg'],['nominal_inertia_body_kg_m2','Nominal body inertia [Ixx, Iyy, Izz]','kg·m²'],['gravity_world_m_s2','Gravity [X, Y, Z]','m/s²'],['timestep_s','Integration step','s'],['integrator','Integrator',''],['main_thrust_max_n','Maximum main rotor thrust','N'],['cyclic_long_torque_max_nm','Pitch cyclic torque scale','N·m'],['cyclic_lat_torque_max_nm','Roll cyclic torque scale','N·m'],['tail_torque_max_nm','Tail torque scale','N·m'],['tail_lever_m','Tail lever arm','m'],['rotor_reaction_nm_per_n','Rotor reaction coefficient','N·m/N'],['linear_drag_n_per_m_s','Linear drag','N/(m/s)'],['angular_damping_nm_per_rad_s','Angular damping','N·m/(rad/s)'],['actuator_lag_s','Actuator lag time constant','s']
    ];
    for(const [key,label,unit] of definitions) if(data.physics?.[key]!==undefined) addParameter(parameters,label,data.physics[key],unit);
    for(const [section,label] of [['position','PID position feedback'],['attitude','Attitude stabilization']]) {
      const values=data.controller?.[section] || {};
      if(!Object.keys(values).length) continue;
      parameterGroup(parameters,label);
      for(const [key,value] of Object.entries(values)) addParameter(parameters,key==='axes' ? 'Axes' : key==='gain_units' ? 'Gain units' : key.toUpperCase(),value);
    }
    if(data.controller?.limits) {
      parameterGroup(parameters,'Command limits');
      if(typeof data.controller.limits==='object' && !Array.isArray(data.controller.limits)) parameterObject(parameters,data.controller.limits);
      else addParameter(parameters,'Runtime limit handling',data.controller.limits);
    }
    if(data.sensors) { parameterGroup(parameters,'Sensor model'); parameterObject(parameters,data.sensors); }
    if(data.hinf?.selected_design) { parameterGroup(parameters,'Selected minimum-entropy H∞ design'); parameterObject(parameters,data.hinf.selected_design); }
    if(data.parameter_estimation) { parameterGroup(parameters,'Online parameter estimation'); parameterObject(parameters,data.parameter_estimation); }
    if(data.initial_observation_hold) { parameterGroup(parameters,'Initial trim observation hold'); parameterObject(parameters,data.initial_observation_hold); }
    for(const key of ['initialization','state','airframe','environment','rotors','motors','limits','baseline','smoothing']) if(data.plant?.[key]) { parameterGroup(parameters,`Reference plant: ${key}`); parameterObject(parameters,data.plant[key]); }
    const coordinates=data.coordinates || data.plant?.coordinates || {};
    put('coordinate-convention',typeof coordinates==='string' ? coordinates : Object.entries(coordinates).map(([key,value])=>`${key.replace(/_/g,' ')}: ${valueText(value)}`).join(' · '));
    const limitations=data.limitations || data.plant?.validity || ['This simulated demonstration does not establish readiness for a physical helicopter.'];
    put('model-limitations',Array.isArray(limitations) ? limitations.join(' ') : valueText(limitations));
    if(data.plant?.boundary) put('entropy-accounting-note',`${data.plant.boundary} Horizon objectives and whole-run totals are distinct. ${data.plant.contact || ''}`);
    updateParameterEstimate(state,data);
    if(state) put('model-status',`${data.name || 'Native model'} · Simulated plant mass ${number(state.mass_kg,3)} kg · Simulated plant inertia [${vector(state.inertia_body_kg_m2,4)}] kg·m². ${data.telemetry?.timing || ''}`);
  } catch(error) { put('model-status',`Model metadata unavailable · ${error.message}`); setTimeout(loadModel,10000); }
  finally { modelPending=false; }
}

async function loadEvaluation() {
  if(evaluationPending)return;
  evaluationPending=true;
  try { renderEvaluation(await request('/api/evaluation')); }
  catch(error) { evaluationUnavailable(error); }
  finally { evaluationPending=false; }
}

el('play-button').addEventListener('click',()=>control(state?.completed || state?.crashed ? 'reset' : state?.running ? 'pause' : 'play'));
el('reset-button').addEventListener('click',()=>control('reset'));
el('gust-button').addEventListener('click',()=>control('gust'));
el('controller-mode').addEventListener('change',event=>control(event.target.value));
el('camera-button').addEventListener('click',()=>view.resetCamera());
el('control-channel').addEventListener('change',event=>charts.setControlChannel(event.target.value));
el('signal-channel').addEventListener('change',event=>{
  charts.setSignalMode(event.target.value);
  put('chart-scope',event.target.value==='controller' ? 'Measured reference-error channels and their reconstruction from the central controller state. This reconstruction is not an unbiased plant-state estimate.' : 'Noisy physical sensor signals versus simulation references at the same measurement timestamp.');
});
for(const button of document.querySelectorAll('[data-chart-group]')) button.addEventListener('click',()=>{
  const group=button.dataset.chartGroup;
  charts.setGroup(group);
  put('chart-scope',group==='thermodynamics' ? 'Secondary physical measurements. Thermal entropy and energy are not the H∞ design objective.' : group==='signals' ? 'Noisy physical sensor signals versus simulation references at the same measurement timestamp. These are distinct from the whitened controller input and dynamic controller state.' : group==='controller' ? 'Live controller measurements. The local-region indicator is a diagnostic, not a nonlinear guarantee.' : 'Measured nonlinear mission tracking; reference agreement does not certify an H∞ norm.');
  el('control-channel-label').hidden=group!=='flight';
  el('signal-channel-label').hidden=group!=='signals';
  if(group==='signals')put('chart-scope',el('signal-channel').value==='controller' ? 'Measured reference-error channels and their reconstruction from the central controller state. This reconstruction is not an unbiased plant-state estimate.' : 'Noisy physical sensor signals versus simulation references at the same measurement timestamp.');
  for(const item of document.querySelectorAll('[data-chart-group]')) item.setAttribute('aria-pressed',String(item===button));
});
charts.clear();
pollState(); pollHistory(); loadModel(); loadEvaluation();
setInterval(loadEvaluation,15000);
setInterval(()=>{
  if(lastStateAt && performance.now()-lastStateAt>2500) {
    el('connection-dot').className='status-dot offline'; put('connection-label','Awaiting simulator · snapshot stale');
    put('state-age',`Last update ${number((performance.now()-lastStateAt)/1000,1)} s ago`); put('flight-label','Snapshot is stale · last known pose'); refreshButtons();
  }
  if(lastHistoryAt && performance.now()-lastHistoryAt>4000) put('history-status','History is stale · displaying last received samples');
},1000);


