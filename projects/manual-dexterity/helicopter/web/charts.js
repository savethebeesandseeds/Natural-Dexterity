import { HINF_MODE } from '/control-evidence.js';

const NS = 'http://www.w3.org/2000/svg';
const C = { orange:'#cf783e', green:'#37866d', blue:'#527fad', purple:'#a26eb0', dark:'#445d51' };
const numeric = value => typeof value === 'number' && Number.isFinite(value);
const fmt = (value, decimals = Math.abs(value) >= 100 ? 0 : 2) => {
  if (!numeric(value)) return '—';
  if (value === 0) return (0).toFixed(decimals);
  return Math.abs(value) < 10 ** -decimals ? value.toExponential(1) : value.toFixed(decimals);
};
const node = (tag, attributes = {}, text) => {
  const element = document.createElementNS(NS, tag);
  for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, value);
  if (text !== undefined) element.textContent = text;
  return element;
};
const line = (name, color, get, dash = false) => ({ name, color, get, dash });
const positionChart = (axis, index) => ({ title: `${axis} position`, unit:'m', series:[line('Actual', C.orange, s => s.position?.[index]), line('Reference', C.green, s => s.target?.[index], true)] });
const templates = [positionChart('X', 0), positionChart('Y', 1), positionChart('Z', 2), {
  title:'Position error', unit:'m', series:[line('Norm', C.dark, s => Array.isArray(s.error) ? Math.hypot(...s.error) : NaN), line('X', C.orange, s => s.error?.[0]), line('Y', C.blue, s => s.error?.[1]), line('Z', C.green, s => s.error?.[2])]
}, {
  title:'Attitude', unit:'deg', series:[line('Roll', C.orange, s => s.attitude_deg?.[0]), line('Pitch', C.blue, s => s.attitude_deg?.[1]), line('Yaw', C.green, s => s.attitude_deg?.[2])]
}];
const thermo = (key, index) => sample => index === undefined ? sample.thermodynamics?.[key] : sample.thermodynamics?.[key]?.[index];
const hinf = key => sample => sample.hinf?.[key];
const norm = value => Array.isArray(value) && value.every(numeric) ? Math.hypot(...value) : NaN;
const thermodynamicCharts = [
  { title:'Physical entropy generation', unit:'W/K', series:[line('Total',C.dark,thermo('entropy_rate_w_per_k')),line('Wake',C.blue,thermo('entropy_wake_w_per_k')),line('Drag',C.purple,thermo('entropy_drag_w_per_k')),line('Motor',C.orange,thermo('entropy_motor_w_per_k')),line('Heat transfer',C.green,thermo('entropy_heat_transfer_w_per_k'))] },
  { title:'Physical entropy accounting', unit:'J/K', series:[line('Generated',C.orange,thermo('cumulative_entropy_j_per_k')),line('Terminal commitment',C.purple,thermo('terminal_entropy_commitment_j_per_k'),true),line('Run + net commitment',C.dark,thermo('entropy_with_terminal_commitment_j_per_k'))] },
  { title:'Power and dissipation', unit:'W', series:[line('Electrical',C.dark,thermo('electrical_power_w')),line('Air loss',C.blue,thermo('air_dissipation_w')),line('Motor loss',C.orange,thermo('motor_loss_w')),line('Heat rejected',C.green,thermo('heat_rejection_w'))] },
  { title:'Motor temperature', unit:'K', includeZero:false, series:[line('Main motor',C.orange,thermo('motor_temperature_k',0)),line('Tail motor',C.blue,thermo('motor_temperature_k',1))] },
  { title:'Energy accounting', unit:'kJ', series:[line('Electrical input',C.orange,s=>numeric(thermo('electrical_energy_j')(s)) ? s.thermodynamics.electrical_energy_j/1000 : NaN),line('Thermal storage',C.purple,s=>numeric(thermo('stored_thermal_energy_j')(s)) ? s.thermodynamics.stored_thermal_energy_j/1000 : NaN),line('Wake storage',C.blue,s=>numeric(thermo('stored_wake_energy_j')(s)) ? s.thermodynamics.stored_wake_energy_j/1000 : NaN)] },
  { title:'Energy balance residual', unit:'W', series:[line('Residual',C.green,thermo('energy_balance_residual_w'))] }
];
const controllerCharts = [
  { title:'Reconstructed weighted output', unit:'normalized', series:[line('Reconstructed ‖z‖',C.green,s=>norm(s.hinf?.weighted_output))] },
  { title:'Position measurement noise', unit:'m', series:[0,1,2].map((index)=>line(['X','Y','Z'][index],[C.orange,C.blue,C.green][index],s=>numeric(s.measured_state?.[index])&&numeric(s.measurement_truth?.[index])?s.measured_state[index]-s.measurement_truth[index]:NaN)) },
  { title:'Dynamic controller timing', unit:'ms', series:[line('Update',C.blue,hinf('update_ms')),line('Control interval',C.orange,hinf('control_interval_ms'),true)] },
  { title:'Command saturation', unit:'0 / 1', range:[0,1.1], series:[line('Saturated',C.orange,s=>typeof s.hinf?.saturated==='boolean'?Number(s.hinf.saturated):NaN)] },
  { title:'Local operating region', unit:'0 / 1', range:[0,1.1], series:[line('Inside declared region',C.green,s=>typeof s.hinf?.local_scope?.within_declared_region==='boolean'?Number(s.hinf.local_scope.within_declared_region):NaN)] },
  { title:'Applied controller', unit:'0 / 1', range:[0,1.1], series:[line('H∞',C.green,s=>s.hinf?.applied_controller?Number(s.hinf.applied_controller===HINF_MODE):NaN),line('PID landing',C.purple,s=>s.hinf?.applied_controller?Number(s.hinf.applied_controller==='pid_landing_support'):NaN),line('PID',C.orange,s=>s.hinf?.applied_controller?Number(s.hinf.applied_controller==='pid_baseline'):NaN),line('Trim hold',C.blue,s=>s.hinf?.applied_controller?Number(s.hinf.applied_controller==='trim_observation_hold'):NaN)] }
];
const signalConfig=(kind,index)=>({
  title:`${['X','Y','Z'][index]} ${kind} measurement`,unit:kind==='position'?'m':'m/s',time:s=>s.measurement_time,series:[line('Measured',C.orange,s=>s.measured_state?.[index+(kind==='velocity'?3:0)]),line('Aligned simulation reference',C.blue,s=>s.measurement_truth?.[index+(kind==='velocity'?3:0)],true)]
});
const reconstructedSignalConfig=(index,model)=>({title:`Controller input · ${model?.hinf?.measurement_labels?.[index]?.replace(/_/g,' ') || `y${index+1}`}`,unit:index<3?'m':'m/s',time:s=>s.hinf?.time,series:[line('Measured error',C.orange,s=>s.hinf?.measurement?.[index]),line('Controller reconstruction',C.green,s=>s.hinf?.measurement_reconstruction?.[index],true)]});
const controlNames = { collective:'Collective', cyclic_long:'Pitch cyclic', cyclic_lat:'Roll cyclic', tail:'Tail rotor' };
const controlConfig = channel => channel === 'mass' ? {
  title:'Online mass estimation', unit:'kg', includeZero:false, series:[line('Estimated',C.orange,s=>s.parameter_estimation?.mass_kg),line('Simulated plant',C.blue,s=>s.mass_kg,true)]
} : channel === 'inflow' ? {
  title:'Induced inflow', unit:'m/s', series:[line('Main rotor',C.orange,s=>s.inflow_m_s?.[0]),line('Tail rotor',C.blue,s=>s.inflow_m_s?.[1])]
} : channel === 'flap' ? {
  title:'Rotor flapping', unit:'rad', series:[line('Longitudinal',C.orange,s=>s.flap_rad?.[0]),line('Lateral',C.blue,s=>s.flap_rad?.[1])]
} : channel === 'thrust' ? {
  title:'Rotor thrust', unit:'N', series:[line('Command', C.green, s => s.solution?.available ? s.solution.commanded_thrust_n : NaN, true), line('Applied', C.orange, s => s.solution?.available ? s.solution.actual_thrust_n : NaN)]
} : channel === 'wind' ? {
  title:'Wind disturbance', unit:'m/s', series:[line('X', C.orange, s => s.wind?.[0]), line('Y', C.blue, s => s.wind?.[1]), line('Z', C.green, s => s.wind?.[2])]
} : {
  title:`${controlNames[channel]} response`, unit:'normalized', range: channel === 'collective' ? [0, 1] : undefined, series:[line('Command', C.green, s => s.controls?.[channel], true), line('Actuator', C.orange, s => s.actuators?.[channel])]
};

export function createCharts(container, inspectLabel) {
  const width = 450, height = 172, box = { left:58, right:12, top:17, bottom:30 };
  const innerWidth = width - box.left - box.right, innerHeight = height - box.top - box.bottom;
  let samples = [], duration = 70, hoverTime = null, model=null, signalMode='physical';
  const charts = [];

  function create(config, group='flight') {
    const card = document.createElement('article'); card.className = 'chart-card';
    const heading = document.createElement('div'); heading.className = 'chart-heading';
    const title = document.createElement('h3'); title.textContent = config.title;
    const unit = document.createElement('span'); unit.textContent = config.unit;
    heading.append(title, unit);
    const svg = node('svg', { viewBox:`0 0 ${width} ${height}`, class:'chart-svg', role:'img', 'aria-label':`${config.title}, ${config.unit}, over time in seconds` });
    const grid = node('g'); const lines = node('g'); const cursor = node('line', { class:'chart-cursor', y1:box.top, y2:height-box.bottom, visibility:'hidden' });
    const empty = node('text', { x:width/2, y:height/2, 'text-anchor':'middle', class:'chart-empty' }, 'Waiting for run history');
    svg.append(grid, lines, cursor, empty);
    const legend = document.createElement('div'); legend.className = 'chart-legend';
    const readout = document.createElement('div'); readout.className = 'chart-readout'; readout.textContent = 'No samples yet';
    card.append(heading, svg, legend, readout); container.append(card);
    const chart = { config, group, card, title, unit, svg, grid, lines, cursor, empty, legend, readout, paths:[] };
    setConfig(chart, config);
    svg.addEventListener('pointermove', event => {
      if (!samples.length) return;
      const bounds = svg.getBoundingClientRect();
      const point = (event.clientX - bounds.left) * width / bounds.width;
      hoverTime = Math.max(0, Math.min(duration, (point-box.left) / innerWidth * duration));
      updateInspection();
    });
    svg.addEventListener('pointerleave', () => { hoverTime = null; updateInspection(); });
    charts.push(chart);
    return chart;
  }

  function setConfig(chart, config) {
    chart.config = config; chart.title.textContent = config.title; chart.unit.textContent = config.unit;
    chart.svg.setAttribute('aria-label', `${config.title}, ${config.unit}, over time in seconds`);
    chart.lines.replaceChildren(); chart.legend.replaceChildren();
    chart.paths = config.series.map(series => {
      const path = node('path', { class:'chart-line', stroke:series.color });
      if (series.dash) path.setAttribute('stroke-dasharray', '5 4');
      chart.lines.append(path);
      const item = document.createElement('span');
      const key = node('svg', { viewBox:'0 0 16 6', 'aria-hidden':'true' });
      const mark = node('line', { x1:0, y1:3, x2:16, y2:3, stroke:series.color, 'stroke-width':2 });
      if (series.dash) mark.setAttribute('stroke-dasharray', '4 3');
      key.append(mark); item.append(key, document.createTextNode(series.name)); chart.legend.append(item);
      return path;
    });
  }

  function axisRange(config) {
    if (config.range) return config.range;
    let lo = config.includeZero === false ? Infinity : 0, hi = config.includeZero === false ? -Infinity : 0;
    for (const sample of samples) for (const series of config.series) {
      const value = series.get(sample);
      if (numeric(value)) { lo = Math.min(lo, value); hi = Math.max(hi, value); }
    }
    if (!numeric(lo) || !numeric(hi)) return [0, 1];
    if (hi === lo) { lo -= 0.005; hi += 0.005; }
    const span = hi-lo;
    const exponent = 10 ** Math.floor(Math.log10(span/3));
    const scaled = span/3/exponent;
    const step = (scaled <= 1 ? 1 : scaled <= 2 ? 2 : scaled <= 5 ? 5 : 10) * exponent;
    return [Math.floor((lo - span * 0.04)/step)*step, Math.ceil((hi + span * 0.04)/step)*step];
  }

  function render(chart) {
    const [lo, hi] = axisRange(chart.config);
    const x = time => box.left + time / duration * innerWidth;
    const y = value => box.top + innerHeight - (value-lo)/(hi-lo)*innerHeight;
    chart.grid.replaceChildren();
    for (let index=0; index<=3; index++) {
      const rawValue = lo+(hi-lo)*index/3;
      const value = Math.abs(rawValue) <= (hi-lo)*Number.EPSILON*8 ? 0 : rawValue;
      const level = y(value);
      chart.grid.append(node('line', { x1:box.left, x2:width-box.right, y1:level, y2:level, class:'chart-grid' }));
      chart.grid.append(node('text', { x:box.left-7, y:level+3, 'text-anchor':'end', class:'chart-tick' }, fmt(value, Math.abs(hi-lo)>5 ? 1 : Math.abs(hi-lo)>=0.1 ? 2 : 3)));
    }
    for (let index=0; index<=4; index++) {
      const time = duration*index/4;
      chart.grid.append(node('text', { x:x(time), y:height-12, 'text-anchor':'middle', class:'chart-tick' }, Number.isInteger(time) ? time.toFixed(0) : time.toFixed(1)));
    }
    chart.grid.append(node('line', { x1:box.left, x2:width-box.right, y1:height-box.bottom, y2:height-box.bottom, class:'chart-axis' }));
    chart.grid.append(node('text', { x:width-box.right, y:height-1, 'text-anchor':'end', class:'chart-unit' }, 'time (s)'));
    const hasValues = samples.some(sample => chart.config.series.some(series => numeric(series.get(sample))));
    chart.empty.textContent = samples.length ? 'Not available in this run' : 'Waiting for run history';
    chart.empty.setAttribute('visibility', hasValues ? 'hidden' : 'visible');
    for (let seriesIndex=0; seriesIndex<chart.config.series.length; seriesIndex++) {
      const series = chart.config.series[seriesIndex];
      let drawing = false, d = '';
      for (const sample of samples) {
        const value = series.get(sample);
        if (!numeric(value)) { drawing=false; continue; }
        const time=chart.config.time ? chart.config.time(sample) : sample.time;
        if(!numeric(time)){drawing=false;continue;}
        d += `${drawing ? 'L' : 'M'}${x(time).toFixed(2)},${y(value).toFixed(2)} `;
        drawing = true;
      }
      chart.paths[seriesIndex].setAttribute('d', d);
    }
  }

  function updateInspection() {
    if (!samples.length) {
      for (const chart of charts) { chart.readout.textContent = 'No samples yet'; chart.cursor.setAttribute('visibility', 'hidden'); }
      inspectLabel.textContent = 'Move over a plot to inspect';
      return;
    }
    const desiredTime = hoverTime === null ? samples[samples.length-1].time : hoverTime;
    let lower=0, upper=samples.length-1;
    while (lower<upper) { const mid=Math.floor((lower+upper)/2); if(samples[mid].time<desiredTime) lower=mid+1; else upper=mid; }
    const index = lower>0 && Math.abs(samples[lower-1].time-desiredTime)<Math.abs(samples[lower].time-desiredTime) ? lower-1 : lower;
    const sample = samples[index];
    inspectLabel.textContent = hoverTime===null ? `Latest sample · ${sample.time.toFixed(1)} s` : `Inspecting · ${sample.time.toFixed(1)} s`;
    for (const chart of charts) {
      const time=chart.config.time ? chart.config.time(sample) : sample.time;
      chart.readout.textContent = `${chart.config.time ? `t = ${fmt(time)} s · ` : ''}${chart.config.series.map(series=>`${series.name} ${fmt(series.get(sample))}`).join(' · ')}`;
      chart.cursor.setAttribute('visibility', hoverTime===null || !numeric(time) ? 'hidden' : 'visible');
      chart.cursor.setAttribute('x1', numeric(time) ? box.left+time/duration*innerWidth : box.left);
      chart.cursor.setAttribute('x2', numeric(time) ? box.left+time/duration*innerWidth : box.left);
    }
  }

  templates.forEach(config => create(config,'flight'));
  const control = create(controlConfig('collective'));
  thermodynamicCharts.forEach(config => create(config,'thermodynamics'));
  controllerCharts.forEach(config => create({...config,time:s=>s.hinf?.time},'controller'));
  const signalCharts=['position','velocity'].flatMap(kind=>[0,1,2].map(index=>({kind,index,chart:create(signalConfig(kind,index),'signals')})));
  const configureSignals=()=>{for(const [channel,{kind,index,chart}] of signalCharts.entries()){setConfig(chart,signalMode==='controller'?reconstructedSignalConfig(channel,model):signalConfig(kind,index));render(chart);}updateInspection();};
  const showGroup = group => charts.forEach(chart => { chart.card.hidden = chart.group!==group; });
  showGroup('controller');
  return {
    update(nextSamples, runDuration=70) {
      samples=nextSamples; duration=Math.max(1, runDuration);
      charts.forEach(render); updateInspection();
    },
    setControlChannel(channel) { setConfig(control, controlConfig(channel)); render(control); updateInspection(); },
    setSignalMode(mode) {signalMode=mode;configureSignals();},
    setModel(value) {model=value;configureSignals();},
    setGroup(group) { showGroup(group); },
    clear() { samples=[]; hoverTime=null; charts.forEach(render); updateInspection(); }
  };
}
