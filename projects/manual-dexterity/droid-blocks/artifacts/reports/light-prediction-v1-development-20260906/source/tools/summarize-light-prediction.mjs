import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';

const PROTOCOL='docs/LIGHT_PREDICTION_EXPERIMENT.md';
const PROTOCOL_SHA='7cc508c8cf1bc88e0b0f417b49df93022e895f8f90c9f1c91e528372696ef6be';
const REFERENCE='artifacts/reports/light-context-v1-development-20260906/report.json';
const REFERENCE_SHA='f3f729120df6b13f79845ecbd4ee72409f111f76b1b525037d3c059e41f7c6ee';
const LEARNED=['current_input','history_input','history_no_input'];
const MODES=[...LEARNED,'persistence'];
const DIMS={current_input:12,history_input:142,history_no_input:132};
const TRAIN=Array.from({length:8},(_,i)=>201+i),TEST=[209,210,211,212],SEEDS=[...TRAIN,...TEST];
const ORIGINS=Array.from({length:23},(_,i)=>10*(i+1)),HORIZONS=[1,4,10];
const PRIMARY=[1,2,3,5,7,9,11],SCALES=[.15,1,1,10,.5,50,50,50,10,10,10,1000];
const hash=s=>crypto.createHash('sha256').update(s).digest('hex');
const hashFile=p=>hash(fs.readFileSync(p));
const readJson=p=>JSON.parse(fs.readFileSync(p,'utf8').replace(/^\uFEFF/,''));
function finite(x,label='numeric value') {assert.equal(typeof x,'number',label);assert.ok(Number.isFinite(x),label);return x;}
function near(a,b,label,tolerance=1e-10) {finite(a,label);finite(b,label);assert.ok(Math.abs(a-b)<=tolerance*Math.max(1,Math.abs(b)),`${label}: ${a} != ${b}`);}
function keys(o,expected,label) {assert.ok(o&&typeof o==='object'&&!Array.isArray(o),label);assert.deepEqual(Object.keys(o).sort(),[...expected].sort(),label);}
function sha(x,label='SHA-256') {assert.match(x,/^[a-f0-9]{64}$/,label);}
function vector(v,n,label) {assert.ok(Array.isArray(v),label);assert.equal(v.length,n,label);v.forEach(x=>finite(x,label));}
function sameNumbers(a,b,label,tolerance=1e-10) {assert.equal(a.length,b.length,label);a.forEach((x,i)=>near(x,b[i],`${label}[${i}]`,tolerance));}
function safeJoin(root,relative) {
 assert.equal(typeof relative,'string');
 assert.ok(relative.length&&!path.isAbsolute(relative)&&!path.win32.isAbsolute(relative)&&!relative.includes(':')&&!relative.includes('\0'),'Unsafe evidence path');
 const parts=relative.split(/[\\/]/);assert.ok(parts.every(s=>s&&s!=='.'&&s!=='..'),'Unsafe evidence path component');
 const base=path.resolve(root),resolved=path.resolve(base,...parts),rel=path.relative(base,resolved);
 assert.ok(rel&&!rel.startsWith('..'+path.sep)&&rel!=='..'&&!path.isAbsolute(rel),'Evidence path escapes archive');
 let current=base;
 for(const part of parts) {current=path.join(current,part);assert.ok(!fs.lstatSync(current).isSymbolicLink(),'Symbolic links are not evidence');}
 const realBase=fs.realpathSync(base),real=fs.realpathSync(resolved),realRel=path.relative(realBase,real);
 assert.ok(realRel&&realRel!=='..'&&!realRel.startsWith('..'+path.sep)&&!path.isAbsolute(realRel),'Resolved evidence escapes archive');
 assert.ok(fs.statSync(real).isFile(),'Evidence must be a regular file');return resolved;
}
// Retain native number spellings (including -0.0) when checking native JSON
// digests. JSON.stringify is deliberately never used as a native serializer.
function valueEnd(text,start) {
 let depth=0,quoted=false,escaped=false;
 for(let i=start;i<text.length;i++) {
  const c=text[i];
  if(quoted) {if(escaped)escaped=false;else if(c==='\\')escaped=true;else if(c==='"'){quoted=false;if(depth===0)return i+1;}continue;}
  if(c==='"'){quoted=true;continue;}
  if(c==='['||c==='{')depth++;
  else if(c===']'||c==='}') {if(depth===0)return i;if(--depth===0)return i+1;}
  else if(depth===0&&(c===','||/\s/.test(c)))return i;
 }
 return text.length;
}
function rawObject(text) {
 const result={};let i=0;while(/\s/.test(text[i]))i++;assert.equal(text[i++],'{');
 for(;;) {
  while(/\s/.test(text[i]))i++;if(text[i]==='}')break;
  assert.equal(text[i],'"');const keyEnd=valueEnd(text,i),key=JSON.parse(text.slice(i,keyEnd));i=keyEnd;
  while(/\s/.test(text[i]))i++;assert.equal(text[i++],':');while(/\s/.test(text[i]))i++;
  const end=valueEnd(text,i);assert.ok(!Object.hasOwn(result,key),'Duplicate native JSON key');result[key]=text.slice(i,end);i=end;
  while(/\s/.test(text[i]))i++;if(text[i]==='}')break;assert.equal(text[i++],',');
 }
 return result;
}
function rawArray(text) {
 const values=[];let i=0;while(/\s/.test(text[i]))i++;assert.equal(text[i++],'[');
 for(;;) {while(/\s/.test(text[i]))i++;if(text[i]===']')break;const end=valueEnd(text,i);values.push(text.slice(i,end));i=end;while(/\s/.test(text[i]))i++;if(text[i]===']')break;assert.equal(text[i++],',');}
 return values;
}
function removeRawKeys(text,removed) {
 const fields=rawObject(text);return '{'+Object.entries(fields).filter(([k])=>!removed.includes(k)).map(([k,v])=>JSON.stringify(k)+':'+v).join(',')+'}';
}
function nonLightRaw(text) {
 const fields=rawObject(text),sensors=rawArray(fields.reward_sensors);
 fields.reward_sensors='['+sensors.map(raw=>JSON.parse(raw).family_id==='ambient_light_v0'?removeRawKeys(raw,['observations']):raw).join(',')+']';
 return '{'+Object.entries(fields).map(([k,v])=>JSON.stringify(k)+':'+v).join(',')+'}';
}
function mechanicalRaw(text) {
 const fields=rawObject(text);delete fields.sun;delete fields.reward;
 fields.diagnostics=removeRawKeys(fields.diagnostics,['instantaneous_light_lux']);
 fields.light_sensor=removeRawKeys(fields.light_sensor,['observations']);
 return '{'+Object.entries(fields).map(([k,v])=>JSON.stringify(k)+':'+v).join(',')+'}';
}
function tape(seed) {
 const mask=(1n<<64n)-1n;let state=BigInt(seed);
 const next=()=>{let z=state=(state+0x9e3779b97f4a7c15n)&mask;z=((z^(z>>30n))*0xbf58476d1ce4e5b9n)&mask;z=((z^(z>>27n))*0x94d049bb133111ebn)&mask;return z^(z>>31n);};
 const out=[];while(out.length<240){const length=[1,2,4,8][Number(next()%4n)],request=Number(next()%3n)-1;for(let i=0;i<length&&out.length<240;i++)out.push(request);}return out;
}
function observation(o,time,effort,fresh=false) {
 keys(o,['schema_version','reward_sensors','actuator_feedback'],'Local observation boundary');assert.equal(o.schema_version,'construction_observation_v2');
 assert.equal(o.reward_sensors.length,2);assert.equal(o.actuator_feedback.length,1);
 const imu=o.reward_sensors.find(s=>s.family_id==='imu_6axis_v0'),light=o.reward_sensors.find(s=>s.family_id==='ambient_light_v0');assert.ok(imu&&light);
 const ids=new Set();
 for(const [s,period,latency] of [[imu,.01,.006],[light,.1,.02]]) {
  keys(s,['module_id','family_id','valid','sequence','sample_time_s','delivered_time_s','age_s','sample_period_s','latency_s','observations'],'Sensor boundary');
  assert.equal(typeof s.module_id,'string');assert.ok(s.module_id.length&&!ids.has(s.module_id));ids.add(s.module_id);
  near(s.sample_period_s,period,'sample period');near(s.latency_s,latency,'sensor latency');
  if(time===0) {assert.equal(s.valid,false);assert.equal(s.sequence,0);for(const key of ['sample_time_s','delivered_time_s','age_s'])assert.equal(s[key],null);}
  else {assert.equal(s.valid,true);const sequence=Math.floor((time-latency+1e-8)/period)+1;assert.equal(s.sequence,sequence,'Native sensor sequence');near(s.sample_time_s,(sequence-1)*period,'Sensor sample time',2e-6);near(s.delivered_time_s,s.sample_time_s+latency,'Delivered time',2e-6);near(s.age_s,time-s.sample_time_s,'Sensor age',2e-6);assert.ok(s.delivered_time_s<=time+2e-6&&s.age_s>=latency-2e-6);}
 }
 if(fresh)near(light.age_s,.02,'Fresh sampled light',2e-6);
 keys(imu.observations,['specific_force_m_s2','angular_velocity_rad_s'],'IMU channels');keys(light.observations,['illuminance_lux','saturated'],'Light channels');
 const force=imu.observations.specific_force_m_s2,gyro=imu.observations.angular_velocity_rad_s;vector(force,3,'Specific force');vector(gyro,3,'Gyro');
 const lux=finite(light.observations.illuminance_lux,'Lux');assert.ok(lux>=0&&lux<=1000);assert.equal(typeof light.observations.saturated,'boolean');
 if(time===0){assert.deepEqual(force,[0,0,0]);assert.deepEqual(gyro,[0,0,0]);assert.equal(lux,0);assert.equal(light.observations.saturated,false);}
 const motor=o.actuator_feedback[0];keys(motor,['module_id','sku_id','valid','feedback'],'Motor boundary');assert.equal(motor.valid,true);assert.equal(motor.sku_id,'rotary_dc_gearmotor_v0');assert.ok(typeof motor.module_id==='string'&&motor.module_id.length&&!ids.has(motor.module_id));
 const m=motor.feedback;keys(m,['position_rad','velocity_rad_s','current_a','bus_voltage_v','temperature_c','output_torque_nm','load_impedance_nm_s_per_rad','stuck_score','stuck','fault_flags'],'Motor feedback');
 for(const key of ['position_rad','velocity_rad_s','current_a','bus_voltage_v','temperature_c','output_torque_nm','load_impedance_nm_s_per_rad','stuck_score'])finite(m[key],key);
 assert.equal(typeof m.stuck,'boolean');assert.deepEqual(m.fault_flags,[]);finite(effort,'Executed effort');assert.ok(Math.abs(effort)<=.15+1e-12);
 return [effort/.15,Math.sin(m.position_rad),Math.cos(m.position_rad),m.velocity_rad_s/10,m.current_a/.5,...force.map(v=>v/50),...gyro.map(v=>v/10),lux/1000];
}
function verifyTrial(trial,directory,assembly) {
 const id=`body-${trial.body_index}-probe-${trial.seed}-${trial.condition}`;
 assert.equal(trial.trial_id,id);assert.ok([0,1,2].includes(trial.body_index)&&SEEDS.includes(trial.seed)&&['A','B'].includes(trial.condition));assert.equal(trial.split,TRAIN.includes(trial.seed)?'train':'test');
 assert.equal(trial.completed,true,id+' incomplete');assert.equal(trial.error,'');assert.deepEqual(trial.snapshot_errors,[]);assert.equal(trial.final_public_physical_snapshot_available,true);
 for(const field of ['completed_steps','accepted_steps','recorded_transition_count','recorded_effort_count','recorded_requested_effort_count'])assert.equal(trial[field],1201,id+' '+field);
 assert.equal(trial.last_attempted_action.step,1201);assert.equal(trial.last_attempted_action.fully_accepted,true);assert.equal(trial.exact_full_transition_replay,true);assert.equal(trial.replay.exact,true);assert.equal(trial.replay.completed_steps,1201);assert.equal(trial.replay.error,'');
 for(const field of ['requested_effort_trace_sha256','effort_trace_sha256','observation_trace_sha256','transition_trace_sha256','mechanical_state_trace_sha256','final_public_physical_snapshot_sha256']) {sha(trial[field]);assert.equal(trial.replay[field],trial[field],id+' replay '+field);}
 const expectedTape=tape(trial.seed);assert.deepEqual(trial.request_tape,expectedTape,'Independently generated SplitMix64 tape');assert.equal(trial.request_tape_sha256,hash('['+expectedTape.map(v=>v.toFixed(1)).join(',')+']'));
 assert.equal(trial.trace.path,`traces/${id}.json`);const file=safeJoin(directory,trial.trace.path),text=fs.readFileSync(file,'utf8');assert.equal(hash(text),trial.trace.sha256,id+' trace bytes');
 const trace=JSON.parse(text),raw=rawObject(text);assert.equal(trace.schema,'light_prediction_trace_v1');assert.equal(trace.trial_id,id);
 keys(trace,['schema','trial_id','initial_observation','initial_public_physical_snapshot','requested_efforts','efforts','transitions','mechanical_state_sha256','final_public_physical_snapshot'],'Full trace schema');
 for(const key of ['requested_efforts','efforts','transitions','mechanical_state_sha256'])assert.equal(trace[key].length,1201);
 for(const [field,key] of [['requested_effort_trace_sha256','requested_efforts'],['effort_trace_sha256','efforts'],['transition_trace_sha256','transitions'],['mechanical_state_trace_sha256','mechanical_state_sha256'],['final_public_physical_snapshot_sha256','final_public_physical_snapshot']])assert.equal(trial[field],hash(raw[key]),id+' native serialized '+key);
 const transitionRaw=rawArray(raw.transitions),observationRaw=transitionRaw.map(t=>rawObject(t).observation),nonLight=observationRaw.map(nonLightRaw);
 assert.equal(trial.observation_trace_sha256,hash('['+observationRaw.join(',')+']'),'Native observation digest');assert.equal(trial.non_light_observation_trace_sha256,hash('['+nonLight.join(',')+']'),'Native non-light digest');
 assert.equal(trial.initial_public_mechanical_snapshot_sha256,hash(mechanicalRaw(raw.initial_public_physical_snapshot)),'Native initial mechanical digest');
 const sun={position_m:[trial.condition==='A'?.3:-.3,0,.2],intensity_lux:1000};assert.deepEqual(trace.initial_public_physical_snapshot.sun,sun);assert.deepEqual(trace.final_public_physical_snapshot.sun,sun);
 assert.deepEqual(trace.initial_public_physical_snapshot.assembly,assembly,'Initial physical assembly');assert.deepEqual(trace.final_public_physical_snapshot.assembly,assembly,'Final physical assembly');
 near(trace.initial_public_physical_snapshot.elapsed_s,0,'Initial elapsed time',0);near(trace.initial_public_physical_snapshot.sim_time,0,'Initial simulation time',0);assert.equal(trace.initial_public_physical_snapshot.tick,0);
 near(trace.final_public_physical_snapshot.elapsed_s,24.02,'Final elapsed time',2e-6);near(trace.final_public_physical_snapshot.sim_time,24.02,'Final physical time',2e-6);assert.equal(trace.final_public_physical_snapshot.tick,12010);
 observation(trace.initial_observation,0,0);let reward=0,maxCurrent=0,maxTemperature=0,maxSpeed=0,vetoes=0,flags=0;const outputs=[];
 for(let i=0;i<1201;i++) {
  const request=i===0?0:.15*expectedTape[Math.floor((i-1)/5)];near(trace.requested_efforts[i],request,'Requested five-control hold',0);
  const before=i===0?trace.initial_observation:trace.transitions[i-1].observation,previous=i===0?0:trace.efforts[i-1];let target=request;
  if(before.reward_sensors.some(s=>!s.valid))target=0;
  else if(target!==0){const sign=target>0?1:-1;target=sign*Math.min(Math.abs(target),.05*Math.max(0,3-sign*before.actuator_feedback[0].feedback.velocity_rad_s));}
  const governed=Math.max(previous-.1,Math.min(previous+.1,target));near(trace.efforts[i],governed,'Independent local governor',1e-12);
  const t=trace.transitions[i];assert.equal(t.terminated,false);assert.equal(t.truncated,false);assert.equal(t.safety.veto,false);assert.deepEqual(t.safety.flags,[]);assert.equal(t.info.physics_steps_executed,10);near(t.info.elapsed_control_dt_s,.02,'Control duration');
  assert.equal(t.reward_components.length,2);near(t.reward,t.reward_components.reduce((sum,c)=>sum+finite(c.transition_reward),0),'Component reward sum');reward+=finite(t.reward);
  const output=observation(t.observation,(i+1)*.02,trace.efforts[i],i%5===0);if(i%5===0)outputs.push(output);
  const m=t.observation.actuator_feedback[0].feedback;maxCurrent=Math.max(maxCurrent,Math.abs(m.current_a));maxTemperature=Math.max(maxTemperature,m.temperature_c);maxSpeed=Math.max(maxSpeed,Math.abs(m.velocity_rad_s));if(t.safety.veto)vetoes++;if(m.fault_flags.length)flags++;
  sha(trace.mechanical_state_sha256[i]);
 }
 near(trial.last_attempted_action.requested_effort,trace.requested_efforts.at(-1),'Last request',0);near(trial.last_attempted_action.effort,trace.efforts.at(-1),'Last effort',0);
 const metrics={steps:1201,duration_s:24.02,integrated_sensor_reward:reward,electrical_energy_j:trace.final_public_physical_snapshot.diagnostics.electrical_energy_j,max_current_a:maxCurrent,max_temperature_c:maxTemperature,max_motor_speed_rad_s:maxSpeed,native_veto_steps:vetoes,feedback_flag_steps:flags};verifyNumbers(trial.metrics,metrics,id+' probe metrics');assert.ok(metrics.electrical_energy_j>=0);
 assert.equal(trial.frames.length,241);assert.equal(outputs.length,241);
 trial.frames.forEach((frame,n)=>{keys(frame,['index','step','time_s','output'],'Sample frame');assert.equal(frame.index,n);assert.equal(frame.step,1+5*n);near(frame.time_s,.02+.1*n,'Sample time');sameNumbers(frame.output,outputs[n],'Strict independently projected output',2e-12);});
 return {outputs,request_tape:expectedTape,pair:{requests:raw.requested_efforts,efforts:raw.efforts,nonLight:'['+nonLight.join(',')+']',mechanicalHashes:raw.mechanical_state_sha256,initial:mechanicalRaw(raw.initial_public_physical_snapshot),final:mechanicalRaw(raw.final_public_physical_snapshot)}};
}
function verifyNumbers(record,expected,label,tolerance=1e-9) {
 keys(record,Object.keys(expected),label);
 for(const [key,value] of Object.entries(expected)) {
  const actual=record[key],where=label+'.'+key;
  if(typeof value==='number')near(actual,value,where,tolerance);
  else if(Array.isArray(value)){assert.ok(Array.isArray(actual),where);assert.equal(actual.length,value.length,where);value.forEach((v,i)=>v===null?assert.equal(actual[i],null,where):near(actual[i],v,where,tolerance));}
  else assert.deepEqual(actual,value,where);
 }
}
function stateAt(data,origin,mode) {
 const p=mode==='current_input'?1:11,out=[];assert.ok(origin>=p-1);
 for(let lag=0;lag<p;lag++)out.push(...data.outputs[origin-lag]);
 if(mode!=='history_no_input')for(let lag=1;lag<p;lag++)out.push(data.request_tape[origin-lag]);
 assert.equal(out.length,DIMS[mode]);return out;
}
function fit(mode,membership,data) {
 const d=DIMS[mode]+1+(mode==='history_no_input'?0:1),n=membership.length;
 const gram=Array.from({length:d},()=>new Float64Array(d)),rhs=Array.from({length:12},()=>new Float64Array(d));
 for(const member of membership) {
  const trial=data.get(member.trial_id),origin=member.origin_index,phi=[1,...stateAt(trial,origin,mode)];
  if(mode!=='history_no_input')phi.push(trial.request_tape[origin]);
  const target=trial.outputs[origin+1];
  for(let i=0;i<d;i++){for(let j=0;j<=i;j++)gram[i][j]+=phi[i]*phi[j];for(let c=0;c<12;c++)rhs[c][i]+=phi[i]*target[c];}
 }
 for(let i=0;i<d;i++){for(let j=0;j<=i;j++){gram[i][j]/=n;gram[j][i]=gram[i][j];}if(i)gram[i][i]+=.001;for(let c=0;c<12;c++)rhs[c][i]/=n;}
 const lower=Array.from({length:d},()=>new Float64Array(d));
 for(let i=0;i<d;i++)for(let j=0;j<=i;j++) {
  let value=gram[i][j];for(let k=0;k<j;k++)value-=lower[i][k]*lower[j][k];
  if(i===j){assert.ok(value>0&&Number.isFinite(value),'Independent regularized Gram is not positive definite');lower[i][j]=Math.sqrt(value);}else lower[i][j]=value/lower[j][j];
 }
 const coefficients=[];
 for(let c=0;c<12;c++) {
  const temp=new Float64Array(d),solution=new Array(d).fill(0);
  for(let i=0;i<d;i++){let value=rhs[c][i];for(let j=0;j<i;j++)value-=lower[i][j]*temp[j];temp[i]=value/lower[i][i];}
  for(let i=d-1;i>=0;i--){let value=temp[i];for(let j=i+1;j<d;j++)value-=lower[j][i]*solution[j];solution[i]=value/lower[i][i];finite(solution[i],'Independently fitted coefficient');}
  coefficients.push(solution);
 }
 return {coefficients,gram,rhs};
}
function verifyModel(model,mode,independent) {
 keys(model,['schema_version','mode','taps','uses_input','state_dimension','output_dimension','coefficient_order','coefficients','state_space','fit'],'Numeric predictor boundary');
 const n=DIMS[mode],p=mode==='current_input'?1:11,input=mode!=='history_no_input';
 assert.equal(model.schema_version,'light_prediction_model_v1');assert.equal(model.mode,mode);assert.equal(model.taps,p);assert.equal(model.uses_input,input);assert.equal(model.state_dimension,n);assert.equal(model.output_dimension,12);assert.equal(model.coefficient_order,'[bias,state...,current_request_if_enabled]');
 assert.equal(model.coefficients.length,12);model.coefficients.forEach((row,c)=>sameNumbers(row,independent.coefficients[c],'Independent ridge coefficient',1e-8));
 let residual=0,rhsMax=0;for(let c=0;c<12;c++)for(let i=0;i<independent.gram.length;i++){let value=0;for(let j=0;j<independent.gram.length;j++)value+=independent.gram[i][j]*model.coefficients[c][j];residual=Math.max(residual,Math.abs(value-independent.rhs[c][i]));rhsMax=Math.max(rhsMax,Math.abs(independent.rhs[c][i]));}
 assert.ok(residual/Math.max(1,rhsMax)<=1e-8,'Native coefficients do not solve declared normal equations');
 verifyNumbers(model.fit,{training_rows:3680,lambda:.001,normal_equation_residual_max_abs:residual,normal_equation_residual_relative:residual/Math.max(1,rhsMax)},'Fit diagnostics',1e-10);
 const space=model.state_space;keys(space,['A','B','C','D','c','equations'],'State-space boundary');assert.equal(space.equations,'z_next=A*z+B*r+c; y=C*z+D*r');
 assert.equal(space.A.length,n);assert.equal(space.B.length,n);assert.equal(space.C.length,12);assert.equal(space.D.length,12);vector(space.c,n,'Affine state bias');
 for(let i=0;i<n;i++) {
  vector(space.A[i],n,'A row');vector(space.B[i],1,'B row');
  for(let j=0;j<n;j++) {
   const expected=i<12?model.coefficients[i][j+1]:(i<12*p?(j===i-12?1:0):(i>12*p&&j===i-1?1:0));
   near(space.A[i][j],expected,'Explicit companion A',0);
  }
  const expectedB=i<12?(input?model.coefficients[i].at(-1):0):(input&&p>1&&i===12*p?1:0);
  near(space.B[i][0],expectedB,'Explicit companion B',0);near(space.c[i],i<12?model.coefficients[i][0]:0,'Explicit affine c',0);
 }
 for(let i=0;i<12;i++){vector(space.C[i],n,'C row');vector(space.D[i],1,'D row');space.C[i].forEach((value,j)=>near(value,i===j?1:0,'Output C selector',0));near(space.D[i][0],0,'No current-output feedthrough',0);}
 return independent.coefficients;
}
function predict(coefficients,mode,state,request) {
 const phi=[1,...state];if(mode!=='history_no_input')phi.push(request);
 const output=coefficients.map(row=>{let sum=0;for(let i=0;i<row.length;i++){const product=row[i]*phi[i];if(!Number.isFinite(product))throw new Error('Nonfinite recursive product');sum+=product;if(!Number.isFinite(sum))throw new Error('Nonfinite recursive sum');}return sum;});
 const p=mode==='current_input'?1:11,next=[...output,...state.slice(0,12*(p-1))];
 if(mode!=='history_no_input'&&p>1)next.push(request,...state.slice(12*p,12*p+p-2));
 return {output,state:next};
}
// Scaled sums keep finite large errors honest without overflowing intermediate
// JS sums that the native long-double score can represent.
function emptyScore() {return {origins:0,failed:0,scales:new Array(12).fill(0),sums:new Array(12).fill(0)};}
function addSquared(score,channel,scale,normalized=1) {
 if(scale===0)return;const old=score.scales[channel];
 if(old<scale){score.sums[channel]=score.sums[channel]*(old/scale)**2+normalized**2;score.scales[channel]=scale;}
 else score.sums[channel]+=(scale/old*normalized)**2;
}
function addScore(score,actual,predicted) {
 score.origins++;if(!predicted){score.failed++;return;}
 for(let c=0;c<12;c++){const difference=predicted[c]-actual[c];if(Number.isFinite(difference))addSquared(score,c,Math.abs(difference));else{const scale=Math.max(Math.abs(predicted[c]),Math.abs(actual[c]));addSquared(score,c,scale,predicted[c]/scale-actual[c]/scale);}}
}
const numberOrNull=value=>Number.isFinite(value)?value:null;
function rms(values) {const scale=Math.max(0,...values);if(scale===0)return 0;if(!Number.isFinite(scale))return Infinity;return scale*Math.sqrt(values.reduce((sum,x)=>sum+(x/scale)**2,0)/values.length);}
function aggregate(scores) {
 const origins=scores.reduce((s,x)=>s+x.origins,0),failed=scores.reduce((s,x)=>s+x.failed,0),eligible=scores.filter(s=>s.origins>0);
 const out={origins,failed_origins:failed,eligible_trials:eligible.length,complete:failed===0&&eligible.length>0,mse:null,rmse:null,channels_mse:null,channels_rmse:null,channels_physical_rmse:null,metric_overflow:false};
 if(failed||!eligible.length)return out;
 const channelRms=Array.from({length:12},(_,c)=>rms(eligible.map(s=>s.scales[c]*Math.sqrt(s.sums[c]/s.origins))));
 const primaryRms=rms(PRIMARY.map(c=>channelRms[c]));
 out.mse=numberOrNull(primaryRms**2);out.rmse=numberOrNull(primaryRms);
 out.channels_mse=channelRms.map(x=>numberOrNull(x*x));out.channels_rmse=channelRms.map(numberOrNull);out.channels_physical_rmse=channelRms.map((x,c)=>numberOrNull(x*SCALES[c]));
 out.metric_overflow=out.mse===null||out.rmse===null||[...out.channels_mse,...out.channels_rmse,...out.channels_physical_rmse].includes(null);out.complete=!out.metric_overflow;return out;
}
function attachChanged(out,changed) {
 for(const [to,from] of Object.entries({changed_request_rmse:'rmse',changed_request_mse:'mse',changed_request_origins:'origins',changed_request_eligible_trials:'eligible_trials',changed_request_failed_origins:'failed_origins',changed_request_complete:'complete',changed_request_channels_rmse:'channels_rmse',changed_request_channels_physical_rmse:'channels_physical_rmse'}))out[to]=changed[from];return out;
}
const relative=(value,reference)=>value===null||reference===null||reference===0?null:1-value/reference;
const atMost=(value,reference,factor)=>value!==null&&reference!==null&&value<=factor*reference;
function range(records) {return !records.length||records.some(r=>r.rmse===null)?null:{min:Math.min(...records.map(r=>r.rmse)),max:Math.max(...records.map(r=>r.rmse))};}
function verifyMetricTree(actual,expected,label) {
 keys(actual,Object.keys(expected),label);
 for(const [key,value] of Object.entries(expected)) {
  const received=actual[key],where=label+'.'+key;
  if(typeof value==='number')near(received,value,where,1e-9);
  else if(Array.isArray(value)){assert.ok(Array.isArray(received),where);assert.equal(received.length,value.length,where);for(let i=0;i<value.length;i++){if(value[i]&&typeof value[i]==='object')verifyMetricTree(received[i],value[i],where+'['+i+']');else if(typeof value[i]==='number')near(received[i],value[i],where+'['+i+']',1e-9);else assert.equal(received[i],value[i],where);}}
  else if(value&&typeof value==='object')verifyMetricTree(received,value,where);
  else assert.equal(received,value,where);
 }
}
function verifyEvaluation(report,data,fitted) {
 const evaluation=report.evaluation;assert.equal(evaluation.complete,true);assert.equal(evaluation.bodies.length,3);let failures=0;const summaries=[];
 for(let body=0;body<3;body++) {
  const result=evaluation.bodies[body];assert.equal(result.body_index,body);assert.equal(result.complete,true);assert.equal(result.forecasts.length,184);
  const trials=report.trials.filter(t=>t.body_index===body&&t.split==='test');assert.equal(trials.length,8);let index=0;
  for(const trial of trials)for(const origin of ORIGINS) {
   const record=result.forecasts[index++],source=data.get(trial.trial_id),requests=source.request_tape.slice(origin,origin+10),actual=source.outputs.slice(origin+1,origin+11),changed=requests.slice(1).some(v=>v!==requests[0]);
   keys(record,['trial_id','seed','condition','origin_index','origin_time_s','initial_output','actual','requests','changed_future_request','predictions','failures'],'Forecast boundary');
   assert.equal(record.trial_id,trial.trial_id);assert.equal(record.seed,trial.seed);assert.equal(record.condition,trial.condition);assert.equal(record.origin_index,origin);near(record.origin_time_s,.02+.1*origin,'Forecast origin time');
   sameNumbers(record.initial_output,source.outputs[origin],'Forecast initial local output',2e-12);assert.deepEqual(record.requests,requests,'Known requests only');assert.equal(record.changed_future_request,changed);assert.equal(record.actual.length,10);record.actual.forEach((y,h)=>sameNumbers(y,actual[h],'Actual target membership',2e-12));
   keys(record.predictions,MODES,'Forecast model membership');assert.ok(record.failures&&typeof record.failures==='object'&&!Array.isArray(record.failures));assert.ok(Object.keys(record.failures).every(id=>LEARNED.includes(id)));
   for(const mode of LEARNED) {
    const predicted=record.predictions[mode];assert.ok(Array.isArray(predicted));assert.ok(predicted.length<=10);let state=stateAt(source,origin,mode),independentFailure=null;
    for(let step=0;step<10;step++) {
     let next;try{next=predict(fitted.get(`${body}:${mode}`),mode,state,requests[step]);}catch(error){independentFailure=step;break;}
     assert.ok(step<predicted.length,`${mode}: recorded failure has no independent numerical cause`);vector(predicted[step],12,'Stored recursive output');sameNumbers(predicted[step],next.output,`${mode} recursive forecast`,1e-7);state=next.state;
    }
    if(independentFailure===null){assert.equal(predicted.length,10);assert.ok(!Object.hasOwn(record.failures,mode),'Spurious forecast failure');}
    else {const failure=record.failures[mode];keys(failure,['error','attempted_step','recorded_predictions'],'Forecast failure');assert.ok(typeof failure.error==='string'&&failure.error.length);assert.equal(failure.attempted_step,independentFailure+1);assert.equal(failure.recorded_predictions,independentFailure);assert.equal(predicted.length,independentFailure);failures++;}
   }
   assert.equal(record.predictions.persistence.length,10);for(const output of record.predictions.persistence)sameNumbers(output,source.outputs[origin],'Persistence holds origin output',2e-12);
  }
  const expectedModels=[];
  for(const mode of MODES) {
   const model={model_id:mode,horizons:[],failed_forecasts:result.forecasts.filter(f=>Object.hasOwn(f.failures,mode)).length};
   for(const horizon of HORIZONS) {
    const all=[],changedAll=[],bySeed=new Map(),changedBySeed=new Map(),trialResults=[];
    for(const trial of trials) {
     const score=emptyScore(),changed=emptyScore();
     for(const record of result.forecasts.filter(f=>f.trial_id===trial.trial_id)) {const prediction=record.predictions[mode][horizon-1]??null;addScore(score,record.actual[horizon-1],prediction);if(record.changed_future_request)addScore(changed,record.actual[horizon-1],prediction);}
     assert.equal(score.origins,23);all.push(score);changedAll.push(changed);if(!bySeed.has(trial.seed)){bySeed.set(trial.seed,[]);changedBySeed.set(trial.seed,[]);}bySeed.get(trial.seed).push(score);changedBySeed.get(trial.seed).push(changed);
     trialResults.push({...attachChanged(aggregate([score]),aggregate([changed])),trial_id:trial.trial_id,seed:trial.seed,condition:trial.condition});
    }
    const metrics={...attachChanged(aggregate(all),aggregate(changedAll)),steps:horizon,seconds:horizon*.1,trials:trialResults,seed_pairs:[]};
    for(const seed of TEST) {assert.equal(bySeed.get(seed).length,2);metrics.seed_pairs.push({...attachChanged(aggregate(bySeed.get(seed)),aggregate(changedBySeed.get(seed))),seed});}
    metrics.trial_rmse_range=range(metrics.trials);metrics.seed_pair_rmse_range=range(metrics.seed_pairs);model.horizons.push(metrics);
   }
   expectedModels.push(model);
  }
  for(let mi=0;mi<MODES.length;mi++) {
   const model=expectedModels[mi];
   for(let h=0;h<HORIZONS.length;h++) {
    const value=model.horizons[h],persistence=expectedModels[3].horizons[h],blind=expectedModels[2].horizons[h];
    value.relative_improvement_over_persistence=relative(value.rmse,persistence.rmse);value.relative_improvement_over_no_input=relative(value.rmse,blind.rmse);value.changed_request_relative_improvement_over_no_input=relative(value.changed_request_rmse,blind.changed_request_rmse);
    value.trials.forEach((trial,i)=>{trial.relative_improvement_over_persistence=relative(trial.rmse,persistence.trials[i].rmse);trial.relative_improvement_over_no_input=relative(trial.rmse,blind.trials[i].rmse);});value.seed_pairs.forEach((pair,i)=>pair.relative_improvement_over_persistence=relative(pair.rmse,persistence.seed_pairs[i].rmse));
   }
   if(mi>=2){model.readiness={candidate:false,passed:false};continue;}
   const medium=model.horizons[1],long=model.horizons[2],persistence=expectedModels[3].horizons[2],blind=expectedModels[2].horizons[2];
   const checks={no_failed_forecasts:model.failed_forecasts===0,rmse_04s_at_most_08_persistence:atMost(medium.rmse,expectedModels[3].horizons[1].rmse,.8),rmse_1s_at_most_08_persistence:atMost(long.rmse,persistence.rmse,.8),every_trial_1s_no_worse_than_persistence:long.trials.every((trial,i)=>atMost(trial.rmse,persistence.trials[i].rmse,1)),rmse_1s_at_most_09_no_input:atMost(long.rmse,blind.rmse,.9),changed_request_subset_nonempty:long.changed_request_origins>0,changed_request_rmse_1s_at_most_09_no_input:atMost(long.changed_request_rmse,blind.changed_request_rmse,.9)};
   model.readiness={candidate:true,passed:Object.values(checks).every(Boolean),checks};
  }
  assert.equal(result.metrics.length,4);result.metrics.forEach((model,i)=>verifyMetricTree(model,expectedModels[i],`Body ${body} ${MODES[i]} metrics`));
  assert.equal(result.readiness_passed,expectedModels[1].readiness.passed);assert.equal(result.simpler_candidate_readiness_passed,expectedModels[0].readiness.passed);
  summaries.push({body_index:body,body_name:report.bodies[body].assembly.name,readiness_passed:expectedModels[1].readiness.passed,simpler_candidate_readiness_passed:expectedModels[0].readiness.passed,models:expectedModels.map(model=>({model_id:model.model_id,failed_forecasts:model.failed_forecasts,readiness:model.readiness,horizons:model.horizons.map(h=>({steps:h.steps,seconds:h.seconds,rmse:h.rmse,changed_request_rmse:h.changed_request_rmse,changed_request_origins:h.changed_request_origins,relative_improvement_over_persistence:h.relative_improvement_over_persistence,relative_improvement_over_no_input:h.relative_improvement_over_no_input,trial_rmse_range:h.trial_rmse_range,seed_pair_rmse_range:h.seed_pair_rmse_range}))}))});
 }
 assert.equal(evaluation.failed_forecasts,failures);const ready=summaries.every(b=>b.readiness_passed),simple=summaries.every(b=>b.simpler_candidate_readiness_passed);assert.equal(evaluation.readiness_passed,ready);assert.equal(evaluation.simpler_candidate_readiness_passed,simple);
 return {by_body:summaries,readiness_passed:ready,simpler_candidate_readiness_passed:simple,failed_forecasts:failures};
}
function compactRaw(text) {
 let out='',quoted=false,escaped=false;for(const c of text){if(quoted){out+=c;if(escaped)escaped=false;else if(c==='\\')escaped=true;else if(c==='"')quoted=false;}else if(c==='"'){quoted=true;out+=c;}else if(!/\s/.test(c))out+=c;}return out;
}
export function summarizeLightPrediction(reportPath) {
 const absolute=path.resolve(reportPath),directory=path.dirname(absolute),reportText=fs.readFileSync(absolute,'utf8'),report=JSON.parse(reportText),reportHash=hash(reportText);
 assert.equal(report.schema,'light_prediction_v1');assert.equal(report.complete,true,'Incomplete native evidence cannot be certified');assert.equal(report.finished,true);assert.equal(report.stage,'evaluated');assert.equal(report.failed_trials,0);assert.equal(report.failed_pairs,0);assert.ok(!report.fatal_error);
 keys(report.checks,['all_trials_complete','all_pairs_valid','all_full_replays','models_immutable_during_evaluation','source_snapshots_unchanged'],'Report integrity gates');for(const [key,value] of Object.entries(report.checks))assert.equal(value,true,key);
 const p=report.protocol;
 for(const [key,value] of Object.entries({path:PROTOCOL,sha256:PROTOCOL_SHA,control_dt_s:.02,sample_dt_s:.1,first_frame_time_s:.02,steps:1201,duration_s:24.02,frame_count:241,requested_intervals:240,body_count:3,trial_count:72,replay_count:72,training_origins:'10..239 inclusive',forecast_origins:'10,20,...,230',request_normalization:.15}))assert.equal(p[key],value,'Locked protocol '+key);
 assert.deepEqual(p.training_seeds,TRAIN);assert.deepEqual(p.heldout_seeds,TEST);assert.deepEqual(p.horizon_steps,HORIZONS);assert.deepEqual(p.primary_channels,PRIMARY);assert.deepEqual(p.physical_channel_scales,SCALES);assert.deepEqual(p.channel_units,['effort','dimensionless','dimensionless','rad/s','A','m/s^2','m/s^2','m/s^2','rad/s','rad/s','rad/s','lux']);
 assert.deepEqual(p.source_A,{position_m:[.3,0,.2],intensity_lux:1000});assert.deepEqual(p.source_B,{position_m:[-.3,0,.2],intensity_lux:1000});assert.equal(p.request_generator,'SplitMix64(seed); duration lengths[next()%4], then amplitude int(next()%3)-1; truncate to 240 intervals');
 const manifest=report.source_sha256;assert.ok(manifest&&Object.keys(manifest).length>0);
 for(const [relative,digest] of Object.entries(manifest)){sha(digest);assert.equal(hashFile(safeJoin(directory,'source/'+relative)),digest,'Archived source bytes '+relative);}
 for(const required of [PROTOCOL,REFERENCE,'src/light_prediction.cpp','src/light_prediction_cli.cpp','include/droid/light_prediction.hpp','tests/native_light_prediction_tests.cpp','src/light_context.cpp','src/light_learner.cpp','src/light_session.cpp','src/construction.cpp','include/droid/light_learner.hpp','CMakeLists.txt','run.sh','Dockerfile','setup.sh','config/module_catalog.json','models/droid.xml','tools/summarize-light-prediction.mjs','config/learning_experiment_v3.json','artifacts/light-search-policy-v3-seed0.json','build/light-prediction-qa/runtime.json'])assert.ok(manifest[required],'Missing source provenance '+required);
 assert.equal(manifest[PROTOCOL],PROTOCOL_SHA);assert.equal(manifest[REFERENCE],REFERENCE_SHA);assert.equal(hashFile(fileURLToPath(import.meta.url)),manifest['tools/summarize-light-prediction.mjs'],'Validator differs from sealed declared source');
 assert.deepEqual(report.reference_report,{path:REFERENCE,sha256:REFERENCE_SHA,snapshot:'source/'+REFERENCE});const reference=readJson(safeJoin(directory,report.reference_report.snapshot));assert.equal(reference.complete,true);assert.equal(reference.failed_trials,0);assert.equal(reference.bodies.length,2);
 for(const preserved of ['src/light_learner.cpp','src/light_session.cpp','src/construction.cpp','src/module_catalog.cpp','src/light_context.cpp','include/droid/light_learner.hpp','include/droid/light_session.hpp','include/droid/construction.hpp','include/droid/light_context.hpp','config/module_catalog.json','models/droid.xml','config/learning_experiment_v3.json','artifacts/light-search-policy-v3-seed0.json'])assert.equal(manifest[preserved],reference.source_sha256[preserved],'Unchanged plant/controller/frozen input '+preserved);
 for(const [relative,digest] of Object.entries(reference.source_sha256))if(/^(src\/|include\/droid\/|tests\/).+\.(cpp|hpp)$/.test(relative))assert.equal(manifest[relative],digest,'Preserved prior native implementation '+relative);
 assert.deepEqual(report.physics_spec,reference.physics_spec,'Unchanged physical specification');assert.deepEqual(report.governor_spec,reference.control_spec.learner,'Unchanged governor/learner specification');assert.deepEqual(report.module_catalog,readJson(safeJoin(directory,'source/config/module_catalog.json')));assert.deepEqual(report.module_catalog,reference.module_catalog);
 const spec=report.predictor_spec;assert.equal(spec.schema_version,'light_prediction_specification_v1');assert.equal(spec.sample_interval_s,.1);assert.equal(spec.output_dimension,12);assert.equal(spec.output_projection,'project_light_context_frame');assert.equal(spec.request_normalization,.15);assert.equal(spec.lambda,.001);for(const field of ['bias_regularized','fitted_scaling','clipping','unit_circle_projection'])assert.equal(spec[field],false);
 assert.deepEqual(spec.modes,LEARNED.map(mode=>({id:mode,taps:mode==='current_input'?1:11,uses_input:mode!=='history_no_input',state_dimension:DIMS[mode]})));
 assert.equal(report.bodies.length,3);const rawBodies=rawArray(rawObject(reportText).bodies);
 for(let body=0;body<3;body++){const item=report.bodies[body];assert.equal(item.body_index,body);const expected=structuredClone(reference.bodies[body===2?0:body].assembly);if(body===2){expected.segments=[4,3];expected.name='Sun creature with a longer first beam';}assert.deepEqual(item.assembly,expected,'Declared body assembly');assert.equal(item.assembly_sha256,hash(compactRaw(rawObject(rawBodies[body]).assembly)),'Native assembly digest');if(body<2)assert.equal(item.assembly_sha256,reference.bodies[body].assembly_sha256);}
 assert.equal(report.trials.length,72);const trials=new Map();let expectedIndex=0;
 for(const body of [0,1,2])for(const seed of SEEDS)for(const condition of ['A','B']){const trial=report.trials[expectedIndex++],id=`body-${body}-probe-${seed}-${condition}`;assert.equal(trial.trial_id,id,'Full declared trial membership/order');assert.equal(trial.body_index,body);assert.equal(trial.seed,seed);assert.equal(trial.condition,condition);assert.equal(trial.split,TRAIN.includes(seed)?'train':'test');trials.set(id,trial);}
 const sealed=report.training_models;assert.equal(sealed.path,'models.json');assert.equal(sealed.saved_before_evaluation,true);assert.equal(sealed.unchanged_after_evaluation,true);assert.equal(sealed.training_rows_per_body,3680);assert.equal(sealed.sha256_after_evaluation,sealed.sha256);sha(sealed.sha256);
 const modelPath=safeJoin(directory,sealed.path);assert.equal(hashFile(modelPath),sealed.sha256,'Sealed model bytes');const models=readJson(modelPath);keys(models,['schema','saved_before_evaluation','bodies'],'Model artifact');assert.equal(models.schema,'light_prediction_models_v1');assert.equal(models.saved_before_evaluation,true);assert.equal(models.bodies.length,3);
 for(let body=0;body<3;body++){const bm=models.bodies[body];keys(bm,['body_index','training_membership','models'],'Body model artifact');assert.equal(bm.body_index,body);const expected=[];for(const trial of report.trials)if(trial.body_index===body&&trial.split==='train')for(let origin=10;origin<=239;origin++)expected.push({trial_id:trial.trial_id,origin_index:origin});assert.equal(expected.length,3680);assert.deepEqual(bm.training_membership,expected,'Entire training membership excludes held-out seeds');assert.equal(bm.models.length,3);bm.models.forEach((model,i)=>assert.equal(model.mode,LEARNED[i]));}
 assert.equal(report.pairs.length,36);const data=new Map();let pairIndex=0;
 for(const body of [0,1,2])for(const seed of SEEDS) {
  const pair=report.pairs[pairIndex++],a=trials.get(`body-${body}-probe-${seed}-A`),b=trials.get(`body-${body}-probe-${seed}-B`);keys(pair,['body_index','seed','trial_a','trial_b','valid','checks'],'Pair record');assert.equal(pair.body_index,body);assert.equal(pair.seed,seed);assert.equal(pair.trial_a,a.trial_id);assert.equal(pair.trial_b,b.trial_id);assert.equal(pair.valid,true);keys(pair.checks,['requests','efforts','non_light_observations','mechanical_states','replays'],'Pair checks');for(const v of Object.values(pair.checks))assert.equal(v,true);
  const da=verifyTrial(a,directory,report.bodies[body].assembly),db=verifyTrial(b,directory,report.bodies[body].assembly);assert.deepEqual(da.pair,db.pair,'Exact A/B requests, emitted efforts, non-light observations and mechanical projections');for(const field of ['request_tape_sha256','requested_effort_trace_sha256','effort_trace_sha256','non_light_observation_trace_sha256','mechanical_state_trace_sha256','initial_public_mechanical_snapshot_sha256'])assert.equal(a[field],b[field],'Matched pair native digest '+field);
  delete da.pair;delete db.pair;data.set(a.trial_id,da);data.set(b.trial_id,db);
 }
 const fitted=new Map();for(let body=0;body<3;body++)for(const mode of LEARNED){const bm=models.bodies[body],independent=fit(mode,bm.training_membership,data);fitted.set(`${body}:${mode}`,verifyModel(bm.models[LEARNED.indexOf(mode)],mode,independent));}
 const evaluated=verifyEvaluation(report,data,fitted);assert.equal(hashFile(modelPath),sealed.sha256,'Model artifact changed during independent validation');assert.equal(hashFile(absolute),reportHash,'Report changed during independent validation');const measurements=report.trials.map(t=>t.metrics);
 return {schema:'light_prediction_summary_v1',complete:true,sources_verified:true,invariants_verified:true,training_membership_verified:true,independent_refits_verified:true,independent_recursive_forecasts_verified:true,independent_metrics_verified:true,report_sha256:reportHash,source_count:Object.keys(manifest).length,protocol_sha256:PROTOCOL_SHA,native_trial_count:72,exact_replay_count:72,training_rows_per_body:3680,withheld_forecast_origins:552,...evaluated,
  native_measurements:{max_current_a:Math.max(...measurements.map(m=>m.max_current_a)),max_temperature_c:Math.max(...measurements.map(m=>m.max_temperature_c)),max_motor_speed_rad_s:Math.max(...measurements.map(m=>m.max_motor_speed_rad_s)),native_veto_steps:measurements.reduce((s,m)=>s+m.native_veto_steps,0),feedback_flag_steps:measurements.reduce((s,m)=>s+m.feedback_flag_steps,0),electrical_energy_j_range:[Math.min(...measurements.map(m=>m.electrical_energy_j)),Math.max(...measurements.map(m=>m.electrical_energy_j))]},
  replay_scope:'Native full requested/emitted command, transition and public mechanical replay; independent validation checks retained bytes and rederives local projection, training, ridge fit, matrices, recursive forecasts and metrics without rerunning hidden MuJoCo state.',
  interpretation_limit:'Per-body affine measured-delay identification under known stationary-source command tapes; no physical-state observability, source-switch prediction, zero-shot body transfer or closed-loop control claim.',report,models};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(path.resolve(process.argv[1])).href) {
 if(process.argv.length<3||process.argv.length>4)throw new Error('Usage: node tools/summarize-light-prediction.mjs REPORT_JSON [NEW_SUMMARY_JSON]');
 const {report,models,...summary}=summarizeLightPrediction(process.argv[2]);const output=JSON.stringify(summary,null,2)+'\n';
 if(process.argv[3]){assert.notEqual(path.resolve(process.argv[3]),path.resolve(process.argv[2]),'Summary cannot overwrite report');fs.writeFileSync(process.argv[3],output,{flag:'wx'});}else process.stdout.write(output);
}
