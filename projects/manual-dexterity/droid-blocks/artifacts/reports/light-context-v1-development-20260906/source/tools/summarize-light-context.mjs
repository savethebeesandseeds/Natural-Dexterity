import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const PROTOCOL = 'docs/LIGHT_CONTEXT_EXPERIMENT.md';
const PROTOCOL_SHA = '3e1218d1e0000cca5020962aeb73ba773f92bdcc1012a178ba4d238389d87dff';
const REFERENCE = 'artifacts/reports/light-learning-v2-development-20260905/report.json';
const REFERENCE_SHA = '931a26b17b399456bb5461293e382ae9880486f1b38157007500eead65998230';
const MODES = ['instantaneous','history','light_only','no_light_history'];
const VIEW_IDS = {instantaneous:'current',history:'history',light_only:'brightness_only',no_light_history:'no_light'};
const DIMS = {instantaneous:12,history:132,light_only:1,no_light_history:121};
const TRAIN = Array.from({length:8},(_,i)=>101+i);
const TEST = [109,110,111,112];
const ENDPOINTS = Array.from({length:22},(_,i)=>10+5*i);
const readJson = p=>JSON.parse(fs.readFileSync(p,'utf8').replace(/^\uFEFF/,''));
const hashFile = p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const median = a=>{const b=[...a].sort((x,y)=>x-y),i=Math.floor(b.length/2);return b.length%2?b[i]:(b[i-1]+b[i])/2;};
function safeJoin(root, relative) {
 assert.equal(typeof relative,'string');
 assert.ok(relative.length && !path.isAbsolute(relative) && !relative.includes(':') && !relative.split(/[\\/]/).includes('..'),'Unsafe evidence path');
 const result=path.resolve(root,relative), rel=path.relative(path.resolve(root),result);
 assert.ok(rel && rel!=='..' && !rel.startsWith('..'+path.sep) && !path.isAbsolute(rel));
 assert.ok(!fs.lstatSync(result).isSymbolicLink(),'Evidence may not be a symbolic link');
 return result;
}
function finite(x,label) {assert.equal(typeof x,'number',label);assert.ok(Number.isFinite(x),label);return x;}
function near(a,b,label,tolerance=1e-10) {finite(a,label);finite(b,label);assert.ok(Math.abs(a-b)<=tolerance*Math.max(1,Math.abs(b)),`${label}: ${a} != ${b}`);}
function sha(x,label) {assert.match(x,/^[a-f0-9]{64}$/,label);}
function exactKeys(o,expected,label) {assert.ok(o && typeof o==='object' && !Array.isArray(o),label);assert.deepEqual(Object.keys(o).sort(),[...expected].sort(),label);}
function assertTrueChecks(checks,label) {assert.ok(checks && Object.keys(checks).length>0,label);for(const [k,v] of Object.entries(checks))assert.equal(v,true,label+':'+k);}
function project(observation, effort, time) {
 exactKeys(observation,['schema_version','reward_sensors','actuator_feedback'],'Local observation boundary');
 assert.equal(observation.schema_version,'construction_observation_v2');
 assert.equal(observation.reward_sensors.length,2);assert.equal(observation.actuator_feedback.length,1);
 const imu=observation.reward_sensors.find(s=>s.family_id==='imu_6axis_v0');
 const light=observation.reward_sensors.find(s=>s.family_id==='ambient_light_v0');
 assert.ok(imu && light);assert.equal(imu.valid,true);assert.equal(light.valid,true);
 for(const [s,period,latency] of [[imu,.01,.006],[light,.1,.02]]) {
  exactKeys(s,['module_id','family_id','valid','sequence','sample_time_s','delivered_time_s','age_s','sample_period_s','latency_s','observations'],'Sensor boundary');
  assert.ok(Number.isInteger(s.sequence)&&s.sequence>0);near(s.sample_period_s,period,'sample period');near(s.latency_s,latency,'latency');
  near(s.sample_time_s+s.age_s,time,'acquisition plus age',2e-6);
  near(s.delivered_time_s-s.sample_time_s,latency,'sample delivery',2e-6);
  assert.ok(s.delivered_time_s<=time+2e-6 && s.age_s>=latency-2e-6);
 }
 near(light.age_s,.02,'Fresh delivered light',2e-6);
 exactKeys(light.observations,['illuminance_lux','saturated'],'Light channels');
 exactKeys(imu.observations,['specific_force_m_s2','angular_velocity_rad_s'],'IMU channels');
 const motor=observation.actuator_feedback[0];assert.equal(motor.valid,true);assert.equal(motor.sku_id,'rotary_dc_gearmotor_v0');
 const m=motor.feedback;assert.deepEqual(m.fault_flags,[]);
 const force=imu.observations.specific_force_m_s2,gyro=imu.observations.angular_velocity_rad_s;
 assert.equal(force.length,3);assert.equal(gyro.length,3);
 const lux=finite(light.observations.illuminance_lux,'Lux');assert.ok(lux>=0 && lux<=1000);
 const result=[effort/.15,Math.sin(m.position_rad),Math.cos(m.position_rad),m.velocity_rad_s/10,m.current_a/.5,...force.map(x=>x/50),...gyro.map(x=>x/10),lux/1000];
 result.forEach(x=>finite(x,'Projected feature'));return result;
}
function features(frames,index,mode) {
 assert.ok(index>=10 && index<frames.length);
 if(mode==='instantaneous')return [...frames[index].features];
 if(mode==='light_only')return [frames[index].features[11]];
 const out=[];
 for(let lag=0;lag<11;lag++)out.push(...frames[index-lag].features.slice(0,mode==='history'?12:11));
 assert.equal(out.length,DIMS[mode]);return out;
}
function predict(training,query) {
 const distances=training.map(e=>{let sum=0;assert.equal(e.features.length,query.length);for(let i=0;i<query.length;i++){const d=query[i]-e.features[i];sum+=d*d;}return sum/query.length;});
 const cutoff=[...distances].sort((a,b)=>a-b)[4];
 let n=0,a=0;distances.forEach((d,i)=>{if(d<=cutoff){n++;if(training[i].is_a)a++;}});
 const p=(a+1)/(n+2);
 return {probability_a:p,predicts_a:p>=.5,neighbor_count:n,squared_mean_distance_cutoff:cutoff};
}
function metrics(predictions) {
 let a=0,b=0,ta=0,fb=0,fa=0,tb=0,brier=0,ambiguous=0;
 for(const p of predictions) {
  const truth=p.context==='A';if(truth){a++;if(p.predicts_a)ta++;else fb++;}else {b++;if(p.predicts_a)fa++;else tb++;}
  brier+=(p.probability_a-Number(truth))**2;
  if(p.probability_a>=.4 && p.probability_a<=.6)ambiguous++;
 }
 assert.ok(a>0 && b>0);
 return {example_count:predictions.length,a_count:a,b_count:b,true_a:ta,false_b:fb,false_a:fa,true_b:tb,balanced_accuracy:.5*(ta/a+tb/b),brier_score:brier/predictions.length,ambiguous_count:ambiguous,ambiguous_fraction:ambiguous/predictions.length};
}
function verifyMetrics(record,expected,label) {exactKeys(record,Object.keys(expected),label);for(const [k,v] of Object.entries(expected))if(Number.isInteger(v))assert.equal(record[k],v,label+'.'+k);else near(record[k],v,label+'.'+k);}
function nonLight(observation) {const result=structuredClone(observation);for(const s of result.reward_sensors)if(s.family_id==='ambient_light_v0')delete s.observations;return result;}

// Report/schema verification is completed against the native writer before collection.
function mechanical(state) {
 const result=structuredClone(state);assert.equal(result.schema,'construction_state_v2');
 delete result.sun;delete result.reward;delete result.diagnostics.instantaneous_light_lux;delete result.light_sensor.observations;return result;
}
function verifyTrial(trial,directory) {
 const id=`body-${trial.body_index}-probe-${trial.probe_seed}-${trial.context}`;
 assert.equal(trial.trial_id,id);assert.ok([0,1].includes(trial.body_index));assert.ok([...TRAIN,...TEST].includes(trial.probe_seed));assert.ok(['A','B'].includes(trial.context));
 assert.equal(trial.split,TRAIN.includes(trial.probe_seed)?'train':'test');assert.equal(trial.mode,'random');
 assert.equal(trial.completed,true,id);assert.equal(trial.completed_steps,600);assert.equal(trial.accepted_steps,600);
 assert.equal(trial.recorded_transition_count,600);assert.equal(trial.recorded_effort_count,600);assert.equal(trial.recorded_requested_effort_count,600);assert.equal(trial.error,'');
 assert.deepEqual(trial.snapshot_errors,[]);assert.equal(trial.final_controller_snapshot_available,true);assert.equal(trial.final_public_physical_snapshot_available,true);
 assert.equal(trial.last_attempted_action.step,600);assert.equal(trial.last_attempted_action.fully_accepted,true);
 assert.equal(trial.exact_controller_and_observation_replay,true);assert.equal(trial.exact_full_transition_replay,true);
 assert.equal(trial.replay.completed_steps,600);assert.equal(trial.replay.error,'');
 for(const field of ['requested_effort_trace_sha256','effort_trace_sha256','observation_trace_sha256','transition_trace_sha256','final_public_physical_snapshot_sha256','final_controller_snapshot_sha256']) {
  sha(trial[field],id+'.'+field);assert.equal(trial.replay[field],trial[field],id+' replay '+field);
 }
 for(const field of ['non_light_observation_trace_sha256','mechanical_state_trace_sha256','initial_public_mechanical_snapshot_sha256'])sha(trial[field],id+'.'+field);
 assert.equal(trial.trace.path,`traces/${id}.json`);const tracePath=safeJoin(directory,trial.trace.path);assert.equal(hashFile(tracePath),trial.trace.sha256,id+' trace bytes');
 const trace=readJson(tracePath);assert.equal(trace.schema,'light_context_trace_v1');assert.equal(trace.trial_id,id);
 assert.equal(trace.efforts.length,600);assert.equal(trace.requested_efforts.length,600);assert.equal(trial.last_attempted_action.requested_effort,trace.requested_efforts.at(-1));assert.equal(trace.transitions.length,600);assert.equal(trace.mechanical_state_sha256.length,600);
 assert.deepEqual(trace.controller_final,trial.controller_final);assert.equal(trial.controller_final.mode,'random');
 assert.equal(trial.last_attempted_action.effort,trace.efforts.at(-1));
 const expectedSun={position_m:[trial.context==='A'?.30:-.30,0,.20],intensity_lux:1000};
 assert.deepEqual(trace.initial_public_physical_snapshot.sun,expectedSun);assert.deepEqual(trace.final_public_physical_snapshot.sun,expectedSun);
 let reward=0,maxCurrent=0,maxTemperature=0,maxSpeed=0,vetoes=0,flags=0,previous=0;
 for(let i=0;i<600;i++) {
  const effort=finite(trace.efforts[i],id+' effort');assert.ok(Math.abs(effort)<=.15+1e-12 && Math.abs(effort-previous)<=.1+1e-12,id+' effort bounds');if(i<5)assert.equal(effort,0);previous=effort;
    const request=finite(trace.requested_efforts[i],'Recorded request');assert.ok([-.15,0,.15].includes(request),'Request is outside ternary probe set');
  if(i<5)assert.equal(request,0);else if((i-5)%20!==0)assert.equal(request,trace.requested_efforts[i-1],'Request hold changed');
  const previousObservation=i===0?trace.initial_observation:trace.transitions[i-1].observation;
  const priorEffort=i===0?0:trace.efforts[i-1];let target=request;
  if(previousObservation.reward_sensors.some(s=>s.valid!==true))target=0;
  else if(target!==0){const sign=target>0?1:-1,velocity=previousObservation.actuator_feedback[0].feedback.velocity_rad_s;target=sign*Math.min(Math.abs(target),.05*Math.max(0,3-sign*velocity));}
  const governed=Math.max(priorEffort-.1,Math.min(priorEffort+.1,target));near(effort,governed,'Independent common governor',1e-12);
  const t=trace.transitions[i];assert.equal(t.terminated,false);assert.equal(t.truncated,false);assert.equal(t.safety.veto,false);assert.deepEqual(t.safety.flags,[]);
  assert.equal(t.info.physics_steps_executed,10);near(t.info.elapsed_control_dt_s,.02,'Control dt');
  assert.equal(t.reward_components.length,2);near(t.reward,t.reward_components.reduce((s,c)=>s+finite(c.transition_reward,'Component reward'),0),'SUM sensor reward');reward+=finite(t.reward,'Reward');
  const feedback=t.observation.actuator_feedback[0].feedback;maxCurrent=Math.max(maxCurrent,Math.abs(feedback.current_a));maxTemperature=Math.max(maxTemperature,feedback.temperature_c);maxSpeed=Math.max(maxSpeed,Math.abs(feedback.velocity_rad_s));
  if(t.safety.veto)vetoes++;if(feedback.fault_flags.length)flags++;
  sha(trace.mechanical_state_sha256[i],'Mechanical state hash');
 }
 const expectedMetrics={steps:600,duration_s:12,integrated_sensor_reward:reward,electrical_energy_j:trace.final_public_physical_snapshot.diagnostics.electrical_energy_j,max_current_a:maxCurrent,max_temperature_c:maxTemperature,max_motor_speed_rad_s:maxSpeed,native_veto_steps:vetoes,feedback_flag_steps:flags};
 verifyMetrics(trial.metrics,expectedMetrics,id+' probe metrics');assert.ok(trial.metrics.electrical_energy_j>=0);
 assert.equal(trial.frames.length,120);let previousSequence=0;
 for(let i=0;i<120;i++) {
  const f=trial.frames[i],step=1+5*i;exactKeys(f,['frame_index','step','time_s','executed_effort','observation','features'],'Frame boundary');
  assert.equal(f.frame_index,i);assert.equal(f.step,step);near(f.time_s,step*.02,'Frame time');assert.equal(f.executed_effort,trace.efforts[step-1]);
  assert.deepEqual(f.observation,trace.transitions[step-1].observation,'Frame must be actual causal observation');
  const projected=project(f.observation,f.executed_effort,f.time_s);assert.equal(f.features.length,12);
  projected.forEach((x,j)=>near(f.features[j],x,id+' feature '+j,2e-12));
  const light=f.observation.reward_sensors.find(s=>s.family_id==='ambient_light_v0');assert.equal(light.sequence,previousSequence+1);previousSequence=light.sequence;
 }
 return trace;
}
function verifyPairs(report,trials,traces) {
 assert.equal(report.pairs.length,24);const seen=new Set();
 for(const pair of report.pairs) {
  const key=`${pair.body_index}:${pair.probe_seed}`;assert.ok(!seen.has(key));seen.add(key);
  assert.equal(pair.valid,true);exactKeys(pair.checks,['identical_requests','identical_commands','identical_non_light_observations','identical_mechanical_states','both_full_replays'],'Pair checks');assertTrueChecks(pair.checks,key);
  assert.equal(pair.trial_a,`body-${pair.body_index}-probe-${pair.probe_seed}-A`);assert.equal(pair.trial_b,`body-${pair.body_index}-probe-${pair.probe_seed}-B`);
  const a=trials.get(pair.trial_a),b=trials.get(pair.trial_b),ta=traces.get(pair.trial_a),tb=traces.get(pair.trial_b);assert.ok(a&&b&&ta&&tb);
  assert.deepEqual(ta.requested_efforts,tb.requested_efforts,'Matched A/B request bits');assert.deepEqual(ta.efforts,tb.efforts,'Matched A/B effort bits');assert.deepEqual(ta.mechanical_state_sha256,tb.mechanical_state_sha256,'Per-step mechanical fingerprints');
  assert.deepEqual(nonLight(ta.initial_observation),nonLight(tb.initial_observation));
  assert.deepEqual(mechanical(ta.initial_public_physical_snapshot),mechanical(tb.initial_public_physical_snapshot));
  assert.deepEqual(mechanical(ta.final_public_physical_snapshot),mechanical(tb.final_public_physical_snapshot));
  for(let i=0;i<600;i++)assert.deepEqual(nonLight(ta.transitions[i].observation),nonLight(tb.transitions[i].observation),'Matched full non-light observations');
  for(let i=0;i<120;i++)assert.deepEqual(a.frames[i].features.slice(0,11),b.frames[i].features.slice(0,11),'Paired non-light features');
  for(const field of ['requested_effort_trace_sha256','effort_trace_sha256','non_light_observation_trace_sha256','mechanical_state_trace_sha256','initial_public_mechanical_snapshot_sha256'])assert.equal(a[field],b[field],key+' '+field);
 }
 assert.equal(seen.size,24);
}
function verifyModels(record,models,trials) {
 assert.equal(models.schema,'light_context_models_v1');assert.equal(models.saved_before_evaluation,true);assert.equal(models.bodies.length,2);
 const lookup=new Map();
 for(let body=0;body<2;body++) {
  const bm=models.bodies[body];assert.equal(bm.body_index,body);exactKeys(bm.models,MODES,'Model modes');
  const expectedMetadata=[];
  for(const trial of record.trials)if(trial.body_index===body&&trial.split==='train')for(const index of ENDPOINTS)expectedMetadata.push({trial_id:trial.trial_id,probe_seed:trial.probe_seed,context:trial.context,frame_index:index,step:trial.frames[index].step,time_s:trial.frames[index].time_s});
  assert.equal(expectedMetadata.length,352);assert.deepEqual(bm.training_example_metadata,expectedMetadata,'Training membership and causal endpoints');
  for(const mode of MODES) {
   const model=bm.models[mode];exactKeys(model,['schema','mode','dimension','k','distance','ties','probability_a','predict_a','training'],'Numeric model boundary');
   assert.equal(model.schema,'light-context-decoder-v1');assert.equal(model.mode,mode);assert.equal(model.dimension,DIMS[mode]);assert.equal(model.k,5);assert.equal(model.training.length,352);
   for(let i=0;i<352;i++) {
    const meta=expectedMetadata[i],trial=trials.get(meta.trial_id);assert.equal(trial.split,'train');assert.ok(TRAIN.includes(trial.probe_seed),'Withheld probe entered training');
    const example=model.training[i];exactKeys(example,['features','is_a'],'Training example has metadata or missing fields');
    assert.equal(example.is_a,trial.context==='A');assert.deepEqual(example.features,features(trial.frames,meta.frame_index,mode),'Model must copy only declared training features');
    example.features.forEach(x=>finite(x,'Model coordinate'));
   }
   lookup.set(`${body}:${mode}`,model.training);
  }
 }
 return lookup;
}
function verifyEvaluation(report,training,trials,reportPath,reportHash) {
 const evaluation=report.evaluation;assert.equal(evaluation.completed,true);assert.equal(evaluation.bodies.length,2);assert.equal(evaluation.negative_control_passed,true);
 const aggregates=Object.fromEntries(MODES.map(m=>[m,[]]));const bodySummaries=[],viewBodies=[];
 for(let body=0;body<2;body++) {
  const result=evaluation.bodies[body];assert.equal(result.body_index,body);exactKeys(result.decoders,MODES,'Evaluation modes');
  const nativeBody=report.bodies[body];const view={body_index:body,body_name:nativeBody.assembly.name,assembly_sha256:nativeBody.assembly_sha256,models:[],probes:TEST.map(seed=>({sequence_id:seed,curves:{}}))};
  const decoderSummaries={};
  for(const mode of MODES) {
   const decoder=result.decoders[mode];assert.equal(decoder.completed,true);assert.equal(decoder.predictions.length,176);assert.equal(decoder.by_probe.length,4);
   const seen=new Set();
   for(const p of decoder.predictions) {
    exactKeys(p,['trial_id','probe_seed','context','frame_index','step','time_s','probability_a','predicts_a','neighbor_count','squared_mean_distance_cutoff'],'Prediction metadata/features boundary');
    const trial=trials.get(p.trial_id);assert.ok(trial);assert.equal(trial.body_index,body);assert.equal(trial.split,'test');assert.ok(TEST.includes(trial.probe_seed));
    assert.equal(p.probe_seed,trial.probe_seed);assert.equal(p.context,trial.context);assert.ok(ENDPOINTS.includes(p.frame_index));
    assert.equal(p.step,trial.frames[p.frame_index].step);assert.equal(p.time_s,trial.frames[p.frame_index].time_s);
    const key=`${p.trial_id}:${p.frame_index}`;assert.ok(!seen.has(key),'Duplicate heldout prediction');seen.add(key);
    const expected=predict(training.get(`${body}:${mode}`),features(trial.frames,p.frame_index,mode));
    assert.equal(p.predicts_a,expected.predicts_a,'Independent predicted class');assert.equal(p.neighbor_count,expected.neighbor_count,'Independent tie-inclusive neighbor count');
    near(p.probability_a,expected.probability_a,'Independent neighbor vote',1e-12);near(p.squared_mean_distance_cutoff,expected.squared_mean_distance_cutoff,'Independent cutoff',1e-12);
   }
   assert.equal(seen.size,176);const overall=metrics(decoder.predictions);verifyMetrics(decoder.overall,overall,'Overall heldout metrics');aggregates[mode].push(...decoder.predictions);
   const groupAccuracies=[];
   for(let i=0;i<4;i++) {
    const seed=TEST[i],group=decoder.by_probe[i];assert.equal(group.probe_seed,seed);
    const subset=decoder.predictions.filter(p=>p.probe_seed===seed);assert.equal(subset.length,44);
    const m=metrics(subset);verifyMetrics(group.metrics,m,'Whole withheld probe metrics');groupAccuracies.push(m.balanced_accuracy);
    const curves={};
    for(const context of ['A','B']) {
     const points=subset.filter(p=>p.context===context).sort((a,b)=>a.frame_index-b.frame_index);assert.equal(points.length,22);
     assert.deepEqual(points.map(p=>p.frame_index),ENDPOINTS);curves[context]=points.map(p=>[p.time_s,p.probability_a]);
    }
    if(mode==='no_light_history') {assert.deepEqual(curves.A,curves.B,'Negative-control paired votes differ');assert.equal(m.balanced_accuracy,.5,'Negative control leaked condition');}
    view.probes[i].curves[VIEW_IDS[mode]]=curves;
   }
   const mean=groupAccuracies.reduce((s,v)=>s+v,0)/4,min=Math.min(...groupAccuracies),max=Math.max(...groupAccuracies);
   near(overall.balanced_accuracy,mean,'Equal group weighting');near(decoder.probe_balanced_accuracy_mean,mean,'Group mean');assert.equal(decoder.probe_balanced_accuracy_min,min);assert.equal(decoder.probe_balanced_accuracy_max,max);
   const ready=overall.balanced_accuracy>=.75 && min>=.65;assert.equal(decoder.readiness_passed,ready,'Predeclared decoder threshold');
   view.models.push({id:VIEW_IDS[mode],accuracy:mean,sequence_accuracies:TEST.map((seed,i)=>({sequence_id:seed,accuracy:groupAccuracies[i]}))});
   decoderSummaries[mode]={overall,by_probe:decoder.by_probe,probe_accuracy_min:min,probe_accuracy_max:max,readiness_passed:ready};
  }
  const delta=decoderSummaries.history.overall.balanced_accuracy-decoderSummaries.instantaneous.overall.balanced_accuracy;
  near(result.history_minus_instantaneous_balanced_accuracy,delta,'Body paired difference');assert.equal(result.readiness_passed,decoderSummaries.history.readiness_passed);assert.equal(result.paired_probe_differences.length,4);
  for(let i=0;i<4;i++){const p=result.paired_probe_differences[i];assert.equal(p.probe_seed,TEST[i]);near(p.history_minus_instantaneous_balanced_accuracy,decoderSummaries.history.by_probe[i].metrics.balanced_accuracy-decoderSummaries.instantaneous.by_probe[i].metrics.balanced_accuracy,'Paired probe delta');}
  bodySummaries.push({body_index:body,body_name:nativeBody.assembly.name,decoders:decoderSummaries,history_minus_instantaneous_balanced_accuracy:delta,paired_probe_differences:result.paired_probe_differences,readiness_passed:result.readiness_passed});viewBodies.push(view);
 }
 exactKeys(evaluation.overall,MODES,'Pooled supplemental modes');for(const mode of MODES)verifyMetrics(evaluation.overall[mode],metrics(aggregates[mode]),'Supplemental pooled metrics');
 const delta=evaluation.overall.history.balanced_accuracy-evaluation.overall.instantaneous.balanced_accuracy;near(evaluation.history_minus_instantaneous_balanced_accuracy,delta,'Pooled delta');
 assert.equal(evaluation.readiness_passed,bodySummaries.every(b=>b.readiness_passed),'Readiness requires both bodies');
 const viewer={schema:'light_context_view_v1',source:{report_path:reportPath,report_sha256:reportHash},protocol:{duration_s:12,history_s:1,training_sequence_ids:TRAIN,test_sequence_ids:TEST},validation:{complete:true,independently_checked:true,native_trial_count:48,exact_replay_count:48},bodies:viewBodies};
 return {by_body:bodySummaries,overall:evaluation.overall,readiness_passed:evaluation.readiness_passed,viewer};
}
export function summarizeLightContext(reportPath) {
 const absolute=path.resolve(reportPath),directory=path.dirname(absolute),report=readJson(absolute),reportHash=hashFile(absolute);
 assert.equal(report.schema,'light_context_v1');assert.equal(report.complete,true,'Context evidence is incomplete');assert.equal(report.finished,true);assert.equal(report.stage,'evaluated');
 assert.equal(report.failed_trials,0);assert.equal(report.failed_pairs,0);assert.ok(!report.fatal_error);
 exactKeys(report.checks,['all_trials_complete','all_pairs_valid','all_full_replays','models_immutable_during_evaluation','negative_control_passed'],'Report checks');assertTrueChecks(report.checks,'Report checks');
 const p=report.protocol;
 assert.equal(p.document,PROTOCOL);assert.equal(p.document_sha256,PROTOCOL_SHA);assert.equal(p.control_dt_s,.02);assert.equal(p.steps,600);assert.equal(p.duration_s,12);
 assert.equal(p.body_count,2);assert.equal(p.trial_count,48);assert.equal(p.replay_count,48);assert.deepEqual(p.training_probe_seeds,TRAIN);assert.deepEqual(p.heldout_probe_seeds,TEST);
 assert.deepEqual(p.source_A,{position_m:[.30,0,.20],intensity_lux:1000});assert.deepEqual(p.source_B,{position_m:[-.30,0,.20],intensity_lux:1000});
 assert.equal(p.frame_count,120);assert.equal(p.frame_steps,'1+5*k, k=0..119');assert.equal(p.endpoint_frames,'10+5*j, j=0..21');assert.equal(p.history_frame_count,11);assert.equal(p.examples_per_trial,22);
 assert.equal(p.training_examples_per_body,352);assert.equal(p.heldout_examples_per_body,176);assert.equal(p.readiness_mean_threshold,.75);assert.equal(p.readiness_every_probe_threshold,.65);assert.equal(p.readiness_requires_both_bodies,true);
 const manifest=report.source_sha256;assert.ok(manifest&&Object.keys(manifest).length>0);
 for(const [relative,expected] of Object.entries(manifest)){sha(expected,'Source digest');assert.equal(hashFile(safeJoin(directory,'source/'+relative)),expected,'Archived source differs: '+relative);}
 for(const required of [PROTOCOL,REFERENCE,'src/light_context.cpp','src/light_context_cli.cpp','include/droid/light_context.hpp','tests/native_light_context_tests.cpp','CMakeLists.txt','run.sh','Dockerfile','setup.sh','config/module_catalog.json','models/droid.xml','tools/summarize-light-context.mjs','config/learning_experiment_v3.json','artifacts/light-search-policy-v3-seed0.json'])assert.ok(manifest[required],'Missing source provenance: '+required);
 assert.equal(manifest[PROTOCOL],PROTOCOL_SHA);assert.equal(manifest[REFERENCE],REFERENCE_SHA);
 assert.equal(hashFile(fileURLToPath(import.meta.url)),manifest['tools/summarize-light-context.mjs'],'Verifier differs from archived declared source');
 assert.equal(report.reference_report.path,REFERENCE);assert.equal(report.reference_report.sha256,REFERENCE_SHA);assert.equal(report.reference_report.snapshot,'source/'+REFERENCE);
 const reference=readJson(safeJoin(directory,report.reference_report.snapshot));assert.equal(reference.complete,true);assert.equal(reference.failed_trials,0);assert.equal(reference.constructions.length,2);
 for(const [relative,expected] of Object.entries(reference.source_sha256))assert.equal(manifest[path.posix.dirname(REFERENCE)+'/source/'+relative],expected,'Missing original-v2 archived source');
 for(const relative of ['config/learning_experiment_v3.json','artifacts/light-search-policy-v3-seed0.json'])assert.equal(manifest[relative],reference.source_sha256[relative],'Frozen original input changed');
 assert.deepEqual(report.physics_spec,reference.physics_spec,'Physics specification changed');assert.deepEqual(report.module_catalog,readJson(safeJoin(directory,'source/config/module_catalog.json')));
 assert.equal(report.decoder_spec.frame_dimension,12);assert.deepEqual(report.decoder_spec.dimensions,DIMS);assert.equal(report.decoder_spec.history.frames,11);assert.equal(report.decoder_spec.history.spacing_s,.1);assert.equal(report.decoder_spec.history.order,'latest to oldest');assert.equal(report.decoder_spec.decoder.k,5);assert.equal(report.decoder_spec.decoder.online_updates,false);
 assert.equal(report.bodies.length,2);for(let i=0;i<2;i++){assert.equal(report.bodies[i].body_index,i);assert.deepEqual(report.bodies[i].assembly,reference.constructions[i].assembly);assert.equal(report.bodies[i].assembly_sha256,reference.constructions[i].assembly_sha256);}
 assert.equal(report.trials.length,48);const trials=new Map(),traces=new Map();
 for(const trial of report.trials){assert.ok(!trials.has(trial.trial_id),'Repeated trial');trials.set(trial.trial_id,trial);traces.set(trial.trial_id,verifyTrial(trial,directory));}
 for(const body of [0,1])for(const seed of [...TRAIN,...TEST])for(const condition of ['A','B'])assert.ok(trials.has(`body-${body}-probe-${seed}-${condition}`),'Declared trial missing');
 verifyPairs(report,trials,traces);
 const modelRecord=report.training_models;assert.equal(modelRecord.path,'training-models.json');assert.equal(modelRecord.saved_before_evaluation,true);assert.equal(modelRecord.unchanged_after_evaluation,true);
 assert.equal(modelRecord.training_examples_per_body,352);assert.equal(modelRecord.heldout_examples_per_body,176);assert.equal(modelRecord.sha256_after_evaluation,modelRecord.sha256);
 const modelPath=safeJoin(directory,modelRecord.path);assert.equal(hashFile(modelPath),modelRecord.sha256,'Sealed model artifact changed');
 const training=verifyModels(report,readJson(modelPath),trials);
 const evaluated=verifyEvaluation(report,training,trials,path.relative(path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..'),absolute).split(path.sep).join('/'),reportHash);
 const values=report.trials.map(t=>t.metrics);
 return {schema:'light_context_summary_v1',complete:true,report_sha256:reportHash,sources_verified:true,source_file_count:Object.keys(manifest).length,invariants_verified:true,training_membership_verified:true,independent_predictions_verified:true,native_trial_count:48,exact_replay_count:48,negative_control_passed:true,...evaluated,
  native_measurements:{max_current_a:Math.max(...values.map(v=>v.max_current_a)),max_temperature_c:Math.max(...values.map(v=>v.max_temperature_c)),max_motor_speed_rad_s:Math.max(...values.map(v=>v.max_motor_speed_rad_s)),native_veto_steps:values.reduce((s,v)=>s+v.native_veto_steps,0),feedback_flag_steps:values.reduce((s,v)=>s+v.feedback_flag_steps,0),electrical_energy_j_range:[Math.min(...values.map(v=>v.electrical_energy_j)),Math.max(...values.map(v=>v.electrical_energy_j))]},
  interpretation_limit:'Supervised decodability under stationary A/B and whole withheld motor probes on known bodies; no automatic context discovery, retrieval, switching-control, new-body or general-observability claim',
  replay_scope:'Native full command/transition/mechanical replay; independent verifier validates retained trace bytes, cross-condition local observations, causal features, training membership and all classifier outputs, not hidden MuJoCo internals'};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(path.resolve(process.argv[1])).href){if(process.argv.length!==3)throw new Error('Usage: node tools/summarize-light-context.mjs REPORT_JSON');const summary=summarizeLightContext(process.argv[2]);const {viewer,...printed}=summary;console.log(JSON.stringify(printed,null,2));}
