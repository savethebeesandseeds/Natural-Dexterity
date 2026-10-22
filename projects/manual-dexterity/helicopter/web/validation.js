import { reportEvidence, HINF_MODE } from '/control-evidence.js';
import { el, put, finite, format, readable } from '/format.js';
const badge=(text,style)=>{const node=document.createElement('span');node.className=`badge ${style}`;node.textContent=text;return node;};
const name=value=>String(value || 'Unnamed scenario').replace(/[_-]+/g,' ');
function appendCell(row,value) {const cell=document.createElement('td');if(value instanceof Node)cell.append(value);else cell.textContent=value;row.append(cell);}

function comparisons(items,verified) {
  const container=el('comparison-results');container.replaceChildren();container.hidden=!Array.isArray(items)||!items.length;
  if(container.hidden)return;
  const heading=document.createElement('h3');heading.textContent='PID and H∞ · measured mission performance';
  const explanation=document.createElement('p');explanation.className='muted';
  explanation.textContent=`${verified?'':'This comparison is not verified for the current build. '}These are measured nonlinear mission outcomes. They do not establish a global H∞ bound or an energy advantage. Any PID support and prepared-input intervals remain part of each mission.`;
  const scroll=document.createElement('div');scroll.className='table-scroll';
  const table=document.createElement('table');table.className='validation-table';
  const head=document.createElement('thead'),headrow=document.createElement('tr');
  for(const label of ['Scenario','Both flight tests pass','PID RMS error (m)','H∞ RMS error (m)','PID peak error (m)','H∞ peak error (m)','H∞ − PID RMS (m)']){const cell=document.createElement('th');cell.textContent=label;headrow.append(cell);}head.append(headrow);
  const body=document.createElement('tbody');
  for(const item of items) {
    const row=document.createElement('tr');
    for(const value of [name(item.scenario),item.both_passed===true?badge('Yes',verified?'passed':'unverified'):badge('No','failed'),format(item.pid_rms_error_m),format(item.hinf_rms_error_m),format(item.pid_max_error_m),format(item.hinf_max_error_m),finite(item.pid_rms_error_m)&&finite(item.hinf_rms_error_m)?format(item.hinf_rms_error_m-item.pid_rms_error_m):'Not reported'])appendCell(row,value);
    body.append(row);
  }
  table.append(head,body);scroll.append(table);container.append(heading,explanation,scroll);
}

export function renderEvaluation(data) {
  const report=data?.report;
  if(!report || !Array.isArray(report.scenarios))throw new Error('Saved evaluation contains no scenario records.');
  const evidence=reportEvidence(data),{methodMatches,verified,passed,complete,failedTrials,hinfPassed,pidFailures,designPassed}=evidence;
  const splitOutcome=verified&&complete&&designPassed&&hinfPassed&&pidFailures>0&&!passed;
  const result=el('validation-badge');result.className=`badge ${!passed?'failed':verified&&complete?'passed':'unverified'}`;
  result.textContent=!methodMatches?'Different method · not H∞ evidence':splitOutcome?'H∞ passes · PID failures':!passed?'Saved evaluation FAILED':!verified?'Saved · unverified':!complete?'Partial evaluation · pass':'Current build · full suite passed';
  put('validation-description',`${report.scenarios.length} saved trials · ${failedTrials?`${failedTrials} failed`:passed?'saved tests passed':'suite failed'}${complete?'':' · incomplete H∞ evidence'}`);
  put('validation-status',`${!methodMatches?'This report is not schema-4 minimum-entropy H∞ validation. Archived thermodynamic-control results cannot validate this method.':verified?'Report method and source match the current controller build.':`Report source is stale or unverified. ${data.verification_reason||''}`} ${splitOutcome?`The numerical design and all H∞ trials passed; ${pidFailures} PID tests failed. The overall suite FAILED.`:passed?'Saved tests passed.':'SAVED TESTS FAILED.'} ${complete?'The declared complete suite is present.':'A complete current-method evaluation is not established.'} Reset replay: ${readable(report.reset_replay_identical)}. The offline linear certificate, nonlinear flight tests and online execution time are separate evidence.`);
  const banner=el('validation-banner');banner.hidden=passed&&verified&&complete;banner.className=`fallback-banner${passed?' warning':''}`;
  banner.textContent=!methodMatches?'The saved results belong to a different controller method. Minimum-entropy H∞ validation has not been established.':splitOutcome?`Current-build evaluation: all H∞ trials and numerical design checks passed, but ${pidFailures} PID tests failed. The overall suite failed; failed comparison flights remain visible below.`:!passed?'The saved overall evaluation FAILED. Inspect the recorded failed checks.':!verified?'Saved H∞ validation does not match this build. Current performance is unverified.':'Only a partial H∞ evaluation is saved.';
  const rows=el('validation-rows');rows.replaceChildren();
  // Keep old reports accessible as raw evidence without relabeling old metrics.
  if(methodMatches)for(const trial of report.scenarios) {
    const row=document.createElement('tr');if(trial.passed!==true)row.className='failed';
    const m=trial.metrics||{},c=trial.controller_checks||{};
    const disabled=trial.mode==='disabled'||trial.autopilot===false;
    const values=[name(trial.name),disabled?'Disabled baseline':trial.controller_mode===HINF_MODE?'Minimum-entropy H∞':trial.controller_mode==='pid_baseline'?'PID baseline':'Unidentified',badge(trial.passed?'Test pass':'TEST FAIL',trial.passed?'passed':'failed'),trial.crashed?'CRASHED':trial.completed?'Completed':'Not completed',format(m.rms_error),format(m.max_error),format(m.endpoint_error),format(m.max_tilt_deg,1),finite(m.saturation_fraction)?format(100*m.saturation_fraction,1):'—',readable(m.physical_constraint_violation_steps),readable(c.passed),trial.controller_mode===HINF_MODE?format(c.applied_fraction!==undefined?100*c.applied_fraction:undefined,1):'Not used',format(c.update_max_ms),readable(c.local_region_exit_steps),format(trial.wall_seconds,2)];
    for(const value of values)appendCell(row,value);rows.append(row);
    const failures=Object.entries(c).filter(([key,value])=>key!=='passed'&&value===false).map(([key])=>key.replace(/_/g,' '));
    if(c.passed===false)failures.push('controller criteria');
    if(trial.autopilot!==false && trial.flight_criteria_passed===false)failures.push('flight criteria');
    if(trial.autopilot!==false && trial.physical_constraints_passed===false)failures.push('physical constraints');
    if(!trial.passed){const note=document.createElement('tr'),cell=document.createElement('td');note.className='failed';cell.colSpan=15;cell.textContent=trial.error||`Failed checks: ${failures.length?failures.join(', '):'see saved report'}.`;note.append(cell);rows.append(note);}
  }
  comparisons(methodMatches?report.comparisons:null,verified);
  const checks=report.design?.checks || (report.design?.passed!==undefined?{numerical_design_passed:report.design.passed}:{});
  put('saved-design-status',methodMatches?`Saved design checks: ${Object.keys(checks).length?Object.entries(checks).map(([key,value])=>`${key.replace(/_/g,' ')}: ${readable(value)}`).join(' · '):'See complete native design record below.'}`:'No current-method design validation in this report.');
  put('validation-thresholds',`Declared flight criteria: ${Object.entries(report.acceptance||report.thresholds||{}).map(([key,value])=>`${key.replace(/_/g,' ')} = ${readable(value)}`).join(' · ') || 'Not reported'}. Local-region exits and sampled timing are diagnostics; nonlinear flight tests are not an H∞ norm certificate.`);
  put('validation-json',JSON.stringify(data,null,2));
  if(!passed)el('validation-panel').open=true;
}

export function evaluationUnavailable(error) {
  el('validation-badge').className='badge unverified';put('validation-badge','Report unavailable');
  put('validation-description','No verified H∞ validation available');put('validation-status',`Saved validation unavailable: ${error.message}`);
  el('validation-rows').replaceChildren();comparisons(null,false);put('validation-thresholds','');put('validation-json',error.message);
  el('validation-banner').hidden=false;el('validation-banner').className='fallback-banner warning';
  put('validation-banner','Saved H∞ validation is unavailable. Live flight alone does not establish the design guarantee.');
}
