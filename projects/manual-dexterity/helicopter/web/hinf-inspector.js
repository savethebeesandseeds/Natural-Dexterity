import { HINF_MODE, landingSupport, observationHold } from '/control-evidence.js';
import { el, put, finite, format, readable, valueTable } from '/format.js';

const controllerFields=[['status','Native status'],['applied_controller','Applied controller'],['update_ms','Dynamic controller update · ms'],['control_interval_ms','Control interval · ms'],['saturated','Command saturation active'],['raw_command','Unclipped command'],['applied_command','Applied command'],['weighted_output','Reconstructed weighted output z'],['disturbance','Disturbance input w'],['control_decisions','All control intervals'],['applied_decisions','H∞ controller intervals'],['landing_support_decisions','PID landing intervals'],['observation_hold_decisions','Prepared trim intervals'],['fallback_decisions','PID fallback intervals'],['saturation_steps','Saturated steps']];
const thermalFields=[['entropy_rate_w_per_k','Physical entropy generation · W/K'],['cumulative_entropy_j_per_k','Physical entropy generated · J/K'],['terminal_entropy_commitment_j_per_k','Cooling + wake commitment · J/K'],['entropy_with_terminal_commitment_j_per_k','Generated + net commitment · J/K'],['electrical_power_w','Electrical power · W'],['electrical_energy_j','Electrical energy · J'],['motor_temperature_k','Motor temperatures · K'],['motor_current_a','Motor currents · A'],['air_dissipation_w','Air dissipation · W'],['motor_loss_w','Motor losses · W'],['heat_rejection_w','Heat rejected · W'],['stored_thermal_energy_j','Thermal storage · J'],['stored_wake_energy_j','Wake storage · J'],['wind_work_j','Wind work · J'],['contact_work_j','Ground-contact work · J'],['energy_balance_residual_w','Instantaneous free-flight energy residual · W'],['integrated_energy_residual_j','Integrated energy residual · J']];

function signalRows(container, labels, values, fallback) {
  valueTable(container,Array.isArray(values) ? values.map((value,index)=>[labels?.[index] || `${fallback} ${index+1}`,value]) : [['Availability','Not reported by this controller']]);
}

export function createHinfInspector() {
  let model=null;
  return {
    setModel(value) {model=value;},
    update(state) {
      const h=state.hinf || {}, selected=state.controller_mode===HINF_MODE, pid=state.controller_mode==='pid_baseline';
      const support=landingSupport(h),hold=observationHold(h),applied=h.applied_controller;
      const active=selected && h.available===true && applied===HINF_MODE;
      const scope=h.local_scope || {};
      const badge=el('controller-badge'),banner=el('fallback-banner');
      banner.hidden=true;banner.className='fallback-banner warning';
      put('controller-name',active ? 'Minimum-entropy H∞' : support ? 'PID landing support' : hold ? 'Prepared trim observation' : applied==='disabled' || state.autopilot===false ? 'Controller disabled' : pid || applied==='pid_baseline' ? 'PID baseline' : 'Controller not identified');
      badge.className='badge unverified';
      if(active) {
        badge.textContent=h.saturated===true ? 'Saturated · outside linear guarantee' : scope.within_declared_region===true ? 'Dynamic feedback · local region' : 'Dynamic feedback · scope unverified';
        if(h.saturated===true || scope.within_declared_region===false) {
          banner.hidden=false;
          banner.textContent=`The linear design guarantee does not cover this operating point. ${h.saturated===true ? 'The command is saturated. ' : ''}${Array.isArray(scope.reasons) ? scope.reasons.join(' ') : scope.reasons || ''}`;
        }
      } else if(support) {
        badge.textContent='PID support · H∞ inactive';banner.hidden=false;
        banner.textContent='PID controls this landing interval. The local airborne H∞ design does not certify ground contact.';
        const handover=state.landing_handover;
        if(handover?.active===true) banner.textContent+=` Measured handover: ${format(handover.time,2)} s; declared settling thresholds ${handover.settled_at_handover===true ? 'met' : handover.settled_at_handover===false ? 'NOT met' : 'not reported'}.`;
      } else if(hold) {
        badge.textContent='Trim hold · feedback inactive';banner.hidden=false;
        banner.textContent='The prepared trim input is held for the declared observation interval. Its time and physical energy remain in mission totals.';
      } else if(pid) badge.textContent='Comparison controller';
      else if(applied==='disabled' || state.autopilot===false) badge.textContent='Disabled baseline';
      else {badge.textContent='No H∞ telemetry';banner.hidden=false;banner.textContent='This service has not identified a minimum-entropy H∞ controller. Legacy thermodynamic optimization is not evidence for this method.';}
      if(selected && applied==='pid_baseline') {badge.textContent='PID fallback';banner.hidden=false;banner.textContent=`PID fallback is applied while H∞ mode is selected. ${h.status || ''}`;}

      put('update-time',active ? `${format(h.update_ms,3)} / ${format(h.control_interval_ms,1)}` : '—');
      put('scope-summary',active ? h.saturated===true ? 'Saturated' : scope.within_declared_region===true ? 'Inside declared region' : scope.within_declared_region===false ? 'Outside declared region' : 'Not established' : 'H∞ inactive');
      put('control-description',selected ? `${h.applied_decisions ?? '—'} / ${h.control_decisions ?? '—'} H∞ intervals · ${h.landing_support_decisions ?? '—'} PID landing · ${h.observation_hold_decisions ?? '—'} trim · ${h.fallback_decisions ?? '—'} fallback` : pid ? 'PID comparison controller' : 'Controller telemetry unavailable');
      put('control-status','The H∞ controller is synthesized offline. Online control updates the dynamic controller state and applies its output; no online nonlinear optimization is performed. The design certificate applies to the declared linear model and assumptions, not to the complete nonlinear mission.');
      put('local-scope-status',`${scope.within_declared_region===true ? 'State is inside the declared diagnostic region.' : scope.within_declared_region===false ? 'State is outside the declared diagnostic region.' : 'Local operating-region diagnostic is unavailable.'} This region check alone does not prove the nonlinear flight satisfies the linear H∞ bound. ${Array.isArray(scope.reasons) ? scope.reasons.join(' ') : scope.reasons || ''}`);
      valueTable(el('controller-values'),controllerFields.map(([key,label])=>[label,h[key]]));
      valueTable(el('thermal-values'),thermalFields.map(([key,label])=>[label,state.thermodynamics?.[key]]));
      signalRows(el('controller-state-values'),model?.hinf?.controller_state_labels,h.controller_state,'Controller state');
      signalRows(el('controller-scaled-values'),model?.hinf?.state_labels,h.controller_coordinate_scaled,'Scaled controller coordinate');
      signalRows(el('measurement-values'),model?.hinf?.measurement_labels,h.measurement,'Measurement');
      signalRows(el('reconstruction-values'),model?.hinf?.measurement_labels,h.measurement_reconstruction,'Reconstructed measurement');
      put('state-estimation-scope',[model?.hinf?.state_coordinate_definition,model?.hinf?.channel_coordinate_definition,h.measurement_reconstruction_kind,h.weighted_output_kind].filter(Boolean).join(' ') || 'The dynamic controller coordinates are not assumed to be unbiased plant-state estimates. Exact simulated pose is a separate diagnostic reference.');
      put('controller-json',JSON.stringify(h,null,2));
      const violations=state.metrics?.physical_constraint_violation_steps;
      el('plant-limit-banner').hidden=!(finite(violations) && violations>0);
      if(violations>0)put('plant-limit-banner',`The simulated plant exceeded its declared physical limits on ${violations} steps; maximum normalized violation ${format(state.metrics.max_physical_constraint_violation,5)}. This is separate from the linear design certificate.`);
      if(selected || pid)el('controller-mode').value=selected ? 'hinf' : 'pid';
    }
  };
}

