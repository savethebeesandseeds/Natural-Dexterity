export const HINF_MODE='minimum_entropy_hinf';

// Source identity alone cannot turn an archived method into H-infinity evidence.
export function reportEvidence(data) {
  const report=data?.report;
  const trials=Array.isArray(report?.scenarios) ? report.scenarios : [];
  const hasHinf=trials.some(trial=>trial.controller_mode===HINF_MODE && trial.autopilot!==false);
  const hasPid=trials.some(trial=>trial.controller_mode==='pid_baseline' && trial.autopilot!==false);
  const methodMatches=report?.schema_version>=4 && report?.method===HINF_MODE && hasHinf;
  const sourceVerified=data?.validation_saved===true && data?.live===false && data?.source_verified===true;
  const failedTrials=trials.filter(trial=>trial.passed!==true).length;
  const passed=report?.passed===true && trials.length>0 && failedTrials===0;
  const complete=report?.complete_evaluation===true && methodMatches && hasPid && Array.isArray(report?.comparisons) && report.comparisons.length>0;
  const hinfTrials=trials.filter(trial=>trial.controller_mode===HINF_MODE&&trial.autopilot!==false);
  const pidTrials=trials.filter(trial=>trial.controller_mode==='pid_baseline');
  const hinfPassed=report?.hinf_trials_passed===true && hinfTrials.length>0 && hinfTrials.every(trial=>trial.passed===true);
  const pidFailures=pidTrials.filter(trial=>trial.passed!==true).length;
  const designPassed=report?.design_passed===true && report?.design?.passed===true;
  return {hasHinf,hasPid,methodMatches,sourceVerified,failedTrials,passed,complete,hinfPassed,pidFailures,designPassed,verified:sourceVerified && methodMatches};
}
export const landingSupport=result=>result?.applied_controller==='pid_landing_support';
export const observationHold=result=>result?.applied_controller==='trim_observation_hold';
export function pidIntegralNotice(state) {
  if(state?.solution?.active!==true)return 'Unused while controller is inactive';
  const applied=state?.hinf?.applied_controller;
  if(observationHold(state?.hinf))return 'Unused during trim hold';
  if(applied===HINF_MODE)return 'Unused by dynamic H∞ controller';
  if(['pid_baseline','pid_landing_support'].includes(applied))return null;
  if(!applied && state?.controller_mode==='pid_baseline')return null;
  return 'Applied controller not identified';
}
