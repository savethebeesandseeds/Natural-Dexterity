const finite=value=>typeof value==='number' && Number.isFinite(value);
const number=(value,digits=3)=>finite(value) ? value.toFixed(digits) : 'Unavailable';
const text=value=>value===undefined || value===null ? 'Not reported' : typeof value==='object' ? JSON.stringify(value) : String(value);
const put=(id,value)=>{const node=document.getElementById(id);if(node)node.textContent=value;};

export function updateParameterEstimate(state,model) {
  const estimate=state?.parameter_estimation, config=model?.parameter_estimation;
  const available=finite(estimate?.mass_kg);
  const panel=document.querySelector('.estimation-inspector');
  if(panel) panel.hidden=!available && !config;
  put('estimated-mass',available ? `${number(estimate.mass_kg)} kg` : 'Unavailable');
  put('estimated-mass-scale',number(estimate?.mass_scale));
  put('plant-mass-reference',finite(state?.mass_kg) ? `${number(state.mass_kg)} kg` : 'Unavailable');
  put('mass-estimate-samples',number(estimate?.accepted_samples,0));
  put('mass-estimate-status',`${!available ? 'This native snapshot does not provide mass-estimation telemetry.' : estimate.accepted_samples===0 ? 'No accepted samples yet; the displayed estimate is the initialization value.' : 'Diagnostic estimate from accepted causal samples. Simulated plant mass is a separate reference.'} ${config?.role || ''}`);
  put('mass-estimate-method',text(estimate?.method ?? config?.method));
  put('mass-estimate-signals',text(config?.source));
  put('mass-inertia-assumption',text(estimate?.inertia_assumption ?? config?.inertia_assumption));
  put('mass-estimate-sharing',text(config?.role ?? config?.sharing));
}
