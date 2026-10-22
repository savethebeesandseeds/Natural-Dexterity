import { el, put, finite, format, readable, valueTable } from '/format.js';
const NS='http://www.w3.org/2000/svg';
function svgNode(tag,attributes={},text) {const node=document.createElementNS(NS,tag);for(const [key,value]of Object.entries(attributes))node.setAttribute(key,value);if(text!==undefined)node.textContent=text;return node;}

function designPlot(container,{title,xLabel,points,series,logX=false}) {
  const card=document.createElement('article');card.className='chart-card';
  const heading=document.createElement('h3');heading.textContent=title;card.append(heading);
  const svg=svgNode('svg',{class:'chart-svg',viewBox:'0 0 450 190',role:'img','aria-label':`${title}; ${xLabel}`});card.append(svg);
  const valid=(Array.isArray(points)?points:[]).filter(point=>finite(point.x)&&(!logX||point.x>0));
  const values=valid.flatMap(point=>series.map(item=>point[item.key])).filter(finite);
  if(!valid.length || !values.length) {svg.append(svgNode('text',{x:225,y:90,'text-anchor':'middle',class:'chart-empty'},'Design data not reported'));container.append(card);return;}
  const transform=x=>logX?Math.log10(x):x;
  let xmin=Math.min(...valid.map(p=>transform(p.x))),xmax=Math.max(...valid.map(p=>transform(p.x)));
  if(xmin===xmax){xmin-=0.5;xmax+=0.5;}
  let ymin=Math.min(0,...values),ymax=Math.max(...values);if(ymin===ymax)ymax=ymin+1;
  const span=ymax-ymin;ymax+=span*.08;
  const x=value=>62+(transform(value)-xmin)/(xmax-xmin)*370,y=value=>153-(value-ymin)/(ymax-ymin)*132;
  for(let i=0;i<4;i++) {
    const v=ymin+(ymax-ymin)*i/3;
    svg.append(svgNode('line',{x1:62,x2:432,y1:y(v),y2:y(v),class:'chart-grid'}),svgNode('text',{x:55,y:y(v)+4,'text-anchor':'end',class:'chart-tick'},format(v,2)));
    const raw=xmin+(xmax-xmin)*i/3, xv=logX?10**raw:raw;
    svg.append(svgNode('text',{x:x(xv),y:173,'text-anchor':i===0?'start':i===3?'end':'middle',class:'chart-tick'},Math.abs(xv)>=10000?xv.toExponential(1):format(xv,1)));
  }
  svg.append(svgNode('text',{x:432,y:189,'text-anchor':'end',class:'chart-unit'},xLabel));
  const legend=document.createElement('div');legend.className='chart-legend';
  for(const item of series) {
    if(!valid.some(point=>finite(point[item.key])))continue;
    let d='',drawing=false;
    for(const point of valid) {if(!finite(point[item.key])){drawing=false;continue;}d+=`${drawing?'L':'M'}${x(point.x)},${y(point[item.key])} `;drawing=true;}
    svg.append(svgNode('path',{d,class:'chart-line',stroke:item.color,...(item.dashed?{'stroke-dasharray':'5 4'}:{})}));
    for(const point of valid)if(finite(point[item.key]))svg.append(svgNode('circle',{cx:x(point.x),cy:y(point[item.key]),r:valid.length<30?2.5:0,fill:item.color}));
    const label=document.createElement('span'),key=svgNode('svg',{viewBox:'0 0 16 6'});key.append(svgNode('line',{x1:0,x2:16,y1:3,y2:3,stroke:item.color,'stroke-width':2}));label.append(key,document.createTextNode(item.label));legend.append(label);
  }
  card.append(legend);container.append(card);
}

export function updateDesign(model) {
  const config=model?.hinf || {},d=config.selected_design || {};
  put('design-method',config.method || 'No minimum-entropy H∞ design metadata received.');
  const status=el('design-status');
  status.className=`badge ${config.passed===false || d.stable===false || d.feasible===false ? 'failed' : config.passed===true?'passed':'unverified'}`;
  status.textContent=config.passed===false || d.stable===false || d.feasible===false ? 'DESIGN CHECK FAILED' : config.passed===true ? 'Numerical linear-design checks passed' : d.stable===true ? 'Linear stability verified · inspect certificate' : 'Design verification not reported';
  put('design-scope',config.certificate_scope || config.scope || 'No linear-design scope has been reported.');
  put('design-assumptions',readable(config.assumptions || model?.controller?.scope));
  put('design-gamma',format(d.gamma));put('design-bound',format(d.hinf_norm_upper_bound));put('design-h2',format(d.h2_norm));put('design-entropy',format(d.entropy_value));
  put('entropy-definition',d.entropy_definition || config.entropy_definition || 'Spectral entropy of the weighted closed-loop disturbance-to-performance transfer. This is not physical entropy in J/K.');
  put('norm-method',d.hinf_norm_method || config.hinf_norm_method || 'A sampled frequency-response peak is a lower estimate, not a certified H∞ upper bound. Numerical certificate details are reported separately.');
  const designFields=[['γ: declared disturbance attenuation',format(d.gamma,6)],['Numerically certified H∞ upper bound',format(d.hinf_norm_upper_bound,6)],['Sampled frequency-response peak',format(d.hinf_norm_sampled,6)],['H₂ norm',d.h2_norm],['Minimum-entropy design value',d.entropy_value],['Closed-loop stability verified',d.stable],['Continuous closed-loop spectral abscissa',format(d.closed_loop_spectral_abscissa,6)],['Sampled closed-loop spectral radius',format(d.sampled_closed_loop_spectral_radius,9)],['Control Riccati residual',d.riccati_residual_x],['Filter Riccati residual',d.riccati_residual_y],['Coupling spectral radius ρ(XY)',d.coupling_spectral_radius],['Coupling ratio ρ(XY) / γ²',format(d.coupling_ratio,6)],['Bounded-real residual',d.bounded_real_residual],['DGKF normalization residual',config.normalized_dgkf_assumption_residual],['Linearization directional error',config.linearization_directional_error]];
  if(finite(d.hinf_norm_lower_bound))designFields.splice(2,0,['H∞ lower bound',format(d.hinf_norm_lower_bound,6)]);
  valueTable(el('design-values'),designFields);
  const plots=el('design-plots');plots.replaceChildren();
  designPlot(plots,{title:'Closed-loop frequency response',xLabel:'ω (rad/s) · logarithmic',logX:true,points:(config.frequency_response || []).map(p=>({x:p.omega_rad_s,sigma:p.sigma_max,gamma:d.gamma})),series:[{key:'sigma',label:'Sampled σmax(Tzw)',color:'#357f6b'},{key:'gamma',label:'Design γ',color:'#d77b46',dashed:true}]});
  const sweep=(config.gamma_sweep || []).map(p=>({...p,x:p.gamma,...(p.feasible===false?{hinf_norm_upper_bound:null,hinf_norm_sampled:null,h2_norm:null,entropy_value:null}:{})}));
  designPlot(plots,{title:'Disturbance attenuation tradeoff',xLabel:'Design γ',points:sweep,series:[{key:'hinf_norm_upper_bound',label:'Certified upper bound',color:'#357f6b'},{key:'hinf_norm_sampled',label:'Sampled peak',color:'#537daf'}]});
  designPlot(plots,{title:'H₂ performance across designs',xLabel:'Design γ',points:sweep,series:[{key:'h2_norm',label:'H₂ norm',color:'#537daf'}]});
  designPlot(plots,{title:'Minimum-entropy design value',xLabel:'Design γ',points:sweep,series:[{key:'entropy_value',label:'Spectral entropy',color:'#a077ad'}]});
  const rows=el('design-sweep-rows');rows.replaceChildren();
  for(const item of config.gamma_sweep || []) {
    const row=document.createElement('tr');
    for(const value of [item.gamma,item.feasible,item.stable,item.hinf_norm_upper_bound,item.hinf_norm_sampled,item.h2_norm,item.entropy_value,item.reason || '']){const cell=document.createElement('td');cell.textContent=readable(value);row.append(cell);}rows.append(row);
  }
  const matrices=el('design-matrices');matrices.replaceChildren();
  for(const [name,matrix] of Object.entries(config.matrices || {})) {
    const detail=document.createElement('details');detail.className='metadata-details';
    const title=document.createElement('summary');title.textContent=name;
    const pre=document.createElement('pre');pre.textContent=JSON.stringify(matrix,null,2);detail.append(title,pre);matrices.append(detail);
  }
  put('design-json',JSON.stringify(config,null,2));
}
