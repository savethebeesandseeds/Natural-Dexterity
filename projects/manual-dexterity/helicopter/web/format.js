export const finite = value => typeof value === 'number' && Number.isFinite(value);
export const format = (value, digits=3) => !finite(value) ? '—' : value === 0 ? (0).toFixed(digits) : Math.abs(value)<10**-digits ? value.toExponential(2) : value.toFixed(digits);
export const readable = value => value===undefined || value===null ? 'Not reported' : typeof value==='boolean' ? value ? 'Yes' : 'No' : finite(value) ? format(value) : Array.isArray(value) ? value.map(readable).join(', ') : typeof value==='object' ? JSON.stringify(value) : String(value);
export const el = id => document.getElementById(id);
export const put = (id,value) => { const target=el(id); if(target) target.textContent=value; };
export function valueTable(container, entries) {
  container.replaceChildren();
  for(const [label,value] of entries) {
    const row=document.createElement('tr'), name=document.createElement('th'), cell=document.createElement('td');
    name.scope='row';name.textContent=label;cell.textContent=readable(value);row.append(name,cell);container.append(row);
  }
}
