/* Local-only weather inspection; all chart resources and data can be embedded. */
'use strict';
const $ = id => document.getElementById(id);
const state = {data: null, cell: null, day: 0, timer: null, runs: [], catalogs: [], request: 0};
const offlineCache = new Map();
const names = {rain: 'Precipitation', tmean: 'Mean temperature', tmin: 'Minimum temperature', tmax: 'Maximum temperature'};
const units = {rain: 'mm', tmean: '°C', tmin: '°C', tmax: '°C'};
const monthNames = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
const chartConfig = {responsive: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d','select2d'], toImageButtonOptions: {scale: 2}};
const palette = {teal: '#087f8c', orange: '#df8957', ink: '#173347', pale: '#b7d9e0'};
const layout = extra => ({font: {family: 'system-ui, sans-serif', size: 11, color: '#647786'},
  paper_bgcolor: 'white', plot_bgcolor: 'white', margin: {l: 55, r: 30, t: 25, b: 40},
  hoverlabel: {bgcolor: '#173347', font: {color: 'white'}}, ...extra});
const finite = value => typeof value === 'number' && Number.isFinite(value);
const average = values => {const good = values.filter(finite); return good.length ? good.reduce((a,b)=>a+b,0)/good.length : null;};
const display = (value, suffix='') => finite(value) ? `${value.toLocaleString(undefined,{maximumFractionDigits:1})}${suffix}` : '—';
const escapeHTML = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

async function api(path) {
  if (window.EGGS_DATA) {
    if (path === '/api/runs') return window.EGGS_DATA.runs;
    if (path === '/api/ensembles') return window.EGGS_DATA.catalogs || [];
    const id=new URL(path, 'http://local').searchParams.get('id');
    if(window.EGGS_DATA.data?.[id])return window.EGGS_DATA.data[id];
    if(offlineCache.has(id))return offlineCache.get(id);
    const encoded=window.EGGS_DATA.compressed_data?.[id];
    if(!encoded)throw new Error('This scenario is not included in the offline file.');
    if(typeof DecompressionStream!=='function')throw new Error('Open this ensemble in a current browser, or use the local viewer.');
    const bytes=Uint8Array.from(atob(encoded),character=>character.charCodeAt(0));
    const stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
    const data=await new Response(stream).json();
    offlineCache.set(id,data);
    if(offlineCache.size>2)offlineCache.delete(offlineCache.keys().next().value);
    return data;
  }
  const response = await fetch(path);
  if (!response.ok) throw new Error(`Could not load this simulation (${response.status}).`);
  return response.json();
}

function selectedCatalog() {
  return state.catalogs.find(catalog=>catalog.scenarios.some(row=>row.id===state.data?.id));
}

function scenarioSelection() {
  const catalog=selectedCatalog(),metric=$('selection-metric').value,percentile=Number($('percentile').value);
  if(!catalog)return null;
  if($('percentile').value===''||!Number.isFinite(percentile)||percentile<0||percentile>100)throw new Error('Choose a percentile between 0 and 100.');
  const rows=catalog.scenarios.filter(row=>finite(row.values[metric])).sort((a,b)=>a.values[metric]-b.values[metric]||a.number-b.number);
  if(!rows.length)throw new Error('No scenario has a valid value for this metric. Choose another metric or review the historical fit checks.');
  const position=(rows.length-1)*percentile/100,lower=Math.floor(position),upper=Math.ceil(position);
  const target=rows[lower].values[metric]+(rows[upper].values[metric]-rows[lower].values[metric])*(position-lower);
  const closest=Math.min(...rows.map(row=>Math.abs(row.values[metric]-target)));
  const tolerance=1e-12*Math.max(1,Math.abs(target),...rows.map(row=>Math.abs(row.values[metric])));
  const chosen=rows.filter(row=>Math.abs(row.values[metric]-target)<=closest+tolerance).sort((a,b)=>a.number-b.number)[0];
  const actual=chosen.values[metric];
  const rank=value=>rows.length===1?50:100*(rows.filter(row=>row.values[metric]<value).length+(rows.filter(row=>row.values[metric]===value).length-1)/2)/(rows.length-1);
  return {catalog,metric,percentile,rows,target,chosen,rank};
}

async function renderScenarioSelection() {
  try {
    const selection=scenarioSelection();
    if(!selection)return;
    const {catalog,metric,percentile,rows,target,chosen,rank}=selection;
    const definition=catalog.metrics.find(item=>item.id===metric),suffix=definition.unit?' '+definition.unit:'';
    const current=catalog.scenarios.find(row=>row.id===state.data.id);
    $('selection-result').textContent=`P${percentile} target: ${display(target,suffix)} → Scenario ${chosen.number}: ${display(chosen.values[metric],suffix)} (ensemble P${display(rank(chosen.values[metric]))}). Viewing Scenario ${current.number}${finite(current.values[metric])?': '+display(current.values[metric],suffix):''}.`;
    $('selection-result').classList.remove('selection-error');
    $('select-scenario').disabled=false;
    const traces=[{type:'scatter',mode:'markers',x:rows.map(row=>rank(row.values[metric])),y:rows.map(row=>row.values[metric]),
      customdata:rows.map(row=>row.id),text:rows.map(row=>'Scenario '+row.number),name:'Generated scenarios',
      marker:{size:rows.map(row=>row.id===state.data.id?12:8),color:rows.map(row=>row.id===state.data.id?palette.orange:palette.teal)},
      hovertemplate:'%{text}<br>Ensemble P%{x:.1f}<br>%{y:.2f}'+suffix+'<extra></extra>'}];
    await Plotly.react('ensemble-chart',traces,layout({margin:{l:65,r:25,t:12,b:45},showlegend:false,
      xaxis:{title:{text:'Percentile within this ensemble'},range:[-3,103],dtick:10,gridcolor:'#edf2f4'},
      yaxis:{title:{text:definition.label+(suffix?' ('+definition.unit+')':'')},gridcolor:'#edf2f4'},
      shapes:[{type:'line',xref:'x',yref:'paper',x0:percentile,x1:percentile,y0:0,y1:1,line:{color:'#c7a290',width:1,dash:'dot'}}]}),chartConfig);
    $('ensemble-chart').removeAllListeners('plotly_click');
    $('ensemble-chart').on('plotly_click',event=>{const id=event.points[0]?.customdata;if(id){$('run').value=id;loadSelected();}});
  } catch(error) {
    $('selection-result').textContent=error.message;
    $('selection-result').classList.add('selection-error');
    $('select-scenario').disabled=true;
  }
}

function renderScenarioPanel() {
  const catalog=selectedCatalog();
  $('scenario-panel').hidden=!catalog;
  if(!catalog){if($('ensemble-chart').data)Plotly.purge('ensemble-chart');return;}
  if($('selection-metric').dataset.experiment!==catalog.experiment) {
    $('selection-metric').replaceChildren();
    catalog.metrics.forEach(metric=>{
      const option=document.createElement('option');option.value=metric.id;option.textContent=metric.label;
      option.disabled=!catalog.scenarios.some(row=>finite(row.values[metric.id]));
      $('selection-metric').append(option);
    });
    $('selection-metric').dataset.experiment=catalog.experiment;
  }
  $('ensemble-count').textContent=catalog.scenario_count+' SCENARIOS';
  $('validation-link').hidden=!!window.EGGS_DATA||!catalog.validation_available;
  $('validation-link').href='/validation?experiment='+encodeURIComponent(catalog.experiment);
  $('ensemble-caption').textContent=`${catalog.start} through ${catalog.last_day} · ${catalog.baseline.land_cells} land cells · metrics use the full simulation period.`;
  $('scenario-method').textContent=`Percentiles rank this ensemble, with low values on the left. Negative SPI/SPEI indicates dry conditions against the ${catalog.baseline.start.slice(0,4)}–${catalog.baseline.last_day.slice(0,4)} reference. SPEI uses temperature-based Hargreaves evaporative demand. The date window and cell selection below change the weather plots; scenario ranking stays regional and uses the full period.`;
  const row=catalog.scenarios.find(item=>item.id===state.data.id);
  const keys=catalog.scales_months.flatMap(scale=>['spi_'+scale,'spei_'+scale]);
  $('scenario-monthly').innerHTML='<table><thead><tr><th>Month</th><th>Rain mm</th><th>PET mm</th>'+keys.map(key=>'<th>'+key.toUpperCase().replace('_','-')+'</th>').join('')+'</tr></thead><tbody>'+
    row.monthly.map(month=>'<tr><td>'+escapeHTML(month.month)+'</td><td>'+display(month.precipitation_mm)+'</td><td>'+display(month.pet_mm)+'</td>'+keys.map(key=>'<td>'+display(month[key])+(month.clipped_indices.includes(key)?'*':'')+'</td>').join('')+'</tr>').join('')+
    '</tbody></table><p>—: insufficient complete simulated months or historical fit support. *: the fitted probability reached the finite tail bound.</p>';
  renderScenarioSelection();
}

async function chooseScenario() {
  try {
    const selection=scenarioSelection();
    if(!selection)return;
    $('run').value=selection.chosen.id;
    await loadSelected();
  } catch(error) {showError(error);}
}

function downloadSelection() {
  const catalog=selectedCatalog();
  if(!catalog)return;
  const selected=catalog.scenarios.find(row=>row.id===state.data.id);
  const selection=scenarioSelection();
  const definition=catalog.metrics.find(metric=>metric.id===selection.metric);
  const result={experiment:catalog.experiment,metric:definition,requested_percentile:selection.percentile,
    target_value:selection.target,selected_scenario:selected,
    selected_metric_ensemble_percentile:finite(selected.values[selection.metric])?selection.rank(selected.values[selection.metric]):null,
    closest_scenario_id:selection.chosen.id,baseline:catalog.baseline,methods:catalog.methods};
  downloadText(JSON.stringify(result,null,2),'scenario_'+selected.number+'_selection.json','application/json');
}

function cells() {
  const result = [], d = state.data;
  d.mask.forEach((row,y)=>row.forEach((valid,x)=>{if(valid) result.push([y,x,Math.cos(d.latitude[y]*Math.PI/180)]);}));
  return result;
}

function spatial(grid, scope=$('scope').value) {
  if (scope === 'cell') return grid[state.cell[0]][state.cell[1]];
  let total=0, weight=0;
  for (const [y,x,w] of cells()) if(finite(grid[y][x])) {total+=grid[y][x]*w;weight+=w;}
  return weight ? total/weight : null;
}

function selectedDays() {
  return state.data.days.map((day,index)=>day >= $('start').value && day <= $('end').value ? index : -1).filter(i=>i>=0);
}

function aggregate(metric, indices) {
  const d=state.data;
  return d.latitude.map((_,y)=>d.longitude.map((_,x)=>{
    if(!d.mask[y][x]) return null;
    const values=indices.map(i=>d.metrics[metric][i][y][x]).filter(finite);
    if(!values.length) return null;
    if(metric==='rain') return values.reduce((a,b)=>a+b,0);
    if(metric==='tmin') return Math.min(...values);
    if(metric==='tmax') return Math.max(...values);
    return average(values);
  }));
}

function boundaryTraces() {
  const geometry=state.data.boundary;
  if(!geometry) return [];
  const polygons=geometry.type==='Polygon' ? [geometry.coordinates] : geometry.coordinates;
  const longitude=[], latitude=[];
  for(const polygon of polygons) for(const ring of polygon) {
    for(const point of ring) {longitude.push(point[0]);latitude.push(point[1]);}
    longitude.push(null);latitude.push(null);
  }
  return [{type:'scatter',mode:'lines',x:longitude,y:latitude,line:{color:'#1d465b',width:1.8},
    hoverinfo:'skip',showlegend:false}];
}

async function drawMap(id, grid, metric, snapshot) {
  const d=state.data, isRain=metric==='rain';
  const values=[];
  if(snapshot) for(const daily of d.metrics[metric]) for(const [y,x] of cells()) {const v=daily[y][x];if(finite(v))values.push(v);}
  else for(const row of grid) for(const value of row) if(finite(value)) values.push(value);
  const low=isRain?0:Math.min(...values), high=Math.max(...values);
  const traces=[{type:'heatmap',x:d.longitude,y:d.latitude,z:grid,
    colorscale:isRain?[[0,'#f4f9fc'],[.15,'#d1e6ed'],[.45,'#6bb6ca'],[.75,'#277d9e'],[1,'#174e70']]:
      [[0,'#3d88a3'],[.35,'#b9dcdf'],[.65,'#f3e3ba'],[1,'#c86447']],
    zmin:low,zmax:high===low?low+1:high,zsmooth:false,hoverongaps:false,
    colorbar:{title:{text:units[metric],side:'right'},thickness:10,len:.85,tickfont:{size:10}},
    hovertemplate:'%{y:.2f}° N, %{x:.2f}° E<br><b>%{z:.2f} '+units[metric]+'</b><extra></extra>'},
    ...boundaryTraces(),{type:'scatter',mode:'markers',x:[d.longitude[state.cell[1]]],y:[d.latitude[state.cell[0]]],
      marker:{symbol:'circle-open',size:12,line:{color:'#153b50',width:2}},hoverinfo:'skip',showlegend:false}];
  const mapLayout=layout({margin:{l:45,r:30,t:10,b:35},uirevision:d.id,
    xaxis:{title:{text:'Longitude °E',font:{size:10}},showgrid:false,zeroline:false},
    yaxis:{title:{text:'Latitude °N',font:{size:10}},showgrid:false,zeroline:false,scaleanchor:'x',scaleratio:1/Math.cos(41.5*Math.PI/180)}});
  await Plotly.react(id,traces,mapLayout,chartConfig);
  const element=$(id);
  element.removeAllListeners('plotly_click');
  element.on('plotly_click',event=>{
    const point=event.points.find(item=>item.data.type==='heatmap');
    if(!point) return;
    const y=d.latitude.reduce((best,value,i)=>Math.abs(value-point.y)<Math.abs(d.latitude[best]-point.y)?i:best,0);
    const x=d.longitude.reduce((best,value,i)=>Math.abs(value-point.x)<Math.abs(d.longitude[best]-point.x)?i:best,0);
    if(!d.mask[y][x]) return;
    state.cell=[y,x];$('scope').value='cell';renderAll();
  });
}

function dailyRows(indices) {
  let cumulative=0;
  return indices.map(i=>{
    const rain=spatial(state.data.metrics.rain[i]);cumulative+=rain||0;
    return {date:state.data.days[i],rain,tmean:spatial(state.data.metrics.tmean[i]),
      tmin:spatial(state.data.metrics.tmin[i]),tmax:spatial(state.data.metrics.tmax[i]),cumulative};
  });
}

function scopeText() {
  const d=state.data;
  return $('scope').value==='area'?'Whole area':`${d.latitude[state.cell[0]].toFixed(2)}° N, ${d.longitude[state.cell[1]].toFixed(2)}° E`;
}

function renderSummary(indices) {
  const rows=dailyRows(indices), d=state.data;
  $('mean-temp').textContent=display(average(rows.map(row=>row.tmean)),' °C');
  let peak=-Infinity, wet=0, total=0;
  const selection=$('scope').value==='cell'?[[...state.cell,1]]:cells();
  for(const index of indices) for(const [y,x,w] of selection) {
    peak=Math.max(peak,d.metrics.tmax[index][y][x]);
    wet+=(d.metrics.rain[index][y][x]>=1?1:0)*w;total+=w;
  }
  $('peak-temp').textContent=display(peak,' °C');
  $('total-rain').textContent=display(rows.reduce((sum,row)=>sum+(row.rain||0),0),' mm');
  $('wet-share').textContent=display(100*wet/total,' %');
  $('summary-scope').textContent=scopeText();
  $('cell-label').textContent=`Selected cell: ${d.latitude[state.cell[0]].toFixed(2)}° N, ${d.longitude[state.cell[1]].toFixed(2)}° E. Click a map to change it.`;
}

function renderSeries(indices) {
  const rows=dailyRows(indices), dates=rows.map(row=>row.date);
  const traces=[
    {type:'scatter',x:dates,y:rows.map(r=>r.tmin),mode:'lines',line:{width:0},hoverinfo:'skip',showlegend:false,name:'Daily minimum'},
    {type:'scatter',x:dates,y:rows.map(r=>r.tmax),mode:'lines',line:{width:0},fill:'tonexty',fillcolor:'rgba(223,137,87,.16)',name:'Daily min–max',hovertemplate:'%{x}<br>Maximum %{y:.1f} °C<extra></extra>'},
    {type:'scatter',x:dates,y:rows.map(r=>r.tmean),line:{color:palette.orange,width:2},name:'Mean temperature',hovertemplate:'%{x}<br>%{y:.1f} °C<extra></extra>'},
    {type:'bar',x:dates,y:rows.map(r=>r.rain),yaxis:'y2',marker:{color:palette.teal},name:'Daily rain',hovertemplate:'%{x}<br>%{y:.1f} mm<extra></extra>'},
    {type:'scatter',x:dates,y:rows.map(r=>r.cumulative),yaxis:'y3',line:{color:'#3d6785',dash:'dot',width:1.5},name:'Cumulative rain',visible:'legendonly',hovertemplate:'%{x}<br>%{y:.1f} mm accumulated<extra></extra>'}
  ];
  $('series-caption').textContent=`${scopeText()} · daily temperature range and rainfall. Click “Cumulative rain” in the legend to show it.`;
  Plotly.react('time-series',traces,layout({margin:{l:50,r:55,t:45,b:40},barmode:'overlay',hovermode:'x unified',
    legend:{orientation:'h',y:1.22,x:0,font:{size:10}},xaxis:{showgrid:false},
    yaxis:{title:{text:'Temperature °C'},gridcolor:'#edf2f4',domain:[.4,1]},
    yaxis2:{title:{text:'Rain mm'},gridcolor:'#edf2f4',domain:[0,.3],rangemode:'tozero'},
    yaxis3:{title:{text:'Cumulative mm'},overlaying:'y2',side:'right',anchor:'free',position:1,showgrid:false}}),chartConfig);
}

function renderMonthly(indices) {
  const d=state.data, rows=dailyRows(indices), groups=new Map();
  rows.forEach(row=>{const month=row.date.slice(0,7);if(!groups.has(month))groups.set(month,[]);groups.get(month).push(row);});
  const keys=[...groups.keys()], labels=keys.map(key=>monthNames[Number(key.slice(5))-1]+' '+key.slice(0,4));
  const rain=keys.map(key=>groups.get(key).reduce((sum,row)=>sum+row.rain,0));
  const temperature=keys.map(key=>average(groups.get(key).map(row=>row.tmean)));
  const traces=[{type:'bar',x:labels,y:rain,name:'Simulated rain',marker:{color:palette.teal}},
    {type:'scatter',x:labels,y:temperature,yaxis:'y2',name:'Simulated temp',mode:'lines+markers',line:{color:palette.orange,width:2}}];
  if(d.climatology) {
    const referenceRain=keys.map(key=>{
      const month=Number(key.slice(5)),year=Number(key.slice(0,4));
      return spatial(d.climatology.rain[month-1])*groups.get(key).length/new Date(year,month,0).getDate();
    });
    const referenceTemperature=keys.map(key=>spatial(d.climatology.temperature[Number(key.slice(5))-1]));
    traces.splice(1,0,{type:'bar',x:labels,y:referenceRain,name:'Historical rain',marker:{color:palette.pale}});
    traces.push({type:'scatter',x:labels,y:referenceTemperature,yaxis:'y2',name:'Reference temp',mode:'lines',line:{color:'#c7a290',width:1.5,dash:'dot'}});
  }
  Plotly.react('monthly-chart',traces,layout({margin:{l:50,r:45,t:45,b:45},legend:{orientation:'h',y:1.25,font:{size:9}},
    barmode:'group',yaxis:{title:{text:'Rain mm'},gridcolor:'#edf2f4',rangemode:'tozero'},
    yaxis2:{title:{text:'°C'},overlaying:'y',side:'right',showgrid:false},xaxis:{showgrid:false}}),chartConfig);
}

function renderHourly() {
  const d=state.data,i=state.day,start=i*24,selected=[],area=[];
  for(let hour=0;hour<24;hour++) {
    const grid=d.hourly_temperature[start+hour];
    selected.push(grid[state.cell[0]][state.cell[1]]);area.push(spatial(grid,'area'));
  }
  const hours=Array.from({length:24},(_,i)=>i);
  $('hourly-caption').textContent=`${d.days[i]} · ${d.latitude[state.cell[0]].toFixed(2)}° N, ${d.longitude[state.cell[1]].toFixed(2)}° E`;
  Plotly.react('hourly-chart',[
    {type:'scatter',x:hours,y:selected,mode:'lines+markers',name:'Selected cell',line:{color:palette.orange,width:2},marker:{size:4}},
    {type:'scatter',x:hours,y:area,name:'Whole area',line:{color:palette.teal,width:1.5,dash:'dot'}}
  ],layout({margin:{l:48,r:20,t:35,b:40},legend:{orientation:'h',y:1.18,font:{size:10}},hovermode:'x unified',
    xaxis:{title:{text:'Hour · UTC'},dtick:3,range:[0,23],showgrid:false},
    yaxis:{title:{text:'Temperature °C'},gridcolor:'#edf2f4'}}),chartConfig);
}

function renderSnapshot() {
  if(!state.data)return;
  const d=state.data,metric=$('metric').value;
  $('day-slider').value=state.day;$('day').value=d.days[state.day];
  $('day-label').textContent=new Date(d.days[state.day]+'T12:00:00Z').toLocaleDateString(undefined,{day:'numeric',month:'short',year:'numeric',timeZone:'UTC'});
  $('daily-caption').textContent=`${names[metric]} · ${units[metric]} · fixed colour scale across all days`;
  drawMap('snapshot-map',d.metrics[metric][state.day],metric,true);
  renderHourly();
}

function renderQuality() {
  const d=state.data,q=d.quality,m=d.metadata;
  $('quality-summary').textContent=q.calibration.length ?
    `Training data & fit checks · ${q.calibration.filter(row=>row.converged).length}/${q.calibration.length} monthly spatial fits meet the 0.005 target` : 'Training data & fit checks';
  const labels=[['Training period',`${m.training_start||'—'} through ${m.training_end||'—'}`],
    ['Retained training days',q.retained_days?.toLocaleString()||'—'],['Modelled land cells',q.land_cells],
    ['Excluded training years',q.excluded_years.length?q.excluded_years.join(', '):'None'],
    ['Excluded incomplete days',q.dropped_days.length],['Random seed',m.random_seed||'Not recorded'],
    ['Sparse seasonal climatology',`${(q.seasonal_support||[]).filter(row=>row.wet_calendar_days_interpolated||row.dry_calendar_days_interpolated).length} cells with calendar-date interpolation`],
    ['Negative rainfall after adjustment',`${m.negative_precipitation_clipped||0} cell-days bounded at zero`],
    ['Temperature reference',d.climatology?.reference_date||'Not recorded']];
  let html='<dl>'+labels.map(([key,value])=>`<dt>${escapeHTML(key)}</dt><dd>${escapeHTML(value)}</dd>`).join('')+'</dl>';
  if(q.dropped_days.length) html+='<p>Incomplete days: '+q.dropped_days.map(row=>escapeHTML(row.date)).join(', ')+'. Rainfall at midnight belongs to the preceding UTC day.</p>';
  if(q.calibration.length) {
    html+='<p>Spatial rainfall calibration: mean absolute error between simulated and historical pairwise correlations. Values closer to zero indicate a closer fit; the target is 0.005. An executable run does not by itself validate regional weather statistics.</p>';
    html+='<table><thead><tr><th>Month</th><th>Occurrence MAE</th><th>Rainfall amount MAE</th></tr></thead><tbody>';
    for(let month=1;month<=12;month++) {
      const occ=q.calibration.find(row=>row.month===month&&row.kind==='occurrence');
      const amt=q.calibration.find(row=>row.month===month&&row.kind==='amounts');
      html+=`<tr><td>${monthNames[month-1]}</td><td>${occ?.mean_absolute_error?.toFixed(4)||'—'}</td><td>${amt?.mean_absolute_error?.toFixed(4)||'—'}</td></tr>`;
    }
    html+='</tbody></table>';
  }
  $('quality-content').innerHTML=html;
}

function renderAll() {
  if(!state.data)return;
  let indices=selectedDays();
  if(!indices.length) {showError('Select a date window containing at least one simulation day.');return;}
  $('error').hidden=true;
  const metric=$('metric').value;
  $('aggregate-caption').textContent=`${metric==='rain'?'Accumulated':metric==='tmin'?'Lowest':metric==='tmax'?'Highest':'Average'} ${names[metric].toLowerCase()} · ${units[metric]}`;
  $('window-label').textContent=indices.length+' DAYS';
  renderSummary(indices);renderSnapshot();drawMap('period-map',aggregate(metric,indices),metric,false);
  renderSeries(indices);renderMonthly(indices);
}

function showError(error) {$('error').hidden=false;$('error').textContent=error.message||String(error);$('status').textContent='Check selection';}

async function loadSelected() {
  const request=++state.request;
  $('status').textContent='Loading…';stopAnimation();
  try {
    const d=await api('/api/run?id='+encodeURIComponent($('run').value));
    if(request!==state.request)return;
    state.data=d;state.day=0;
    const valid=cells();state.cell=valid[Math.floor(valid.length/2)].slice(0,2);
    for(const id of ['start','end','day']) {$(id).min=d.days[0];$(id).max=d.days.at(-1);}
    $('start').value=d.days[0];$('end').value=d.days.at(-1);$('day-slider').max=d.days.length-1;
    $('run-info').textContent=`${d.quality.land_cells} available land cells · hourly temperature + daily rainfall`;
    $('subtitle').textContent=`${d.days[0]} to ${d.days.at(-1)} · ${d.days.length} simulated days · ${d.quality.land_cells} land cells`;
    $('status').textContent=window.EGGS_DATA?'Offline explorer':'Local explorer';
    $('reference').textContent='Historical training '+(d.metadata.training_start||'').slice(0,4)+'–'+(d.metadata.training_end||'').slice(0,4);
    renderQuality();renderAll();renderScenarioPanel();
  } catch(error) {showError(error);}
}

function stopAnimation() {if(state.timer)clearInterval(state.timer);state.timer=null;$('play').textContent='▶';}
function downloadCSV() {
  const rows=dailyRows(selectedDays()),csv=['date,temperature_mean_C,temperature_min_C,temperature_max_C,precipitation_mm,cumulative_precipitation_mm'];
  rows.forEach(row=>csv.push([row.date,row.tmean,row.tmin,row.tmax,row.rain,row.cumulative].join(',')));
  downloadText(csv.join('\n'),($('scope').value==='area'?'area':'selected_cell')+'_daily_weather.csv','text/csv');
}

function downloadText(content,filename,type) {
  const blobURLs=typeof URL.createObjectURL==='function';
  const url=blobURLs?URL.createObjectURL(new Blob([content],{type:type+';charset=utf-8'})):
    'data:'+type+';charset=utf-8,'+encodeURIComponent(content);
  const link=document.createElement('a');link.href=url;link.download=filename;link.click();
  if(blobURLs)setTimeout(()=>URL.revokeObjectURL(url),1000);
}

$('run').addEventListener('change',loadSelected);
$('selection-metric').addEventListener('change',renderScenarioSelection);
$('percentile').addEventListener('input',renderScenarioSelection);
$('select-scenario').addEventListener('click',chooseScenario);
$('download-selection').addEventListener('click',()=>{try{downloadSelection();}catch(error){showError(error);}});
for(const id of ['metric','scope','start','end'])$(id).addEventListener('change',renderAll);
$('day-slider').addEventListener('input',event=>{state.day=Number(event.target.value);renderSnapshot();});
$('day').addEventListener('change',event=>{const index=state.data.days.indexOf(event.target.value);if(index>=0){state.day=index;renderSnapshot();}});
$('reset').addEventListener('click',()=>{$('start').value=state.data.days[0];$('end').value=state.data.days.at(-1);renderAll();});
$('download').addEventListener('click',downloadCSV);
$('play').addEventListener('click',()=>{if(state.timer){stopAnimation();return;}$('play').textContent='Ⅱ';state.timer=setInterval(()=>{state.day=(state.day+1)%state.data.days.length;renderSnapshot();},650);});
window.addEventListener('beforeunload',stopAnimation);

(async()=>{
  try {
    if(typeof window.Plotly?.react!=='function')throw new Error('The chart library could not start. Open the complete explorer.html file directly in a browser.');
    [state.runs,state.catalogs]=await Promise.all([api('/api/runs'),api('/api/ensembles')]);
    if(!state.runs.length)throw new Error('No completed simulations found. Run Main.py with a case-study configuration first.');
    state.runs.forEach(item=>{const option=document.createElement('option');option.value=item.id;option.textContent=item.label;$('run').append(option);});
    const preferred=state.runs.find(item=>item.id.startsWith('capitanata_ensemble_apr_oct_2026/'))||state.runs.find(item=>item.id.startsWith('capitanata_apr_oct_2026/'));
    if(preferred)$('run').value=preferred.id;
    await loadSelected();
  } catch(error) {showError(error);}
})();
