'use strict';
const R=window.VALIDATION_DATA;
const $=id=>document.getElementById(id);
const finite=value=>typeof value==='number'&&Number.isFinite(value);
const fmt=(value,digits=2)=>finite(value)?value.toLocaleString(undefined,{maximumFractionDigits:digits}):'—';
const escapeHTML=value=>String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const colors={historical:'#c77c4d',ensemble:'#087f8c'};
const config={responsive:true,displaylogo:false,modeBarButtonsToRemove:['lasso2d','select2d']};
const layout=extra=>({font:{family:'system-ui,sans-serif',size:11,color:'#647786'},paper_bgcolor:'white',plot_bgcolor:'white',margin:{l:65,r:25,t:35,b:50},legend:{orientation:'h',y:1.12,font:{size:10}},...extra});
const labels={cumulative_rain:['Cumulative precipitation','mm'],cumulative_pet:['Cumulative estimated PET','mm'],cumulative_balance:['Cumulative P − PET','mm'],temperature_7day:['Trailing 7-day mean temperature','°C']};

function bands(x,data,group,inner=false){
  const historical=group==='historical',color=colors[group],name=historical?'Historical':'Ensemble';
  const traces=[{type:'scatter',mode:'lines',x,y:data['5'],line:{width:0},hoverinfo:'skip',showlegend:false},
    {type:'scatter',mode:'lines',x,y:data['95'],line:{width:0},fill:'tonexty',fillcolor:historical?'rgba(199,124,77,.16)':'rgba(8,127,140,.15)',name:name+' P5–P95',hoverinfo:'skip'}];
  if(inner)traces.push({type:'scatter',mode:'lines',x,y:data['25'],line:{width:0},hoverinfo:'skip',showlegend:false},
    {type:'scatter',mode:'lines',x,y:data['75'],line:{width:0},fill:'tonexty',fillcolor:'rgba(8,127,140,.19)',name:'Ensemble P25–P75',hoverinfo:'skip'});
  traces.push({type:'scatter',mode:'lines',x,y:data['50'],line:{color,width:2,dash:historical?'dash':'solid'},name:name+' median',hovertemplate:'%{x}<br>%{y:.2f}<extra>'+name+'</extra>'});
  return traces;
}

function renderEnvelope(){
  const key=$('trajectory').value,entry=R.envelopes[key],traces=[...bands(R.dates,entry.historical,'historical'),...bands(R.dates,entry.ensemble,'ensemble',true)];
  const selected=R.historical.find(row=>String(row.year)===$('historical-year').value);
  if(selected)traces.push({type:'scatter',mode:'lines',x:R.dates,y:selected.series[key],name:'Historical '+selected.year,line:{color:'#6c4ca0',width:2},hovertemplate:'%{x}<br>%{y:.2f}<extra>Historical '+selected.year+'</extra>'});
  Plotly.react('envelope',traces,layout({hovermode:'x unified',xaxis:{tickformat:'%b',showgrid:false},yaxis:{title:{text:labels[key][0]+' · '+labels[key][1]},gridcolor:'#edf2f4'}}),config);
  $('envelope-note').textContent=`${fmt(entry.historical_pointwise_containment_percent,1)}% of historical season-days fall inside the ensemble's pointwise P5–P95 band. This is not the fraction of whole seasonal trajectories contained. Shaded bands are quantiles, not single weather scenarios.`;
}

function renderMonthly(){
  const key=$('monthly-variable').value,data=R.monthly[key];
  const names={rain:'Monthly precipitation · mm',temperature:'Monthly temperature · °C',pet:'Monthly estimated PET · mm'};
  Plotly.react('monthly',[...bands(R.months,data.historical,'historical'),...bands(R.months,data.ensemble,'ensemble')],
    layout({xaxis:{tickformat:'%b',showgrid:false},yaxis:{title:{text:names[key]},gridcolor:'#edf2f4'}}),config);
}

function values(group,key){return R[group].map(row=>row.values[key]).filter(finite).sort((a,b)=>a-b);}
function quantile(sorted,p){const i=(sorted.length-1)*p,a=Math.floor(i),b=Math.ceil(i);return sorted[a]+(sorted[b]-sorted[a])*(i-a);}
function renderDistributions(){
  const key=$('distribution-metric').value,metric=R.comparisons[key],h=values('historical',key),g=values('ensemble',key);
  if(!h.length||!g.length)return;
  const axis=metric.label+(metric.unit?' · '+metric.unit:'');
  const ecdf=(data,name,color)=>({type:'scatter',mode:'lines',x:[data[0],...data],y:[0,...data.map((_,i)=>100*(i+1)/data.length)],name,line:{color,width:2,shape:'hv'},hovertemplate:'%{x:.2f}<br>%{y:.1f}%<extra>'+name+'</extra>'});
  Plotly.react('ecdf',[ecdf(h,'Historical',colors.historical),ecdf(g,'Ensemble',colors.ensemble)],layout({xaxis:{title:{text:axis},gridcolor:'#edf2f4'},yaxis:{title:{text:'Cumulative fraction of seasons · %'},range:[0,101],gridcolor:'#edf2f4'}}),config);
  const p=Array.from({length:19},(_,i)=>(i+1)/20),x=p.map(q=>quantile(h,q)),y=p.map(q=>quantile(g,q)),lo=Math.min(...x,...y),hi=Math.max(...x,...y);
  Plotly.react('qq',[{type:'scatter',mode:'lines',x:[lo,hi],y:[lo,hi],name:'Equal quantiles',line:{color:'#b8c6cc',dash:'dot'},hoverinfo:'skip'},
    {type:'scatter',mode:'markers',x,y,text:p.map(q=>'P'+Math.round(q*100)),name:'Matching quantiles',marker:{color:colors.ensemble,size:8},hovertemplate:'%{text}<br>Historical %{x:.2f}<br>Ensemble %{y:.2f}<extra></extra>'}],
    layout({xaxis:{title:{text:'Historical '+(metric.unit||'value')},gridcolor:'#edf2f4'},yaxis:{title:{text:'Ensemble '+(metric.unit||'value')},gridcolor:'#edf2f4'}}),config);
  const ci=metric.mean_bias_bootstrap_95;
  $('metric-score').textContent=`Historical n=${metric.historical_n}; ensemble n=${metric.ensemble_n}. Mean bias ${fmt(metric.mean_bias)} ${metric.unit}; SD ratio ${fmt(metric.spread_ratio)}; historical values inside ensemble P5–P95: ${fmt(metric.historical_inside_ensemble_p5_p95_percent,1)}%.`+(ci?` Whole-season resampling interval for mean bias: ${fmt(ci[0])} to ${fmt(ci[1])} ${metric.unit}.`:'');
}

function boundaryTraces(){
  if(!R.boundary)return [];
  const polygons=R.boundary.type==='Polygon'?[R.boundary.coordinates]:R.boundary.coordinates,x=[],y=[];
  for(const polygon of polygons)for(const ring of polygon){for(const point of ring){x.push(point[0]);y.push(point[1]);}x.push(null);y.push(null);}
  return [{type:'scatter',mode:'lines',x,y,line:{color:'#294b60',width:1.6},hoverinfo:'skip',showlegend:false}];
}
function renderMap(){
  const key=$('map-metric').value,z=R.bias_maps[key],limit=Math.max(...z.flat().filter(finite).map(Math.abs),1e-10);
  const units={precipitation_mm:'mm',temperature_C:'°C',wet_day_percent:'pp',dry_spell_days:'days'};
  Plotly.react('bias-map',[{type:'heatmap',x:R.longitude,y:R.latitude,z,zmin:-limit,zmax:limit,hoverongaps:false,
    colorscale:[[0,'#4087a5'],[.5,'#fafafa'],[1,'#c46d4c']],colorbar:{title:{text:units[key]},thickness:11},
    hovertemplate:'%{y:.2f}° N, %{x:.2f}° E<br>Bias %{z:.2f} '+units[key]+'<extra></extra>'},...boundaryTraces()],
    layout({margin:{l:50,r:35,t:10,b:40},xaxis:{title:{text:'Longitude'},showgrid:false},yaxis:{title:{text:'Latitude'},showgrid:false,scaleanchor:'x',scaleratio:1/Math.cos(R.latitude.reduce((a,b)=>a+b,0)/R.latitude.length*Math.PI/180)}}),config);
}
function renderJoint(){
  const xkey=$('joint-x').value,ykey=$('joint-y').value;
  const traces=['historical','ensemble'].map(group=>{const rows=R[group].filter(row=>finite(row.values[xkey])&&finite(row.values[ykey]));return {type:'scatter',mode:'markers',x:rows.map(row=>row.values[xkey]),y:rows.map(row=>row.values[ykey]),text:rows.map(row=>group==='historical'?'Historical '+row.year:'Scenario '+row.number),name:group==='historical'?'Historical':'Ensemble',marker:{color:colors[group],size:group==='historical'?9:6,opacity:.7,symbol:group==='historical'?'diamond':'circle'},hovertemplate:'%{text}<br>%{x:.2f}, %{y:.2f}<extra></extra>'};});
  const axis=key=>R.comparisons[key].label+(R.comparisons[key].unit?' · '+R.comparisons[key].unit:'');
  Plotly.react('joint',traces,layout({xaxis:{title:{text:axis(xkey)},gridcolor:'#edf2f4'},yaxis:{title:{text:axis(ykey)},gridcolor:'#edf2f4'}}),config);
}
function downloadScores(){
  const keys=['id','historical_mean','ensemble_mean','mean_bias','spread_ratio','historical_inside_ensemble_p5_p95_percent','wasserstein_distance','ks_distance'];
  const text=[keys.join(','),...Object.values(R.comparisons).map(row=>keys.map(key=>row[key]??'').join(','))].join('\n');
  const link=document.createElement('a');link.href='data:text/csv;charset=utf-8,'+encodeURIComponent(text);link.download='distribution_comparison.csv';link.click();
}

try{
  if(typeof Plotly==='undefined')throw new Error('The embedded chart library could not start. Open the complete report.html in a browser.');
  $('subtitle').textContent=`${R.ensemble_members} generated seasons · ${R.historical_seasons} historical seasons (${R.historical_years[0]}–${R.historical_years.at(-1)}) · ${R.land_cells} land cells · ${R.start.slice(5)} through ${R.last_day.slice(5)}`;
  const modes={in_sample:'Training-data comparison',out_of_sample:'Held-out-year comparison',mixed:'Mixed training / held-out comparison'};
  $('mode').textContent=modes[R.comparison_mode];
  $('comparison-note').textContent=R.comparison_mode==='in_sample'?'These historical seasons were used to fit the generator. This report checks climatology fidelity; independent validation requires held-out years.':R.comparison_mode==='out_of_sample'?'These historical seasons are outside the model training years. The historical trend correction and index fits use the training model only.':'Some evaluation seasons overlap model training. Interpret this as a mixed-sample diagnostic.';
  $('reference-note').textContent=`Historical ERA5-Land temperatures are adjusted to ${R.temperature_reference_date} using the model's trained trend. Rainfall uses the prepared daily totals. All comparisons use the same ${R.land_cells} cells and matching calendar windows. The empirical P5–P95 intervals are not confidence intervals for the mean.`;
  const rain=R.comparisons.total_precipitation_mm,temp=R.comparisons.mean_temperature_C;
  $('rain-bias').textContent=fmt(rain.mean_bias,1)+' mm';$('rain-spread').textContent=fmt(rain.spread_ratio)+'×';
  $('rain-coverage').textContent=fmt(rain.historical_inside_ensemble_p5_p95_percent,1)+'%';$('temperature-bias').textContent=fmt(temp.mean_bias)+' °C';
  for(const row of R.historical){const option=document.createElement('option');option.value=row.year;option.textContent=row.year;$('historical-year').append(option);}
  for(const metric of Object.values(R.comparisons))for(const id of ['distribution-metric','joint-x','joint-y']){const option=document.createElement('option');option.value=metric.id;option.textContent=metric.label;option.disabled=!metric.available;$(id).append(option);}
  $('joint-x').value='total_precipitation_mm';$('joint-y').value='pet_mm';
  $('score-table').innerHTML=Object.values(R.comparisons).map(row=>`<tr data-metric="${escapeHTML(row.id)}"><td>${escapeHTML(row.label)}${row.unit?' · '+escapeHTML(row.unit):''}</td><td>${fmt(row.historical_mean)}</td><td>${fmt(row.ensemble_mean)}</td><td>${fmt(row.mean_bias)}</td><td>${fmt(row.spread_ratio)}</td><td>${fmt(row.historical_inside_ensemble_p5_p95_percent,1)}%</td><td>${fmt(row.wasserstein_distance)}</td><td>${fmt(row.ks_distance,3)}</td></tr>`).join('');
  $('spatial-table').innerHTML=R.spatial_correlation.map(row=>`<tr><td>${row.month}</td><td>${fmt(row.rainfall_correlation_MAE,3)}</td><td>${fmt(row.wet_occurrence_correlation_MAE,3)}</td><td>${row.rainfall_pairs}</td><td>${row.wet_occurrence_pairs}</td></tr>`).join('');
  $('fit-note').textContent=R.spatial_calibration.length?`${R.spatial_calibration.filter(row=>row.converged).length}/${R.spatial_calibration.length} original monthly spatial fits met their calibration target. The table above compares the resulting generated weather with history using separately calculated daily correlations.`:'No stored spatial-calibration report was available.';
  $('method-list').innerHTML=Object.entries(R.methods).filter(([key])=>key!=='sources').map(([key,value])=>'<li><strong>'+escapeHTML(key)+':</strong> '+escapeHTML(value)+'</li>').join('')+'<li><strong>Method references:</strong> '+R.methods.sources.map((url,i)=>'<a href="'+escapeHTML(url)+'" target="_blank" rel="noopener">Research reference '+(i+1)+'</a>').join(' · ')+'</li>';
  $('sample-note').textContent=`Evaluated historical seasons: ${R.historical_years.join(', ')}. Skipped seasons: ${R.skipped_historical_seasons.length}. Training years: ${R.training_years.join(', ')}. Historical trends are not refitted to the evaluation sample.`;
  for(const id of ['trajectory','historical-year'])$(id).addEventListener('change',renderEnvelope);
  $('monthly-variable').addEventListener('change',renderMonthly);$('distribution-metric').addEventListener('change',renderDistributions);
  $('map-metric').addEventListener('change',renderMap);for(const id of ['joint-x','joint-y'])$(id).addEventListener('change',renderJoint);
  $('download-scores').addEventListener('click',downloadScores);
  $('score-table').addEventListener('click',event=>{const row=event.target.closest('[data-metric]');if(row&&R.comparisons[row.dataset.metric].available){$('distribution-metric').value=row.dataset.metric;renderDistributions();$('distribution-metric').scrollIntoView({behavior:'smooth',block:'center'});}});
  renderEnvelope();renderMonthly();renderDistributions();renderMap();renderJoint();
}catch(error){$('error').hidden=false;$('error').textContent=error.message;}
