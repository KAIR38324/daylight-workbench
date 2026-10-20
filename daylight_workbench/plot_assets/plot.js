'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const colors = {front:'#0b9b91', other:'#98abb7', all:'#3d8eb7', selected:'#f2a430'};
  const state = {data:null, selected:[], mode:'2d', tool:'lasso', x:null, y:null, z:null, constraints:{}, bridge:null, scatterEvents:false, parallelEvents:false, webgl:false, view:'frontier'};
  try { const c=document.createElement('canvas'); state.webgl=!!(c.getContext('webgl2')||c.getContext('webgl')); } catch (_) {}
  const escapeHtml = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const isSelected = key => state.selected.includes(key);
  const formatNumber = v => v === null || v === undefined ? '—' : Number(v).toLocaleString(undefined,{maximumSignificantDigits:7});
  const label = s => escapeHtml(s);
  // These are presentation-only labels. Raw names remain the dictionary keys,
  // bridge contract and hover/tooltips, so no data semantics are inferred.
  const displayName = s => String(s).replace(/^predicted::/, '').replace(/_/g, ' ');
  const shortName = (s,n=24) => displayName(s).length>n ? displayName(s).slice(0,n-1)+'…' : displayName(s);
  const axisLabel = s => {
    const words=displayName(s).split(' '); let lines=[''];
    for(const w of words){if((lines.at(-1)+' '+w).trim().length>23 && lines.at(-1))lines.push(w);else lines[lines.length-1]=(lines.at(-1)+' '+w).trim();}
    return lines.map(label).join('<br>');
  };
  function fillAxes() {
    const names = state.data.objectives;
    for (const [node, preferred, fallback] of [[$('axisX'),state.x,0],[$('axisY'),state.y,1],[$('axisZ'),state.z,2]]) {
      node.replaceChildren();
      names.forEach(name => { const o=document.createElement('option'); o.value=name; o.textContent=displayName(name); o.title=name; node.appendChild(o); });
      node.value = names.includes(preferred) ? preferred : names[Math.min(fallback,names.length-1)] || '';
    }
    state.x=$('axisX').value; state.y=$('axisY').value; state.z=$('axisZ').value;
    if(names.length < 3 || !state.webgl) state.mode='2d';
    $('mode3').disabled=names.length<3 || !state.webgl;
    $('mode3').title=state.webgl?'':'3D needs WebGL; 2D and parallel-coordinate brushing remain available.';
    $('scatterHint').textContent=state.webgl?'拖拽框选；点击单个方案。3D 模式可旋转并点击。':'当前环境无 WebGL：3D 不可用；2D 与平行坐标仍可交互。';
    $('mode2').classList.toggle('active',state.mode==='2d');
    $('mode3').classList.toggle('active',state.mode==='3d');
    $('zWrap').style.display=state.mode==='3d'?'flex':'none';
    $('interactionTools').style.display=state.mode==='3d'?'none':'flex';
  }
  function legend() {
    const entries = state.data.has_pareto ? [['Pareto 前沿',colors.front],['其他方案',colors.other]] : [['候选方案',colors.all]];
    entries.push(['已选',colors.selected]);
    $('legend').replaceChildren();
    for(const [name,color] of entries) { const e=document.createElement('span'); e.className='key-item'; const d=document.createElement('span'); d.className='key-dot'; d.style.background=color; e.appendChild(d); e.appendChild(document.createTextNode(name)); $('legend').appendChild(e); }
  }
  function pointTrace(rows, name, color, selected=false) {
    const dims=[state.x,state.y,...(state.mode==='3d'?[state.z]:[])];
    const visible=rows.filter(r=>dims.every(d=>r.values[d]!==null));
    const is3=state.mode==='3d';
    const t={type:is3?'scatter3d':(state.webgl && visible.length>2000?'scattergl':'scatter'), mode:'markers', name,
      x:visible.map(r=>r.values[state.x]), y:visible.map(r=>r.values[state.y]),
      customdata:visible.map(r=>r.id_key),
      hovertext:visible.map(r=>`<b>ID:</b> ${r.hover_id}<br><b>${label(state.x)}:</b> ${formatNumber(r.values[state.x])}<br><b>${label(state.y)}:</b> ${formatNumber(r.values[state.y])}`+(is3?`<br><b>${label(state.z)}:</b> ${formatNumber(r.values[state.z])}`:'')),
      hovertemplate:'%{hovertext}<extra></extra>',
      marker:{color,size:selected?10:7,opacity:selected?1:(state.selected.length?.17:.8),line:{color:selected?'#7e530c':'#fff',width:selected?2:1}},
      showlegend:false};
    if(is3) t.z=visible.map(r=>r.values[state.z]);
    return t;
  }
  async function drawScatter() {
    if(!state.data) return;
    const rows=state.data.rows;
    const traces=state.data.has_pareto ? [pointTrace(rows.filter(r=>!r.pareto),'Other',colors.other),pointTrace(rows.filter(r=>r.pareto),'Pareto',colors.front)] : [pointTrace(rows,'Candidates',colors.all)];
    traces.push(pointTrace(rows.filter(r=>isSelected(r.id_key)),'Selected',colors.selected,true));
    const visibleCount=rows.filter(r=>[state.x,state.y,...(state.mode==='3d'?[state.z]:[])].every(d=>r.values[d]!==null)).length;
    $('scope').textContent=`散点显示 ${visibleCount}/${state.data.count} 个方案（无抽样；其余目标值缺失）；图中选择不会缩小表格或导出范围。`;
    const axis={title:{font:{size:11,color:'#60798d'},standoff:13},gridcolor:'#e9eef3',zerolinecolor:'#dbe5ed',linecolor:'#c5d2dd',tickfont:{size:10,color:'#8192a1'},automargin:true};
    const layout={margin:{l:62,r:25,b:66,t:27},paper_bgcolor:'#fff',plot_bgcolor:'#fff',showlegend:false,hovermode:'closest',dragmode:state.mode==='3d'?'orbit':state.tool,font:{family:'Segoe UI, Arial, sans-serif'},hoverlabel:{bgcolor:'#fff',bordercolor:'#cddae4',font:{size:12,color:'#2f4b60'}},uirevision:`${state.data.version}-${state.mode}-${state.x}-${state.y}-${state.z}`};
    if(state.mode==='3d') { layout.scene={xaxis:{...axis,title:{...axis.title,text:axisLabel(state.x)}},yaxis:{...axis,title:{...axis.title,text:axisLabel(state.y)}},zaxis:{...axis,title:{...axis.title,text:axisLabel(state.z)}},bgcolor:'#fff',aspectmode:'cube'}; layout.margin={l:0,r:0,b:0,t:5}; }
    else { layout.xaxis={...axis,title:{...axis.title,text:axisLabel(state.x)}}; layout.yaxis={...axis,title:{...axis.title,text:axisLabel(state.y)}}; }
    await Plotly.react($('scatter'),traces,layout,{displaylogo:false,responsive:true,modeBarButtonsToRemove:['toImage','sendDataToCloud'],scrollZoom:false});
    if(!state.scatterEvents) {
      state.scatterEvents=true;
      $('scatter').on('plotly_click', ev=>{const id=ev.points?.[0]?.customdata; if(id===undefined)return; setSelection(isSelected(id)?state.selected.filter(v=>v!==id):[...state.selected,id],true);});
      $('scatter').on('plotly_selected',ev=>{if(!ev?.points)return; setSelection([...new Set(ev.points.filter(p=>p.customdata!==undefined).map(p=>p.customdata))],true);});
    }
  }

  // SVG parallel coordinates keep the analysis usable when Chromium has no WebGL
  // (notably offscreen/remote desktops). All rows remain plotted, never sampled.
  function drawParallelSvg() {
    const host=$('parallel'), data=state.data;
    if(!data)return;
    host.replaceChildren();
    const dims=data.dimensions;
    if(!dims.length)return;
    const width=Math.max(host.clientWidth,dims.length*108+40), height=Math.max(host.clientHeight-16,180);
    const top=72, bottom=height-34, left=58, right=width-58;
    $('parallelHint').textContent=`${dims.length} 个维度 · 横向滚动查看全部；拖拽轴刷选，双击轴清除该维度。`;
    const bounds=dims.map(d=>{
      let lo=Infinity, hi=-Infinity;
      for(const r of data.rows) {const v=r.values[d.name];if(v!==null){lo=Math.min(lo,v);hi=Math.max(hi,v);}}
      if(!Number.isFinite(lo) || !Number.isFinite(hi)){lo=0;hi=1;}
      if(lo===hi){lo-=.5;hi+=.5;}
      return [lo,hi];
    });
    const ns='http://www.w3.org/2000/svg';
    const elem=(tag,attrs={})=>{const e=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,String(v));return e;};
    const svg=elem('svg',{viewBox:`0 0 ${width} ${height}`,width,height,'aria-label':'Parallel coordinates with draggable axis brushes'});
    svg.style.touchAction='none'; host.appendChild(svg);
    const x=i=>left+(right-left)*(dims.length===1?.5:i/(dims.length-1));
    const y=(i,v)=>bottom-(v-bounds[i][0])/(bounds[i][1]-bounds[i][0])*(bottom-top);
    const clippedY=raw=>Math.max(top,Math.min(bottom,raw));
    const makeLine=(r,foreground)=>{
      let path='',segment=false;
      dims.forEach((d,i)=>{const v=r.values[d.name];if(v===null){segment=false;return;}path+=`${segment?'L':'M'}${x(i).toFixed(1)},${y(i,v).toFixed(1)}`;segment=true;});
      const line=elem('path',{d:path,fill:'none',stroke:foreground?colors.selected:(data.has_pareto?(r.pareto?colors.front:colors.other):colors.all),'stroke-width':foreground?2.6:1.1,'stroke-opacity':foreground?.9:.18});
      const title=elem('title');title.textContent=r.display_id;line.appendChild(title);
      line.style.cursor='pointer';line.addEventListener('click',()=>setSelection(isSelected(r.id_key)?state.selected.filter(v=>v!==r.id_key):[...state.selected,r.id_key],true));
      svg.appendChild(line);
    };
    data.rows.filter(r=>!isSelected(r.id_key)).forEach(r=>makeLine(r,false));
    data.rows.filter(r=>isSelected(r.id_key)).forEach(r=>makeLine(r,true));
    dims.forEach((d,i)=>{
      const xx=x(i), [lo,hi]=bounds[i];
      const axis=elem('line',{x1:xx,y1:top,x2:xx,y2:bottom,stroke:'#6e8792','stroke-width':1.2});svg.appendChild(axis);
      const topText=elem('text',{x:xx,y:21,'text-anchor':'middle',fill:'#526d81','font-size':11,'font-weight':600});
      const words=displayName(d.name).split(' ');let lines=[''];
      for(const w of words){if((lines.at(-1)+' '+w).trim().length>15&&lines.at(-1))lines.push(w);else lines[lines.length-1]=(lines.at(-1)+' '+w).trim();}
      lines.slice(0,3).forEach((line,j)=>{const t=elem('tspan',{x:xx,dy:j?14:0});t.textContent=line.length>16?line.slice(0,15)+'…':line;topText.appendChild(t);});
      const tt=elem('title');tt.textContent=d.name;topText.appendChild(tt);svg.appendChild(topText);
      for(const [v,yy] of [[hi,top],[lo,bottom]]){const tick=elem('text',{x:xx+5,y:yy+3,fill:'#8196a5','font-size':10});tick.textContent=d.kind==='category'?(d.categories[Math.round(v)]||''):Number(v).toLocaleString(undefined,{maximumSignificantDigits:3});svg.appendChild(tick);}
      const c=state.constraints[d.name];
      if(c && Array.isArray(c) && !Array.isArray(c[0])){const ya=y(i,c[0]),yb=y(i,c[1]);svg.appendChild(elem('rect',{x:xx-11,y:Math.min(ya,yb),width:22,height:Math.abs(ya-yb),fill:'#f2a430','fill-opacity':.24,stroke:'#df8e13'}));}
      const hit=elem('rect',{x:xx-11,y:top,width:22,height:bottom-top,fill:'transparent','data-brush-dimension':d.name});hit.style.cursor='crosshair';
      let start=null,preview=null;
      const pos=e=>{const box=svg.getBoundingClientRect();return clippedY((e.clientY-box.top)*height/box.height);};
      hit.addEventListener('pointerdown',e=>{start=pos(e);preview=elem('rect',{x:xx-10,y:start,width:20,height:0,fill:colors.selected,'fill-opacity':.22,'pointer-events':'none'});svg.appendChild(preview);try{hit.setPointerCapture(e.pointerId);}catch(_){/* Synthetic events have no active pointer. */}e.preventDefault();});
      hit.addEventListener('pointermove',e=>{if(start===null||!preview)return;const end=pos(e);preview.setAttribute('y',Math.min(start,end));preview.setAttribute('height',Math.abs(end-start));});
      hit.addEventListener('pointercancel',()=>{start=null;if(preview)preview.remove();preview=null;});
      hit.addEventListener('pointerup',e=>{if(start===null)return;const finish=pos(e);if(preview)preview.remove();preview=null;if(Math.abs(finish-start)<4){start=null;return;}
        const toValue=yy=>lo+(bottom-yy)/(bottom-top)*(hi-lo);
        state.constraints[d.name]=[Math.min(toValue(start),toValue(finish)),Math.max(toValue(start),toValue(finish))];start=null;
        setSelection(brushedIds(),true);drawParallelSvg();e.preventDefault();});
      hit.addEventListener('dblclick',e=>{state.constraints[d.name]=null;setSelection(brushedIds(),true);drawParallelSvg();e.preventDefault();});
      svg.appendChild(hit);
    });
  }

  function inside(v,c) {
    if(v===null) return false;
    if(Array.isArray(c?.[0])) return c.some(range=>v>=range[0]&&v<=range[1]);
    return Array.isArray(c) && v>=c[0] && v<=c[1];
  }
  function brushedIds() {
    const active=Object.entries(state.constraints).filter(([,range])=>range!==null && range!==undefined);
    if(!active.length) return [];
    return state.data.rows.filter(r=>active.every(([name,range])=>inside(r.values[name],range))).map(r=>r.id_key);
  }
  function parallelLine() {
    const base=state.data.has_pareto?colors.other:colors.all;
    // Numeric categories keep selected lines visibly distinct in WebGL parcoords.
    return {color:state.data.rows.map(r=>isSelected(r.id_key)?2:(r.pareto?1:0)),
      cmin:0,cmax:2,showscale:false,
      colorscale:[[0,base],[.2499,base],[.25,colors.front],[.7499,colors.front],[.75,colors.selected],[1,colors.selected]]};
  }
  function updateParallelSelection() {
    if(!state.webgl || state.view!=='frontier')return;
    const plot=$('parallel');
    if(plot.data?.[0]?.type!=='parcoords')return;
    // Updating line.color alone preserves all brush constraints. Its restyle
    // event is ignored by the constraintrange-only handler below.
    Plotly.restyle(plot,{'line.color':[parallelLine().color]},[0]);
  }
  async function drawParallel() {
    if(!state.data)return;
    if(!state.webgl) { drawParallelSvg(); return; }
    const ds=state.data.dimensions.map(d=>{
      const dim={label:label(shortName(d.name,18)),values:state.data.rows.map(r=>r.values[d.name])};
      if(d.kind==='category') { dim.tickvals=d.categories.map((_,i)=>i); dim.ticktext=d.categories.map(label); dim.range=[-.25,Math.max(.75,d.categories.length-.75)]; }
      if(state.constraints[d.name]!==undefined)dim.constraintrange=state.constraints[d.name];
      return dim;
    });
    const line=parallelLine();
    const traces=ds.length?[{type:'parcoords',dimensions:ds,line,labelfont:{size:11,color:'#405a67'},tickfont:{size:9,color:'#8297a0'}}]:[];
    $('parallelHint').textContent=`${ds.length} 个维度 · 横向滚动查看全部；拖拽轴范围刷选。`;
    $('parallelHint').title=state.data.dimensions.map(d=>d.name).join(' · ');
    await Plotly.react($('parallel'),traces,{width:Math.max($('parallel').clientWidth,ds.length*108+40),height:Math.max($('parallel').clientHeight-16,180),margin:{l:58,r:58,t:62,b:30},paper_bgcolor:'#fff',font:{family:'Segoe UI, Arial, sans-serif'},uirevision:state.data.version},{displaylogo:false,responsive:false,displayModeBar:false});
    updateParallelSelection();
    if(!state.parallelEvents) {
      state.parallelEvents=true;
      $('parallel').on('plotly_restyle',updates=>{
        const change=updates?.[0]||{};
        if(!Object.keys(change).some(k=>k.includes('constraintrange')))return;
        const dims=$('parallel').data?.[0]?.dimensions||[];
        dims.forEach((dim,i)=>{const name=state.data.dimensions[i]?.name; if(name) state.constraints[name]=dim.constraintrange??null;});
        setSelection(brushedIds(),true);
      });
    }
  }
  function setSelection(ids,send) {
    if(!state.data)return;
    const allowed=new Set(state.data.rows.map(r=>r.id_key));
    state.selected=[...new Set(ids)].filter(k=>typeof k==='string'&&allowed.has(k));
    $('selected').textContent=String(state.selected.length);
    $('clear').disabled=!state.selected.length && !Object.values(state.constraints).some(Boolean);
    if(state.view==='frontier') {drawScatter();if(state.webgl)updateParallelSelection();else drawParallelSvg();}
    else renderDetail();
    if(send && state.bridge) state.bridge.reportSelection(JSON.stringify({version:state.data.version,keys:state.selected}));
  }

  const svgNS='http://www.w3.org/2000/svg';
  const sv=(tag,attrs={})=>{const e=document.createElementNS(svgNS,tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,String(v));return e;};
  const addText=(parent,value,attrs={})=>{const e=sv('text',attrs);e.textContent=String(value);parent.appendChild(e);return e;};
  const addTitle=(parent,value)=>{const e=sv('title');e.textContent=String(value);parent.appendChild(e);};
  function makeSvg(host,width,height) {
    host.replaceChildren();
    const svg=sv('svg',{width,height,viewBox:`0 0 ${width} ${height}`});
    svg.style.fontFamily='Arial, sans-serif';host.appendChild(svg);return svg;
  }
  function range(name) {
    let lo=Infinity,hi=-Infinity;
    for(const r of state.data.rows) {const v=r.values[name];if(v!==null){lo=Math.min(lo,v);hi=Math.max(hi,v);}}
    if(!Number.isFinite(lo)){lo=0;hi=1;}
    if(lo===hi){lo-=.5;hi+=.5;}
    return [lo,hi];
  }
  function selectToggle(key) {setSelection(isSelected(key)?state.selected.filter(v=>v!==key):[...state.selected,key],true);}
  function emptyDetail(host,message) {host.replaceChildren();const d=document.createElement('div');d.className='detail-empty';d.textContent=message;host.appendChild(d);}
  function drawMatrix() {
    const host=$('matrixChart'), names=state.data.objectives.slice(0,4), n=names.length;
    $('matrixNote').textContent=`两两散点：显示前 ${n}/${state.data.objectives.length} 个目标、全部 ${state.data.count} 个方案（无行抽样）；点选联动。`;
    if(!state.data.rows.length){emptyDetail(host,'暂无候选方案。');return;}
    const cell=Math.max(135,Math.min(230,Math.floor((host.clientHeight-65)/n),Math.floor((host.clientWidth-105)/n)));
    const pad=80,width=pad+n*cell+8,height=pad+n*cell+12;
    const svg=makeSvg(host,width,height), ranges=names.map(range);
    for(let i=0;i<n;i++)for(let j=0;j<n;j++) {
      const xx=pad+j*cell,yy=pad+i*cell,inner=cell-16;
      svg.appendChild(sv('rect',{x:xx,y:yy,width:cell-5,height:cell-5,fill:'#fff',stroke:'#e3eaed'}));
      if(i===j) {
        const title=addText(svg,shortName(names[i],Math.floor(cell/6.5)),{x:xx+9,y:yy+22,fill:'#345965','font-size':11,'font-weight':600});addTitle(title,names[i]);
        const bins=12,count=Array(bins).fill(0),[lo,hi]=ranges[i];
        for(const r of state.data.rows){const v=r.values[names[i]];if(v!==null)count[Math.min(bins-1,Math.floor((v-lo)/(hi-lo)*bins))]++;}
        const max=Math.max(1,...count);
        count.forEach((c,b)=>svg.appendChild(sv('rect',{x:xx+8+b*inner/bins,y:yy+cell-15-c/max*(cell-53),width:inner/bins-1,height:c/max*(cell-53),fill:'#bbd9d6'})));
      } else {
        const [xlo,xhi]=ranges[j],[ylo,yhi]=ranges[i];
        for(const r of [...state.data.rows.filter(r=>!isSelected(r.id_key)),...state.data.rows.filter(r=>isSelected(r.id_key))]){const xv=r.values[names[j]],yv=r.values[names[i]];if(xv===null||yv===null)continue;
          const selected=isSelected(r.id_key);
          const circle=sv('circle',{cx:xx+8+(xv-xlo)/(xhi-xlo)*(cell-23),cy:yy+cell-13-(yv-ylo)/(yhi-ylo)*(cell-24),r:selected?3.5:2.2,fill:selected?colors.selected:(state.data.has_pareto?(r.pareto?colors.front:colors.other):colors.all),'fill-opacity':selected?1:(state.selected.length?.23:.62),'data-candidate-key':r.id_key});
          circle.style.cursor='pointer';addTitle(circle,`${r.display_id} · ${names[j]}: ${formatNumber(xv)} · ${names[i]}: ${formatNumber(yv)}`);
          circle.addEventListener('click',()=>selectToggle(r.id_key));svg.appendChild(circle);
        }
      }
    }
    names.forEach((name,i)=>{const top=addText(svg,shortName(name,Math.floor(cell/6.5)),{x:pad+i*cell+5,y:60,fill:'#5b747e','font-size':10});addTitle(top,name);
      const side=addText(svg,shortName(name,Math.floor(cell/6)),{x:25,y:pad+i*cell+cell/2,'text-anchor':'middle',transform:`rotate(-90 25 ${pad+i*cell+cell/2})`,fill:'#5b747e','font-size':10});addTitle(side,name);});
  }
  function drawDistributions() {
    const host=$('distributionChart'), names=state.data.objectives;
    if(!state.data.rows.length){emptyDetail(host,'暂无候选方案。');return;}
    const width=Math.max(430,host.clientWidth-38),h=132,height=names.length*h+10;
    const svg=makeSvg(host,width,height),left=58,right=width-38;
    names.forEach((name,i)=>{
      const y=i*h,[lo,hi]=range(name),bins=14,count=Array(bins).fill(0),members=Array.from({length:bins},()=>[]);
      for(const r of state.data.rows){const v=r.values[name];if(v===null)continue;const b=Math.min(bins-1,Math.floor((v-lo)/(hi-lo)*bins));count[b]++;members[b].push(r.id_key);}
      const max=Math.max(1,...count),base=y+108;
      const heading=addText(svg,displayName(name),{x:left,y:y+16,fill:'#345965','font-size':12,'font-weight':600});addTitle(heading,name);
      svg.appendChild(sv('line',{x1:left,y1:base,x2:right,y2:base,stroke:'#b8c9cf'}));
      for(let b=0;b<bins;b++){const barW=(right-left)/bins,barH=count[b]/max*66;
        const bar=sv('rect',{x:left+b*barW+1,y:base-barH,width:Math.max(1,barW-2),height:barH,fill:'#9acac6','fill-opacity':state.selected.length?.52:.82,'data-hist-bin':`${i}-${b}`,'data-count':count[b]});
        bar.style.cursor='pointer';addTitle(bar,`${name}: ${formatNumber(lo+(hi-lo)*b/bins)}–${formatNumber(lo+(hi-lo)*(b+1)/bins)} · ${count[b]} 个方案`);
        bar.addEventListener('click',()=>setSelection(members[b],true));svg.appendChild(bar);
      }
      for(const key of state.selected){const r=state.data.rows.find(v=>v.id_key===key);if(!r)continue;const v=r.values[name];if(v===null)continue;
        const xx=left+(v-lo)/(hi-lo)*(right-left),marker=sv('line',{x1:xx,y1:y+27,x2:xx,y2:base,stroke:colors.selected,'stroke-width':2,'stroke-opacity':.9});
        addTitle(marker,`${r.display_id} · ${name}: ${formatNumber(v)}`);svg.appendChild(marker);
      }
      addText(svg,formatNumber(lo),{x:left,y:base+14,fill:'#879aa2','font-size':9});
      addText(svg,formatNumber(hi),{x:right,y:base+14,fill:'#879aa2','font-size':9,'text-anchor':'end'});
    });
  }
  function drawComparison() {
    const host=$('comparisonChart'),keys=state.selected.slice(0,8),palette=[colors.selected,'#427ca5','#408f88','#8a6faa','#c36970','#62915f','#9d7161','#617f8a'];
    $('comparisonNote').textContent='每个目标保留原始单位；横线为全部方案范围，右侧显示已选方案的精确值。'+(state.selected.length>8?` 显示前 8/${state.selected.length} 个选择，表格与导出保留全部。`:'');
    if(!keys.length){emptyDetail(host,'先在目标空间、矩阵、分布或结果表中选择方案。');return;}
    const legend=document.createElement('div');legend.className='detail-legend';
    keys.forEach((key,i)=>{const r=state.data.rows.find(v=>v.id_key===key);if(!r)return;const e=document.createElement('span'),dot=document.createElement('i');dot.style.background=palette[i];e.appendChild(dot);e.appendChild(document.createTextNode(r.display_id));legend.appendChild(e);});
    host.replaceChildren();host.appendChild(legend);
    const width=Math.max(460,host.clientWidth-38),rowH=64+keys.length*24,height=state.data.objectives.length*rowH+5,left=90,right=width-115;
    const svg=sv('svg',{width,height,viewBox:`0 0 ${width} ${height}`});svg.style.fontFamily='Segoe UI, Arial, sans-serif';host.appendChild(svg);
    state.data.objectives.forEach((name,i)=>{
      const y=i*rowH,[lo,hi]=range(name);
      const heading=addText(svg,displayName(name),{x:left,y:y+18,fill:'#345965','font-size':12,'font-weight':600});addTitle(heading,name);
      keys.forEach((key,j)=>{
        const r=state.data.rows.find(v=>v.id_key===key);if(!r)return;const v=r.values[name],yy=y+43+j*24;
        svg.appendChild(sv('line',{x1:left,y1:yy,x2:right,y2:yy,stroke:'#e1e8ee','stroke-width':1.4}));
        const id=addText(svg,r.display_id.length>12?r.display_id.slice(0,11)+'…':r.display_id,{x:left-12,y:yy+4,'text-anchor':'end',fill:palette[j],'font-size':10});addTitle(id,r.display_id);
        if(v===null){addText(svg,'—',{x:right+16,y:yy+4,fill:'#8295a3','font-size':11});return;}
        const xx=left+(v-lo)/(hi-lo)*(right-left),c=sv('circle',{cx:xx,cy:yy,r:5,fill:palette[j],stroke:'#fff','stroke-width':1.3});
        addTitle(c,`${r.display_id} · ${name}: ${formatNumber(v)}`);svg.appendChild(c);
        addText(svg,formatNumber(v),{x:right+16,y:yy+4,fill:'#36566e','font-size':11,'font-weight':600});
      });
      addText(svg,formatNumber(lo),{x:left,y:y+rowH-7,fill:'#879aa2','font-size':10});
      addText(svg,formatNumber(hi),{x:right,y:y+rowH-7,fill:'#879aa2','font-size':10,'text-anchor':'end'});
    });
  }
  function drawCorrelation() {
    const host=$('correlationChart'), rows=state.data.correlations||[],targets=state.data.objectives;
    const excluded=state.data.excluded_inputs||[];
    $('correlationNote').textContent=`当前 ${state.data.count} 个候选方案的 Pearson r，仅描述线性关联，不代表因果关系。`+(excluded.length?` 排除 ${excluded.length} 个非数值输入。`:'')+(state.selected.length?` 已选 ${state.selected.length} 个，系数仍基于全部方案。`:'');
    if(!rows.length){emptyDetail(host,'没有可计算相关性的数值输入变量。');return;}
    const left=205,cell=Math.min(180,Math.max(115,(host.clientWidth-left-55)/targets.length)),rowH=42,width=left+targets.length*cell+20,height=78+rows.length*rowH+12;
    const svg=makeSvg(host,width,height);
    targets.forEach((name,j)=>{const t=addText(svg,shortName(name,Math.floor(cell/6.5)),{x:left+j*cell+cell/2,y:35,'text-anchor':'middle',fill:'#55727d','font-size':10,'font-weight':600});addTitle(t,name);});
    rows.forEach((row,i)=>{const y=68+i*rowH;
      const t=addText(svg,shortName(row.input,28),{x:8,y:y+25,fill:'#365764','font-size':10});addTitle(t,row.input);
      row.cells.forEach((c,j)=>{const x=left+j*cell,r=c.r;
        const fill=r===null?'#f2f5f6':r>=.6?'#80beb8':r>=.2?'#d0eae6':r<=-.6?'#9ebfdb':r<=-.2?'#dce9f3':'#f3f6f6';
        const cellRect=sv('rect',{x,y,width:cell-4,height:rowH-4,fill,stroke:'#e6ecee'});addTitle(cellRect,`${row.input} × ${c.objective}: ${r===null?'未定义':r.toFixed(3)} (n=${c.n})`);svg.appendChild(cellRect);
        addText(svg,r===null?'—':r.toFixed(2),{x:x+(cell-4)/2,y:y+25,'text-anchor':'middle',fill:'#264451','font-size':11,'font-weight':600});
      });
    });
  }
  function renderDetail() {
    if(!state.data)return;
    if(state.view==='matrix')drawMatrix();
    else if(state.view==='distribution')drawDistributions();
    else if(state.view==='comparison')drawComparison();
    else if(state.view==='correlation')drawCorrelation();
  }
  function showView(view) {
    state.view=view;
    for(const name of ['frontier','matrix','distribution','comparison','correlation']) {
      $(name+'View').hidden=name!==view;
      document.querySelector(`[data-analysis-view="${name}"]`).classList.toggle('active',name===view);
    }
    if(!state.data)return;
    if(view==='frontier'){drawScatter();drawParallel();}
    else renderDetail();
  }

  function newData(payload) {
    state.data=payload; state.constraints={};
    state.selected=[];
    $('total').textContent=String(payload.count);
    $('selected').textContent='0';
    $('scope').textContent=`显示全部 ${payload.count} 个方案（无抽样）；图中选择不会缩小表格或导出范围。`;
    fillAxes(); legend(); showView(state.view);
  }
  for(const button of document.querySelectorAll('[data-analysis-view]')) button.addEventListener('click',()=>showView(button.dataset.analysisView));
  window.addEventListener('resize',()=>{if(state.view==='frontier') {drawScatter();drawParallel();}else renderDetail();});
  for(const id of ['axisX','axisY','axisZ']) $(id).addEventListener('change',()=>{state.x=$('axisX').value;state.y=$('axisY').value;state.z=$('axisZ').value;drawScatter();});
  for(const [id,mode] of [['mode2','2d'],['mode3','3d']]) $(id).addEventListener('click',()=>{if(mode==='3d'&&state.data?.objectives.length<3)return;state.mode=mode;fillAxes();drawScatter();});
  for(const [id,tool] of [['toolLasso','lasso'],['toolBox','select'],['toolZoom','zoom'],['toolPan','pan']]) $(id).addEventListener('click',()=>{state.tool=tool;for(const other of ['toolLasso','toolBox','toolZoom','toolPan'])$(other).classList.toggle('active',other===id);drawScatter();});
  $('clear').addEventListener('click',()=>{state.constraints={};drawParallel();setSelection([],true);});
  if(typeof qt!=='undefined' && typeof QWebChannel!=='undefined') new QWebChannel(qt.webChannelTransport,channel=>{
    state.bridge=channel.objects.plotBridge;
    state.bridge.payloadChanged.connect(raw=>newData(JSON.parse(raw)));
    state.bridge.selectionChanged.connect(raw=>{const msg=JSON.parse(raw);if(state.data&&msg.version===state.data.version)setSelection(msg.keys,false);});
    state.bridge.ready();
  });
})();
