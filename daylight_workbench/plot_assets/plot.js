'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const colors = {front:'#0b9b91', other:'#98abb7', all:'#3d8eb7', selected:'#f2a430'};
  const state = {data:null, selected:[], mode:'2d', x:null, y:null, z:null, constraints:{}, bridge:null, scatterEvents:false, parallelEvents:false, webgl:false, view:'frontier'};
  try { const c=document.createElement('canvas'); state.webgl=!!(c.getContext('webgl2')||c.getContext('webgl')); } catch (_) {}
  const escapeHtml = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const isSelected = key => state.selected.includes(key);
  const formatNumber = v => v === null || v === undefined ? '—' : Number(v).toLocaleString(undefined,{maximumSignificantDigits:7});
  const label = s => escapeHtml(s);
  function fillAxes() {
    const names = state.data.objectives;
    for (const [node, preferred, fallback] of [[$('axisX'),state.x,0],[$('axisY'),state.y,1],[$('axisZ'),state.z,2]]) {
      node.replaceChildren();
      names.forEach(name => { const o=document.createElement('option'); o.value=name; o.textContent=name; node.appendChild(o); });
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
    const axis={title:{font:{size:11,color:'#5d7483'}},gridcolor:'#e5edf0',zerolinecolor:'#dbe6e9',linecolor:'#bacdd4',tickfont:{size:10,color:'#7e929a'},automargin:true};
    const layout={margin:{l:58,r:20,b:48,t:18},paper_bgcolor:'#fff',plot_bgcolor:'#fff',showlegend:false,hovermode:'closest',dragmode:state.mode==='3d'?'orbit':'lasso',font:{family:'Arial, sans-serif'},uirevision:state.data.version};
    if(state.mode==='3d') { layout.scene={xaxis:{...axis,title:{...axis.title,text:label(state.x)}},yaxis:{...axis,title:{...axis.title,text:label(state.y)}},zaxis:{...axis,title:{...axis.title,text:label(state.z)}},bgcolor:'#fff',aspectmode:'cube'}; layout.margin={l:0,r:0,b:0,t:5}; }
    else { layout.xaxis={...axis,title:{...axis.title,text:label(state.x)}}; layout.yaxis={...axis,title:{...axis.title,text:label(state.y)}}; }
    await Plotly.react($('scatter'),traces,layout,{displaylogo:false,responsive:true,modeBarButtonsToRemove:['toImage','sendDataToCloud','select2d'],scrollZoom:false});
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
    const width=Math.max(host.clientWidth,300), height=Math.max(host.clientHeight,200);
    const top=54, bottom=height-38, left=45, right=width-35;
    const dims=data.dimensions;
    if(!dims.length)return;
    const bounds=dims.map(d=>{
      let lo=Infinity, hi=-Infinity;
      for(const r of data.rows) {const v=r.values[d.name];if(v!==null){lo=Math.min(lo,v);hi=Math.max(hi,v);}}
      if(!Number.isFinite(lo) || !Number.isFinite(hi)){lo=0;hi=1;}
      if(lo===hi){lo-=.5;hi+=.5;}
      return [lo,hi];
    });
    const ns='http://www.w3.org/2000/svg';
    const elem=(tag,attrs={})=>{const e=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,String(v));return e;};
    const svg=elem('svg',{viewBox:`0 0 ${width} ${height}`,width:'100%',height:'100%','aria-label':'Parallel coordinates with draggable axis brushes'});
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
      const topText=elem('text',{x:xx,y:top-15,'text-anchor':'middle',fill:'#54707a','font-size':10,'font-weight':600});
      topText.textContent=d.name.length>12?d.name.slice(0,11)+'…':d.name;const tt=elem('title');tt.textContent=d.name;topText.appendChild(tt);svg.appendChild(topText);
      for(const [v,yy] of [[hi,top],[lo,bottom]]){const tick=elem('text',{x:xx+4,y:yy+3,fill:'#9badb3','font-size':9});tick.textContent=d.kind==='category'?(d.categories[Math.round(v)]||''):Number(v).toLocaleString(undefined,{maximumSignificantDigits:3});svg.appendChild(tick);}
      const c=state.constraints[d.name];
      if(c && Array.isArray(c) && !Array.isArray(c[0])){const ya=y(i,c[0]),yb=y(i,c[1]);svg.appendChild(elem('rect',{x:xx-11,y:Math.min(ya,yb),width:22,height:Math.abs(ya-yb),fill:'#f2a430','fill-opacity':.24,stroke:'#df8e13'}));}
      const hit=elem('rect',{x:xx-11,y:top,width:22,height:bottom-top,fill:'transparent','data-brush-dimension':d.name});hit.style.cursor='crosshair';
      let start=null;
      const pos=e=>{const box=svg.getBoundingClientRect();return clippedY((e.clientY-box.top)*height/box.height);};
      hit.addEventListener('pointerdown',e=>{start=pos(e);try{hit.setPointerCapture(e.pointerId);}catch(_){/* Synthetic events have no active pointer. */}e.preventDefault();});
      hit.addEventListener('pointerup',e=>{if(start===null)return;const finish=pos(e);if(Math.abs(finish-start)<4){start=null;return;}
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
      const dim={label:label(d.name),values:state.data.rows.map(r=>r.values[d.name])};
      if(d.kind==='category') { dim.tickvals=d.categories.map((_,i)=>i); dim.ticktext=d.categories.map(label); dim.range=[-.25,Math.max(.75,d.categories.length-.75)]; }
      if(state.constraints[d.name]!==undefined)dim.constraintrange=state.constraints[d.name];
      return dim;
    });
    const line=parallelLine();
    const traces=ds.length?[{type:'parcoords',dimensions:ds,line,labelfont:{size:11,color:'#405a67'},tickfont:{size:9,color:'#8297a0'}}]:[];
    await Plotly.react($('parallel'),traces,{margin:{l:35,r:35,t:42,b:26},paper_bgcolor:'#fff',font:{family:'Arial, sans-serif'},uirevision:state.data.version},{displaylogo:false,responsive:true,displayModeBar:false});
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
    const cell=Math.max(120,Math.min(190,Math.floor((host.clientWidth-55)/n)));
    const pad=42,width=pad+n*cell+8,height=pad+n*cell+12;
    const svg=makeSvg(host,width,height), ranges=names.map(range);
    for(let i=0;i<n;i++)for(let j=0;j<n;j++) {
      const xx=pad+j*cell,yy=pad+i*cell,inner=cell-16;
      svg.appendChild(sv('rect',{x:xx,y:yy,width:cell-5,height:cell-5,fill:'#fff',stroke:'#e3eaed'}));
      if(i===j) {
        addText(svg,names[i],{x:xx+9,y:yy+22,fill:'#345965','font-size':11,'font-weight':600});
        const bins=12,count=Array(bins).fill(0),[lo,hi]=ranges[i];
        for(const r of state.data.rows){const v=r.values[names[i]];if(v!==null)count[Math.min(bins-1,Math.floor((v-lo)/(hi-lo)*bins))]++;}
        const max=Math.max(1,...count);
        count.forEach((c,b)=>svg.appendChild(sv('rect',{x:xx+8+b*inner/bins,y:yy+cell-15-c/max*(cell-53),width:inner/bins-1,height:c/max*(cell-53),fill:'#bbd9d6'})));
      } else {
        const [xlo,xhi]=ranges[j],[ylo,yhi]=ranges[i];
        for(const r of state.data.rows){const xv=r.values[names[j]],yv=r.values[names[i]];if(xv===null||yv===null)continue;
          const selected=isSelected(r.id_key);
          const circle=sv('circle',{cx:xx+8+(xv-xlo)/(xhi-xlo)*(cell-23),cy:yy+cell-13-(yv-ylo)/(yhi-ylo)*(cell-24),r:selected?3.5:2.2,fill:selected?colors.selected:(state.data.has_pareto?(r.pareto?colors.front:colors.other):colors.all),'fill-opacity':selected?1:(state.selected.length?.23:.62),'data-candidate-key':r.id_key});
          circle.style.cursor='pointer';addTitle(circle,`${r.display_id} · ${names[j]}: ${formatNumber(xv)} · ${names[i]}: ${formatNumber(yv)}`);
          circle.addEventListener('click',()=>selectToggle(r.id_key));svg.appendChild(circle);
        }
      }
    }
    names.forEach((name,i)=>{addText(svg,name,{x:pad+i*cell+4,y:28,fill:'#5b747e','font-size':10});
      addText(svg,name,{x:5,y:pad+i*cell+15,fill:'#5b747e','font-size':10});});
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
      addText(svg,name,{x:left,y:y+16,fill:'#345965','font-size':11,'font-weight':600});
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
    const host=$('comparisonChart'), keys=state.selected.slice(0,8), palette=['#df9223','#087f79','#507fb5','#8a6faa','#c36970','#62915f','#9d7161','#617f8a'];
    $('comparisonNote').textContent=state.selected.length>8?`每个目标使用原始单位；显示前 8/${state.selected.length} 个已选方案，表格与导出仍保留全部。`:'每个目标单独使用原始单位与刻度；灰线为全部方案范围。';
    if(!keys.length){emptyDetail(host,'先在目标空间、矩阵、分布或结果表中选择方案。');return;}
    const legend=document.createElement('div');legend.className='detail-legend';
    keys.forEach((key,i)=>{const r=state.data.rows.find(v=>v.id_key===key);if(!r)return;const e=document.createElement('span'),dot=document.createElement('i');dot.style.background=palette[i];e.appendChild(dot);e.appendChild(document.createTextNode(r.display_id));legend.appendChild(e);});
    host.replaceChildren();host.appendChild(legend);
    const width=Math.max(450,host.clientWidth-38),rowH=82,height=state.data.objectives.length*rowH+12,left=70,right=width-60;
    const svg=sv('svg',{width,height,viewBox:`0 0 ${width} ${height}`});svg.style.fontFamily='Arial, sans-serif';host.appendChild(svg);
    state.data.objectives.forEach((name,i)=>{const y=i*rowH,[lo,hi]=range(name),yy=y+47;
      addText(svg,name,{x:left,y:y+17,fill:'#345965','font-size':11,'font-weight':600});
      svg.appendChild(sv('line',{x1:left,y1:yy,x2:right,y2:yy,stroke:'#b6c8ce','stroke-width':2}));
      keys.forEach((key,j)=>{const r=state.data.rows.find(v=>v.id_key===key);if(!r)return;const v=r.values[name];if(v===null)return;
        const xx=left+(v-lo)/(hi-lo)*(right-left),c=sv('circle',{cx:xx,cy:yy+(j%3-1)*5,r:5,fill:palette[j],stroke:'#fff','stroke-width':1.2});
        addTitle(c,`${r.display_id} · ${name}: ${formatNumber(v)}`);svg.appendChild(c);
      });
      addText(svg,formatNumber(lo),{x:left,y:yy+22,fill:'#879aa2','font-size':9});
      addText(svg,formatNumber(hi),{x:right,y:yy+22,fill:'#879aa2','font-size':9,'text-anchor':'end'});
    });
  }
  function drawCorrelation() {
    const host=$('correlationChart'), rows=state.data.correlations||[],targets=state.data.objectives;
    const excluded=state.data.excluded_inputs||[];
    $('correlationNote').textContent=`当前 ${state.data.count} 个候选方案的 Pearson r，仅描述线性关联，不代表因果关系。`+(excluded.length?` 排除 ${excluded.length} 个非数值输入。`:'')+(state.selected.length?` 已选 ${state.selected.length} 个，系数仍基于全部方案。`:'');
    if(!rows.length){emptyDetail(host,'没有可计算相关性的数值输入变量。');return;}
    const left=170,cell=100,rowH=39,width=left+targets.length*cell+20,height=58+rows.length*rowH+12;
    const svg=makeSvg(host,width,height);
    targets.forEach((name,j)=>{const t=addText(svg,name.length>15?name.slice(0,14)+'…':name,{x:left+j*cell+cell/2,y:35,'text-anchor':'middle',fill:'#55727d','font-size':10,'font-weight':600});addTitle(t,name);});
    rows.forEach((row,i)=>{const y=50+i*rowH;
      const t=addText(svg,row.input.length>24?row.input.slice(0,23)+'…':row.input,{x:8,y:y+25,fill:'#365764','font-size':10});addTitle(t,row.input);
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
  window.addEventListener('resize',()=>{if(state.view==='frontier') {drawScatter();if(state.webgl)updateParallelSelection();else drawParallelSvg();}else renderDetail();});
  for(const id of ['axisX','axisY','axisZ']) $(id).addEventListener('change',()=>{state.x=$('axisX').value;state.y=$('axisY').value;state.z=$('axisZ').value;drawScatter();});
  for(const [id,mode] of [['mode2','2d'],['mode3','3d']]) $(id).addEventListener('click',()=>{if(mode==='3d'&&state.data?.objectives.length<3)return;state.mode=mode;fillAxes();drawScatter();});
  $('clear').addEventListener('click',()=>{state.constraints={};drawParallel();setSelection([],true);});
  if(typeof qt!=='undefined' && typeof QWebChannel!=='undefined') new QWebChannel(qt.webChannelTransport,channel=>{
    state.bridge=channel.objects.plotBridge;
    state.bridge.payloadChanged.connect(raw=>newData(JSON.parse(raw)));
    state.bridge.selectionChanged.connect(raw=>{const msg=JSON.parse(raw);if(state.data&&msg.version===state.data.version)setSelection(msg.keys,false);});
    state.bridge.ready();
  });
})();
