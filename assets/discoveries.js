const $ = id => document.getElementById(id);
const NS = 'http://www.w3.org/2000/svg';
const palette = ['#63caff','#ff758f','#ba97ff','#54dcc1','#ffcc69','#e58dff','#8be28e','#ffac79','#7c9bff','#ecb8a5'];
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
function svg(parent, tag, attrs = {}, text) {
  const el = document.createElementNS(NS, tag);
  for (const [key,value] of Object.entries(attrs)) el.setAttribute(key,value);
  if (text !== undefined) el.textContent = text;
  parent.append(el); return el;
}
function button(label, action) {const el = document.createElement('button'); el.type = 'button'; el.textContent = label; el.addEventListener('click',action); return el;}
function seededRandom(seed) {return () => {seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7,61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296;};}
const rng = seededRandom(42);
let graph, results, byId, adjacency, active, groups;
let view = 'root', focused = null, mappedNodes = new Map(), positions = new Map();
let displayedEdges = new Map();
let playing = false, visible = false, animation = null, walkers = [], particleLayer;
let selectedMotif = '300', exampleIndex = 0, motifPlotReady = false;
let motifDrawing = false, motifDirty = false;
const name = id => byId.get(id)?.name || id;
const initials = label => label.replace(/\([^)]*\)/g, '').trim().split(/\s+/)
  .map(word => word.match(/[\p{L}\p{N}]/u)?.[0] || '')
  .filter(Boolean).slice(0, 3).join('').toUpperCase();
const nodeColor = id => {
  const top = results.atlas.nodePaths[id]?.[0];
  return top === 'isolates' ? '#ffd43b' : palette[(Number(top?.split('-')[1]) - 1) % palette.length] || '#63caff';
};
function locate(ids,label) {document.dispatchEvent(new CustomEvent('ram:select-original-nodes',{detail:{ids,label}}));}
function lineCurve(a,b,offset = 14) {
  const dx = b.x-a.x, dy = b.y-a.y, length = Math.hypot(dx,dy) || 1;
  const start = {x:a.x+dx/length*(a.r+3),y:a.y+dy/length*(a.r+3)};
  const end = {x:b.x-dx/length*(b.r+5),y:b.y-dy/length*(b.r+5)};
  return `M${start.x},${start.y} Q${(a.x+b.x)/2-dy/length*offset},${(a.y+b.y)/2+dx/length*offset} ${end.x},${end.y}`;
}
function arrow(parent,id,color='#b5c9ec') {
  const defs = svg(parent,'defs');
  const marker = svg(defs,'marker',{id,viewBox:'0 0 10 10',refX:8,refY:5,markerWidth:5,markerHeight:5,orient:'auto-start-reverse'});
  svg(marker,'path',{d:'M0 0 L10 5 L0 10 Z',fill:color});
}

// Deterministic circle packing; the geometry encodes membership size, not distance.
function pack(items) {
  const placed = [];
  for (const item of [...items].sort((a,b) => b.r-a.r || a.id.localeCompare(b.id))) {
    let p;
    for (let step=0;step<12000;step++) {
      const radius = Math.sqrt(step)*7, angle = step*2.399963229728653;
      p = {...item,x:radius*Math.cos(angle),y:radius*Math.sin(angle)};
      if (placed.every(q => Math.hypot(p.x-q.x,p.y-q.y)>p.r+q.r+10)) break;
    }
    placed.push(p);
  }
  const minX = Math.min(...placed.map(p=>p.x-p.r)), maxX = Math.max(...placed.map(p=>p.x+p.r));
  const minY = Math.min(...placed.map(p=>p.y-p.r)), maxY = Math.max(...placed.map(p=>p.y+p.r));
  const scale = Math.min(1.7,820/(maxX-minX),475/(maxY-minY));
  return new Map(placed.map(p=>[p.id,{...p,x:450+(p.x-(minX+maxX)/2)*scale,y:280+(p.y-(minY+maxY)/2)*scale,r:p.r*scale}]));
}
function setAtlasView(id, focus = null) {
  view = id; focused = focus; stopAnimation();
  if (!focus) $('atlas-find').value = '';
  const group = groups[view];
  const items = group.children.length ? group.children.map(id => ({id,members:groups[id].members,group:true,r:Math.sqrt(groups[id].members.length)*12,color:nodeColor(groups[id].members[0])}))
    : group.members.map(id => ({id,members:[id],group:false,r:9+Math.sqrt(byId.get(id).incoming)*2,color:nodeColor(id)}));
  positions = pack(items); mappedNodes = new Map();
  for (const item of items) for (const node of item.members) mappedNodes.set(node,item.id);
  const root = $('atlas-svg'); root.replaceChildren(); arrow(root,'atlas-arrow');
  const scene = svg(root,'g',{'data-view':view});
  const links = new Map();
  displayedEdges = new Map();
  for (const [source,target] of graph.edges) {
    const a = mappedNodes.get(source), b = mappedNodes.get(target);
    if (a && b && a !== b) {const key = `${a}\t${b}`; links.set(key,(links.get(key)||0)+1);}
  }
  for (const [key,count] of links) {
    const [a,b] = key.split('\t');
    const path = svg(scene,'path',{d:lineCurve(positions.get(a),positions.get(b)),fill:'none',stroke:'#94b6e6',opacity:.18,'stroke-width':Math.min(4,.5+Math.sqrt(count)*.35),'data-source':a,'data-target':b});
    displayedEdges.set(key,{path,length:path.getTotalLength()});
    svg(path,'title',{},`${positions.get(a).group ? groups[a].label : name(a)} → ${positions.get(b).group ? groups[b].label : name(b)}: ${count} directed links`);
  }
  for (const p of positions.values()) {
    const label = p.group ? groups[p.id].label : name(p.id);
    const action = p.group ? groups[p.id].children.length ? `Open ${groups[p.id].children.length} subgroups` : 'Open individual characters' : 'Inspect character';
    const el = svg(scene,'g',{class:'atlas-node',transform:`translate(${p.x} ${p.y})`,tabindex:0,role:'button','aria-label':`${label}, ${p.members.length} character(s). ${action}`,'data-id':p.id});
    svg(el,'circle',{r:p.r,fill:p.color,'fill-opacity':p.id === focused ? .5 : .15,stroke:p.id === focused ? '#fff' : p.color,'stroke-width':p.id === focused ? 3 : 1.5});
    svg(el,'circle',{r:Math.max(1,p.r-6),fill:'none',stroke:p.color,'stroke-opacity':.15,'pointer-events':'none'});
    svg(el,'title',{},p.group ? `${label} · ${p.members.length} characters\n${action}\n${p.id==='isolates'?'Display group only; no simulated visits.':`Highest simulated visit probability: ${groups[p.id].leaders.map(name).join(', ')}`}` : `${label}\n${byId.get(p.id).incoming} incoming · ${byId.get(p.id).outgoing} outgoing`);
    if (p.group) {
      svg(el,'text',{'text-anchor':'middle',y:-8,'font-size':p.r<30?10:13},p.id==='isolates'?'Isolates':`C${p.id.slice(2).replaceAll('-','.')}`);
      svg(el,'text',{'text-anchor':'middle',y:15,class:'bubble-count'},p.members.length);
      if (p.r>48) svg(el,'text',{'text-anchor':'middle',y:34,class:'bubble-leader'},p.id==='isolates'?'display group':name(groups[p.id].leaders[0]).split(' (')[0].slice(0,18));
    } else if (p.r>14) svg(el,'text',{'text-anchor':'middle',y:5,'font-size':11},initials(name(p.id)));
    const click = () => p.group ? setAtlasView(p.id) : inspectCharacter(p.id);
    const highlight = on => {
      for (const edge of scene.querySelectorAll('path[data-source]')) {
        const incident = edge.dataset.source === p.id || edge.dataset.target === p.id;
        edge.setAttribute('opacity',on ? incident ? .75 : .04 : .18);
        if (on && incident) edge.setAttribute('marker-end','url(#atlas-arrow)');
        else edge.removeAttribute('marker-end');
      }
    };
    el.addEventListener('mouseenter',()=>highlight(true));el.addEventListener('mouseleave',()=>highlight(false));
    el.addEventListener('focus',()=>highlight(true));el.addEventListener('blur',()=>highlight(false));
    el.addEventListener('click',click);
    el.addEventListener('keydown',event => {if (event.key==='Enter'||event.key===' ') {event.preventDefault();click();}});
  }
  particleLayer = svg(root,'g',{'aria-hidden':'true','pointer-events':'none'});
  if (!reducedMotion.matches) scene.animate([{opacity:0,transform:'scale(.97)',transformOrigin:'450px 280px'},{opacity:1,transform:'scale(1)',transformOrigin:'450px 280px'}],{duration:300,easing:'ease-out'});
  $('atlas-up').disabled = !group.parent;
  $('atlas-breadcrumb').replaceChildren();
  const ancestry = []; for (let g=group;g;g=groups[g.parent]) ancestry.unshift(g);
  for (const g of ancestry) {$('atlas-breadcrumb').append(button(g.label,()=>setAtlasView(g.id)));}
  $('atlas-group').replaceChildren(new Option(group.children.length?'Choose a community':'Character level',''));
  for (const id of group.children) $('atlas-group').append(new Option(`${groups[id].label} · ${groups[id].members.length} members · ${groups[id].children.length ? `${groups[id].children.length} subgroups` : 'opens characters'}`,id));
  $('atlas-group').disabled = !group.children.length;
  $('atlas-status').textContent = `${group.members.length} characters · ${group.internalEdges} internal directed links · ${group.children.length ? `${group.children.length} groups in view` : 'individual characters'}. ${group.kind==='display-group'?'Isolates are grouped for display, not a detected community.':''}`;
  $('atlas-detail-title').textContent = group.label;
  $('atlas-detail-text').textContent = `${group.incomingEdges} links enter from outside; ${group.outgoingEdges} links leave. ${group.kind==='display-group'?'No links to follow.':`Highest simulated visit probability: ${group.leaders.map(name).join(', ')}.`}`;
  $('atlas-members').replaceChildren();
  for (const id of group.members) {const li=document.createElement('li');li.append(button(name(id),()=>findCharacter(id)));$('atlas-members').append(li);}
  $('atlas-flow').disabled = group.kind==='display-group';
  if (group.kind==='display-group') {playing=false;updatePlayButton();}
  if (focus) inspectCharacter(focus);
  updateSensitivity();
  resetWalkers(); startAnimation();
  root.dataset.ready = 'true'; root.dataset.view = view;
}
function inspectCharacter(id) {
  focused = id; $('atlas-find').value = id;
  for (const el of $('atlas-svg').querySelectorAll('.atlas-node')) {
    const circle = el.querySelector('circle');circle.setAttribute('stroke',el.dataset.id===id?'#fff':positions.get(el.dataset.id).color);circle.setAttribute('stroke-width',el.dataset.id===id?3:1.5);
  }
  const node=byId.get(id), bridge=results.atlas.bridges.find(n=>n.id===id);
  $('atlas-detail-title').textContent = name(id);
  $('atlas-detail-text').textContent = `${node.incoming} incoming · ${node.outgoing} outgoing · ${bridge?.crossNeighbors||0} distinct neighbors in ${bridge?.otherCommunities||0} other top-level communities.`;
  updateSensitivity();
}
function overlap(reference, alternative) {
  const a=new Set(reference),b=new Set(alternative),shared=reference.filter(id=>b.has(id));
  return {shared, lost:reference.filter(id=>!b.has(id)), gained:alternative.filter(id=>!a.has(id)), score:shared.length/(a.size+b.size-shared.length||1)};
}
function bestOverlap(reference, run) {
  return Object.entries(run.groups).map(([id,members])=>({id,...overlap(reference,members)})).sort((a,b)=>b.score-a.score||a.id.localeCompare(b.id))[0];
}
function updateSensitivity() {
  const run=results.atlas.sensitivity.find(r=>r.seed===Number($('atlas-seed').value));
  $('atlas-seed-summary').textContent=`Seed ${run.seed}: ${Object.keys(run.groups).length} top-level groups. Same-group pairs shared with the displayed run: ${(100*run.pairJaccard).toFixed(1)}% of the pairs grouped together in either run.`;
  const changes=$('atlas-seed-changes');changes.replaceChildren();
  if(view==='isolates'||(focused&&results.atlas.nodePaths[focused][0]==='isolates')){
    $('atlas-seed-detail').textContent='Isolates are excluded from community detection in every run. Their display group is not part of this comparison.';return;
  }
  if(view==='root'&&!focused){
    $('atlas-seed-detail').textContent=run.seed===42?'This is the reference run shown in the map. Choose another seed to inspect membership changes.':'Groups with the lowest overlap with their best-matching alternative group. Open one to see which members change.';
    // Preserve reference IDs: alternative group numbers are not comparable labels.
    const sorted=groups.root.children.filter(id=>id!=='isolates').map(id=>({reference:id,match:bestOverlap(groups[id].members,run)})).filter(item=>item.match.score<1).sort((a,b)=>a.match.score-b.match.score);
    const buttons=document.createElement('div');buttons.className='seed-buttons';
    for(const item of sorted.slice(0,5))buttons.append(button(`${groups[item.reference].label}: ${item.match.lost.length} leave, ${item.match.gained.length} join`,()=>setAtlasView(item.reference)));
    changes.append(buttons);return;
  }
  const top=results.atlas.nodePaths[focused||groups[view].members[0]][0];
  let match;
  if(focused){
    const alternative=Object.values(run.groups).find(members=>members.includes(focused));
    match=overlap(groups[top].members.filter(id=>id!==focused),alternative.filter(id=>id!==focused));
    $('atlas-seed-detail').textContent=`${name(focused)}: ${match.shared.length} companions stay in the same group, ${match.lost.length} leave, and ${match.gained.length} join when changing from seed 42 to ${run.seed}.`;
  }else{
    match=bestOverlap(groups[top].members,run);
    $('atlas-seed-detail').textContent=`Top-level ${groups[top].label}, matched by greatest member overlap: ${match.shared.length} members retained, ${match.lost.length} leave, ${match.gained.length} join. ${view!==top?'This compares its parent community, not the smaller subgroup currently open.':''}`;
  }
  for(const [label,ids]of [['Leave',match.lost],['Join',match.gained]]){
    const p=document.createElement('p');p.textContent=`${label}: ${ids.length?ids.map(name).join(', '):'none'}.`;changes.append(p);
  }
}
function findCharacter(id) {const path=results.atlas.nodePaths[id];if(path)setAtlasView(path.at(-1),id);}
function chooseNext(walker) {
  walker.from = walker.to;
  const neighbors=adjacency.get(walker.from);
  walker.teleport = !neighbors.length || rng()<.15;
  const choices = walker.teleport ? active : neighbors;
  walker.to=choices[Math.floor(rng()*choices.length)];walker.progress=0;
}
function resetWalkers() {
  particleLayer.replaceChildren();
  const seeds=groups[view].members.filter(id=>results.atlas.nodePaths[id][0]!=='isolates');
  walkers=seeds.length?Array.from({length:Number($('atlas-walkers').value)},(_,index)=>{
    const walker={to:seeds[Math.floor(rng()*seeds.length)],progress:rng(),offset:index*2.399963229728653};chooseNext(walker);walker.progress=rng();
    walker.el=svg(particleLayer,'circle',{r:3,fill:'#fff6c4',opacity:0});return walker;
  }):[];
}
function updatePlayButton() {$('atlas-flow').textContent=playing?'Ⅱ Pause simulation':'▶ Simulate browsing';$('atlas-flow').setAttribute('aria-pressed',String(playing));}
function stopAnimation() {if(animation!==null)cancelAnimationFrame(animation);animation=null;}
function startAnimation() {
  if(!playing||!visible||document.hidden||animation!==null)return;
  let previous;
  const step=time=>{
    const dt=previous?Math.min(time-previous,80):0;previous=time;
    for(const w of walkers){
      w.progress+=dt/1500;if(w.progress>=1)chooseNext(w);
      const a=positions.get(mappedNodes.get(w.from)),b=positions.get(mappedNodes.get(w.to));
      let x,y,opacity=0,radius=3;
      if(w.teleport){const p=w.progress<.5?a:b;if(p){x=p.x;y=p.y;opacity=Math.abs(w.progress-.5)*1.6;}}
      else if(a&&b){
        if(a===b){x=a.x+Math.cos(w.offset)*a.r*.3;y=a.y+Math.sin(w.offset)*a.r*.3;radius=2+4*Math.sin(w.progress*Math.PI);}
        else {const edge=displayedEdges.get(`${mappedNodes.get(w.from)}\t${mappedNodes.get(w.to)}`);if(edge){const p=edge.path.getPointAtLength(w.progress*edge.length);x=p.x;y=p.y;}}opacity=.9;
      }
      if(opacity&&x!==undefined){w.el.setAttribute('cx',x);w.el.setAttribute('cy',y);}else opacity=0;
      w.el.setAttribute('r',radius);w.el.setAttribute('fill',w.teleport?'#ba97ff':'#fff6c4');w.el.setAttribute('opacity',opacity);
    }
    animation=requestAnimationFrame(step);
  };
  animation=requestAnimationFrame(step);
}

function triangle(root,edges,large=false) {
  root.replaceChildren();const marker=`triad-arrow-${root.id||Math.random().toString(36).slice(2)}`;arrow(root,marker,'#a5c6f6');
  const points=large?{a:{x:200,y:38,r:17},b:{x:75,y:190,r:17},c:{x:325,y:190,r:17}}:{a:{x:35,y:12,r:5},b:{x:12,y:58,r:5},c:{x:58,y:58,r:5}};
  for(const [a,b] of edges)svg(root,'path',{d:lineCurve(points[a],points[b],large?16:5),fill:'none',stroke:'#a5c6f6','stroke-width':large?2:1.2,'marker-end':`url(#${marker})`});
  for(const [id,p]of Object.entries(points)){svg(root,'circle',{cx:p.x,cy:p.y,r:p.r,fill:'#ffd43b'});if(large)svg(root,'text',{x:p.x,y:p.y+5,'text-anchor':'middle',fill:'#080e20','font-size':14,'font-weight':700},id.toUpperCase());}
}
const plotLayout={paper_bgcolor:'#080e20',plot_bgcolor:'#080e20',font:{family:'Arial, sans-serif',color:'#fff8de',size:13},margin:{l:60,r:20,t:25,b:55},modebar:{color:'#bdc8e2',activecolor:'#ffd43b',bgcolor:'#080e20'},autosize:true};
const plotConfig={responsive:true,displaylogo:false};
function availableMotifs() {
  const motifs=results.fingerprint.motifs.filter(m=>$('motif-filter').value==='all'||m.connected);
  const order=$('motif-order').value;
  return motifs.sort(order==='deviation'?(a,b)=>Math.abs(b.z||0)-Math.abs(a.z||0):order==='count'?(a,b)=>b.observed-a.observed:(a,b)=>a.id.localeCompare(b.id));
}
async function drawMotifs() {
  const motifs=availableMotifs();if(!motifs.some(m=>m.id===selectedMotif))selectedMotif=motifs[0].id;
  const restoreFocus = $('motif-gallery').contains(document.activeElement);
  $('motif-gallery').replaceChildren();
  for(const m of motifs){
    const el=button('',()=>{selectedMotif=m.id;exampleIndex=0;requestMotifs();});el.className='motif-card';el.setAttribute('aria-pressed',String(m.id===selectedMotif));el.setAttribute('aria-label',`Triad ${m.id}, ${m.observed} observed`);el.dataset.motif=m.id;
    const diagram=svg(el,'svg',{viewBox:'0 0 70 70','aria-hidden':'true'});triangle(diagram,m.diagramEdges);
    const info=document.createElement('div');
    for(const [cls,text]of [['motif-code',m.id],['motif-card-count',`${m.observed.toLocaleString()} observed`],['motif-card-z'+(m.z<0?' negative':''),m.z===null?'z = N/A':`z = ${m.z>0?'+':''}${m.z.toFixed(1)}`]]){const span=document.createElement('span');span.className=cls;span.textContent=text;info.append(span);}
    el.append(info);$('motif-gallery').append(el);
  }
  if (restoreFocus) $('motif-gallery').querySelector('[aria-pressed="true"]').focus({preventScroll:true});
  await Plotly.react($('motif-profile'),[{type:'bar',x:motifs.map(m=>m.id),y:motifs.map(m=>m.z),marker:{color:motifs.map(m=>m.id===selectedMotif?'#ffd43b':m.z>=0?'#ff758f':'#63caff')},customdata:motifs.map(m=>[m.observed,m.nullMean]),hovertemplate:'%{x}<br>Observed: %{customdata[0]}<br>Random mean: %{customdata[1]:.1f}<br>z: %{y:.2f}<extra></extra>'}],{...plotLayout,xaxis:{title:{text:'Triad type'},type:'category'},yaxis:{title:{text:'Standardized difference (z)'},zerolinecolor:'#ffffff70'}},plotConfig);
  if(!motifPlotReady){$('motif-profile').on('plotly_click',event=>{const id=event.points?.[0]?.x;if(results.fingerprint.motifs.some(m=>m.id===id)){selectedMotif=id;exampleIndex=0;requestMotifs();}});motifPlotReady=true;}
  const m=results.fingerprint.motifs.find(m=>m.id===selectedMotif);
  $('motif-title').textContent=`Triad ${m.id} · ${m.diagramEdges.length} directed links`;
  $('motif-score').textContent=m.z===null?'z = N/A':`z = ${m.z>0?'+':''}${m.z.toFixed(2)}`;
  $('motif-stats').textContent=`Observed: ${m.observed.toLocaleString()} triples · random mean: ${m.nullMean.toFixed(2)} · sample SD: ${m.nullStd.toFixed(2)}. ${m.z===null?'No variation in the null sample; the z-score is undefined.':''}`;
  const min=Math.min(m.observed,...m.nullCounts),max=Math.max(m.observed,...m.nullCounts),padding=Math.max(1,(max-min)*.07);
  await Plotly.react($('motif-null'),[{type:'histogram',x:m.nullCounts,marker:{color:'#63caff'},nbinsx:18,hovertemplate:'Count: %{x}<br>Random graphs: %{y}<extra></extra>'}],{...plotLayout,xaxis:{title:{text:'Number of triples of this type'},range:[min-padding,max+padding]},yaxis:{title:{text:'Random graphs'},rangemode:'tozero'},shapes:[{type:'line',x0:m.observed,x1:m.observed,y0:0,y1:1,yref:'paper',line:{color:'#ffd43b',width:3}}],annotations:[{x:m.observed,y:1,yref:'paper',text:'Observed',showarrow:false,font:{color:'#ffd43b'},yshift:12}]},plotConfig);
  $('motif-example').replaceChildren();
  m.examples.forEach((ids,i)=>$('motif-example').append(new Option(`Example ${i+1} of ${m.examples.length}`,String(i))));
  if(!m.examples.length)$('motif-example').append(new Option('No observed examples',''));
  exampleIndex=Math.min(exampleIndex,Math.max(0,m.examples.length-1));$('motif-example').value=String(exampleIndex);
  drawExample();$('motif-profile').dataset.ready='true';$('motif-profile').dataset.selected=selectedMotif;
}
async function requestMotifs(){motifDirty=true;if(motifDrawing)return;motifDrawing=true;try{while(motifDirty){motifDirty=false;await drawMotifs();}}catch(error){showError(error);}finally{motifDrawing=false;}}
function drawExample(){
  const m=results.fingerprint.motifs.find(m=>m.id===selectedMotif),ids=m.examples[exampleIndex];
  $('motif-example-names').replaceChildren();$('motif-example-svg').replaceChildren();
  $('motif-next').disabled=m.examples.length<2;$('motif-locate').disabled=!ids;$('motif-example').disabled=!ids;
  if(!ids){$('motif-example-names').textContent='No triples of this type occur in the snapshot.';return;}
  const labels=new Map(ids.map((id,i)=>[id,'abc'[i]]));
  const edges=graph.edges.filter(([a,b])=>labels.has(a)&&labels.has(b)).map(([a,b])=>[labels.get(a),labels.get(b)]);
  triangle($('motif-example-svg'),edges,true);
  ids.forEach((id,i)=>{const li=document.createElement('li'),label=document.createElement('span');label.textContent='ABC'[i];li.append(label,document.createTextNode(name(id)));$('motif-example-names').append(li);});
  $('motif-example-svg').setAttribute('aria-label',`Triad ${selectedMotif}: ${ids.map(name).join(', ')}. ${edges.length} directed links.`);
}
function showError(error){$('discovery-error').textContent='The atlas or motif analysis could not load. Rebuild the notebook and include the generated discovery files when publishing.';console.error(error);}

async function start(){
  const responses=await Promise.all(['assets/plots/explorer.json','assets/plots/discoveries.json'].map(url=>fetch(url).then(r=>{if(!r.ok)throw Error(`${url}: ${r.status}`);return r.json();})));
  [graph,results]=responses;byId=new Map(graph.nodes.map(n=>[n.id,n]));groups=results.atlas.groups;
  adjacency=new Map(graph.nodes.map(n=>[n.id,[]]));for(const[a,b]of graph.edges)adjacency.get(a).push(b);
  active=graph.nodes.filter(n=>results.atlas.nodePaths[n.id][0]!=='isolates').map(n=>n.id);
  for(const n of [...graph.nodes].sort((a,b)=>a.name.localeCompare(b.name)))$('atlas-find').append(new Option(n.name,n.id));
  $('atlas-find').addEventListener('change',()=>{if($('atlas-find').value)findCharacter($('atlas-find').value);});
  $('atlas-group').addEventListener('change',()=>{if($('atlas-group').value)setAtlasView($('atlas-group').value);});
  $('atlas-up').addEventListener('click',()=>{if(groups[view].parent)setAtlasView(groups[view].parent);});
  $('atlas-home').addEventListener('click',()=>{$('atlas-find').value='';setAtlasView('root');});
  $('atlas-seed').addEventListener('change',updateSensitivity);
  $('atlas-flow').addEventListener('click',()=>{playing=!playing;updatePlayButton();if(playing)startAnimation();else stopAnimation();});
  $('atlas-walkers').addEventListener('input',resetWalkers);
  $('atlas-locate').addEventListener('click',()=>locate(focused?[focused]:groups[view].members,focused?`Atlas: ${name(focused)}`:`Atlas: ${groups[view].label}`));
  const observer=new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;if(visible)startAnimation();else stopAnimation();},{threshold:0});observer.observe($('atlas-svg'));
  document.addEventListener('visibilitychange',()=>document.hidden?stopAnimation():startAnimation());
  for(const bridge of results.atlas.bridges.slice(0,10)){const li=document.createElement('li');li.append(button(`${name(bridge.id)} · ${bridge.crossNeighbors} cross-community neighbors`,()=>findCharacter(bridge.id)));$('atlas-bridges').append(li);}
  const method=results.atlas.method;
  const deeper=groups.root.children.filter(id=>id!=='isolates'&&groups[id].children.length);
  $('atlas-hierarchy-guide').textContent=`Click to open a group: ${method.topModules-deeper.length} open directly into characters; ${deeper.length} ${deeper.length===1?'has':'have'} smaller detected groups${deeper.length?` (${deeper.map(id=>`${groups[id].label}, ${groups[id].members.length} characters`).join('; ')})`:''}. Isolates are a separate display group.`;
  $('atlas-method').textContent=`Infomap ${method.version} · directed links · ${method.trials} trials · seed ${method.seed}. ${method.topModules} top-level communities on ${method.activeNodes} linked characters; ${method.isolatesExcluded} isolates excluded. Recorded teleportation: 15%, uniform active-node targets. One-level codelength: ${method.oneLevelCodelength.toFixed(3)}; fitted hierarchy: ${method.codelength.toFixed(3)} bits per step. Geometry is a circle packing, not a measure of network distance.`;
  $('motif-method').textContent=`Exact census of ${results.fingerprint.method.triads.toLocaleString()} unordered triples. Each of 100 random graphs starts from the snapshot and completes ${results.fingerprint.method.successfulSwapsPerGraph.toLocaleString()} directed swaps (10 per edge), seeds 4200–4299. Finite rewiring does not prove uniform sampling or adequate mixing. Z-scores are exploratory, not significance claims.`;
  for(const id of ['motif-filter','motif-order'])$(id).addEventListener('change',requestMotifs);
  $('motif-example').addEventListener('change',()=>{exampleIndex=Number($('motif-example').value);drawExample();});
  $('motif-next').addEventListener('click',()=>{const m=results.fingerprint.motifs.find(m=>m.id===selectedMotif);if(m.examples.length){exampleIndex=(exampleIndex+1)%m.examples.length;$('motif-example').value=String(exampleIndex);drawExample();}});
  $('motif-locate').addEventListener('click',()=>{const m=results.fingerprint.motifs.find(m=>m.id===selectedMotif);if(m.examples[exampleIndex])locate(m.examples[exampleIndex],`Triad ${m.id} · example ${exampleIndex+1}`);});
  setAtlasView('root');await requestMotifs();
}
start().catch(showError);
