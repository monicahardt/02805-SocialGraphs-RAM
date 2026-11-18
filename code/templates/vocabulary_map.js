const chart = document.getElementById('map');
const detail = document.getElementById('detail');
const characterSelect = document.getElementById('character');
const statusLine = document.getElementById('map-status');
let layer = 'both';
let selected = null;
let expanded = false;
let paintQueue = Promise.resolve();
let paintVersion = 0;
let hovered = null;
const mediumName = value => value === 'books' ? 'Book' : 'Film';
const safe = value => String(value).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const visible = id => layer === 'both' || data.nodes[id].medium === layer;
const counterpart = id => id % 2 === 0 ? id + 1 : id - 1;
const visibleConnections = id => data.connections[id].filter(connection => visible(connection.id));

function edgeCoordinates(a, b) {
  return {x: [data.positions[a][0], data.positions[b][0]], y: [data.positions[a][1], data.positions[b][1]]};
}

function networkCoordinates() {
  const x = [], y = [];
  for (const [a, b] of data.edges) {
    if (!visible(a) || !visible(b)) continue;
    x.push(data.positions[a][0], data.positions[b][0], null);
    y.push(data.positions[a][1], data.positions[b][1], null);
  }
  return {x, y};
}

function ranges(ids, focus) {
  const points = ids.map(id => data.positions[id]);
  const xs = points.map(p => p[0]), ys = points.map(p => p[1]);
  let loX = Math.min(...xs), hiX = Math.max(...xs), loY = Math.min(...ys), hiY = Math.max(...ys);
  const span = Math.max(hiX - loX, hiY - loY, focus ? .2 : .5);
  const pad = span * (focus ? .36 : .15);
  if (hiX - loX < span * .3) {const center = (hiX + loX) / 2; loX = center - span * .15; hiX = center + span * .15;}
  if (hiY - loY < span * .3) {const center = (hiY + loY) / 2; loY = center - span * .15; hiY = center + span * .15;}
  return {'xaxis.range': [loX - pad, hiX + pad], 'yaxis.range': [loY - pad, hiY + pad],
          'xaxis.autorange': false, 'yaxis.autorange': false};
}

function overlaps(a, b) {
  return Math.max(0, Math.min(a.right,b.right) - Math.max(a.left,b.left)) * Math.max(0, Math.min(a.bottom,b.bottom) - Math.max(a.top,b.top));
}

function nodeLabels(ids, overview = false) {
  const full = chart._fullLayout, xaxis = full.xaxis, yaxis = full.yaxis;
  const context = document.createElement('canvas').getContext('2d');
  context.font = '600 14px Arial';
  const occupied = ids.map(id => {
    const p = data.positions[id], x = xaxis.l2p(p[0]) + xaxis._offset, y = yaxis.l2p(p[1]) + yaxis._offset;
    return {left:x-12,right:x+12,top:y-12,bottom:y+12};
  });
  const annotations = [];
  for (const id of ids) {
    const node = data.nodes[id], p = data.positions[id];
    const x = xaxis.l2p(p[0]) + xaxis._offset, y = yaxis.l2p(p[1]) + yaxis._offset;
    const width = context.measureText(node.character).width + 14, height = overview ? 27 : 43;
    const offsets = [[width/2+20,0],[-width/2-20,0],[0,-height/2-22],[0,height/2+22],
      [width/2+15,-39],[-width/2-15,-39],[width/2+15,39],[-width/2-15,39],
      [0,-85],[0,85],[width/2+32,-72],[-width/2-32,-72],[width/2+32,72],[-width/2-32,72]];
    let best = null;
    for (const [dx,dy] of offsets) {
      const box = {left:x+dx-width/2,right:x+dx+width/2,top:y+dy-height/2,bottom:y+dy+height/2};
      const outside = Math.max(0,xaxis._offset+3-box.left) + Math.max(0,box.right-xaxis._offset-xaxis._length+3)
        + Math.max(0,yaxis._offset+3-box.top) + Math.max(0,box.bottom-yaxis._offset-yaxis._length+3);
      const score = occupied.reduce((sum,old)=>sum+overlaps(box,old),0) + outside*10000;
      if (!best || score < best.score) best = {score,dx,dy,box};
      if (score === 0) break;
    }
    occupied.push(best.box);
    annotations.push({x:p[0],y:p[1],xref:'x',yref:'y',ax:best.dx,ay:best.dy,
      text:'<b>'+safe(node.character)+'</b>'+(overview?'':'<br><span style="font-size:12px">'+mediumName(node.medium)+'</span>'),
      showarrow:true,arrowhead:0,arrowwidth:.8,arrowcolor:'#7a9085',
      bgcolor:'rgba(247,245,239,.97)',borderpad:4,xanchor:'center',yanchor:'middle',
      font:{size:overview?12:14,color:'#203e35'},captureevents:false});
  }
  return annotations;
}

function renderPanel() {
  if (selected === null) {
    detail.innerHTML = '<div class="empty"><p class="eyebrow">Read the connections</p><h2>Choose a book or film profile</h2>'
      +'<p>Every line connects vocabularies with shared words. Click a node to reveal:</p><ol><li>The names at the other ends.</li><li>Similarity scores and contributing words.</li><li>The character’s counterpart in the other medium.</li></ol>'
      +'<p class="muted">The map will zoom to a small, labeled neighborhood. You can follow a connection by clicking its name.</p></div>';
    return;
  }
  const node = data.nodes[selected], other = data.nodes[counterpart(selected)], connections = visibleConnections(selected);
  const shown = expanded ? connections : connections.slice(0,3);
  detail.innerHTML = '<p class="eyebrow">Selected '+mediumName(node.medium)+' profile</p><h2>'+safe(node.character)+'</h2>'
    +'<p class="muted">'+safe(node.role)+' · '+node.words.toLocaleString()+' words<br>Matched installments '+safe(node.installments)+'</p>'
    +'<div class="profile-tabs"><button type="button" data-profile="books" class="'+(node.medium==='books'?'active':'')+'" aria-pressed="'+(node.medium==='books')+'">○ Book profile</button>'
    +'<button type="button" data-profile="movies" class="'+(node.medium==='movies'?'active':'')+'" aria-pressed="'+(node.medium==='movies')+'">▲ Film profile</button></div>'
    +'<div class="counterpart"><span>Book–film vocabulary similarity</span><strong>'+node.counterpart_cosine.toFixed(3)+'</strong></div>'
    +'<h3>Connected characters <span class="muted">('+connections.length+')</span></h3>'
    +'<p class="small muted">'+(connections.length>3?'Map shows the three strongest connections. ':'')+'Scores are cosine similarity, from 0 to 1. Click a name to follow the connection.</p>'
    +'<div class="connection-list">'+shown.map(connection => {
      const neighbor = data.nodes[connection.id];
      return '<button type="button" class="connection" data-neighbor="'+connection.id+'" aria-label="Explore '+safe(neighbor.character)+' '+mediumName(neighbor.medium)+' profile">'
        +'<span class="connection-heading"><span class="connection-name">'+safe(neighbor.character)+'</span><span class="score">'+connection.cosine.toFixed(3)+'</span></span>'
        +'<span class="connection-medium">'+(neighbor.medium==='books'?'○ Book':'▲ Film')+' · '+safe(neighbor.role)+'</span>'
        +'<span class="shared">Shared words: '+connection.shared_terms.map(safe).join(', ')+'</span></button>';
    }).join('')+'</div>'
    +(connections.length===0?'<p class="muted">No other-character graph connections are visible in this layer. Try Both.</p>':'')
    +(connections.length>3?'<button type="button" class="list-toggle" id="list-toggle">'+(expanded?'Show the three strongest':'List all '+connections.length+' connections')+'</button>':'')
    +'<h3>Distinctive words in this profile</h3><p class="muted">'+safe(node.top_words)+'</p>';
  detail.querySelectorAll('[data-profile]').forEach(button=>button.addEventListener('click',()=>{
    const id = selected - selected%2 + (button.dataset.profile==='movies'?1:0);
    if (layer!=='both' && data.nodes[id].medium!==layer) layer=data.nodes[id].medium;
    selectNode(id);
  }));
  detail.querySelectorAll('[data-neighbor]').forEach(button=>{
    button.addEventListener('click',()=>selectNode(Number(button.dataset.neighbor)));
    button.addEventListener('mouseenter',()=>highlightConnection(Number(button.dataset.neighbor)));
    button.addEventListener('mouseleave',()=>highlightConnection(null));
    button.addEventListener('focus',()=>highlightConnection(Number(button.dataset.neighbor)));
    button.addEventListener('blur',()=>highlightConnection(null));
  });
  const toggle = document.getElementById('list-toggle');
  if (toggle) toggle.addEventListener('click',()=>{expanded=!expanded;renderPanel();});
}

function highlightConnection(id) {
  hovered = id;
  if (selected === null || id === null || !visible(id)) {
    Plotly.restyle(chart,{x:[[]],y:[[]]},[data.hoverTrace]);return;
  }
  const line = edgeCoordinates(selected,id);
  Plotly.restyle(chart,{x:[line.x],y:[line.y]},[data.hoverTrace]);
}

function requestPaint(fit = true) {
  // Serialize updates so rapid node/layer clicks cannot leave stale highlights.
  const version = ++paintVersion;
  paintQueue = paintQueue.catch(()=>{}).then(()=>version===paintVersion?paint(fit,version):undefined);
  return paintQueue;
}

async function paint(fit, version) {
  chart.dataset.rendering = 'true';
  for (const button of document.querySelectorAll('[data-layer]')) {
    const active = button.dataset.layer===layer;button.classList.toggle('active',active);button.setAttribute('aria-pressed',String(active));
  }
  const connections = selected===null?[]:visibleConnections(selected).slice(0,3);
  const focus = new Set(selected===null?[]:[selected,...connections.map(c=>c.id)]);
  const other = selected===null?null:counterpart(selected);
  if (other!==null && visible(other)) focus.add(other);
  const background = networkCoordinates();
  await Plotly.restyle(chart,{x:[background.x],y:[background.y],opacity:selected===null?.24:.045},[0]);
  for (const trace of data.traceNodes) {
    const opacity = trace.ids.map(id=>selected===null?1:focus.has(id)?1:.11);
    const size = trace.ids.map(id=>id===selected?21:focus.has(id)?16:selected===null?13:8);
    await Plotly.restyle(chart,{visible:layer==='both'||trace.medium===layer,'marker.opacity':[opacity],'marker.size':[size]},[trace.trace]);
  }
  for (let i=0;i<data.focusTraces.length;i++) {
    const connection=connections[i], line=connection?edgeCoordinates(selected,connection.id):{x:[],y:[]};
    await Plotly.restyle(chart,{x:[line.x],y:[line.y],'line.width':connection?1.8+3*connection.cosine:2.6},[data.focusTraces[i]]);
  }
  const identity=selected!==null&&visible(other)?edgeCoordinates(selected,other):{x:[],y:[]};
  await Plotly.restyle(chart,{x:[identity.x],y:[identity.y]},[1]);
  await Plotly.restyle(chart,{x:[[]],y:[[]]},[data.hoverTrace]);
  if (version !== paintVersion) return;
  const height = window.innerWidth < 760 ? 460 : Math.max(510, Math.min(700, window.innerHeight - 280));
  chart.style.height = height + 'px';
  await Plotly.relayout(chart,{height,width:chart.clientWidth});
  if (fit) {
    const ids=selected===null?data.nodes.map(n=>n.id).filter(visible):[...focus];
    await Plotly.relayout(chart,ranges(ids,selected!==null));
  }
  const overviewNames=new Set(['Voldemort','Bellatrix Lestrange','Dolores Umbridge','Lucius Malfoy','Harry Potter','Rubeus Hagrid','Albus Dumbledore']);
  const labels=selected===null?data.nodes.filter(n=>visible(n.id)&&overviewNames.has(n.character)&&(layer!=='both'||n.medium==='books')).map(n=>n.id):[...focus];
  await Plotly.relayout(chart,{annotations:nodeLabels(labels,selected===null)});
  if (version !== paintVersion) return;
  const node=selected===null?null:data.nodes[selected];
  statusLine.textContent=selected===null?'○ Books · ▲ Films. Click a node to inspect a labeled neighborhood.':mediumName(node.medium)+' profile: '+node.character+' · '+connections.length+' named connections shown'+(visible(other)?' · dotted line: same character in the other medium':'');
  chart.dataset.selectedNode=selected===null?'':String(selected);
  chart.dataset.connectedNodes=JSON.stringify(connections.map(c=>c.id));
  chart.dataset.rendering='false';
}

function selectNode(id) {
  selected=Number(id);expanded=false;hovered=null;
  if (!visible(selected)) layer=data.nodes[selected].medium;
  characterSelect.value=String(selected-selected%2);
  renderPanel();return requestPaint();
}

function clearSelection() {
  selected=null;expanded=false;hovered=null;characterSelect.value='';
  renderPanel();return requestPaint();
}

chart.on('plotly_click',event=>{const point=event.points[0];if(Number.isInteger(point.customdata))selectNode(point.customdata);});
characterSelect.addEventListener('change',event=>{
  if (event.target.value==='') clearSelection();
  else selectNode(Number(event.target.value)+(layer==='movies'?1:selected!==null&&data.nodes[selected].medium==='movies'?1:0));
});
document.getElementById('clear').addEventListener('click',clearSelection);
document.querySelectorAll('[data-layer]').forEach(button=>button.addEventListener('click',()=>{
  layer=button.dataset.layer;
  if (selected!==null&&!visible(selected)) selected=counterpart(selected);
  expanded=false;renderPanel();requestPaint();
}));
let resizeTimer;
window.addEventListener('resize',()=>{
  clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>Plotly.Plots.resize(chart).then(()=>requestPaint()),180);
});
renderPanel();requestPaint();
