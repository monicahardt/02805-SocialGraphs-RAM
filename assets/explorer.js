import {survivingGraph, shortestPath} from './network-model.mjs';

const $ = id => document.getElementById(id);
const nav = $('section-nav'), toggle = document.querySelector('.nav-toggle');
toggle.addEventListener('click', () => {
  const open = toggle.getAttribute('aria-expanded') !== 'true';
  toggle.setAttribute('aria-expanded', String(open)); nav.classList.toggle('is-open', open);
});
nav.addEventListener('click', event => {
  if (event.target.closest('a')) {nav.classList.remove('is-open'); toggle.setAttribute('aria-expanded', 'false');}
});
document.addEventListener('keydown', event => {
  if (event.key === 'Escape') {nav.classList.remove('is-open'); toggle.setAttribute('aria-expanded', 'false');}
});
const sections = [...document.querySelectorAll('main > section')];
let scrollQueued = false;
function trackSection() {
  const current = sections.filter(s => s.getBoundingClientRect().top < window.innerHeight * .4).at(-1) || sections[0];
  for (const link of nav.querySelectorAll('a')) {
    if (link.hash === `#${current.id}`) link.setAttribute('aria-current', 'location');
    else link.removeAttribute('aria-current');
  }
  scrollQueued = false;
}
window.addEventListener('scroll', () => {if (!scrollQueued) {scrollQueued = true; requestAnimationFrame(trackSection);}}, {passive:true});
trackSection();

const colors = {giant:'#ff536d', island:'#62bcff', isolate:'#ffd43b', muted:'#63718d'};
const baseLayout = {paper_bgcolor:'#080e20', plot_bgcolor:'#080e20',
  font:{family:'Arial, sans-serif', color:'#fff8de', size:13},
  modebar:{color:'#bdc8e2',activecolor:'#ffd43b',bgcolor:'#080e20'},
  margin:{l:30,r:20,t:20,b:20}, autosize:true};
const config = {responsive:true, displaylogo:false, scrollZoom:false};
let data, byId, graph, path = null, routeRequested = false, group = null, groupLabel = '';
let state = {hero:'', direction:'both', component:'all', strategy:'target', count:0};
let drawing = false, dirty = false, ready = false;

function name(id) {return byId.get(id)?.name || id;}
function removedIds() {return (state.strategy === 'target' ? data.attack.targetOrder : data.attack.randomOrder).slice(0, state.count);}
function appendNames(element, ids, selectable = false) {
  element.replaceChildren();
  for (const id of ids) {
    const item = document.createElement('li');
    if (selectable) {
      const button = document.createElement('button'); button.type = 'button';
      button.textContent = name(id); button.addEventListener('click', () => selectHero(id)); item.append(button);
    } else item.textContent = name(id);
    element.append(item);
  }
}
function selectHero(id) {
  state.hero = id; state.component = 'all'; group = null; routeRequested = false;
  $('hero').value = id; $('component').value = 'all'; requestRender();
}
function selectGroup(ids, label, status) {
  group = new Set(ids); groupLabel = label; state.hero = ''; state.component = 'all'; routeRequested = false;
  $('hero').value = ''; $('component').value = 'all';
  status.textContent = `${group.size} characters highlighted on the network map. Use “Network lab” in the sidebar to inspect them.`;
  requestRender();
}
function visibleByComponent(n) {
  return state.component === 'all' || (state.component === 'giant' && n.component === 0) ||
    (state.component === 'island' && n.component === 1) || (state.component === 'isolates' && n.component > 1);
}
function edgeTrace(edges, color, width) {
  const x = [], y = [];
  for (const [a,b] of edges) {x.push(byId.get(a).x, byId.get(b).x, null); y.push(byId.get(a).y, byId.get(b).y, null);}
  return {type:'scatter', mode:'lines', x, y, line:{color,width}, hoverinfo:'skip', showlegend:false};
}

async function render() {
  graph = survivingGraph(data, new Set(removedIds()));
  const hero = state.hero, activeHero = graph.ids.has(hero);
  path = routeRequested ? shortestPath(graph, hero, $('destination').value, $('path-mode').value === 'directed') : null;
  let spotlight = null;
  if (path) spotlight = new Set(path);
  else if (group) spotlight = new Set([...group].filter(id => graph.ids.has(id)));
  else if (activeHero) {
    const adjacency = state.direction === 'both' ? graph.neighbors : graph[state.direction];
    spotlight = new Set([hero, ...adjacency.get(hero)]);
  }
  const visible = graph.nodes.filter(visibleByComponent), visibleIds = new Set(visible.map(n => n.id));
  const edges = graph.edges.filter(([a,b]) => visibleIds.has(a) && visibleIds.has(b));
  const highlighted = activeHero && !path && !group ? edges.filter(([a,b]) =>
    (state.direction !== 'incoming' && a === hero) || (state.direction !== 'outgoing' && b === hero)) : [];
  const traces = [edgeTrace(edges, spotlight ? 'rgba(125,169,233,.05)' : 'rgba(125,169,233,.20)', .7)];
  if (highlighted.length) traces.push(edgeTrace(highlighted, 'rgba(255,212,59,.55)', 1.5));
  if (path) traces.push(edgeTrace(path.slice(1).map((id,i) => [path[i],id]), colors.isolate, 3));
  traces.push({type:'scatter', mode:'markers', x:visible.map(n => n.x), y:visible.map(n => n.y),
    customdata:visible.map(n => [n.id, name(n.id), graph.incoming.get(n.id).size, graph.outgoing.get(n.id).size]),
    marker:{size:visible.map(n => 7 + 2*Math.sqrt(n.incoming)),
      color:visible.map(n => {
        if (n.id === hero || path?.includes(n.id)) return colors.isolate;
        if (activeHero && spotlight?.has(n.id) && !group) {
          const incoming = graph.incoming.get(hero).has(n.id), outgoing = graph.outgoing.get(hero).has(n.id);
          return incoming && outgoing ? colors.isolate : incoming ? colors.island : colors.giant;
        }
        return n.component === 0 ? colors.giant : n.component === 1 ? colors.island : colors.isolate;
      }), opacity:visible.map(n => !spotlight || spotlight.has(n.id) ? .95 : .10),
      line:{color:visible.map(n => n.id === hero ? '#fff' : '#080e20'), width:visible.map(n => n.id === hero ? 2 : .6)}},
    hovertemplate:'<b>%{customdata[1]}</b><br>Surviving incoming: %{customdata[2]}<br>Surviving outgoing: %{customdata[3]}<extra></extra>', showlegend:false});
  const annotations = path ? path.slice(1).map((id,i) => ({
    x:byId.get(id).x, y:byId.get(id).y, ax:byId.get(path[i]).x, ay:byId.get(path[i]).y,
    xref:'x',yref:'y',axref:'x',ayref:'y',text:'',showarrow:true,arrowhead:3,arrowsize:1.3,arrowwidth:2,arrowcolor:colors.isolate,standoff:9,
  })) : [];
  const networkLayout = {...baseLayout, uirevision:`component-${state.component}`, dragmode:'pan', showlegend:false,
    xaxis:{visible:false}, yaxis:{visible:false,scaleanchor:'x'}, annotations};
  await Plotly.react($('network-lab'), traces, networkLayout, config);
  if (routeRequested) {
    if (!hero || !$('destination').value) $('path-status').textContent = 'Choose both a starting character and a destination.';
    else if (!graph.ids.has(hero) || !graph.ids.has($('destination').value)) $('path-status').textContent = 'An endpoint has been removed. Reduce the removal count or choose surviving characters.';
    else $('path-status').textContent = path ? `${path.length - 1} hop(s) · ${path.map(name).join(' → ')}` : 'No route exists with this direction setting and removal count. Try “in either direction” or another character.';
  } else $('path-status').textContent = 'Select a starting character above, then choose a destination. Gold arrows follow the route; “either direction” may traverse a link backwards.';
  const selectedIds = spotlight ? visible.filter(n => spotlight.has(n.id)).map(n => n.id) : visible.map(n => n.id);
  if (hero && !activeHero) $('selection-status').textContent = `${name(hero)} has been removed by this experiment. Reduce the slider or select another character.`;
  else if (activeHero) $('selection-status').textContent = `${name(hero)} · ${graph.incoming.get(hero).size} incoming / ${graph.outgoing.get(hero).size} outgoing surviving links · ${selectedIds.length} nodes highlighted.`;
  else if (group) $('selection-status').textContent = `${groupLabel} · ${selectedIds.length} of ${group.size} selected characters remain visible.`;
  else $('selection-status').textContent = `${visible.length} visible characters · ${edges.length} visible directed links. Select a character or click a node to inspect its neighborhood.`;
  appendNames($('selection-list'), selectedIds, true);
  $('remaining-nodes').textContent = graph.nodes.length;
  $('remaining-edges').textContent = graph.edges.length.toLocaleString();
  $('largest-component').textContent = graph.largest;
  $('current-isolates').textContent = graph.isolates;
  $('removed-count').textContent = state.count;
  $('nav-status').textContent = `${graph.nodes.length} characters · ${graph.edges.length.toLocaleString()} links remain`;
  $('experiment-status').textContent = `${state.count} removed · ${graph.components.length} weak components remain. The network map and route finder use these surviving nodes. The three snapshot charts below keep the original data.`;
  appendNames($('removed-list'), removedIds());
  await renderRobustness();
  syncComparisonHighlight();
  $('network-lab').dataset.renderedCount = state.count;
  $('network-lab').dataset.renderedHero = state.hero;
}

async function requestRender() {
  dirty = true;
  if (!ready || drawing) return;
  drawing = true;
  try {while (dirty) {dirty = false; await render();}}
  catch (error) {showError(error);}
  finally {drawing = false;}
}
function showError(error) {
  $('lab-error').textContent = 'The explorer could not load or update. Build the notebook, serve the site over HTTP, and reload. The notebook and data links below remain available.';
  console.error(error);
}
async function renderRobustness() {
  const a = data.attack, x = a.target.map((_,i) => i);
  const traces = [
    {x,y:a.randomMin,mode:'lines',line:{width:0},showlegend:false,hoverinfo:'skip'},
    {x,y:a.randomMax,mode:'lines',line:{width:0},fill:'tonexty',fillcolor:'rgba(98,188,255,.14)',name:'Random range',hoverinfo:'skip'},
    {x,y:a.randomMean,mode:'lines',line:{color:colors.island,width:2},name:'Random mean',hovertemplate:'%{x} removed<br>Random mean: %{y:.1f}<extra></extra>'},
    {x,y:a.target,mode:'lines',line:{color:colors.giant,width:3},name:'Targeted',hovertemplate:'%{x} removed<br>Largest: %{y}<extra></extra>'},
    {x,y:a.randomExample,mode:'lines',line:{color:'#b8cbe9',width:1,dash:'dot'},name:'Seed 42',hovertemplate:'%{x} removed<br>Seed 42: %{y}<extra></extra>'},
    {x:[state.count],y:[graph.largest],mode:'markers',marker:{size:12,color:'#fff',line:{color:'#080e20',width:2}},name:'Current experiment',showlegend:false,hovertemplate:'Current: %{x} removed<br>Largest: %{y}<extra></extra>'},
  ];
  await Plotly.react($('robustness-plot'),traces,{...baseLayout,margin:{l:58,r:15,t:55,b:55},uirevision:'robustness',
    xaxis:{title:{text:'Characters removed'},range:[0,a.limit],fixedrange:true},yaxis:{title:{text:'Largest weak component'},range:[0,data.nodes.length],fixedrange:true},
    legend:{orientation:'h',x:0,y:1.2,font:{size:11}},hovermode:'closest',
    shapes:[{type:'line',x0:state.count,x1:state.count,y0:0,y1:1,yref:'paper',line:{color:'#ffffff40',width:1,dash:'dot'}}]},config);
}

// Linked views: interact with the original notebook figures without replacing their Python source.
function plotFromFrame(frame) {return frame.contentDocument?.querySelector('.js-plotly-plot');}
function syncComparisonHighlight() {
  const frame = document.querySelector('#comparison iframe'), plot = plotFromFrame(frame);
  if (!plot?.data || !frame.contentWindow.Plotly) return;
  const selected = group || (state.hero ? new Set([state.hero]) : null);
  const texts = plot.data[0].text;
  const selectedpoints = selected ? texts.flatMap((text,i) => selected.has(data.nodes.find(n => n.name === text)?.id) ? [i] : []) : null;
  frame.contentWindow.Plotly.restyle(plot,{selectedpoints:[selectedpoints], 'selected.marker':{color:'#ffd43b',opacity:1,size:13}, 'unselected.marker':{opacity:.2}},[0]);
}
async function connectFrame(sectionId, instruction) {
  const section = $(sectionId), frame = section.querySelector('iframe');
  const hint = document.createElement('p'); hint.className = 'linked-instruction'; hint.textContent = instruction;
  const status = document.createElement('p'); status.className = 'linked-status'; status.setAttribute('aria-live','polite');
  section.querySelector('figure').before(hint); section.querySelector('figure').after(status);
  const hook = () => {
    const plot = plotFromFrame(frame);
    if (!plot?.on || plot.dataset.linked) return;
    plot.dataset.linked = 'true';
    plot.on('plotly_click', event => {
      const point = event.points?.[0]; if (!point) return;
      if (sectionId === 'degrees') {
        const key = point.data.name === 'Incoming' ? 'incoming' : 'outgoing';
        selectGroup(data.nodes.filter(n => n[key] === point.x).map(n => n.id), `${key} degree ${point.x}`, status);
      } else {
        const node = data.nodes.find(n => n.name === point.text || n.id === point.customdata ||
          (sectionId === 'rankings' && n.name.split(' (')[0].replace('Cloak and Dagger','Cloak & Dagger') === point.y));
        if (node) {selectHero(node.id); status.textContent = `${node.name} selected on the network map.`;}
      }
    });
    if (sectionId === 'comparison') {
      plot.on('plotly_selected', event => {
        if (!event?.points) return;
        const ids = event.points.filter(p => p.curveNumber === 0).map(p => data.nodes.find(n => n.name === p.text)?.id).filter(Boolean);
        selectGroup(ids, 'Scatter selection', status);
      });
      plot.on('plotly_deselect', () => {group = null; state.hero = ''; $('hero').value = ''; status.textContent = ''; requestRender();});
      syncComparisonHighlight();
    }
  };
  frame.addEventListener('load', hook); hook();
}

async function start() {
  const response = await fetch('assets/plots/explorer.json');
  if (!response.ok) throw Error(`Explorer data: HTTP ${response.status}`);
  data = await response.json(); byId = new Map(data.nodes.map(n => [n.id,n]));
  if (!window.Plotly) throw Error('Plotly bundle missing');
  for (const node of [...data.nodes].sort((a,b) => a.name.localeCompare(b.name))) {
    for (const id of ['hero','destination']) {const option = document.createElement('option'); option.value = node.id; option.textContent = node.name; $(id).append(option);}
  }
  $('hero').addEventListener('change', () => selectHero($('hero').value));
  for (const id of ['direction','component','strategy']) $(id).addEventListener('change', () => {
    state[id] = $(id).value;
    if (id === 'component') routeRequested = false;
    requestRender();
  });
  $('remove-count').max = data.attack.limit;
  $('remove-count').addEventListener('input', () => {state.count = Number($('remove-count').value); $('removed-count').textContent = state.count; requestRender();});
  $('trace-path').addEventListener('click', () => {routeRequested = true; group = null; state.component = 'all'; $('component').value = 'all'; requestRender();});
  for (const id of ['destination','path-mode']) $(id).addEventListener('change', () => {if (routeRequested) requestRender();});
  $('clear-path').addEventListener('click', () => {routeRequested = false; requestRender();});
  $('reset-lab').addEventListener('click', () => {
    state = {hero:'',direction:'both',component:'all',strategy:'target',count:0};
    group = null; routeRequested = false;
    for (const id of ['hero','direction','component','strategy']) $(id).value = state[id];
    $('remove-count').value = 0; $('destination').value = ''; $('path-mode').value = 'directed';
    document.querySelectorAll('.linked-status').forEach(el => el.textContent = '');
    Plotly.relayout($('network-lab'),{'xaxis.autorange':true,'yaxis.autorange':true}); requestRender();
  });
  $('export-experiment').addEventListener('click', () => {
    const current = survivingGraph(data,new Set(removedIds()));
    const report = {snapshot:'2026-08-26',strategy:state.strategy,count:state.count,ranking:'original in-degree, node ID breaks ties',randomSeed:state.strategy === 'random' ? 42 : null,
      removed:removedIds(),remainingNodes:current.nodes.length,remainingEdges:current.edges.length,largestWeakComponent:current.largest,isolates:current.isolates,
      route:routeRequested ? {source:state.hero,target:$('destination').value,directed:$('path-mode').value === 'directed',path:shortestPath(current,state.hero,$('destination').value,$('path-mode').value === 'directed')} : null};
    const url = URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json'}));
    const link = document.createElement('a'); link.href = url; link.download = 'ram-network-experiment.json'; link.click(); setTimeout(() => URL.revokeObjectURL(url),1000);
  });
  ready = true; await requestRender();
  document.addEventListener('ram:select-original-nodes', event => {
    const ids = event.detail.ids.filter(id => byId.has(id));
    state.count = 0; $('remove-count').value = 0;
    selectGroup(ids, event.detail.label, $('selection-status'));
    $('map').scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth'});
  });
  $('network-lab').on('plotly_click', event => {const id = event.points?.[0]?.customdata?.[0]; if (byId.has(id)) selectHero(id);});
  $('robustness-plot').on('plotly_click', event => {
    const count = Number(event.points?.[0]?.x); if (!Number.isFinite(count)) return;
    state.count = Math.max(0,Math.min(data.attack.limit,Math.round(count))); $('remove-count').value = state.count; requestRender();
  });
  connectFrame('comparison','Click a character, or use the lasso / box-select toolbar to send a group to the network map. These are original snapshot degrees.');
  connectFrame('degrees','Click a degree-frequency point to highlight that entire group on the network map.');
  connectFrame('rankings','Click a bar to select that character on the network map. Rankings use the original snapshot.');
  $('network-lab').dataset.ready = 'true';
}
start().catch(showError);
