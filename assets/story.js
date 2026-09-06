const $ = id => document.getElementById(id);
const ns = 'http://www.w3.org/2000/svg';
function draw(parent, tag, attrs, text) {
  const element = document.createElementNS(ns, tag);
  for (const [key, value] of Object.entries(attrs)) element.setAttribute(key, value);
  if (text !== undefined) element.textContent = text;
  parent.append(element); return element;
}
fetch('assets/week1-analysis.json').then(r => {if (!r.ok) throw Error(r.status); return r.json();}).then(data => {
  const byId = new Map(data.nodes.map(n => [n.node_id, n]));
  function distribution() {
    const svg = $('distribution'); svg.replaceChildren();
    const log = $('scale').value === 'log';
    const series = ['incoming', 'outgoing'].map(key => {
      const counts = new Map();
      data.nodes.forEach(n => counts.set(n[key], (counts.get(n[key]) || 0) + 1));
      return [...counts].filter(([k]) => !log || k > 0).sort((a,b) => a[0]-b[0]);
    });
    const maxY = Math.max(...series.flat().map(p => p[1]));
    const x = k => 65 + (log ? Math.log10(k)/Math.log10(110) : k/110)*795;
    const y = n => 320 - (log ? Math.log10(n)/Math.log10(maxY) : n/maxY)*280;
    draw(svg,'path',{d:'M65 30 V320 H860',fill:'none',stroke:'#bdc8e2'});
    for (const k of log ? [1,2,5,10,20,50,100] : [0,20,40,60,80,100]) draw(svg,'text',{x:x(k),y:343,fill:'#bdc8e2','font-size':14,'text-anchor':'middle'},k);
    for (const n of log ? [1,2,5,10,20,50,100].filter(n => n<=maxY) : [0,Math.round(maxY/2),maxY]) {
      draw(svg,'line',{x1:65,x2:860,y1:y(n),y2:y(n),stroke:'#ffffff18'});
      draw(svg,'text',{x:53,y:y(n)+5,fill:'#bdc8e2','font-size':14,'text-anchor':'end'},n);
    }
    draw(svg,'text',{x:470,y:373,fill:'#bdc8e2','text-anchor':'middle','font-size':16},'Degree (links per character)');
    draw(svg,'text',{x:65,y:20,fill:'#bdc8e2','font-size':14},'Characters');
    series.forEach((points,i) => points.forEach(([k,n]) => {
      const circle = draw(svg,'circle',{cx:x(k),cy:y(n),r:i?3:5,fill:i?'#62bcff':'#ff536d',opacity:.8});
      draw(circle,'title',{},`${i?'Out':'In'}-degree ${k}: ${n} characters`);
    }));
    $('scale-note').textContent = log ? 'Logarithmic axes omit degree zero (log 0 is undefined); zero-degree characters remain in the dataset.' : 'Linear axes include degree zero.';
  }
  $('scale').addEventListener('change',distribution); distribution();
  $('island').textContent = data.components[1].map(id => byId.get(id).name).sort().join(' · ');
  $('isolates').textContent = data.nodes.filter(n => !n.incoming && !n.outgoing).map(n => n.name).join(' · ');
  const giant = data.components[0].map((id,i,arr) => ({id,x:300+180*Math.cos(i*2*Math.PI/arr.length),y:300+180*Math.sin(i*2*Math.PI/arr.length)}));
  const positions = new Map(giant.map(n => [n.id,n]));
  const giantEdges = data.edges.filter(([a,b]) => positions.has(a)&&positions.has(b));
  for (let iteration=0;iteration<160;iteration++) {
    const force = new Map(giant.map(n => [n.id,{x:(300-n.x)*.015,y:(300-n.y)*.015}]));
    for(let i=0;i<giant.length;i++) for(let j=i+1;j<giant.length;j++) {
      const a=giant[i],b=giant[j],dx=a.x-b.x,dy=a.y-b.y,d2=Math.max(16,dx*dx+dy*dy),f=90/d2;
      force.get(a.id).x+=dx*f;force.get(a.id).y+=dy*f;force.get(b.id).x-=dx*f;force.get(b.id).y-=dy*f;
    }
    for(const [a,b] of giantEdges){const p=positions.get(a),q=positions.get(b),dx=q.x-p.x,dy=q.y-p.y;force.get(a).x+=dx*.007;force.get(a).y+=dy*.007;force.get(b).x-=dx*.007;force.get(b).y-=dy*.007;}
    for(const n of giant){const f=force.get(n.id);n.x=Math.max(25,Math.min(575,n.x+Math.max(-5,Math.min(5,f.x))));n.y=Math.max(50,Math.min(570,n.y+Math.max(-5,Math.min(5,f.y))));}
  }
  data.components[1].forEach((id,i,arr)=>positions.set(id,{x:745+85*Math.cos(i*2*Math.PI/arr.length),y:170+85*Math.sin(i*2*Math.PI/arr.length)}));
  data.components.slice(2).flat().forEach((id,i)=>positions.set(id,{x:660+(i%5)*40,y:370+Math.floor(i/5)*45}));
  const svg=$('network');
  for(const [a,b] of data.edges){const p=positions.get(a),q=positions.get(b);draw(svg,'line',{x1:p.x,y1:p.y,x2:q.x,y2:q.y,stroke:'#7da9e9','stroke-opacity':.13});}
  const circles=new Map();
  for(const n of data.nodes){const p=positions.get(n.node_id),circle=draw(svg,'circle',{cx:p.x,cy:p.y,r:3+Math.sqrt(n.incoming)*.8,fill:n.incoming?'#ff536d':'#ffd43b',stroke:'#080e20','stroke-width':.6});draw(circle,'title',{},`${n.name}: ${n.incoming} in, ${n.outgoing} out`);circles.set(n.node_id,circle);}
  for(const [x,y,label] of [[25,30,'Giant · 277 characters'],[640,40,'Island · 9 characters'],[640,330,'Isolates · 17 characters']])draw(svg,'text',{x,y,fill:'#fff8de','font-size':18},label);
  for(const n of [...data.nodes].sort((a,b)=>a.name.localeCompare(b.name))){const option=document.createElement('option');option.value=n.node_id;option.textContent=n.name;$('character').append(option);}
  function inspect(){for(const [id,c] of circles){c.setAttribute('stroke',id===$('character').value?'#ffffff':'#080e20');c.setAttribute('stroke-width',id===$('character').value?3:.6);}const n=byId.get($('character').value);$('character-detail').textContent=`${n.name}: ${n.incoming} incoming links · ${n.outgoing} outgoing links · weak component of ${data.components.find(c=>c.includes(n.node_id)).length} character(s).`;}
  $('character').value='Spider-Man';$('character').addEventListener('change',inspect);inspect();
}).catch(error => {$('error').textContent='Interactive charts could not load. Serve this folder over HTTP and reload; the findings and tables remain available above.';console.error(error);});
