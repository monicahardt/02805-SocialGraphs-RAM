const chart=document.getElementById('polarity'),detail=document.getElementById('detail');
const character=document.getElementById('character'),roleSelect=document.getElementById('role');
let selected=null,role='all',queue=Promise.resolve(),version=0;
const safe=value=>String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const visible=n=>role==='all'||n.role===role;
const xy=n=>[n.negative_per_1000,n.positive_per_1000];

function panel(){
  if(selected===null){
    detail.innerHTML='<div class="empty"><p class="eyebrow">59 characters · 118 profiles</p><h2>Do heroes and villains use different words?</h2>'
      +'<p>The colors show their story roles. Their positions come only from the words attributed to them.</p>'
      +'<ul><li>Choose a character to compare book and film.</li><li>See positive and negative counts separately.</li><li>Click a contributing word to read the dialogue behind it.</li></ul>'
      +'<p class="muted">A villain can use positive words. A hero can describe terrible events. This map lets you inspect that overlap.</p></div>';return;
  }
  const b=data.nodes[selected],f=data.nodes[selected+1];
  detail.innerHTML='<p class="eyebrow">'+safe(b.role)+' · matched stories '+safe(b.installments)+'</p><h2>'+safe(b.character)+'</h2>'
    +'<p class="coverage">Book '+b.words.toLocaleString()+' words · Film '+f.words.toLocaleString()+' words</p>'
    +((b.low_word_count||f.low_word_count)?'<p class="warning">At least one profile has fewer than 300 words. A few word choices can move its position substantially.</p>':'')
    +'<div class="scores">'+[b,f].map(n=>'<div class="score '+(n.medium==='movies'?'film':'')+'"><b>'+(n.medium==='books'?'○ Book':'▲ Film')+'</b>'
      +'<p>Positive <strong>'+n.positive_per_1000.toFixed(1)+'</strong><br>Negative <strong>'+n.negative_per_1000.toFixed(1)+'</strong></p>'
      +'<p class="small">'+n.positive+' positive · '+n.negative+' negative matches</p></div>').join('')+'</div>'
    +'<p class="small">Rates per 1,000 dialogue words. Click any word below for original lines. Numbers beside words are raw counts.</p>'
    +'<div class="word-columns">'+[b,f].map(n=>'<section><h3>'+(n.medium==='books'?'Book words':'Film words')+'</h3>'
      +['positive','negative'].map(p=>'<p class="small">'+p.toUpperCase()+'</p>'+n.terms[p].map(t=>'<button type="button" class="word '+p+'" data-id="'+n.id+'" data-word="'+safe(t.word)+'">'
        +safe(t.word)+'<span>'+t.count+'</span></button>').join('')).join('')+'</section>').join('')+'</div>'
    +'<section id="evidence"><p class="small">Choose a word to inspect its context.</p></section>';
  detail.querySelectorAll('[data-word]').forEach(button=>button.addEventListener('click',()=>showEvidence(Number(button.dataset.id),button.dataset.word)));
}

function showEvidence(id,word){
  detail.querySelectorAll('[data-word]').forEach(button=>button.classList.toggle('active',button.dataset.word===word));
  const node=data.nodes[id],term=[...node.terms.positive,...node.terms.negative].find(t=>t.word===word);
  const evidence=document.getElementById('evidence'),polarity=node.terms.positive.includes(term)?'positive':'negative';
  evidence.innerHTML='<h3>“'+safe(word)+'” · '+(node.medium==='books'?'Book':'Film')+'</h3><p class="small">'+term.count+' occurrences · '+term.per_1000.toFixed(2)+' per 1,000 words · '+polarity+' lexicon entry</p>'
    +'<div class="quotes">'+term.examples.map(e=>'<blockquote>'+safe(e.text)+'<footer>'+safe(e.source)+' · chapter '+safe(e.chapter)+' · parsed record '+e.record_index+'</footer></blockquote>').join('')+'</div>'
    +'<p class="small">Up to three original lines are shown. The label describes the dictionary entry; the sentence may use it differently.</p>';
  evidence.scrollIntoView({behavior:'smooth',block:'nearest'});
}

async function paint(current){
  chart.dataset.rendering='true';
  const pair=selected===null?[]:[data.nodes[selected],data.nodes[selected+1]];
  const background={x:[],y:[]};
  for(let i=0;i<data.nodes.length;i+=2){
    const b=data.nodes[i],f=data.nodes[i+1];if(!visible(b))continue;
    background.x.push(b.negative_per_1000,f.negative_per_1000,null);
    background.y.push(b.positive_per_1000,f.positive_per_1000,null);
  }
  await Plotly.restyle(chart,{x:[background.x],y:[background.y],opacity:selected===null?.32:.07},[0]);
  await Plotly.restyle(chart,{x:[pair.map(n=>n.negative_per_1000)],y:[pair.map(n=>n.positive_per_1000)]},[1]);
  for(const group of data.groups){
    await Plotly.restyle(chart,{visible:role==='all'||group.role===role,
      'marker.opacity':[group.ids.map(id=>selected===null||id===selected||id===selected+1?1:.13)],
      'marker.size':[group.ids.map(id=>id===selected||id===selected+1?20:data.nodes[id].low_word_count?10:14)]},[group.trace]);
  }
  if(current!==version)return;
  const shown=data.nodes.filter(visible),xmax=Math.max(...shown.map(n=>n.negative_per_1000))*1.18,ymax=Math.max(...shown.map(n=>n.positive_per_1000))*1.18;
  const height=innerWidth<760?460:Math.max(520,Math.min(740,innerHeight-260));
  chart.style.height=height+'px';
  const labels=pair.map((n,i)=>({x:n.negative_per_1000,y:n.positive_per_1000,text:'<b>'+safe(n.character)+'</b><br>'+(i===0?'Book':'Film'),
    ax:0,ay:i===0?-48:48,showarrow:true,arrowhead:0,arrowcolor:'#8a9e8e',bgcolor:'rgba(247,244,236,.96)',borderpad:4,font:{size:13,color:'#243e3b'}}));
  await Plotly.relayout(chart,{height,width:chart.clientWidth,'xaxis.range':[0,xmax],'yaxis.range':[0,ymax],annotations:labels});
  if(current!==version)return;
  chart.dataset.selectedCharacter=selected===null?'':data.nodes[selected].character;
  chart.dataset.rendering='false';
  document.getElementById('status').textContent=shown.length/2+' character pairs visible · ○ Book → ▲ Film · Small markers: fewer than 300 words';
}

function refresh(){const current=++version;queue=queue.catch(()=>{}).then(()=>current===version?paint(current):undefined);return queue;}
function select(id){selected=id===null?null:Number(id)-Number(id)%2;
  if(selected!==null&&!visible(data.nodes[selected])){role='all';roleSelect.value='all';}
  character.value=selected===null?'':String(selected);panel();return refresh();}
chart.on('plotly_click',event=>{const point=event.points[0];if(Number.isInteger(point.customdata))select(point.customdata);});
character.addEventListener('change',()=>select(character.value===''?null:Number(character.value)));
document.getElementById('clear').addEventListener('click',()=>select(null));
roleSelect.addEventListener('change',()=>{role=roleSelect.value;if(selected!==null&&!visible(data.nodes[selected]))selected=null;character.value=selected===null?'':String(selected);panel();refresh();});
let resizeTimer;window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(refresh,180);});
panel();refresh();
