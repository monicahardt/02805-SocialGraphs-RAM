const safe=value=>String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const controls=Object.fromEntries(['spell','medium','story','kind','search','variants','reviewed'].map(id=>[id,document.getElementById(id)]));
const result=document.getElementById('results'),more=document.getElementById('more');
let limit=25,filtered=[];
const labels={casting_attempt:'Casting attempt supported by book narration',casting_attempt_in_transcript:'Casting attempt inferred from transcript',discussion:'Discussion / explanation',reported_casting:'Reported casting by someone else',unreviewed:'Casting context unreviewed'};

function render(){
  for(const [id,medium] of [['book-rate','books'],['film-rate','movies']]){
    const count=events.filter(e=>e.medium===medium&&e.kind==='incantation'&&(controls.variants.checked||e.status==='exact')).length;
    document.getElementById(id).textContent=(10000*count/wordBudgets[medium]).toFixed(2);
  }
  const query=controls.search.value.toLowerCase().trim();
  filtered=events.filter(e=>(!controls.spell.value||e.spell===controls.spell.value)
    &&(!controls.medium.value||e.medium===controls.medium.value)
    &&(!controls.story.value||e.installment===Number(controls.story.value))
    &&(!controls.kind.value||e.kind===controls.kind.value)
    &&(controls.variants.checked||e.status==='exact')
    &&(!controls.reviewed.checked||e.reviewed_context!=='unreviewed')
    &&(!query||[e.text,e.source_speaker,e.character,e.reviewed_speaker,e.spell].join(' ').toLowerCase().includes(query)));
  document.getElementById('result-count').textContent=filtered.length+' matches in '+new Set(filtered.map(e=>e.medium+':'+e.record_index)).size+' dialogue lines · showing '+Math.min(limit,filtered.length);
  result.innerHTML=filtered.slice(0,limit).map(e=>{
    const marked=safe(e.text).replace(safe(e.matched_text),'<mark>'+safe(e.matched_text)+'</mark>');
    const context=labels[e.reviewed_context]||e.reviewed_context;
    return '<article class="result" data-id="'+safe(e.id)+'"><div class="result-head"><strong class="'+(e.medium==='books'?'book':'film')+'">'+safe(e.spell)+'</strong>'
      +'<span class="badge">'+(e.medium==='books'?'Book':'Film')+' · story '+e.installment+'</span><span class="badge">'+safe(e.kind)+'</span>'
      +'<span class="badge">'+safe(e.status.replaceAll('_',' '))+'</span></div>'
      +'<p class="source">'+safe(e.source)+' · '+safe(e.chapter)+' · parsed record '+e.record_index+'<br>Source speaker: '+safe(e.source_speaker)+'</p>'
      +(e.reviewed_context!=='unreviewed'?'<div class="review"><b>'+safe(context)+'</b><br>Reviewed speaker: '+safe(e.reviewed_speaker)+'<div class="source">'+safe(e.review_evidence)+'</div></div>':'<p class="source">'+safe(context)+' · '+(e.utterance_shape==='incantation_only'?'Incantation-only line':'Embedded in dialogue')+'</p>')
      +'<blockquote>'+marked+'</blockquote><p class="kwic">'+safe(e.left_context)+' <b>['+safe(e.matched_text)+']</b> '+safe(e.right_context)+'</p>'
      +'<details class="neighbors"><summary>Read nearby dialogue</summary>'+e.neighbors.map(n=>'<div class="neighbor"><b>'+safe(n.speaker)+'</b> · record '+n.record_index+'<br>'+safe(n.text)+'</div>').join('')+'</details></article>';
  }).join('')||(filtered.length===0?'<p class="intro">No matches for these filters. Absence from this dialogue inventory does not establish absence from the story.</p>':'');
  more.hidden=filtered.length<=limit;
  result.dataset.matches=String(filtered.length);
}

for(const [id,control] of Object.entries(controls))control.addEventListener(id==='search'?'input':'change',()=>{limit=25;render();});
more.addEventListener('click',()=>{limit+=25;render();});
document.getElementById('reset').addEventListener('click',()=>{
  for(const id of ['spell','medium','story','search'])controls[id].value='';
  controls.kind.value='incantation';controls.variants.checked=true;controls.reviewed.checked=false;limit=25;render();
});
const params=new URLSearchParams(location.search);
if(params.get('reviewed')==='1')controls.reviewed.checked=true;
for(const id of ['spell','medium','story'])if(params.has(id))controls[id].value=params.get(id);
render();
