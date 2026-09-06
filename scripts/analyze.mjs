import fs from 'node:fs';
import assert from 'node:assert/strict';

const root = new URL('../', import.meta.url);
const read = name => fs.readFileSync(new URL(name, root), 'utf8').trim().split(/\r?\n/).filter(l => !l.startsWith('#'));
const [header, ...rows] = read('week1_nodes.tsv');
const nodes = rows.map(row => Object.fromEntries(header.split('\t').map((key, i) => [key, row.split('\t')[i]])));
const edges = read('week1_edges.tsv').map(row => row.split('\t'));
const byId = new Map(nodes.map(n => [n.node_id, {...n, incoming: 0, outgoing: 0, neighbors: new Set()}]));
assert.equal(byId.size, nodes.length, 'Node IDs must be unique');
assert.equal(new Set(edges.map(e => e.join('\t'))).size, edges.length, 'Edges must be unique');
for (const [a, b] of edges) {
  assert(byId.has(a) && byId.has(b), 'Every endpoint must be in the roster');
  byId.get(a).outgoing++; byId.get(b).incoming++;
  byId.get(a).neighbors.add(b); byId.get(b).neighbors.add(a);
}
const seen = new Set(), components = [];
for (const id of byId.keys()) {
  if (seen.has(id)) continue;
  const stack = [id], component = []; seen.add(id);
  while (stack.length) {
    const current = stack.pop(); component.push(current);
    for (const next of byId.get(current).neighbors) if (!seen.has(next)) { seen.add(next); stack.push(next); }
  }
  components.push(component);
}
components.sort((a, b) => b.length - a.length);
const result = {
  snapshot: '2026-08-26',
  nodes: [...byId.values()].map(({neighbors, ...n}) => n), edges, components,
};
assert.equal(result.nodes.reduce((sum, n) => sum + n.incoming, 0), edges.length);
assert.equal(result.nodes.reduce((sum, n) => sum + n.outgoing, 0), edges.length);
fs.writeFileSync(new URL('assets/week1-analysis.json', root), JSON.stringify(result));
for (const key of ['incoming', 'outgoing']) console.log(key, [...result.nodes].sort((a,b) => b[key]-a[key]).slice(0, 5).map(n => `${n.name}: ${n[key]}`).join(', '));
console.log(`${nodes.length} nodes, ${edges.length} edges. Component sizes: ${components.map(c => c.length).join(', ')}`);
console.log('Isolates:', result.nodes.filter(n => !n.incoming && !n.outgoing).map(n => n.name).join(', '));
