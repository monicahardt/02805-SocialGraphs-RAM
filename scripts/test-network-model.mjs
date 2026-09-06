import assert from 'node:assert/strict';
import fs from 'node:fs';
import {survivingGraph, shortestPath} from '../assets/network-model.mjs';

// Directed paths must not silently traverse backwards; roster-only nodes survive.
const fixture = {nodes:['a','b','c','d'].map(id => ({id})),edges:[['a','b'],['b','c']]};
const full = survivingGraph(fixture);
assert.deepEqual(shortestPath(full,'a','c'),['a','b','c']);
assert.equal(shortestPath(full,'c','a'),null);
assert.deepEqual(shortestPath(full,'c','a',false),['c','b','a']);
assert.deepEqual(shortestPath(full,'d','d'),['d']);
assert.equal(shortestPath(full,'a','d',false),null);
const cut = survivingGraph(fixture,new Set(['b']));
assert.equal(cut.isolates,3);
assert.equal(cut.largest,1);
assert.equal(shortestPath(cut,'a','c',false),null);
assert.equal(shortestPath(cut,'b','b'),null);
assert.equal(survivingGraph(fixture,new Set(['a','b','c','d'])).largest,0);

// Independently compare the live JavaScript calculations with NetworkX exports.
const data = JSON.parse(fs.readFileSync(new URL('../assets/plots/explorer.json',import.meta.url),'utf8'));
assert.equal(survivingGraph(data).nodes.length,303);
assert.equal(survivingGraph(data).edges.length,1784);
assert.equal(survivingGraph(data).isolates,17);
for (const [order, expected] of [[data.attack.targetOrder,data.attack.target],[data.attack.randomOrder,data.attack.randomExample]]) {
  for (let k = 0; k <= data.attack.limit; k++) {
    const actual = survivingGraph(data,new Set(order.slice(0,k)));
    assert.equal(actual.largest,expected[k],`NetworkX comparison at removal ${k}`);
    assert.equal(actual.nodes.length,303-k);
  }
}
console.log('Passed: directed paths, unreachable/removed endpoints, isolates, empty graph, and 122 NetworkX removal comparisons.');
