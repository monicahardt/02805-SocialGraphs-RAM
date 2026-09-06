// Small, pure graph operations used by the live explorer.
export function survivingGraph(data, removed = new Set()) {
  const nodes = data.nodes.filter(n => !removed.has(n.id));
  const ids = new Set(nodes.map(n => n.id));
  const edges = data.edges.filter(([a, b]) => ids.has(a) && ids.has(b));
  const incoming = new Map(nodes.map(n => [n.id, new Set()]));
  const outgoing = new Map(nodes.map(n => [n.id, new Set()]));
  const neighbors = new Map(nodes.map(n => [n.id, new Set()]));
  for (const [a, b] of edges) {
    outgoing.get(a).add(b); incoming.get(b).add(a);
    neighbors.get(a).add(b); neighbors.get(b).add(a);
  }
  const seen = new Set(), components = [];
  for (const {id} of nodes) {
    if (seen.has(id)) continue;
    const component = [], queue = [id]; seen.add(id);
    for (let i = 0; i < queue.length; i++) {
      const current = queue[i]; component.push(current);
      for (const next of neighbors.get(current)) if (!seen.has(next)) {seen.add(next); queue.push(next);}
    }
    components.push(component);
  }
  components.sort((a, b) => b.length - a.length);
  return {nodes, ids, edges, incoming, outgoing, neighbors, components,
    largest: components[0]?.length ?? 0,
    isolates: nodes.filter(n => !incoming.get(n.id).size && !outgoing.get(n.id).size).length};
}

export function shortestPath(graph, source, target, directed = true) {
  if (!graph.ids.has(source) || !graph.ids.has(target)) return null;
  const neighbors = directed ? graph.outgoing : graph.neighbors;
  const previous = new Map([[source, null]]), queue = [source];
  for (let i = 0; i < queue.length; i++) {
    const current = queue[i];
    if (current === target) {
      const path = [];
      for (let n = target; n !== null; n = previous.get(n)) path.push(n);
      return path.reverse();
    }
    for (const next of neighbors.get(current)) if (!previous.has(next)) {previous.set(next, current); queue.push(next);}
  }
  return null;
}
