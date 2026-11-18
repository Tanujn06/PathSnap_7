"""graph.py - road-graph connectivity and hop tables (DATA-03).

Owner: Simon C A. Branch: feature/road-graph.

Per-trip directed arc graph used by the HMM transition model (plan 8.3):
- successors[arc]: arcs whose from_node equals this arc's to_node
  (head-to-tail, directed).
- reverse_of[arc]: the opposite-direction arc of the same road, if it
  exists.
- Hop distance (BFS, max 3 hops) between candidate arcs of consecutive
  observations, cached per source arc.
- Connected-component and isolated-node counts for the EDA.
"""

from collections import deque

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

MAX_HOPS = 3


class RoadGraph:
    """Directed arc-to-arc connectivity for one trip's road network."""

    def __init__(self, nodes: np.ndarray, arcs: np.ndarray):
        self.nodes = nodes
        self.arcs = arcs
        self.n_arcs = len(arcs)
        # Arcs sorted by from_node so successors(a) is a searchsorted range
        # on the from_node of a's to_node.
        self._order = np.argsort(arcs[:, 0], kind="stable")
        self._sorted_from = arcs[self._order, 0]
        # Packed (from, to) -> arc id for reverse lookup.
        n_nodes = len(nodes)
        self._arc_by_edge = {
            int(f) * n_nodes + int(t): i for i, (f, t) in enumerate(arcs)
        }
        self._n_nodes = n_nodes
        self._hop_cache: dict[int, dict[int, int]] = {}

    def successors(self, arc: int) -> np.ndarray:
        """Arcs whose from_node is this arc's to_node."""
        to_node = int(self.arcs[arc, 1])
        lo = np.searchsorted(self._sorted_from, to_node, side="left")
        hi = np.searchsorted(self._sorted_from, to_node, side="right")
        return self._order[lo:hi]

    def reverse_of(self, arc: int) -> int:
        """The opposite-direction arc id, or -1 when the road is one-way."""
        f, t = int(self.arcs[arc, 0]), int(self.arcs[arc, 1])
        return self._arc_by_edge.get(t * self._n_nodes + f, -1)

    def hop_map(self, src: int, max_hops: int = MAX_HOPS) -> dict[int, int]:
        """BFS from src arc: {arc_id: hop distance} within max_hops (cached)."""
        cached = self._hop_cache.get(src)
        if cached is not None:
            return cached
        dist: dict[int, int] = {}
        queue: deque = deque()
        for nxt in self.successors(src):
            dist[int(nxt)] = 1
            queue.append(int(nxt))
        while queue:
            cur = queue.popleft()
            d = dist[cur]
            if d >= max_hops:
                continue
            for nxt in self.successors(cur):
                nxt = int(nxt)
                if nxt not in dist:
                    dist[nxt] = d + 1
                    queue.append(nxt)
        self._hop_cache[src] = dist
        return dist

    def pairwise_hops(
        self, src_arcs: np.ndarray, dst_arcs: np.ndarray, max_hops: int = MAX_HOPS
    ) -> np.ndarray:
        """Hop distance matrix between candidate sets of consecutive observations.

        Entry [i, j]: 0 when src_arcs[i] == dst_arcs[j] (same arc), 1..max_hops
        when connected within that many hops, -1 when not reachable.
        """
        out = np.full((len(src_arcs), len(dst_arcs)), -1, dtype=np.int8)
        dst_index = {int(a): j for j, a in enumerate(dst_arcs)}
        for i, src in enumerate(src_arcs):
            src = int(src)
            if src in dst_index:
                out[i, dst_index[src]] = 0
            for arc, d in self.hop_map(src, max_hops).items():
                if arc == src:
                    continue  # same arc stays 0 even if a U-turn loop reaches it
                j = dst_index.get(arc)
                if j is not None:
                    out[i, j] = d
        return out

    def component_stats(self) -> dict[str, int]:
        """Connected components and isolated nodes of the undirected node graph."""
        n = self._n_nodes
        edges = np.concatenate([self.arcs, self.arcs[:, ::-1]])
        graph = sparse.csr_matrix(
            (np.ones(len(edges), dtype=np.int8), (edges[:, 0], edges[:, 1])),
            shape=(n, n),
        )
        n_components, labels = csgraph.connected_components(graph, directed=False)
        degree = np.asarray(graph.sum(axis=0)).ravel() + np.asarray(graph.sum(axis=1)).ravel()
        return {
            "n_components": int(n_components),
            "n_isolated_nodes": int((degree == 0).sum()),
            "largest_component_nodes": int(np.bincount(labels).max()) if n else 0,
        }


def route_hop_stats(route: np.ndarray, graph: RoadGraph) -> dict[str, float]:
    """Hop distance between consecutive arcs of a reference route.

    Used by the DATA-03 acceptance check: on reference routes, at least
    99 percent of consecutive arc pairs should be 1 hop apart.
    """
    hops = np.array(
        [graph.pairwise_hops([int(a)], [int(b)])[0, 0] for a, b in zip(route[:-1], route[1:])],
        dtype=np.int8,
    )
    one_hop = int((hops == 1).sum())
    total = len(hops)
    return {
        "n_transitions": total,
        "one_hop": one_hop,
        "one_hop_frac": one_hop / total if total else 1.0,
        "same_arc": int((hops == 0).sum()),
        "unreachable": int((hops == -1).sum()),
    }
