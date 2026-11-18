"""Tests for src/pathsnap/graph.py (DATA-03): one success, one failure case."""

import numpy as np

from pathsnap.graph import RoadGraph, route_hop_stats


def _toy_graph():
    # Chain 0->1->2->3 (arcs 0,1,2), reverse arcs 3,4,5, and a detached
    # component 4->5 (arc 6). Node 6 is isolated.
    nodes = np.array(
        [
            [77.000, 13.000],
            [77.001, 13.000],
            [77.002, 13.000],
            [77.003, 13.000],
            [77.010, 13.010],
            [77.011, 13.010],
            [77.020, 13.020],
        ]
    )
    arcs = np.array(
        [
            [0, 1],  # 0
            [1, 2],  # 1
            [2, 3],  # 2
            [1, 0],  # 3: reverse of 0
            [2, 1],  # 4: reverse of 1
            [3, 2],  # 5: reverse of 2
            [4, 5],  # 6: detached component
        ]
    )
    return RoadGraph(nodes, arcs)


def test_successors_reverse_and_hops():
    g = _toy_graph()
    # Successors include the reverse arc: it also starts at this arc's to_node.
    assert set(g.successors(0)) == {1, 3}
    assert set(g.successors(1)) == {2, 4}
    assert g.reverse_of(0) == 3
    assert g.reverse_of(6) == -1  # one-way detached road

    hops = g.pairwise_hops(np.array([0]), np.array([0, 1, 2, 6]))
    assert hops[0, 0] == 0  # same arc
    assert hops[0, 1] == 1  # head-to-tail
    assert hops[0, 2] == 2  # two hops
    assert hops[0, 3] == -1  # unreachable: detached component

    stats = g.component_stats()
    assert stats["n_components"] == 3  # chain, detached edge, isolated node
    assert stats["n_isolated_nodes"] == 1


def test_unreachable_pair_is_minus_one():
    # Failure case: hops requested between arcs in different components
    # stay -1 and route stats count them as unreachable.
    g = _toy_graph()
    route = np.array([0, 1, 6])
    stats = route_hop_stats(route, g)
    assert stats["n_transitions"] == 2
    assert stats["one_hop"] == 1
    assert stats["unreachable"] == 1
