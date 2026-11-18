"""Tests for src/pathsnap/hmm.py (HMM-01): one success, one failure case."""

import numpy as np
import pandas as pd

from pathsnap.graph import RoadGraph
from pathsnap.hmm import HMMParams, decode, standin_emission_logp


def _two_way_graph():
    # One two-way road: arc 0 = 0->1 (east), arc 1 = 1->0 (west).
    nodes = np.array([[77.0, 13.0], [77.001, 13.0]])
    arcs = np.array([[0, 1], [1, 0]])
    return RoadGraph(nodes, arcs)


def _cands(dist_arc0, dist_arc1, obs=(0, 1, 2, 3, 4)):
    rows = []
    for i, o in enumerate(obs):
        rows.append((o, 0, dist_arc0[i]))
        rows.append((o, 1, dist_arc1[i]))
    return pd.DataFrame(rows, columns=["obs_idx", "arc_id", "dist_m"])


def test_viterbi_smooths_noisy_nearest_arc():
    # The nearest arc flips to the wrong direction at obs 3 (4 m vs 5 m);
    # the HMM should stay on arc 0 the whole way because two direction
    # switches cost 2 * lambda_per_hop each, more than the emission gain,
    # and arc 1 is a poor match (20 m) at every other observation.
    g = _two_way_graph()
    c = _cands([5, 5, 5, 5, 5], [20, 20, 4, 20, 20])
    em = standin_emission_logp(c["dist_m"].to_numpy(), 12.5)
    decoded = decode(c, em, g, HMMParams())
    assert decoded.tolist() == [0, 0, 0, 0, 0]

    # With transitions free (zero lambda and epsilon), the decoder falls
    # back to per-observation argmin, which flips at obs 2 - proving the
    # smoothing above came from the transition model.
    hard = decode(c, em, g, HMMParams(lambda_per_hop=0.0, epsilon=0.0))
    assert hard.tolist() == [0, 0, 1, 0, 0]


def test_missing_candidates_filled_and_unreachable_handled():
    # Failure case: obs 1 has no candidates at all, and the arcs on either
    # side of it live in disconnected components (unreachable pair) - the
    # decoder must still return one arc per observation.
    nodes = np.array([[77.0, 13.0], [77.001, 13.0], [77.010, 13.010], [77.011, 13.010]])
    arcs = np.array([[0, 1], [2, 3]])
    g = RoadGraph(nodes, arcs)
    c = pd.DataFrame(
        [(0, 0, 5.0), (2, 1, 5.0)], columns=["obs_idx", "arc_id", "dist_m"]
    )
    em = standin_emission_logp(c["dist_m"].to_numpy(), 12.5)
    decoded = decode(c, em, g, HMMParams(), n_obs=3)
    assert decoded.shape == (3,)
    assert (decoded >= 0).all()
    assert decoded[0] == 0 and decoded[2] == 1
    assert decoded[1] in (0, 1)  # filled from the nearest decoded arc


def test_gap_starts_new_segment():
    g = _two_way_graph()
    c = _cands([5, 5, 5, 5], [6, 6, 6, 6], obs=(0, 1, 2, 3))
    em = standin_emission_logp(c["dist_m"].to_numpy(), 12.5)
    times = np.array([1.0, 2.0, 100.0, 101.0])  # 98 s gap between obs 1 and 2
    decoded = decode(c, em, g, HMMParams(gap_threshold_s=5.0), obs_times=times)
    assert decoded.tolist() == [0, 0, 0, 0]
