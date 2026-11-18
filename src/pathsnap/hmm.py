"""hmm.py - HMM/Viterbi decoder (HMM-01: v1 with stand-in emissions).

Owner: Simon C A. Branch: feature/hmm-viterbi.

Design (sprint plan 8.3):
- Hidden states: at each observation, its candidate arcs (up to 10). Arc ids
  are kept so the decoded path maps back to road segments.
- Emission: log(clip(p_match, 1e-6)) from the selected model times weight w
  in HMM-02. The v1 stand-in is -d^2 / (2 sigma^2) from dist_m.
- Transition: same arc = best score; connected in 1 to 3 hops = score
  reduced by lambda per hop; unreachable = a large finite penalty epsilon,
  so decoding never fails.
- Missing candidates / GPS gaps: when dt exceeds gap_threshold_s or an
  observation has no candidates, a new segment starts with no transition
  across the gap; empty observations are filled from the nearest decoded
  arc.

Interface: decode(cands, emission_logp, graph, params) returns one arc per
observation.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pathsnap.graph import RoadGraph


@dataclass
class HMMParams:
    """Decoder parameters. Frozen at gate G8 (HMM-02); defaults are the v1 stand-in."""

    sigma_m: float = 12.5  # stand-in emission width (plan: about 10 to 15 m)
    lambda_per_hop: float = 2.0  # transition penalty per hop (1 to 3 hops)
    epsilon: float = 50.0  # large finite penalty for unreachable transitions
    w: float = 1.0  # emission weight (tuned on train + student_test in HMM-02)
    gap_threshold_s: float = 5.0  # dt above this starts a new segment


def standin_emission_logp(dist_m: np.ndarray, sigma_m: float) -> np.ndarray:
    """HMM-01 stand-in emission log-probability from candidate distance."""
    return -(np.asarray(dist_m, dtype=np.float64) ** 2) / (2.0 * sigma_m**2)


def _viterbi_segment(
    cand_arcs: list[np.ndarray], emission: list[np.ndarray], graph: RoadGraph, params: HMMParams
) -> list[np.ndarray]:
    """Log-space Viterbi with back-pointers over one gap-free segment."""
    T = len(cand_arcs)
    delta = [emission[0] * params.w]
    backptr: list[np.ndarray] = []
    for t in range(1, T):
        hops = graph.pairwise_hops(cand_arcs[t - 1], cand_arcs[t])
        trans = np.where(
            hops == 0,
            0.0,
            np.where(hops > 0, -params.lambda_per_hop * hops, -params.epsilon),
        )
        scores = delta[t - 1][:, None] + trans
        best = np.argmax(scores, axis=0)
        backptr.append(best)
        delta.append(scores[best, np.arange(len(cand_arcs[t]))] + emission[t] * params.w)
    path = [int(np.argmax(delta[-1]))]
    for t in range(T - 2, -1, -1):
        path.append(int(backptr[t][path[-1]]))
    path.reverse()
    return [cand_arcs[t][path[t]] for t in range(T)]


def decode(
    cands: pd.DataFrame,
    emission_logp: np.ndarray,
    graph: RoadGraph,
    params: HMMParams | None = None,
    obs_times: np.ndarray | None = None,
    n_obs: int | None = None,
) -> np.ndarray:
    """Decode one trip: returns one arc id per observation (0..n_obs-1).

    cands: candidate rows for one trip (contract columns; obs_idx selects
    the observation). emission_logp: one log-probability per cands row,
    already clipped; pass standin_emission_logp(dist_m, sigma) for v1.
    obs_times: per-observation timestamps (seconds) for gap detection; when
    None every observation is treated as 1 s apart. Observations with no
    candidates are filled from the nearest decoded arc in the same segment.
    """
    if params is None:
        params = HMMParams()
    if n_obs is None:
        n_obs = int(cands["obs_idx"].max()) + 1 if len(cands) else 0
    if obs_times is None:
        obs_times = np.arange(n_obs, dtype=np.float64)

    by_obs: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for obs_idx, grp in zip(cands["obs_idx"].to_numpy(), range(len(cands))):
        by_obs.setdefault(int(obs_idx), []).append(grp)
    order_rows = cands[["obs_idx", "arc_id"]].to_numpy()

    # Segment boundaries: start a new segment after a big time gap or at an
    # observation with no candidates.
    segments: list[list[int]] = []
    current: list[int] = []
    prev_t = None
    for i in range(n_obs):
        if i not in by_obs:
            if current:
                segments.append(current)
                current = []
            prev_t = None
            continue
        t_i = float(obs_times[i])
        if prev_t is not None and t_i - prev_t > params.gap_threshold_s:
            if current:
                segments.append(current)
                current = []
        current.append(i)
        prev_t = t_i
    if current:
        segments.append(current)

    decoded = np.full(n_obs, -1, dtype=np.int64)
    for seg in segments:
        cand_arcs, emission = [], []
        for obs in seg:
            rows = by_obs[obs]
            cand_arcs.append(order_rows[rows, 1].astype(np.int64))
            emission.append(np.asarray(emission_logp)[rows].astype(np.float64))
        arcs = _viterbi_segment(cand_arcs, emission, graph, params)
        for obs, arc in zip(seg, arcs):
            decoded[obs] = arc

    # Fill empty observations from the nearest decoded arc.
    if np.any(decoded < 0) and np.any(decoded >= 0):
        have = np.where(decoded >= 0)[0]
        for i in np.where(decoded < 0)[0]:
            nearest = have[np.argmin(np.abs(have - i))]
            decoded[i] = decoded[nearest]
    return decoded
