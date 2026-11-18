"""Tests for src/pathsnap/candidates.py (DATA-02): one success, one failure case."""

import numpy as np
import pytest

from pathsnap.candidates import COLUMNS, candidates_for_trip

# A tiny two-way road: node 0 -> node 1 (arc 0) and node 1 -> node 0 (arc 1),
# about 111 m long, running east at lat 13. GPS points sit ~5 m north of it.
LON0, LAT0 = 77.0, 13.0
LON1 = LON0 + 0.001  # ~108 m east at this latitude
GPS_OFFSET_LAT = 5.0 / 110574.0  # 5 m north


def _write_trip(tmp_path, trip_id="00000003", gps_far=False):
    d = tmp_path / trip_id
    d.mkdir()
    lats = [LAT0 + GPS_OFFSET_LAT] * 3
    if gps_far:
        lats[2] = LAT0 + 0.01  # ~1.1 km away: no arc within 150 m
    track = [[LON0 + 0.0002, lats[0], 1.0], [LON0 + 0.0005, lats[1], 2.0], [LON0 + 0.0008, lats[2], 3.0]]
    np.savetxt(d / f"{trip_id}.track", track)
    np.savetxt(d / f"{trip_id}.nodes", [[LON0, LAT0], [LON1, LAT0]])
    np.savetxt(d / f"{trip_id}.arcs", [[0, 1], [1, 0]], fmt="%d")
    np.savetxt(d / f"{trip_id}.route", [0], fmt="%d")
    return d


def test_candidates_success(tmp_path):
    df = candidates_for_trip(_write_trip(tmp_path))
    assert list(df.columns) == COLUMNS
    obs0 = df[df["obs_idx"] == 0].sort_values("cand_rank")
    # Both directions of the two-way road are candidates.
    assert set(obs0["arc_id"]) == {0, 1}
    # Rank 0 is a nearest arc, about 5 m off, halfway along fractions valid.
    assert obs0["dist_m"].iloc[0] == pytest.approx(5.0, abs=1.0)
    assert obs0["dist_m"].is_monotonic_increasing
    assert obs0["along_frac"].between(0.0, 1.0).all()
    assert obs0["cand_rank"].tolist() == list(range(len(obs0)))


def test_candidates_far_point_gets_none(tmp_path):
    df = candidates_for_trip(_write_trip(tmp_path, gps_far=True))
    # The far-away third observation is beyond the 150 m widened radius:
    # no rows for obs_idx 2, while the near observations still have theirs.
    assert 2 not in set(df["obs_idx"])
    assert {0, 1} <= set(df["obs_idx"])
