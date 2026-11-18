"""Tests for src/pathsnap/data_io.py (DATA-01): one success, one failure case."""

import numpy as np
import pytest

from pathsnap.data_io import (
    arc_geometry,
    haversine_m,
    load_trip,
    project,
    trip_origin,
)


def _write_trip(tmp_path, trip_id="00000000"):
    d = tmp_path / trip_id
    d.mkdir()
    # Small synthetic trip: two nodes, one arc, three GPS points, one route arc.
    np.savetxt(d / f"{trip_id}.track", [[77.0000, 13.0000, 1.0], [77.0005, 13.0005, 2.0], [77.0010, 13.0010, 3.0]])
    np.savetxt(d / f"{trip_id}.nodes", [[77.0000, 13.0000], [77.0010, 13.0010]])
    np.savetxt(d / f"{trip_id}.arcs", [[0, 1]], fmt="%d")
    np.savetxt(d / f"{trip_id}.route", [0], fmt="%d")
    return d


def test_load_trip_success(tmp_path):
    trip = load_trip(_write_trip(tmp_path))
    assert trip.trip_id == "00000000"
    assert trip.track.shape == (3, 3)
    assert trip.nodes.shape == (2, 2)
    assert trip.arcs.shape == (1, 2)
    assert trip.route is not None and trip.route.tolist() == [0]

    lon0, lat0 = trip_origin(trip)
    geom = arc_geometry(trip.nodes, trip.arcs, lon0, lat0)
    # Projected arc length matches haversine within 0.5 percent.
    hav = haversine_m(
        trip.nodes[0, 0], trip.nodes[0, 1], trip.nodes[1, 0], trip.nodes[1, 1]
    )
    assert geom["length"][0] == pytest.approx(hav, rel=0.005)
    # Arc runs north-east, so bearing is about 45 degrees.
    assert 30.0 < geom["bearing"][0] < 60.0


def test_load_trip_missing_file_raises(tmp_path):
    d = _write_trip(tmp_path, trip_id="00000001")
    (d / "00000001.track").unlink()
    with pytest.raises(FileNotFoundError):
        load_trip(d)


def test_missing_route_is_none(tmp_path):
    d = _write_trip(tmp_path, trip_id="00000002")
    (d / "00000002.route").unlink()
    trip = load_trip(d)
    assert trip.route is None


def test_projection_matches_haversine():
    # Two points ~1.5 km apart inside one trip's extent.
    lon1, lat1 = 77.5946, 12.9716
    lon2, lat2 = 77.6046, 12.9816
    x, y = project(np.array([lon1, lon2]), np.array([lat1, lat2]), lon1, lat1)
    proj = float(np.hypot(x[1] - x[0], y[1] - y[0]))
    hav = float(haversine_m(lon1, lat1, lon2, lat2))
    assert proj == pytest.approx(hav, rel=0.005)
