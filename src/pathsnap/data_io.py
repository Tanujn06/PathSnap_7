"""data_io.py - trip loaders and local projection (DATA-01).

Owner: Simon C A. Branch: feature/data-loaders.

Loads the four per-trip raw files described in the Dataset Exploration PDF:

- ``<trip>.track``  : longitude, latitude, timestamp (seconds), ~1 Hz
- ``<trip>.nodes``  : longitude, latitude; 0-based row index is the node id
- ``<trip>.arcs``   : from_node_id, to_node_id; 0-based row index is the arc id
- ``<trip>.route``  : chronological arc_id sequence (optional)

and builds the per-trip local metric projection agreed in the sprint plan
(section 4.1): x = (lon - lon0) * cos(lat0) * 111320, y = (lat - lat0) *
110574, with lon0/lat0 at the trip centroid. Trips are spread across the
world, so no global projected CRS applies; at trip scale (a few km) this
local projection is accurate to well within GPS noise.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

M_PER_DEG_LAT = 110574.0
M_PER_DEG_LON = 111320.0
EARTH_RADIUS_M = 6371000.0

TRACK_DTYPE = np.float64
INT_DTYPE = np.int64


@dataclass
class Trip:
    """One loaded trip. ``route`` is None when the .route file is absent."""

    trip_id: str
    track: np.ndarray  # (n_obs, 3): lon, lat, t
    nodes: np.ndarray  # (n_nodes, 2): lon, lat
    arcs: np.ndarray  # (n_arcs, 2): from_node_id, to_node_id
    route: np.ndarray | None  # (n_route_arc,): arc_id sequence


def _load(path: Path, dtype, ndmin: int, required: bool = True) -> np.ndarray | None:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"missing required trip file: {path}")
        return None
    return np.loadtxt(path, dtype=dtype, ndmin=ndmin)


def load_trip(trip_dir: str | Path) -> Trip:
    """Load one trip folder containing the four raw files."""
    trip_dir = Path(trip_dir)
    if not trip_dir.is_dir():
        raise FileNotFoundError(f"not a trip directory: {trip_dir}")
    trip_id = trip_dir.name
    track = _load(trip_dir / f"{trip_id}.track", TRACK_DTYPE, ndmin=2)
    nodes = _load(trip_dir / f"{trip_id}.nodes", TRACK_DTYPE, ndmin=2)
    arcs = _load(trip_dir / f"{trip_id}.arcs", INT_DTYPE, ndmin=2)
    route = _load(trip_dir / f"{trip_id}.route", INT_DTYPE, ndmin=1, required=False)
    return Trip(trip_id=trip_id, track=track, nodes=nodes, arcs=arcs, route=route)


def list_trips(root: str | Path) -> list[Path]:
    """Return the sorted trip directories under a split root."""
    root = Path(root)
    return sorted(p for p in root.iterdir() if p.is_dir())


def trip_origin(trip: Trip) -> tuple[float, float]:
    """Projection origin (lon0, lat0): the centroid of the trip's GPS track."""
    lon0 = float(np.mean(trip.track[:, 0]))
    lat0 = float(np.mean(trip.track[:, 1]))
    return lon0, lat0


def project(
    lon: np.ndarray, lat: np.ndarray, lon0: float, lat0: float
) -> tuple[np.ndarray, np.ndarray]:
    """Project lon/lat to local metres around (lon0, lat0)."""
    x = (np.asarray(lon, dtype=np.float64) - lon0) * np.cos(np.radians(lat0)) * M_PER_DEG_LON
    y = (np.asarray(lat, dtype=np.float64) - lat0) * M_PER_DEG_LAT
    return x, y


def arc_geometry(
    nodes: np.ndarray, arcs: np.ndarray, lon0: float, lat0: float
) -> dict[str, np.ndarray]:
    """Projected arc geometry: start xy, end xy, length (m), bearing (deg).

    Bearing is degrees clockwise from north, in [0, 360).
    """
    start_lonlat = nodes[arcs[:, 0]]
    end_lonlat = nodes[arcs[:, 1]]
    sx, sy = project(start_lonlat[:, 0], start_lonlat[:, 1], lon0, lat0)
    ex, ey = project(end_lonlat[:, 0], end_lonlat[:, 1], lon0, lat0)
    dx = ex - sx
    dy = ey - sy
    length = np.hypot(dx, dy)
    bearing = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0
    start_xy = np.column_stack([sx, sy])
    end_xy = np.column_stack([ex, ey])
    return {"start_xy": start_xy, "end_xy": end_xy, "length": length, "bearing": bearing}


def haversine_m(
    lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray
) -> np.ndarray:
    """Great-circle distance in metres; reference for projection checks."""
    lon1, lat1, lon2, lat2 = (
        np.radians(np.asarray(a, dtype=np.float64)) for a in (lon1, lat1, lon2, lat2)
    )
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    h = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(h))
