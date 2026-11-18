"""EDA-01: dataset inventory, data-quality checks and first Folium map.

Reads the raw PathSnap trips with plain numpy (no dependency on DATA-01
loaders) and writes tables, plots and one Folium HTML map to results/eda/.

Usage:
    python viz/eda.py --data-root "data/Student folder" --out results/eda
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csgraph

M_PER_DEG_LAT = 110_574.0
M_PER_DEG_LON_AT_EQUATOR = 111_320.0


def trip_dirs(root, partition):
    base = os.path.join(root, partition)
    return sorted(
        (os.path.join(base, d) for d in os.listdir(base)
         if os.path.isdir(os.path.join(base, d))),
        key=lambda p: os.path.basename(p),
    )


def load_trip(tdir):
    """Load one trip's raw files. Returns dict of arrays (route may be None)."""
    tid = os.path.basename(tdir)
    out = {"trip_id": tid}
    for ext, kw in (("track", 3), ("nodes", 2), ("arcs", 2)):
        path = os.path.join(tdir, f"{tid}.{ext}")
        out[ext] = np.loadtxt(path, ndmin=2) if os.path.exists(path) else None
        out[f"{ext}_bytes"] = os.path.getsize(path) if os.path.exists(path) else 0
    rpath = os.path.join(tdir, f"{tid}.route")
    if os.path.exists(rpath):
        out["route"] = np.loadtxt(rpath, dtype=np.int64, ndmin=1)
        out["route_bytes"] = os.path.getsize(rpath)
    else:
        out["route"] = None
        out["route_bytes"] = 0
    return out


def local_xy(lon, lat, lon0, lat0):
    """Local metric projection centred on (lon0, lat0), per the team contract."""
    x = (lon - lon0) * np.cos(np.deg2rad(lat0)) * M_PER_DEG_LON_AT_EQUATOR
    y = (lat - lat0) * M_PER_DEG_LAT
    return x, y


def network_stats(nodes, arcs):
    """Connected components and isolated nodes of the trip road graph."""
    n = len(nodes)
    a = arcs.astype(np.int64)
    # undirected graph for connectivity
    rows = np.concatenate([a[:, 0], a[:, 1]])
    cols = np.concatenate([a[:, 1], a[:, 0]])
    g = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    n_comp, labels = csgraph.connected_components(g, directed=False)
    deg = np.asarray(g.sum(axis=1)).ravel()
    return n_comp, int((deg == 0).sum())


def route_connectivity(route, arcs):
    """Share of consecutive route arcs that join head-to-tail."""
    if route is None or len(route) < 2:
        return 1.0, 0, 0
    heads = arcs[route[:-1], 1]
    tails = arcs[route[1:], 0]
    ok = heads == tails
    return float(ok.mean()), int(ok.sum()), int(len(ok))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True,
                    help="folder containing train/ and student_test/")
    ap.add_argument("--out", default="results/eda")
    ap.add_argument("--map-partition", default="train")
    ap.add_argument("--map-trip", default=None,
                    help="trip id for the Folium map (default: first train trip)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    inv_rows, dt_rows, route_rows, net_rows, qual_rows = [], [], [], [], []
    all_dt = []
    for partition in ("train", "student_test"):
        for tdir in trip_dirs(args.data_root, partition):
            t = load_trip(tdir)
            track, nodes, arcs, route = t["track"], t["nodes"], t["arcs"], t["route"]
            n_obs = len(track)
            inv_rows.append({
                "partition": partition, "trip_id": t["trip_id"],
                "n_gps_points": n_obs,
                "n_nodes": len(nodes), "n_arcs": len(arcs),
                "n_route_arcs": 0 if route is None else len(route),
                "track_bytes": t["track_bytes"], "nodes_bytes": t["nodes_bytes"],
                "arcs_bytes": t["arcs_bytes"], "route_bytes": t["route_bytes"],
            })
            # time gaps
            ts = track[:, 2]
            dt = np.diff(ts)
            all_dt.append(dt)
            big = dt[dt > 5]
            dt_rows.append({
                "partition": partition, "trip_id": t["trip_id"],
                "n_steps": len(dt), "n_dt_gt_5s": int(len(big)),
                "max_dt_s": float(dt.max()) if len(dt) else 0.0,
                "pct_dt_1s": float((dt == 1).mean() * 100) if len(dt) else 100.0,
                "timestamps_monotonic": bool((dt > 0).all()),
            })
            # data quality
            lon, lat = track[:, 0], track[:, 1]
            qual_rows.append({
                "partition": partition, "trip_id": t["trip_id"],
                "nan_values": int(np.isnan(track).sum() + np.isnan(nodes).sum()
                                  + np.isnan(arcs).sum()),
                "lon_min": float(lon.min()), "lon_max": float(lon.max()),
                "lat_min": float(lat.min()), "lat_max": float(lat.max()),
                "lon_plausible": bool(((lon >= -180) & (lon <= 180)).all()),
                "lat_plausible": bool(((lat >= -90) & (lat <= 90)).all()),
            })
            # road network connectivity
            n_comp, n_iso = network_stats(nodes, arcs)
            net_rows.append({
                "partition": partition, "trip_id": t["trip_id"],
                "connected_components": n_comp, "isolated_nodes": n_iso,
            })
            # reference route
            frac, n_ok, n_trans = route_connectivity(route, arcs)
            route_rows.append({
                "partition": partition, "trip_id": t["trip_id"],
                "route_arcs": 0 if route is None else len(route),
                "connected_transitions": n_ok, "total_transitions": n_trans,
                "frac_head_to_tail": frac,
            })

    inv = pd.DataFrame(inv_rows)
    dt_df = pd.DataFrame(dt_rows)
    qual = pd.DataFrame(qual_rows)
    net = pd.DataFrame(net_rows)
    rts = pd.DataFrame(route_rows)

    inv.to_csv(os.path.join(args.out, "inventory.csv"), index=False)
    dt_df.to_csv(os.path.join(args.out, "time_gaps.csv"), index=False)
    qual.to_csv(os.path.join(args.out, "data_quality.csv"), index=False)
    net.to_csv(os.path.join(args.out, "network_connectivity.csv"), index=False)
    rts.to_csv(os.path.join(args.out, "route_connectivity.csv"), index=False)

    summary = {
        "trips_train": int((inv.partition == "train").sum()),
        "trips_student_test": int((inv.partition == "student_test").sum()),
        "total_gps_points_train": int(inv.loc[inv.partition == "train", "n_gps_points"].sum()),
        "total_gps_points_student_test": int(inv.loc[inv.partition == "student_test", "n_gps_points"].sum()),
        "gps_points_per_trip_train_min": int(inv.loc[inv.partition == "train", "n_gps_points"].min()),
        "gps_points_per_trip_train_max": int(inv.loc[inv.partition == "train", "n_gps_points"].max()),
        "arcs_per_trip_min": int(inv.n_arcs.min()),
        "arcs_per_trip_max": int(inv.n_arcs.max()),
        "arcs_per_trip_median": float(inv.n_arcs.median()),
        "nodes_per_trip_median": float(inv.n_nodes.median()),
        "nan_values_total": int(qual.nan_values.sum()),
        "all_lon_plausible": bool(qual.lon_plausible.all()),
        "all_lat_plausible": bool(qual.lat_plausible.all()),
        "lon_range": [float(qual.lon_min.min()), float(qual.lon_max.max())],
        "lat_range": [float(qual.lat_min.min()), float(qual.lat_max.max())],
        "steps_dt_gt_5s_total": int(dt_df.n_dt_gt_5s.sum()),
        "max_dt_s": float(dt_df.max_dt_s.max()),
        "max_dt_trip": str(dt_df.loc[dt_df.max_dt_s.idxmax(), "trip_id"]),
        "max_dt_partition": str(dt_df.loc[dt_df.max_dt_s.idxmax(), "partition"]),
        "route_transitions_head_to_tail": int(rts.connected_transitions.sum()),
        "route_transitions_total": int(rts.total_transitions.sum()),
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- plots (train partition unless noted) ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_dt_cat = np.concatenate(all_dt) if all_dt else np.array([1.0])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(np.clip(all_dt_cat, 0, 10), bins=np.arange(0, 11) - 0.5, log=True)
    ax[0].set_xlabel("dt (s, clipped at 10)")
    ax[0].set_ylabel("count (log)")
    ax[0].set_title("Sampling interval distribution (both partitions)")
    ax[1].hist(inv.loc[inv.partition == "train", "n_gps_points"], bins=30)
    ax[1].set_xlabel("GPS points per trip")
    ax[1].set_ylabel("trips")
    ax[1].set_title("Trip length distribution (train)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "sampling_and_lengths.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for partition, marker in (("train", "."), ("student_test", "x")):
        q = qual[qual.partition == partition]
        ax.plot((q.lon_min + q.lon_max) / 2, (q.lat_min + q.lat_max) / 2,
                marker, label=partition)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("Trip centre locations (both partitions)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "trip_locations.png"), dpi=120)
    plt.close(fig)

    # ---- Folium map of one trip (train partition by default) ----
    import folium
    map_partition = args.map_partition
    if args.map_trip:
        map_dir = os.path.join(args.data_root, map_partition, args.map_trip)
    else:
        map_dir = trip_dirs(args.data_root, map_partition)[0]
    t = load_trip(map_dir)
    track, nodes, arcs, route = t["track"], t["nodes"], t["arcs"], t["route"]
    lat0, lon0 = float(track[:, 1].mean()), float(track[:, 0].mean())
    m = folium.Map(location=[lat0, lon0], zoom_start=14)

    # clip road network to ~300 m around the track bounding box
    pad = 300.0
    xn, yn = local_xy(nodes[:, 0], nodes[:, 1], lon0, lat0)
    xt, yt = local_xy(track[:, 0], track[:, 1], lon0, lat0)
    in_box = ((xn > xt.min() - pad) & (xn < xt.max() + pad)
              & (yn > yt.min() - pad) & (yn < yt.max() + pad))
    keep_nodes = np.where(in_box)[0]
    keep_set = set(keep_nodes.tolist())
    a = arcs.astype(np.int64)
    seg = [(a[i, 0], a[i, 1]) for i in range(len(a))
           if a[i, 0] in keep_set and a[i, 1] in keep_set]
    MAX_SEG = 20000
    if len(seg) > MAX_SEG:
        idx = np.random.RandomState(42).choice(len(seg), MAX_SEG, replace=False)
        seg = [seg[i] for i in idx]
    fg_net = folium.FeatureGroup(name="road network (clipped 300 m)", show=True)
    for u, v in seg:
        folium.PolyLine(
            [[nodes[u, 1], nodes[u, 0]], [nodes[v, 1], nodes[v, 0]]],
            color="grey", weight=1, opacity=0.5).add_to(fg_net)
    fg_net.add_to(m)

    if route is not None:
        fg_ref = folium.FeatureGroup(name="reference route", show=True)
        rcoords = []
        for aid in route:
            u, v = arcs[int(aid)].astype(np.int64)
            rcoords.append([nodes[u, 1], nodes[u, 0]])
            rcoords.append([nodes[v, 1], nodes[v, 0]])
        folium.PolyLine(rcoords, color="green", weight=3,
                        opacity=0.9).add_to(fg_ref)
        fg_ref.add_to(m)

    fg_gps = folium.FeatureGroup(name="GPS observations", show=True)
    for lo, la, _ in track[:: max(1, len(track) // 3000)]:
        folium.CircleMarker([la, lo], radius=2, color="blue",
                            fill=True, fill_opacity=0.7).add_to(fg_gps)
    fg_gps.add_to(m)
    folium.LayerControl().add_to(m)
    map_path = os.path.join(args.out, f"map_{map_partition}_{t['trip_id']}.html")
    m.save(map_path)

    print("EDA summary:")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote tables, plots and Folium map to {args.out}/")
    print(f"Map: {map_path} (partition={map_partition}, trip={t['trip_id']})")


if __name__ == "__main__":
    main()
