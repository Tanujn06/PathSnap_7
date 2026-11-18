import numpy as np
import pytest
from pathsnap.data_io import Trip
from pathsnap.labels import align_route


def tiny(route=None):
    return Trip('test',np.array([[0.,0.,0.],[.0001,0.,1.],[.0002,0.,2.]]),
                np.array([[0.,0.],[.0001,0.],[.0002,0.],[.0003,0.]]),
                np.array([[0,1],[1,2],[2,3]]),np.array([0,1,2]) if route is None else route)


def test_known_monotone_route():
    frame, positions = align_route(tiny(),return_positions=True)
    assert list(frame.columns)==['trip_id','obs_idx','ref_arc','ref_dist_m']
    assert len(frame)==3
    assert np.all(frame.ref_dist_m < 1e-6)
    assert np.all(np.diff(positions)>=0)
    assert np.all(np.diff(positions)<=3)


def test_missing_reference_fails():
    trip=tiny();trip.route=None
    with pytest.raises(ValueError,match='required'):
        align_route(trip)


def test_bad_arc_fails():
    with pytest.raises(ValueError,match='invalid arc'):
        align_route(tiny(np.array([99])))


def test_zero_length_arc_is_finite():
    trip=tiny(np.array([0]));trip.arcs[0]=[0,0]
    frame=align_route(trip)
    assert np.isfinite(frame.ref_dist_m).all()


def test_optimum_matches_exhaustive_small_lattice():
    from itertools import product
    trip=tiny()
    frame,pos=align_route(trip,max_advance=1,return_positions=True)
    from pathsnap.data_io import trip_origin, project, arc_geometry
    origin=trip_origin(trip)
    x,y=project(trip.track[:,0],trip.track[:,1],*origin)
    pts=np.column_stack([x,y]);g=arc_geometry(trip.nodes,trip.arcs,*origin)
    def cost(seq):
        total=0.
        for p,j in zip(pts,seq):
            a=g['start_xy'][j];v=g['end_xy'][j]-a
            f=np.clip(np.dot(p-a,v)/np.dot(v,v),0,1)
            total+=np.linalg.norm(p-a-f*v)
        return total
    valid=[s for s in product(range(3),repeat=3) if all(0<=b-a<=1 for a,b in zip(s,s[1:]))]
    assert cost(pos)==pytest.approx(min(cost(s) for s in valid))
