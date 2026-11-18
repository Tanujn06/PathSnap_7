import numpy as np
import pandas as pd
import pytest
from pathsnap.data_io import Trip
from pathsnap.features import build_features, candidate_coverage, FEATURE_COLUMNS


def sample():
    t=Trip('00000001',np.array([[0.,0.,0.],[.0001,0.,1.],[.0002,0.,2.]]),np.array([[0.,0.],[.0001,0.],[.0002,0.]]),np.array([[0,1],[1,0],[1,2]]),None)
    c=pd.DataFrame({'trip_id':['00000001']*4,'obs_idx':[0,0,1,1],'arc_id':[0,1,0,1],'dist_m':[1.,1.,2.,2.],'along_frac':[0.,1.,1.,0.],'cand_rank':[0,1,0,1]})
    l=pd.DataFrame({'trip_id':['00000001']*3,'obs_idx':[0,1,2],'ref_arc':[0,0,2]})
    return t,c,l


def test_geometry_heading_and_schema():
    t,c,l=sample();f=build_features(t,c,l)
    assert list(f.columns)==['trip_id','obs_idx','arc_id','label']+FEATURE_COLUMNS
    assert list(f.label)==[1,0,1,0]
    assert f.loc[0,'heading_cos']==pytest.approx(1.)
    assert f.loc[1,'heading_cos']==pytest.approx(-1.)
    assert f.has_reverse_arc.eq(1).all()
    assert f.n_within_20m.eq(2).all()
    assert np.isfinite(f[FEATURE_COLUMNS]).all().all()


def test_inference_has_no_label_and_does_not_read_route():
    t,c,l=sample();t.route=np.array([999999])
    f=build_features(t,c)
    assert 'label' not in f
    assert list(f.columns)==['trip_id','obs_idx','arc_id']+FEATURE_COLUMNS


def test_invalid_candidate_fails():
    t,c,l=sample();c.loc[0,'arc_id']=99
    with pytest.raises(ValueError,match='invalid arc_id'):
        build_features(t,c,l)


def test_missing_label_fails():
    t,c,l=sample()
    with pytest.raises(ValueError,match='missing labels'):
        build_features(t,c,l.iloc[:1])


def test_all_observation_coverage_counts_empty():
    t,c,l=sample();s=candidate_coverage(c,l)
    assert s['n_observations']==3
    assert s['n_missed']==1
    assert s['n_empty_candidate_observations']==1
    assert s['candidate_recall']==pytest.approx(2/3)


def test_stationary_and_empty_candidates():
    t,c,l=sample();t.track[:,0:2]=0
    f=build_features(t,c)
    assert f.heading_cos.eq(0).all()
    assert f.speed_mps.eq(0).all()
    empty=build_features(t,c.iloc[:0])
    assert empty.empty
