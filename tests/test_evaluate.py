import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace
from pathsnap.evaluate import (macro_f1_candidates,log_loss_candidates,
    candidate_miss_rate,top1_obs_accuracy,top1_predictions,validate_probabilities,
    route_continuity,continuity_counts,sequence_level_f1,summary)


def fixture_tables():
    f=pd.DataFrame({'trip_id':['00000001']*4,'obs_idx':[0,0,1,1],
                    'arc_id':[0,1,0,1],'label':[1,0,0,1]})
    p=f[['trip_id','obs_idx','arc_id']].copy();p['p_match']=[.9,.1,.8,.2]
    l=pd.DataFrame({'trip_id':['00000001']*3,'obs_idx':[0,1,2],'ref_arc':[0,1,2]})
    return f,p,l


def test_hand_calculated_metrics():
    f,p,l=fixture_tables();s=summary(f,p,l)
    assert s['top1']==pytest.approx(1/3)
    assert s['candidate_miss_rate']==pytest.approx(1/3)
    assert s['macro_f1']==pytest.approx(.5)
    assert s['log_loss']==pytest.approx(-np.log([.9,.9,.2,.2]).mean())


def test_probability_join_uses_keys_not_row_order():
    f,p,l=fixture_tables()
    assert summary(f,p,l)==summary(f,p.iloc[::-1],l)
    with pytest.raises(ValueError,match='exactly'):
        validate_probabilities(f,p.iloc[:3])
    with pytest.raises(ValueError,match='unique'):
        validate_probabilities(f,pd.concat([p,p.iloc[:1]]))


def test_ties_choose_smallest_arc_id():
    f,p,l=fixture_tables();p['p_match']=.5
    assert top1_predictions(p).pred_arc.tolist()==[0,0]


def test_continuity_collapses_repeats_and_uses_direct_adjacency():
    g=SimpleNamespace(arcs=np.array([[0,1],[1,2],[2,3],[9,10]]))
    s=continuity_counts([0,0,1,3],g)
    assert s=={'connected_transitions':1,'n_transitions':2,'route_continuity':.5}
    assert route_continuity([0,2],g)==0 # two graph hops, not one
    assert route_continuity([0,0],g)==1
    assert route_continuity([],g) is None


def test_sequence_set_f1_not_sequence_order_score():
    assert sequence_level_f1([0,0,1],[1,2])==.5
    assert sequence_level_f1([1,0],[0,1])==1
    assert sequence_level_f1([],[])==1
    assert sequence_level_f1([],[1])==0


def test_bad_inputs_fail():
    with pytest.raises(ValueError):log_loss_candidates([0,1],[.2,1.1])
    with pytest.raises(ValueError):macro_f1_candidates([],[])
    with pytest.raises(ValueError):macro_f1_candidates([0],[2])
    g=SimpleNamespace(arcs=np.array([[0,1]]))
    with pytest.raises(ValueError):route_continuity([-1],g)
    with pytest.raises(ValueError):route_continuity([1],g)


def test_absent_class_fixed_macro_average_and_extreme_probability():
    assert macro_f1_candidates([0,0],[0,0])==.5
    assert np.isfinite(log_loss_candidates([0,1],[0,1]))


def test_candidate_miss_counts_all_labelled_observations():
    f,p,l=fixture_tables()
    assert candidate_miss_rate(f,l)==pytest.approx(1/3)
    with pytest.raises(ValueError):candidate_miss_rate(f,pd.concat([l,l.iloc[:1]]))
