"""EVAL-01: explicit candidate, observation and route metrics.

Development evaluation only. Reference labels never enter prediction.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss

KEYS = ['trip_id', 'obs_idx', 'arc_id']
OBS_KEYS = ['trip_id', 'obs_idx']


def _binary(y_true, values, probabilities=False):
    y=np.asarray(y_true)
    v=np.asarray(values)
    if y.ndim!=1 or v.ndim!=1 or len(y)!=len(v) or not len(y):
        raise ValueError('inputs must be nonempty equal-length vectors')
    if not np.isin(y,[0,1]).all() or not np.isfinite(v).all():
        raise ValueError('labels must be binary and scores finite')
    if probabilities:
        if ((v<0)|(v>1)).any():
            raise ValueError('probabilities must be in [0,1]')
    elif not np.isin(v,[0,1]).all():
        raise ValueError('predictions must be binary')
    return y.astype(int),v


def macro_f1_candidates(y_true, y_pred):
    """Mean F1 of fixed classes 0 and 1, absent-class F1=0."""
    y,p=_binary(y_true,y_pred)
    return float(f1_score(y,p,labels=[0,1],average='macro',zero_division=0))


def log_loss_candidates(y_true, p_match):
    """Binary natural-log loss; clip to [1e-7,1-1e-7] after validation."""
    y,p=_binary(y_true,p_match,probabilities=True)
    p=np.clip(p.astype(float),1e-7,1-1e-7)
    return float(log_loss(y,np.column_stack((1-p,p)),labels=[0,1]))


def validate_probabilities(features, probabilities):
    """Join G4 scores by exact keys, never positional order. Reject missing/extra."""
    if not set(KEYS).issubset(features.columns) or not set(KEYS+['p_match']).issubset(probabilities.columns):
        raise ValueError('missing probability/feature key columns')
    for name,frame in [('features',features),('probabilities',probabilities)]:
        if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
            raise ValueError(f'{name} keys must be nonnull and unique')
        if not frame.trip_id.map(lambda x:isinstance(x,str)).all():
            raise ValueError('trip_id must be strings, preserving leading zeros')
    if 'p_match' in features.columns:
        raise ValueError('features must not already contain p_match')
    out=features.merge(probabilities[KEYS+['p_match']],on=KEYS,how='outer',indicator=True,validate='one_to_one',sort=False)
    if not out['_merge'].eq('both').all():
        raise ValueError('probabilities must have exactly the feature keys')
    scores=out.p_match.to_numpy()
    if not np.isfinite(scores).all() or ((scores<0)|(scores>1)).any():
        raise ValueError('p_match must be finite probabilities in [0,1]')
    return out.drop(columns='_merge')


def _labels(labels):
    if not set(OBS_KEYS+['ref_arc']).issubset(labels.columns) or labels.empty:
        raise ValueError('nonempty observation reference labels required')
    if labels[OBS_KEYS+['ref_arc']].isna().any().any() or labels.duplicated(OBS_KEYS).any():
        raise ValueError('reference observation keys must be unique and nonnull')


def candidate_miss_rate(candidates, labels):
    """Denominator: all reference observations, including no-candidate points."""
    _labels(labels)
    if not set(KEYS).issubset(candidates.columns):
        raise ValueError('candidate keys missing')
    covered=labels[OBS_KEYS+['ref_arc']].merge(candidates[KEYS].drop_duplicates(),left_on=OBS_KEYS+['ref_arc'],right_on=KEYS,how='left',indicator=True)
    return float(covered['_merge'].eq('left_only').mean())


def top1_predictions(scored):
    """Highest p_match per observation; ties choose smallest arc_id."""
    if not set(KEYS+['p_match']).issubset(scored.columns) or scored.duplicated(KEYS).any():
        raise ValueError('unique scored candidate keys required')
    p=scored.p_match.to_numpy()
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('scores must be probabilities')
    return (scored.sort_values(OBS_KEYS+['p_match','arc_id'],ascending=[True,True,False,True],kind='stable')
            .drop_duplicates(OBS_KEYS)[KEYS].rename(columns={'arc_id':'pred_arc'}))


def top1_obs_accuracy(scored, labels):
    """All-reference-observation denominator. Empty-candidate points are wrong."""
    _labels(labels)
    pred=top1_predictions(scored)
    out=labels.merge(pred,on=OBS_KEYS,how='left',validate='one_to_one')
    return float(out.ref_arc.eq(out.pred_arc).mean())


def _arc_sequence(values, graph=None):
    a=np.asarray(values)
    if a.ndim==1 and len(a)==0:
        return a.astype(np.int64)
    if a.ndim!=1 or not np.issubdtype(a.dtype,np.integer):
        raise ValueError('arc sequence must be a one-dimensional integer array')
    if (a<0).any() or (graph is not None and (a>=len(graph.arcs)).any()):
        raise ValueError('arc id outside road graph')
    return a


def continuity_counts(predicted_arcs, graph):
    """Collapse consecutive repeats; count direct directed arc transitions.

    Empty sequences have no prediction and return score None. A nonempty
    single arc/run has no invalid transition and returns score 1.
    """
    a=_arc_sequence(predicted_arcs,graph)
    if not len(a):
        return {'connected_transitions':0,'n_transitions':0,'route_continuity':None}
    a=a[np.r_[True,a[1:]!=a[:-1]]]
    total=max(len(a)-1,0)
    connected=int((graph.arcs[a[:-1],1]==graph.arcs[a[1:],0]).sum())
    return {'connected_transitions':connected,'n_transitions':total,
            'route_continuity':connected/total if total else 1.0}


def route_continuity(predicted_arcs, graph):
    return continuity_counts(predicted_arcs,graph)['route_continuity']


def sequence_level_f1(predicted_arcs, reference_route):
    """Set overlap F1; ignores traversal order and repeated arc multiplicity."""
    p=set(_arc_sequence(predicted_arcs).tolist())
    r=set(_arc_sequence(reference_route).tolist())
    if not p and not r:
        return 1.0
    return 2*len(p&r)/(len(p)+len(r))


def summary(features, probabilities, labels):
    """Candidate+observation metrics after exact G4 key validation.

    Route metrics are separate per-trip calls: never join different trips or
    bridge missing observations when constructing an arc sequence.
    """
    _labels(labels)
    scored=validate_probabilities(features,probabilities)
    if 'label' not in scored.columns:
        raise ValueError('candidate ground-truth label column required')
    known=scored[OBS_KEYS].drop_duplicates().merge(labels[OBS_KEYS],on=OBS_KEYS,how='left',indicator=True)
    if not known['_merge'].eq('both').all():
        raise ValueError('candidate observation has no reference label')
    truth=scored[KEYS+['label']].merge(labels[OBS_KEYS+['ref_arc']],on=OBS_KEYS,how='left',validate='many_to_one')
    if not truth.label.eq(truth.arc_id.eq(truth.ref_arc).astype(int)).all():
        raise ValueError('candidate labels disagree with observation reference labels')
    y=scored.label.to_numpy();p=scored.p_match.to_numpy()
    miss=candidate_miss_rate(scored,labels)
    return {'n_candidate_rows':len(scored),'n_observations':len(labels),
            'top1':top1_obs_accuracy(scored,labels),
            'macro_f1':macro_f1_candidates(y,(p>=.5).astype(int)),
            'log_loss':log_loss_candidates(y,p),
            'candidate_miss_rate':miss,'candidate_recall':1-miss}
