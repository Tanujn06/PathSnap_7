"""FEAT-01: geometric, motion and graph features for candidate arcs.

Labels are optional development-only inputs, never derived at inference.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .data_io import load_trip, trip_origin, project, arc_geometry

FEATURE_COLUMNS = ['dist_m','along_frac','cand_rank','dist_minus_min','arc_length',
                   'gps_heading','heading_cos','speed_mps','dt_s','n_within_20m',
                   'arc_in_degree','arc_out_degree','has_reverse_arc']
KEYS = ['trip_id','obs_idx','arc_id']


def build_features(trip, candidates, labels=None):
    """Return one row per supplied candidate in the frozen feature order.

    Motion uses centred neighbours after a three-point coordinate moving
    average. At endpoints use available neighbours. Speed is zero when
    elapsed time is nonpositive. A stationary heading is represented as
    zero degrees and heading_cos=0, indicating no directional evidence.
    Arc degree means in-degree of the start node and out-degree of end node.
    n_within_20m counts only the supplied candidate set, not all nearby arcs.
    """
    required=KEYS+['dist_m','along_frac','cand_rank']
    if not set(required).issubset(candidates.columns):
        raise ValueError('candidate schema is incomplete')
    c=candidates[required].copy()
    if c.duplicated(KEYS).any():
        raise ValueError('duplicate candidate keys')
    if not c.trip_id.eq(str(trip.trip_id)).all():
        raise ValueError('candidate trip_id does not match trip')
    if not all(np.isfinite(c[col]).all() for col in required[1:]):
        raise ValueError('candidate values must be finite')
    for col,limit in [('obs_idx',len(trip.track)),('arc_id',len(trip.arcs))]:
        if ((c[col]<0)|(c[col]>=limit)|(c[col]!=np.floor(c[col]))).any():
            raise ValueError(f'invalid {col}')
    if ((c.dist_m<0)|(c.along_frac<0)|(c.along_frac>1)|(c.cand_rank<0)).any():
        raise ValueError('invalid candidate geometry/rank')
    if len(trip.track)==0 or not np.isfinite(trip.track).all():
        raise ValueError('track must be nonempty and finite')
    obs=c.obs_idx.to_numpy(dtype=np.int64); arc=c.arc_id.to_numpy(dtype=np.int64)
    origin=trip_origin(trip)
    x,y=project(trip.track[:,0],trip.track[:,1],*origin)
    points=np.column_stack((x,y));n=len(points)
    # Edge-padded 3-point mean, then central difference over neighbours.
    pad=np.pad(points,((1,1),(0,0)),mode='edge')
    smooth=(pad[:-2]+pad[1:-1]+pad[2:])/3
    index=np.arange(n);left=np.maximum(index-1,0);right=np.minimum(index+1,n-1)
    movement=smooth[right]-smooth[left]
    norm=np.linalg.norm(movement,axis=1)
    elapsed=trip.track[right,2]-trip.track[left,2]
    speed=np.divide(norm,elapsed,out=np.zeros(n),where=elapsed>0)
    heading=np.mod(np.degrees(np.arctan2(movement[:,0],movement[:,1])),360)
    dt=np.r_[0.,np.diff(trip.track[:,2])]
    geom=arc_geometry(trip.nodes,trip.arcs,*origin)
    cosine=np.cos(np.deg2rad(heading[obs]-geom['bearing'][arc]))
    cosine[(norm[obs]<1e-8)|(geom['length'][arc]<1e-8)]=0
    indeg=np.bincount(trip.arcs[:,1],minlength=len(trip.nodes))
    outdeg=np.bincount(trip.arcs[:,0],minlength=len(trip.nodes))
    edges=set(map(tuple,trip.arcs.tolist()))
    reverse=np.fromiter(((int(v),int(u)) in edges for u,v in trip.arcs),dtype=np.int8,count=len(trip.arcs))
    c['dist_minus_min']=c.dist_m-c.groupby('obs_idx').dist_m.transform('min')
    c['arc_length']=geom['length'][arc]
    c['gps_heading']=heading[obs]
    c['heading_cos']=cosine
    c['speed_mps']=speed[obs]
    c['dt_s']=dt[obs]
    c['n_within_20m']=c.dist_m.le(20).groupby(c.obs_idx).transform('sum').astype(np.int64)
    c['arc_in_degree']=indeg[trip.arcs[arc,0]]
    c['arc_out_degree']=outdeg[trip.arcs[arc,1]]
    c['has_reverse_arc']=reverse[arc]
    columns=KEYS.copy()
    if labels is not None:
        if not {'trip_id','obs_idx','ref_arc'}.issubset(labels.columns):
            raise ValueError('label schema is incomplete')
        lab=labels[labels.trip_id.eq(str(trip.trip_id))]
        if lab.duplicated(['trip_id','obs_idx']).any():
            raise ValueError('duplicate label keys')
        c=c.merge(lab[['trip_id','obs_idx','ref_arc']],on=['trip_id','obs_idx'],how='left',validate='many_to_one')
        if c.ref_arc.isna().any():
            raise ValueError('missing labels for candidate observations')
        c['label']=(c.arc_id==c.ref_arc).astype(np.int8)
        columns+=['label']
    return c[columns+FEATURE_COLUMNS]


def candidate_coverage(candidates, labels):
    """All-label denominator, counting observations with no candidate as misses."""
    if labels.duplicated(['trip_id','obs_idx']).any():
        raise ValueError('duplicate label keys')
    pairs=candidates[['trip_id','obs_idx','arc_id']].drop_duplicates()
    matched=labels.merge(pairs,left_on=['trip_id','obs_idx','ref_arc'],right_on=KEYS,how='left',indicator=True)
    covered=labels[['trip_id','obs_idx']].merge(candidates[['trip_id','obs_idx']].drop_duplicates(),on=['trip_id','obs_idx'],how='left',indicator=True)
    total=len(labels);hits=int((matched['_merge']=='both').sum())
    return {'n_observations':total,'n_reference_in_candidates':hits,'n_missed':total-hits,
            'n_empty_candidate_observations':int((covered['_merge']=='left_only').sum()),
            'candidate_recall':hits/total if total else None,
            'candidate_miss_rate':1-hits/total if total else None}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',required=True)
    p.add_argument('--cache',default='data_cache')
    p.add_argument('--out',default='results/features')
    args=p.parse_args();cache=Path(args.cache);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    for split in ['train','student_test']:
        c=pd.read_parquet(cache/f'cands_{split}.parquet')
        labels=pd.read_parquet(cache/f'labels_{split}.parquet')
        tables=[]
        for tid,group in c.groupby('trip_id',sort=True):
            trip=load_trip(Path(args.data_root)/split/tid)
            tables.append(build_features(trip,group,labels))
        table=pd.concat(tables,ignore_index=True)
        if not np.isfinite(table[FEATURE_COLUMNS].to_numpy()).all():
            raise ValueError('nonfinite generated features')
        table.to_parquet(cache/f'features_{split}.parquet',index=False)
        summary=candidate_coverage(c,labels)
        summary.update(n_candidate_rows=len(table),n_positive_rows=int(table.label.sum()),feature_columns=FEATURE_COLUMNS)
        (out/f'{split}_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(split,summary,flush=True)
    Path('configs').mkdir(exist_ok=True)
    Path('configs/feature_columns.json').write_text(json.dumps({'feature_columns':FEATURE_COLUMNS},indent=2)+'\n')

if __name__=='__main__':
    main()
