"""Reproduce nearest-reference coverage checks on high-residual TRAIN trips.
This is development analysis only, never an inference component.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .data_io import load_trip, trip_origin, project, arc_geometry
from .labels import align_route


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-root',required=True)
    p.add_argument('--out',default='results/labels/reference_coverage.csv')
    args=p.parse_args();rows=[]
    for tid in ['00000054','00000033','00000002']:
        trip=load_trip(Path(args.train_root)/tid)
        origin=trip_origin(trip)
        x,y=project(trip.track[:,0],trip.track[:,1],*origin)
        points=np.column_stack([x,y])
        g=arc_geometry(trip.nodes,trip.arcs,*origin)
        a=g['start_xy'][trip.route];v=g['end_xy'][trip.route]-a
        denominator=np.maximum((v*v).sum(1),1e-30)
        distances=[]
        for offset in range(0,len(points),128):
            d=points[offset:offset+128,None,:]-a
            fraction=np.clip(np.einsum('cmi,mi->cm',d,v)/denominator,0,1)
            distances.extend(np.linalg.norm(d-fraction[:,:,None]*v,axis=2).min(1))
        aligned=align_route(trip)
        rows.append(dict(trip_id=tid,n_obs=len(points),
                         nearest_reference_median_m=float(np.median(distances)),
                         nearest_reference_p95_m=float(np.percentile(distances,95)),
                         alignment_median_m=float(aligned.ref_dist_m.median()),
                         alignment_p95_m=float(aligned.ref_dist_m.quantile(.95))))
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out,index=False)
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':
    main()
