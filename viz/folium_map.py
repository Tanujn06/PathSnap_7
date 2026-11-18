"""PIPE-01: map and evaluate development predictions AFTER inference.

Windows cmd: set PYTHONPATH=src
python viz/folium_map.py --data-root "data/Student folder/student_test"
Use --allow-version-mismatch only for a clearly labelled diagnostic run.
"""
import argparse
import html
import json
from pathlib import Path
import numpy as np
import pandas as pd
import folium
from pathsnap.pipeline import Predictor
from pathsnap.evaluate import route_continuity, sequence_level_f1, top1_predictions
from pathsnap.labels import align_route
from pathsnap.data_io import Trip


def render_map(result, trip, reference, output, scores, case):
    gps = result['gps']
    m = folium.Map(location=gps[0], zoom_start=14, tiles='OpenStreetMap')
    fg = folium.FeatureGroup(name='GPS observations (blue)', show=True)
    for coord in gps:
        folium.CircleMarker(coord, radius=2, color='#2764ad', fill=True, fill_opacity=.6).add_to(fg)
    fg.add_to(m)
    # Clip to the track bbox + approximately300m. No long fictitious bridges.
    from pathsnap.data_io import trip_origin, project
    origin = trip_origin(trip)
    gx, gy = project(trip.track[:,0],trip.track[:,1],*origin)
    nx, ny = project(trip.nodes[:,0],trip.nodes[:,1],*origin)
    keep = (nx>=min(gx)-300)&(nx<=max(gx)+300)&(ny>=min(gy)-300)&(ny<=max(gy)+300)
    net = folium.FeatureGroup(name='Road network (grey,300m bbox)', show=False)
    for u,v in trip.arcs[keep[trip.arcs[:,0]] & keep[trip.arcs[:,1]]]:
        folium.PolyLine(trip.nodes[[u,v],::-1].tolist(),color='#888',weight=1,opacity=.35).add_to(net)
    net.add_to(m)
    if reference is not None:
        ref = folium.FeatureGroup(name='Reference route (green, evaluation only)',show=False)
        for a in np.unique(reference):
            folium.PolyLine(trip.nodes[trip.arcs[a],::-1].tolist(),color='#268747',weight=5,opacity=.6).add_to(ref)
        ref.add_to(m)
    pred = folium.FeatureGroup(name='Model + HMM arcs (red)',show=True)
    for a in np.unique(result['predicted_arcs']):
        folium.PolyLine(trip.nodes[trip.arcs[a],::-1].tolist(),color='#c83439',weight=3,opacity=.85).add_to(pred)
    pred.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[min(x[0] for x in gps),min(x[1] for x in gps)],
                  [max(x[0] for x in gps),max(x[1] for x in gps)]])
    title=html.escape(f"{case}: trip {trip.trip_id}")
    warning=('Tested sklearn1.7.2 resave; original1.9.1 equivalence unproved' if result.get('model_provenance') else
             'DIAGNOSTIC: sklearn version mismatch; numeric audit only' if result['warnings'] else
             'Actual handed-over model + tuned HMM; not GPS stub')
    text=(f'<b>{title}</b><br>{warning}<br>'
          f'All-observation top1: classifier {scores["baseline_top1"]:.3f}, HMM {scores["hmm_top1"]:.3f}<br>'
          f'Whole-trip continuity: {scores["continuity"]:.3f}; set F1: {scores["sequence_f1"]:.3f}<br>'
          'Blue: GPS | grey: roads | green: reference | red: prediction<br>'
          'Arcs drawn separately; gaps are not filled with invented connector roads.<br>'
          'Reference loaded after prediction for development evaluation only.')
    panel=f'<div style="position:fixed;bottom:22px;left:15px;z-index:9999;background:white;padding:12px;border:1px solid #ccc;max-width:540px;font:12px/1.5 Arial">{text}</div>'
    m.get_root().html.add_child(folium.Element(panel));m.save(str(output))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',required=True)
    p.add_argument('--out',default='results/maps')
    p.add_argument('--allow-version-mismatch',action='store_true')
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    model=Predictor(allow_version_mismatch=args.allow_version_mismatch)
    rows=[];saved={};predictions=[];probabilities=[]
    for folder in sorted(Path(args.data_root).iterdir()):
        if not folder.is_dir():continue
        result,trip,scored,graph=model.predict_trip(folder,return_details=True)
        predictions.append(pd.DataFrame({'trip_id':trip.trip_id,'obs_idx':np.arange(len(trip.track)),
                                         'pred_arc':result['predicted_arcs']}))
        probabilities.append(scored[['trip_id','obs_idx','arc_id','p_match']])
        # Prediction complete BEFORE any attempt to read a reference file.
        route_path=folder/(folder.name+'.route')
        if not route_path.is_file():continue
        route=np.loadtxt(route_path,dtype=np.int64,ndmin=1)
        labelled_trip=Trip(trip.trip_id,trip.track,trip.nodes,trip.arcs,route)
        labels=align_route(labelled_trip)
        truth=labels.ref_arc.to_numpy()
        baseline=top1_predictions(scored).set_index('obs_idx').pred_arc.reindex(range(len(truth)))
        row=dict(trip_id=trip.trip_id,n_obs=len(truth),baseline_top1=float(baseline.eq(truth).mean()),
                 hmm_top1=float(np.mean(np.array(result['predicted_arcs'])==truth)),
                 continuity=route_continuity(result['predicted_arcs'],graph),
                 sequence_f1=sequence_level_f1(result['predicted_arcs'],route),
                 runtime_ms=result['latency_ms'],version_mismatch=bool(model.version_warnings),
                 resaved_model=bool(model.provenance))
        rows.append(row);saved[trip.trip_id]=(result,trip,route,row)
        print(json.dumps(row),flush=True)
    if len(rows)<3:raise ValueError('At least3labelled development trips required for distinct examples')
    # Deterministic illustrative selection, never used for model/HMM tuning.
    good=max(rows,key=lambda r:(r['hmm_top1'],r['trip_id']))
    remaining=[r for r in rows if r['trip_id']!=good['trip_id']]
    helped=max(remaining,key=lambda r:(r['hmm_top1']-r['baseline_top1'],r['trip_id']))
    remaining=[r for r in remaining if r['trip_id']!=helped['trip_id']]
    failure=min(remaining,key=lambda r:(r['hmm_top1'],r['trip_id']))
    selected=[]
    for case,row in [('good',good),('hmm_helped',helped),('failure',failure)]:
        result,trip,route,score=saved[row['trip_id']]
        name=f'{case}_{trip.trip_id}.html';render_map(result,trip,route,out/name,score,case)
        selected.append(dict(case=case,path=name,**row))
    pd.DataFrame(rows).to_csv(out/'pipeline_student_test.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_parquet(out/'pipeline_predictions.parquet',index=False)
    pd.concat(probabilities,ignore_index=True).to_parquet(out/'pipeline_probabilities.parquet',index=False)
    (out/'selection.json').write_text(json.dumps({'selection':'deterministic post-hoc illustration, not held-out model selection',
        'reference_labels':'DATA-04 derived monotone alignment; all-observation denominator',
        'continuity':'whole trip, collapsed adjacent repeats; includes cross-gap transitions',
        'hmm_params':vars(model.params),'runtime_version_warnings':model.version_warnings,
        'model_provenance':model.provenance,'maps':selected},indent=2)+'\n')

if __name__=='__main__':main()
