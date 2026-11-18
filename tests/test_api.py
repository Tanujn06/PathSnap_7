"""API-02 tests use tiny fitted classifier fixtures, not model experiment evidence."""
from pathlib import Path
import shutil
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pathsnap.pipeline import Predictor
from api.main import create_app
from test_pipeline import setup


def client_for(setup,tmp_path):
    folder,predictor=setup
    return TestClient(create_app(folder.parent,tmp_path/'api.log',lambda:predictor)),folder,predictor


def test_real_api_success_no_reference(setup,tmp_path):
    client,folder,p=client_for(setup,tmp_path)
    with client:
        assert client.get('/health').json()['mock'] is False
        assert folder.name in client.get('/trips').json()['trips']
        r=client.post('/predict',json={'trip_id':folder.name})
        assert r.status_code==200
        result=r.json();assert not result['mock']
        assert result['prediction_mode']=='model_hmm'
        assert len(result['predicted_arcs'])==3
        assert all(len(point)==2 and isinstance(point[0],float) for point in result['predicted_path'])
        assert all(len(segment)==2 and len(segment[0])==2 for segment in result['predicted_segments'])
        assert result['metrics']['top1'] is None
        assert result['metrics']['macro_f1'] is None
        assert result['metrics']['route_continuity'] is not None
        assert result['evaluation_status']=='reference_unavailable'
    assert '"status": 200' in (tmp_path/'api.log').read_text()


def test_reference_only_after_prediction(setup,tmp_path,monkeypatch):
    client,folder,p=client_for(setup,tmp_path)
    (folder/(folder.name+'.route')).write_text('0\n1\n')
    predicted=False;original=p.predict_trip
    def prediction(*args,**kwargs):
        nonlocal predicted
        r=original(*args,**kwargs);predicted=True;return r
    monkeypatch.setattr(p,'predict_trip',prediction)
    import api.main as api
    load=api.np.loadtxt
    def guarded(path,*args,**kwargs):
        if str(path).endswith('.route'):assert predicted
        return load(path,*args,**kwargs)
    monkeypatch.setattr(api.np,'loadtxt',guarded)
    with client:
        r=client.post('/predict',json={'trip_id':folder.name}).json()
        assert r['evaluation_status']=='reference_evaluated_after_prediction'
        assert r['metrics']['top1'] is not None
        assert r['metrics']['macro_f1'] is not None
        assert 'ref_arc' not in str(r) and 'reference_route' not in r


def test_api_failures_and_invalid_reference(setup,tmp_path):
    client,folder,p=client_for(setup,tmp_path)
    with client:
        assert client.post('/predict',json={'trip_id':'missing'}).status_code==404
        assert client.post('/predict',json={'trip_id':'../escape'}).status_code==422
        assert client.post('/predict',json={'trip_id':folder.name,'label':1}).status_code==422
        (folder/(folder.name+'.route')).write_text('malformed')
        r=client.post('/predict',json={'trip_id':folder.name}).json()
        assert r['evaluation_status']=='invalid_reference_ignored' and r['metrics']['top1'] is None
        (folder/(folder.name+'.track')).write_text('bad')
        assert client.post('/predict',json={'trip_id':folder.name}).status_code==422


def test_startup_model_failure_not_faked(setup,tmp_path):
    def broken():raise RuntimeError('model guard failed')
    with pytest.raises(RuntimeError,match='model guard'):
        with TestClient(create_app(tmp_path,tmp_path/'api.log',broken)):pass


def test_symlink_trip_not_exposed(setup,tmp_path):
    client,folder,p=client_for(setup,tmp_path)
    (folder.parent/'alias').symlink_to(folder,target_is_directory=True)
    with client:
        assert 'alias' not in client.get('/trips').json()['trips']
        assert client.post('/predict',json={'trip_id':'alias'}).status_code==404
