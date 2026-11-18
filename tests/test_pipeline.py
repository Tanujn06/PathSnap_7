"""PIPE-01 contract tests. Tiny fitted model is a test fixture, not experiment evidence."""
import builtins
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from pathsnap.features import FEATURE_COLUMNS
from pathsnap.pipeline import Predictor, load_prediction_trip, prediction_candidates
from pathsnap.candidates import candidates_for_trip


@pytest.fixture
def setup(tmp_path):
    folder=tmp_path/'00000001';folder.mkdir()
    (folder/'00000001.track').write_text('0 0 0\n0.0001 0 1\n0.0002 0 2\n')
    (folder/'00000001.nodes').write_text('0 0\n0.0001 0\n0.0002 0\n')
    (folder/'00000001.arcs').write_text('0 1\n1 2\n')
    model=DummyClassifier(strategy='prior')
    model.fit(pd.DataFrame(np.zeros((4,13)),columns=FEATURE_COLUMNS),[0,1,0,1])
    joblib.dump(model,tmp_path/'model.joblib')
    (tmp_path/'model.json').write_text(json.dumps({'feature_columns':FEATURE_COLUMNS}))
    (tmp_path/'hmm.json').write_text(json.dumps(dict(sigma_m=12.5,lambda_per_hop=4.,epsilon=100.,w=1.,gap_threshold_s=10.)))
    p=Predictor(tmp_path/'model.joblib',tmp_path/'model.json',tmp_path/'hmm.json')
    return folder,p


def test_success_without_reference_and_never_read_route(setup,monkeypatch):
    folder,p=setup
    without=p.predict_trip(folder)
    (folder/'00000001.route').write_text('invalid reference must not be read')
    original=builtins.open
    def guarded(file,*args,**kwargs):
        if str(file).endswith('.route'):raise AssertionError('reference leakage')
        return original(file,*args,**kwargs)
    monkeypatch.setattr(builtins,'open',guarded)
    with_invalid=p.predict_trip(folder)
    assert without['predicted_arcs']==with_invalid['predicted_arcs']
    assert len(with_invalid['predicted_arcs'])==3
    assert with_invalid['mock'] is False
    assert all(v is None for v in with_invalid['metrics'].values())
    assert with_invalid['hmm_params']['lambda_per_hop']==4
    json.dumps(with_invalid,allow_nan=False)


def test_candidates_equal_upstream(setup):
    folder,p=setup
    pd.testing.assert_frame_equal(prediction_candidates(load_prediction_trip(folder)),candidates_for_trip(folder))


def test_missing_input_fails(setup):
    folder,p=setup
    (folder/'00000001.track').unlink()
    with pytest.raises(FileNotFoundError):p.predict_trip(folder)


def test_no_candidates_fails_instead_of_faking_route(setup):
    folder,p=setup
    (folder/'00000001.track').write_text('1 1 0\n1.001 1 1\n')
    with pytest.raises(ValueError,match='No candidate'):p.predict_trip(folder)


@pytest.mark.parametrize('table,value',[
 ('track','nan 0 1\n'),('track','0 0 2\n0 0 1\n'),
 ('arcs','0 99\n'),('track','1 2\n')])
def test_invalid_inputs_fail(setup,table,value):
    folder,p=setup
    (folder/('00000001.'+table)).write_text(value)
    with pytest.raises(ValueError):p.predict_trip(folder)


def test_model_schema_rejected(setup,tmp_path):
    folder,p=setup
    (tmp_path/'model.json').write_text(json.dumps({'feature_columns':FEATURE_COLUMNS[::-1]}))
    with pytest.raises(ValueError,match='feature order'):
        Predictor(tmp_path/'model.joblib',tmp_path/'model.json',tmp_path/'hmm.json')


def test_runtime_mismatch_is_not_silent(setup,monkeypatch):
    import warnings
    from sklearn.exceptions import InconsistentVersionWarning
    folder,p=setup
    load=joblib.load
    def warn_load(path):
        warnings.warn(InconsistentVersionWarning(estimator_name='fixture',current_sklearn_version='current',original_sklearn_version='other'))
        return load(path)
    monkeypatch.setattr(joblib,'load',warn_load)
    with pytest.raises(RuntimeError,match='versions differ'):
        Predictor(p.model_path,p.model_path.parent/'model.json',p.model_path.parent/'hmm.json')
    diagnostic=Predictor(p.model_path,p.model_path.parent/'model.json',p.model_path.parent/'hmm.json',allow_version_mismatch=True)
    assert diagnostic.predict_trip(folder)['warnings']


def test_default_resave_provenance_guards(setup,tmp_path,monkeypatch):
    import pathsnap.pipeline as pipeline
    import hashlib,shutil,sklearn
    folder,p=setup
    (tmp_path/'models').mkdir();(tmp_path/'configs').mkdir()
    shutil.copy(p.model_path,tmp_path/'models/selected.joblib')
    shutil.copy(p.model_path,tmp_path/'models/original.joblib')
    shutil.copy(tmp_path/'model.json',tmp_path/'models/config.json')
    shutil.copy(tmp_path/'hmm.json',tmp_path/'configs/hmm_params.json')
    digest=hashlib.sha256(p.model_path.read_bytes()).hexdigest()
    runtime=dict(artifact='models/selected.joblib',sha256=digest,
                 original_artifact='models/original.joblib',original_sha256=digest,
                 tested_sklearn=sklearn.__version__,limitation='fixture provenance caveat')
    config=tmp_path/'models/runtime.json';config.write_text(json.dumps(runtime))
    monkeypatch.setattr(pipeline,'ROOT',tmp_path)
    result=pipeline.Predictor().predict_trip(folder)
    assert result['model_provenance']==runtime
    assert 'fixture provenance caveat' in result['warnings']
    runtime['tested_sklearn']='untested';config.write_text(json.dumps(runtime))
    with pytest.raises(RuntimeError,match='validated only'):pipeline.Predictor()
    runtime['tested_sklearn']=sklearn.__version__;config.write_text(json.dumps(runtime))
    (tmp_path/'models/selected.joblib').write_bytes(b'modified')
    with pytest.raises(ValueError,match='hash mismatch'):pipeline.Predictor()
