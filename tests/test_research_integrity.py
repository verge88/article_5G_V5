import hashlib
import hmac
import json
import time
import numpy as np
import pandas as pd
import pytest
from experiment.data import features, feature_sets, audit_exclusion, REPORT, OBSERVED, KIE
from experiment.kie import decode_response, key_for
from experiment.models import empirical_threshold, make_detectors
from experiment.evaluate import metrics, event_metrics
from experiment.report import bootstrap, report


@pytest.fixture
def frame():
    rng = np.random.default_rng(5)
    rows = []
    for episode in ['a', 'b']:
        for tick in range(5):
            for role in ['SMF', 'UDM', 'PCF']:
                row = dict(episode=episode, tick=tick, role=role, label=0, scenario='normal')
                row.update({c: float(rng.uniform(.1, 20)) for c in REPORT+OBSERVED+KIE})
                rows.append(row)
    return pd.DataFrame(rows)


def test_full_source_exclusion_including_derived_and_historical_values(frame):
    audits = audit_exclusion(frame)
    assert all(a['proposed_max_change'] == 0 for a in audits)
    assert all(a['naive_max_change'] > 0 for a in audits)


def test_labels_and_controller_metadata_cannot_change_features(frame):
    a = features(frame)
    frame['label'] = 1
    frame['scenario'] = 'stale_report'
    frame['fault_us'] = 123
    frame['target'] = 'SMF'
    pd.testing.assert_frame_equal(a, features(frame))


def test_histories_reset_between_core_restarts(frame):
    x = features(frame)
    first = frame.tick == 0
    assert (x.loc[first, [f'lag_{c}' for c in REPORT+OBSERVED]] == 0).all().all()


def signed_payload(master='test', seq=1):
    now = int(time.time()*1e6)
    payload = f'127.0.0.4|{seq}|'+('a'*32)+f'|{now-250000}|{now}|8|8|0|4000'
    mac = hmac.new(key_for(master, '127.0.0.4'), payload.encode(), hashlib.sha256).hexdigest()
    return json.dumps(dict(payload=payload, mac=mac)).encode()


def test_kie_rejects_tampering_and_wrong_challenge():
    body = signed_payload()
    row = decode_response(body, 'test', '127.0.0.4', 1, 'a'*32)
    assert row['authenticated'] and row['reported_responses'] == 8
    for corrupted, seq, nonce in [(body.replace(b'4000', b'4001'), 1, 'a'*32),
                                  (body, 2, 'a'*32), (body, 1, 'b'*32)]:
        with pytest.raises(ValueError):
            decode_response(corrupted, 'test', '127.0.0.4', seq, nonce)


def test_threshold_has_strict_upper_tail_and_is_finite():
    assert empirical_threshold(np.arange(100), .01) == 99
    with pytest.raises(ValueError):
        empirical_threshold([], .01)


def test_absent_class_not_reported_as_perfect_accuracy():
    m = metrics(pd.DataFrame(dict(label=[0,0], alarm=[False,True], score=[0.,1.])))
    assert m['recall'] is None and m['auroc'] is None and m['fpr'] == .5


def test_missing_seed_blocks_publication(tmp_path):
    cfg = tmp_path/'config.json'
    cfg.write_text(json.dumps(dict(seeds=[1,2], profile='full')))
    with pytest.raises(ValueError, match='Incomplete'):
        report(tmp_path, tmp_path/'report', cfg)


def test_bootstrap_requires_independent_replicates():
    mean, low, high = bootstrap([.5], np.random.default_rng(0))
    assert mean == .5 and low is None and high is None


def test_models_have_no_label_or_identity_inputs():
    for model in make_detectors(1):
        assert not {'label','scenario','episode','tick','fault_us','seed','target'} & set(model.columns)
    assert not set(REPORT) & set(feature_sets()['domain'])


def test_model_family_smoke(frame):
    x = features(frame)
    for detector in make_detectors(1):
        detector.fit(x)
        assert np.isfinite(detector.score(x)).all()
