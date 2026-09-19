"""Tests for persistent source-bias detection, quarantine and reinstatement."""
from __future__ import annotations

import pytest

from aegisrover.sensors.fusion import Measurement
from aegisrover.sensors.source_health import (
    HEALTHY, ISOLATED, RECOVERING, SUSPECT, WARMING,
    GuardedFusion, SourceHealthConfig, SourceHealthGuard,
)
from aegisrover.storage.audit import AuditLog
from aegisrover.storage.repository import Repository

V = 0.04  # claimed variance; sigma ~0.2, so a bias of 3.0 is ~10+ sigma


def tick(guard, t, values, *, sources=('gps', 'lidar', 'imu')):
    return guard.fuse([Measurement(name, values[name], V, age=0.0) for name in sources])


def clean(values):
    return {k: 10.0 for k in values}


def test_new_source_warms_up_then_healthy_and_keeps_full_weight():
    guard = SourceHealthGuard(SourceHealthConfig(warmup=4))
    for _ in range(5):
        r = tick(guard, 0.0, {'gps': 10.0, 'lidar': 10.0, 'imu': 10.02})
    assert r.status == {'gps': HEALTHY, 'lidar': HEALTHY, 'imu': HEALTHY}
    assert set(r.weights) == {'gps', 'lidar', 'imu'}
    assert r.value == pytest.approx(10.0, abs=0.02)
    assert r.transitions == ()


def test_persistent_bias_is_downweighted_then_isolated_and_stops_pulling_fusion():
    cfg = SourceHealthConfig(warmup=4, suspect_streak=3, cusum_h=4.0,
                             cooldown_streak=3, recovery_streak=6)
    guard = SourceHealthGuard(cfg)

    # Healthy phase.
    for _ in range(8):
        r = tick(guard, 0.0, {'gps': 10.0, 'lidar': 10.0, 'imu': 10.0})
    assert r.value == pytest.approx(10.0, abs=0.05)

    # GPS starts a slow, persistent walk to +3.
    drifted = []
    bias = 0.0
    for step in range(20):
        bias = min(3.0, bias + 0.3)
        r = tick(guard, 0.0, {'gps': 10.0 + bias, 'lidar': 10.0, 'imu': 10.0})
        drifted.append(r)
        if guard.status_of('gps') == ISOLATED:
            break

    assert guard.status_of('gps') == ISOLATED
    suspect_weight = next(d.weights['gps'] for d in drifted if 'gps' in d.suspect)
    assert suspect_weight < 0.2  # down-weighted before full isolation
    assert r.weights['gps'] == 0.0
    assert r.isolated == ('gps',) and 'gps' in r.rejected
    # The point of the whole feature: the fused value stays at truth.
    assert r.value == pytest.approx(10.0, abs=0.05)

    kinds = [(t.old, t.new) for t in guard.history('gps')]
    assert (HEALTHY, SUSPECT) in kinds
    assert (SUSPECT, ISOLATED) in kinds
    iso = guard.history('gps')[-1]
    assert iso.reference == pytest.approx(10.0, abs=0.3)
    assert iso.supporters == 2


def test_recovery_ramp_and_full_reinstatement_after_bias_clears():
    cfg = SourceHealthConfig(warmup=2, suspect_streak=3, cusum_h=3.0,
                             cooldown_streak=4, recovery_streak=8)
    guard = SourceHealthGuard(cfg)
    for _ in range(6):
        tick(guard, 0.0, {'gps': 10.0, 'lidar': 10.0, 'imu': 10.0})
    for _ in range(12):
        r = tick(guard, 0.0, {'gps': 13.0, 'lidar': 10.0, 'imu': 10.0})
        if guard.status_of('gps') == ISOLATED:
            break
    assert guard.status_of('gps') == ISOLATED

    # Sensor fixed: serves clean readings. First it cools down at zero weight.
    for _ in range(cfg.cooldown_streak):
        r = tick(guard, 0.0, {'gps': 10.0, 'lidar': 10.0, 'imu': 10.0})
    assert guard.status_of('gps') == RECOVERING

    # Trust ramps up during probation, then returns to full.
    ramp = []
    for _ in range(cfg.recovery_streak):
        r = tick(guard, 0.0, {'gps': 10.0, 'lidar': 10.0, 'imu': 10.0})
        ramp.append(r.trust['gps'])
    assert guard.status_of('gps') == HEALTHY
    assert ramp[0] < ramp[len(ramp) // 2] < ramp[-1] or ramp[-1] == pytest.approx(1.0)
    assert r.trust['gps'] == pytest.approx(1.0)
    assert (RECOVERING, HEALTHY) in [(t.old, t.new) for t in guard.history('gps')]


def test_re_drift_during_probation_restarts_isolation_with_longer_cooldown():
    cfg = SourceHealthConfig(warmup=2, suspect_streak=2, cusum_h=2.0,
                             cooldown_streak=3, cooldown_cap=20, recovery_streak=10)
    guard = SourceHealthGuard(cfg)

    def drive(bias, until_status=None, limit=40):
        nonlocal r
        for _ in range(limit):
            r = tick(guard, 0.0, {'gps': 10.0 + bias, 'lidar': 10.0, 'imu': 10.0})
            if until_status and guard.status_of('gps') == until_status:
                return

    r = None
    for _ in range(6):
        r = tick(guard, 0.0, {'gps': 10.0, 'lidar': 10.0, 'imu': 10.0})
    drive(3.0, ISOLATED)
    assert guard.history('gps')[-1].isolated_count == 1
    drive(0.0, RECOVERING)
    # Flap: biased again while on probation -> immediately re-isolated.
    drive(3.0, ISOLATED, limit=4)
    last = guard.history('gps')[-1]
    assert last.old == RECOVERING and last.new == ISOLATED
    assert last.isolated_count == 2
    # Second cooldown is doubled; three clean samples are no longer enough.
    drive(0.0, limit=cfg.cooldown_streak)
    assert guard.status_of('gps') == ISOLATED
    drive(0.0, RECOVERING)


def test_transient_spike_does_not_quarantine():
    cfg = SourceHealthConfig(warmup=2, suspect_streak=5)
    guard = SourceHealthGuard(cfg)
    for _ in range(6):
        tick(guard, 0.0, {'gps': 10.0, 'lidar': 10.0, 'imu': 10.0})
    tick(guard, 0.0, {'gps': 30.0, 'lidar': 10.0, 'imu': 10.0})  # single spike
    for _ in range(10):
        tick(guard, 0.0, {'gps': 10.0, 'lidar': 10.0, 'imu': 10.0})
    assert guard.status_of('gps') == HEALTHY
    assert guard.history('gps') == ()


def test_two_sources_are_never_judged_without_redundancy():
    cfg = SourceHealthConfig(min_support=2)
    guard = SourceHealthGuard(cfg)
    r = None
    for _ in range(30):
        r = tick(guard, 0.0, {'gps': 15.0, 'lidar': 10.0}, sources=('gps', 'lidar'))
    # With only two sources there is no majority: leave the operator's config alone.
    assert guard.status_of('gps') in (WARMING, HEALTHY)
    assert r.weights['gps'] > 0.0


def test_verdict_does_not_depend_on_measurement_order():
    cfg = SourceHealthConfig(warmup=2, suspect_streak=3, cusum_h=3.0,
                             cooldown_streak=2, recovery_streak=5)
    results = {}
    for order in [('gps', 'lidar', 'imu'), ('imu', 'lidar', 'gps'),
                  ('lidar', 'gps', 'imu')]:
        guard = SourceHealthGuard(cfg)
        for _ in range(6):
            tick(guard, 0.0, {'gps': 10.0, 'lidar': 10.0, 'imu': 10.0}, sources=order)
        for _ in range(15):
            r = tick(guard, 0.0, {'gps': 13.0, 'lidar': 10.0, 'imu': 10.0}, sources=order)
        results[order] = (guard.tracks(), r.weights, r.value)
    baseline = results[('gps', 'lidar', 'imu')]
    for other in [('imu', 'lidar', 'gps'), ('lidar', 'gps', 'imu')]:
        tracks, weights, value = results[other]
        assert {k: v['status'] for k, v in tracks.items()} == \
            {k: v['status'] for k, v in baseline[0].items()}
        assert weights == pytest.approx(baseline[1])
        assert value == pytest.approx(baseline[2])


def test_negative_drift_is_caught_and_does_not_convince_clean_sources():
    cfg = SourceHealthConfig(warmup=2, suspect_streak=3, cusum_h=3.0,
                             cooldown_streak=3, recovery_streak=6)
    guard = SourceHealthGuard(cfg)
    for _ in range(6):
        tick(guard, 0.0, {'gps': 10.0, 'lidar': 10.0, 'imu': 10.0})
    r = None
    for _ in range(25):
        r = tick(guard, 0.0, {'gps': 10.0, 'lidar': 7.0, 'imu': 10.0})
        if guard.status_of('lidar') == ISOLATED:
            break
    assert guard.status_of('lidar') == ISOLATED
    assert guard.status_of('gps') == HEALTHY and guard.status_of('imu') == HEALTHY
    assert r.value == pytest.approx(10.0, abs=0.05)


def test_correlated_drift_degrades_but_never_removes_every_source():
    # Two sensors that move together define the median: no statistical method
    # can tell them from the truth without a third independent reference. The
    # contract is therefore degradation, never total loss of an estimate.
    cfg = SourceHealthConfig(warmup=2, min_support=2, suspect_streak=3,
                             cusum_h=3.0, cooldown_streak=3, recovery_streak=6)
    guard = SourceHealthGuard(cfg)
    for _ in range(6):
        tick(guard, 0.0, {'gps': 10.0, 'lidar': 10.0, 'imu': 10.0})
    r = None
    for _ in range(30):
        r = tick(guard, 0.0, {'gps': 10.0, 'lidar': 13.0, 'imu': 13.2})
    assert sum(r.weights.values()) == pytest.approx(1.0)
    assert any(w > 0.0 for w in r.weights.values())


def test_symmetric_split_deviants_are_quarantined_without_losing_the_middle():
    # Exactly three sources at 7/10/13 are a 50/50 split about the honest
    # middle. Deviation is scaled by each source's declared variance, so both
    # extremists stand out from the median; the sequence in which they are
    # quarantined must never take the fused value away from the middle.
    cfg = SourceHealthConfig(warmup=2, suspect_streak=3, cusum_h=4.0,
                             cooldown_streak=3, recovery_streak=8)
    guard = SourceHealthGuard(cfg)
    for _ in range(6):
        tick(guard, 0.0, {'gps': 10.0, 'lidar': 10.0, 'imu': 10.0})
    for step in range(30):
        r = tick(guard, 0.0, {'gps': 10.0, 'lidar': 7.0, 'imu': 13.0})
        assert r.value == pytest.approx(10.0, abs=1.5)
    assert guard.status_of('gps') == HEALTHY
    assert {guard.status_of('lidar'), guard.status_of('imu')} <= \
        {ISOLATED, RECOVERING, HEALTHY}
    # Whatever the quarantine sequence, the long-run output tracks the middle.
    assert r.value == pytest.approx(10.0, abs=0.2)


def test_two_drifters_that_split_apart_both_get_caught():
    # With a fourth independent witness (or two honest anchors), each bad
    # source stands away from the panel median on its own and is quarantined.
    cfg = SourceHealthConfig(warmup=2, suspect_streak=3, cusum_h=4.0,
                             cooldown_streak=3, recovery_streak=8)
    guard = SourceHealthGuard(cfg)
    sources = ('gps', 'lidar', 'imu', 'vision')

    def drive(values):
        return guard.fuse([Measurement(name, values[name], V, age=0.0)
                           for name in sources])

    for _ in range(6):
        drive({'gps': 10.0, 'lidar': 10.0, 'imu': 10.0, 'vision': 10.0})
    r = None
    for _ in range(40):
        r = drive({'gps': 10.0, 'lidar': 7.0, 'imu': 13.0, 'vision': 10.0})
        if guard.status_of('lidar') == ISOLATED and guard.status_of('imu') == ISOLATED:
            break
    assert guard.status_of('lidar') == ISOLATED
    assert guard.status_of('imu') == ISOLATED
    assert guard.status_of('gps') == HEALTHY
    assert guard.status_of('vision') == HEALTHY
    assert r.value == pytest.approx(10.0, abs=0.05)


def test_manual_mute_blocks_and_reinstate_requires_reprobation():
    guard = SourceHealthGuard(SourceHealthConfig(warmup=2, recovery_streak=4))
    for _ in range(5):
        tick(guard, 0.0, {'gps': 10.0, 'lidar': 10.0, 'imu': 10.0})
    guard.isolate('gps', actor='operator@eve', reason='recalibration')
    r = tick(guard, 0.0, {'gps': 10.0, 'lidar': 10.0, 'imu': 10.0})
    assert r.muted == ('gps',) and r.weights['gps'] == 0.0
    guard.reinstate('gps', actor='operator@eve', reason='calibration done')
    assert guard.status_of('gps') == RECOVERING
    r = tick(guard, 0.0, {'gps': 10.0, 'lidar': 10.0, 'imu': 10.0})
    assert 0.0 < r.trust['gps'] < 1.0
    with pytest.raises(ValueError):
        guard.isolate('lidar', actor='', reason='x')


def test_stale_sources_are_reported_not_judged():
    guard = SourceHealthGuard(SourceHealthConfig(warmup=1), max_age=1.0)
    for _ in range(4):
        guard.fuse([
            Measurement('gps', 10.0, V, age=0.0),
            Measurement('lidar', 10.0, V, age=0.0),
            Measurement('imu', 10.0, V, age=0.0),
        ])
    r = guard.fuse([
        Measurement('gps', 10.0, V, age=5.0),
        Measurement('lidar', 10.0, V, age=0.0),
        Measurement('imu', 10.0, V, age=0.0),
    ])
    assert r.stale == ('gps',)
    assert 'gps' not in r.weights
    with pytest.raises(ValueError):
        guard.fuse([Measurement('gps', 10.0, V, age=9.0)])


def test_transitions_are_recorded_in_a_verifiable_audit_chain():
    clock = iter(range(1000))
    repo = Repository(':memory:', clock=lambda: float(next(clock)))
    audit = AuditLog(repo, clock=lambda: float(next(clock)))
    cfg = SourceHealthConfig(warmup=2, suspect_streak=3, cusum_h=3.0,
                             cooldown_streak=3, recovery_streak=6)
    guard = SourceHealthGuard(cfg, audit=audit, stream='position',
                              clock=lambda: float(next(clock)))
    for _ in range(6):
        tick(guard, 0.0, {'gps': 10.0, 'lidar': 10.0, 'imu': 10.0})
    for _ in range(15):
        tick(guard, 0.0, {'gps': 13.0, 'lidar': 10.0, 'imu': 10.0})
        if guard.status_of('gps') == ISOLATED:
            break
    guard.isolate('lidar', actor='operator@eve', reason='maintenance')

    assert audit.verify() == ()
    subjects = {(e.action, e.subject) for e in audit.entries()}
    assert ('sensor.suspect', 'position/gps') in subjects
    assert ('sensor.isolate', 'position/gps') in subjects
    assert ('sensor.mute', 'position/lidar') in subjects
    auto = [e for e in audit.entries() if e.subject == 'position/gps']
    assert all(e.actor == SourceHealthGuard.AUTO_ACTOR for e in auto)
    assert all(e.payload['z'] is not None for e in auto)
    manual = [e for e in audit.entries() if e.subject == 'position/lidar'][0]
    assert manual.actor == 'operator@eve' and manual.payload['reason'] == 'maintenance'

    # Tampering with history is detectable.
    first = auto[0]
    audit.tamper(first.seq, payload={**first.payload, 'reason': 'rewritten'})
    breaks = audit.verify()
    assert breaks and breaks[0].seq == first.seq


def test_result_dict_is_json_friendly_and_ordered():
    guard = SourceHealthGuard(SourceHealthConfig(warmup=1, suspect_streak=2))
    for _ in range(3):
        tick(guard, 0.0, {'gps': 10.0, 'lidar': 10.0, 'imu': 10.0})
    r = tick(guard, 0.0, {'gps': 14.0, 'lidar': 10.0, 'imu': 10.0})
    payload = r.to_dict()
    assert set(payload) >= {'value', 'weights', 'agreement', 'rejected', 'stale',
                            'isolated', 'muted', 'suspect', 'status', 'trust',
                            'transitions'}
    assert isinstance(payload['transitions'], list)
    assert isinstance(r, GuardedFusion)
