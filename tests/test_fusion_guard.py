"""FusionGuard: persistent drift is identified, down-weighted, isolated,
re-admitted after recovery, and every step is auditable."""
import pytest

from aegisrover.sensors.fusion import Measurement, fuse
from aegisrover.sensors.fusion_guard import (
    DEGRADED, ISOLATED, PROBATION, TRUSTED, FusionGuard, GuardConfig)
from aegisrover.storage.audit import AuditLog
from aegisrover.storage.repository import Repository


def cycle(guard, values, variance=0.01):
    return guard.update([Measurement(name, value, variance) for name, value in values.items()])


# ------------------------------------------------------------------ drift
def test_persistent_drift_is_downweighted_then_isolated():
    guard = FusionGuard()
    guarded = []
    for step in range(30):
        drift = 0.0 if step < 5 else 0.05 * (step - 4)
        guarded.append(cycle(guard, {'a': 10.0, 'b': 10.05, 'c': 10.0 + drift}).value)
    assert guard.status()['c']['status'] == ISOLATED
    assert guard.status()['a']['status'] == TRUSTED
    assert guard.status()['b']['status'] == TRUSTED
    # the fused value stayed with the healthy sources instead of following c
    assert guarded[-1] == pytest.approx(10.025, abs=0.05)
    kinds = [t.status for t in guard.history()]
    assert DEGRADED in kinds and ISOLATED in kinds
    assert kinds.index(DEGRADED) < kinds.index(ISOLATED)  # down-weighted first


def test_two_sources_drag_plain_fuse_but_anchor_holds():
    # today's behaviour: nothing arbitrates between two sources, the fused
    # value is silently pulled to the midpoint
    dragged = fuse([Measurement('anchor', 10.0, 0.01), Measurement('d', 11.0, 0.01)])
    assert dragged.value == pytest.approx(10.5)

    guard = FusionGuard(GuardConfig(anchor='anchor'))
    result = None
    for _ in range(20):
        result = cycle(guard, {'anchor': 10.0, 'd': 10.5})
    assert guard.status()['anchor']['status'] == TRUSTED
    assert guard.status()['d']['status'] == ISOLATED
    assert result.value == pytest.approx(10.0)


def test_two_sources_without_anchor_degrade_symmetrically_and_alert():
    guard = FusionGuard()
    for _ in range(10):
        cycle(guard, {'a': 10.0, 'b': 10.4})
    statuses = guard.status()
    # undecidable from data alone: both are penalised and the dispute is logged
    assert statuses['a']['status'] != TRUSTED
    assert statuses['b']['status'] != TRUSTED
    assert guard.history()


# ------------------------------------------------------------------ recovery
def test_recovers_through_probation_after_stability_returns():
    guard = FusionGuard()
    for _ in range(15):
        cycle(guard, {'a': 10.0, 'b': 10.05, 'c': 11.0})  # c stuck high
    assert guard.status()['c']['status'] == ISOLATED

    for _ in range(4):
        cycle(guard, {'a': 10.0, 'b': 10.05, 'c': 10.02})  # healthy again
    assert guard.status()['c']['status'] == ISOLATED  # not re-admitted yet

    result = cycle(guard, {'a': 10.0, 'b': 10.05, 'c': 10.02})
    assert guard.status()['c']['status'] == PROBATION
    assert result.fusion.weights['c'] < 0.2  # small re-entry weight

    for _ in range(5):
        result = cycle(guard, {'a': 10.0, 'b': 10.05, 'c': 10.02})
    assert guard.status()['c']['status'] == TRUSTED
    assert result.fusion.weights['c'] == pytest.approx(1 / 3, abs=0.02)


def test_reoffence_during_probation_goes_straight_back_to_isolated():
    guard = FusionGuard()
    for _ in range(3):
        cycle(guard, {'a': 10.0, 'b': 10.05, 'c': 11.0})
    for _ in range(5):
        cycle(guard, {'a': 10.0, 'b': 10.05, 'c': 10.02})
    assert guard.status()['c']['status'] == PROBATION

    result = cycle(guard, {'a': 10.0, 'b': 10.05, 'c': 12.0})
    assert guard.status()['c']['status'] == ISOLATED
    assert result.value == pytest.approx(10.025, abs=0.05)


def test_single_spike_does_not_isolate_the_source():
    guard = FusionGuard()
    for _ in range(3):
        cycle(guard, {'a': 10.0, 'b': 10.05, 'c': 10.02})
    result = cycle(guard, {'a': 10.0, 'b': 10.05, 'c': 10.4})  # one bad read
    assert 'c' in result.fusion.rejected
    assert guard.status()['c']['status'] == TRUSTED
    result = cycle(guard, {'a': 10.0, 'b': 10.05, 'c': 10.02})
    assert result.fusion.weights['c'] == pytest.approx(1 / 3, abs=0.02)


# ------------------------------------------------------------------ fallback
def test_fallback_when_everything_isolated_then_recovery_path():
    guard = FusionGuard()
    result = None
    for _ in range(15):
        result = cycle(guard, {'a': 10.0, 'b': 12.0})
    assert guard.status()['a']['status'] == ISOLATED
    assert guard.status()['b']['status'] == ISOLATED
    assert result.fallback
    assert result.value in (10.0, 12.0)  # least-bad source, clearly flagged

    for _ in range(12):
        result = cycle(guard, {'a': 10.0, 'b': 10.0})
    assert not result.fallback
    statuses = {s['status'] for s in guard.status().values()}
    assert statuses <= {PROBATION, TRUSTED}


# ------------------------------------------------------------------ audit
def test_transitions_are_written_to_the_hash_chained_audit_log():
    audit = AuditLog(Repository(':memory:'))
    guard = FusionGuard(audit=audit)
    for _ in range(12):
        cycle(guard, {'a': 10.0, 'b': 10.05, 'c': 11.0})
    for _ in range(12):
        cycle(guard, {'a': 10.0, 'b': 10.05, 'c': 10.02})

    entries = list(audit.entries())
    actions = [e.action for e in entries]
    assert 'fusion.source.isolated' in actions
    assert 'fusion.source.probation' in actions
    assert 'fusion.source.trusted' in actions
    assert {e.subject for e in entries} == {'c'}
    assert all(e.actor == 'fusion-guard' for e in entries)
    assert all('reason' in e.payload and 'score' in e.payload for e in entries)
    assert audit.verify() == ()
    # the in-memory history and the persisted trail tell the same story
    assert [t.status for t in guard.history()] == [
        e.action.rsplit('.', 1)[1] for e in entries]


def test_operator_reset_is_audited():
    audit = AuditLog(Repository(':memory:'))
    guard = FusionGuard(audit=audit)
    for _ in range(10):
        cycle(guard, {'a': 10.0, 'b': 10.05, 'c': 11.0})
    assert guard.status()['c']['status'] == ISOLATED
    guard.reset('c')
    assert guard.status()['c']['status'] == TRUSTED
    assert any(e.action == 'fusion.source.reset' for e in audit.entries())
    with pytest.raises(KeyError):
        guard.reset('nonexistent')


# ------------------------------------------------------------------ config
def test_config_validation():
    with pytest.raises(ValueError):
        GuardConfig(degrade_score=3.0, isolate_score=2.0)
    with pytest.raises(ValueError):
        GuardConfig(score_half_life=0.0)
    with pytest.raises(ValueError):
        GuardConfig(min_weight=0.0)
    with pytest.raises(ValueError):
        FusionGuard().update([])
