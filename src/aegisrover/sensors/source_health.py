"""Persistent source-bias guard for sensor fusion.

The plain outlier rejection in :func:`aegisrover.sensors.fusion.fuse` catches a
single spike, but not a sensor that *walks away* from the others: a slowly
drifting source stays inside the pack on every tick and drags the weighted mean
with it until a human notices. This module closes that loop.

Each source is tracked over time against a robust consensus of the panel: the
reference is a trust-weighted median (one extremist cannot move it) and each
source's deviation is normalised by the panel's declared variances. Normalised
deviation feeds a streak/CUSUM persistence detector, so a slow bias that is
small on any single tick accumulates into evidence. The resulting health state
machine first scales the source's fusion weight down (suspect), then removes it
(isolated), and finally admits it back through a cool-down plus a probation
ramp once it has been clean long enough:

    warming -> healthy -> suspect -> isolated -> recovering -> healthy
                                  ^                |
                                  +---- re-drift --+   (cool-down restarts,
                                                       longer each time)

Judgement uses statuses snapshotted at the start of each tick, so verdicts do
not depend on the order measurements arrive in. A healthy source is only
challenged by a healthy majority; a source already on probation can be tested
against a single remaining trusted witness, which also means a correlated
majority failure can degrade the fusion to low-trust suspicion but can never
remove every source and leave the system with no estimate.

Every automatic transition and every manual mute/reinstate is appended to the
hash-chained :class:`~aegisrover.storage.audit.AuditLog`, so the full sequence
of *why* a source lost trust and *when* it earned it back can be reconstructed
and verified afterwards.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from aegisrover.estimation.outliers import hampel
from aegisrover.sensors.fusion import Measurement, agreement
from aegisrover.storage.audit import AuditLog

__all__ = ('SourceHealthConfig', 'Transition', 'SourceHealthGuard', 'GuardedFusion',
           'HEALTHY', 'WARMING', 'SUSPECT', 'ISOLATED', 'RECOVERING', 'MUTED')

WARMING = 'warming'
HEALTHY = 'healthy'
SUSPECT = 'suspect'
ISOLATED = 'isolated'
RECOVERING = 'recovering'
MUTED = 'muted'

_AUDIT_ACTIONS = {
    SUSPECT: 'sensor.suspect',
    ISOLATED: 'sensor.isolate',
    RECOVERING: 'sensor.recovering',
    HEALTHY: 'sensor.recover',
    MUTED: 'sensor.mute',
}


@dataclass(frozen=True)
class SourceHealthConfig:
    """Tuning knobs. Defaults assume roughly periodic ``fuse`` calls.

    Detection
    ---------
    warmup: samples a brand-new source gets on probation trust before it is
        judged at all (it can still contribute, with reduced weight).
    k_sigma: a sample deviating from the panel consensus by more than this
        many combined sigmas counts as "deviating".
    cusum_k: CUSUM slack in sigmas; absorbs small wobbles around the band.
    cusum_h: CUSUM threshold (sigmas accumulated) that promotes suspect ->
        isolated. Set well above ``k_sigma`` so a source must stay outside the
        band for several samples -- the point is to catch a *persistent* bias,
        not the same spike the Hampel pass already removes.
    suspect_streak: consecutive deviating samples that promote healthy ->
        suspect.
    clear_streak: consecutive clean samples that demote suspect -> healthy.
    min_suspect: minimum samples a source must spend in suspect state before
        it can be isolated, so quarantine always follows an observed dwell at
        reduced weight rather than a single decisive tick.
    min_support: minimum number of *other* trusted sources that must agree on
        the reference before a *healthy* source can come under suspicion. This
        is a majority requirement: with two sources nobody gets convicted, and
        a healthy source cannot be flagged by a single contradictory source.
        Sources already on probation (suspect/recovering/isolated) can be
        judged or cleared with just one trusted other, so a quarantined
        sensor is not stuck when it is the last source left to test it.

    Recovery
    --------
    cooldown_streak: clean samples an isolated source must serve at zero weight
        before it is readmitted on probation. Doubles (up to ``cooldown_cap``)
        after every repeated isolation so a flapping source stays out longer.
    recovery_streak: clean samples needed across isolated+recovering before full
        weight is restored.
    probation_trust, suspect_trust, recovering_trust: weight multipliers (the
        recovering one ramps linearly up to 1.0 over the probation window).
    """
    warmup: int = 6
    k_sigma: float = 4.0
    cusum_k: float = 0.5
    cusum_h: float = 12.0
    suspect_streak: int = 8
    clear_streak: int = 8
    min_suspect: int = 4
    min_support: int = 2
    cooldown_streak: int = 8
    cooldown_cap: int = 128
    recovery_streak: int = 16
    probation_trust: float = 0.25
    suspect_trust: float = 0.25
    recovering_trust: float = 0.25


@dataclass(frozen=True)
class Transition:
    seq: int
    tick: int
    at: float
    source: str
    old: str
    new: str
    reason: str
    actor: str
    z: float | None
    cusum: float | None
    reference: float | None
    sigma: float | None
    supporters: int
    isolated_count: int

    def to_dict(self) -> dict:
        return {
            'seq': self.seq, 'tick': self.tick, 'at': round(self.at, 6),
            'source': self.source, 'old': self.old, 'new': self.new,
            'reason': self.reason, 'actor': self.actor,
            'z': None if self.z is None else round(self.z, 6),
            'cusum': None if self.cusum is None else round(self.cusum, 6),
            'reference': self.reference, 'sigma': self.sigma,
            'supporters': self.supporters, 'isolated_count': self.isolated_count,
        }


@dataclass
class _Track:
    source: str
    status: str = WARMING
    seen: int = 0
    deviate_streak: int = 0
    clean_streak: int = 0
    cusum_pos: float = 0.0
    cusum_neg: float = 0.0
    last_z: float = 0.0
    max_abs_z: float = 0.0
    suspect_age: int = 0
    isolated_count: int = 0

    def snapshot(self) -> dict:
        return {'status': self.status, 'seen': self.seen,
                'deviate_streak': self.deviate_streak,
                'clean_streak': self.clean_streak,
                'suspect_age': self.suspect_age,
                'cusum': round(max(self.cusum_pos, self.cusum_neg), 6),
                'last_z': round(self.last_z, 6),
                'max_abs_z': round(self.max_abs_z, 6),
                'isolated_count': self.isolated_count}


@dataclass(frozen=True)
class GuardedFusion:
    value: float
    weights: dict[str, float]
    agreement: float
    rejected: tuple[str, ...]
    stale: tuple[str, ...]
    isolated: tuple[str, ...]
    muted: tuple[str, ...]
    suspect: tuple[str, ...]
    status: dict[str, str]
    trust: dict[str, float]
    transitions: tuple[Transition, ...]

    def to_dict(self) -> dict:
        return {
            'value': round(self.value, 9),
            'weights': {k: round(v, 6) for k, v in self.weights.items()},
            'agreement': round(self.agreement, 6),
            'rejected': list(self.rejected), 'stale': list(self.stale),
            'isolated': list(self.isolated), 'muted': list(self.muted),
            'suspect': list(self.suspect), 'status': dict(self.status),
            'trust': {k: round(v, 6) for k, v in self.trust.items()},
            'transitions': [t.to_dict() for t in self.transitions],
        }


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    half = 0.5 * cumulative[-1]
    # First weight bucket reaching half, and first bucket strictly past it:
    # when they differ the boundary splits two buckets evenly and the median
    # is the midpoint of those two values.
    lower = int(np.searchsorted(cumulative, half, side='left'))
    upper = int(np.searchsorted(cumulative, half, side='right'))
    lower = min(lower, len(values) - 1)
    upper = min(upper, len(values) - 1)
    if lower == upper:
        return float(values[lower])
    return float(0.5 * (values[lower] + values[upper]))


def _consensus(items: Sequence[tuple[float, float, float]],
               half_life: float | None) -> tuple[float, float]:
    """Robust consensus of trusted sources: weighted median plus 1-sigma band.

    ``items`` holds ``(value, effective_variance, age)`` triples; the caller
    inflates the variance of partially-trusted sources so a suspect cannot
    drag the reference centre. The centre is a weighted median (one extremist
    cannot move it much), and the band width comes from the sources' declared
    variances -- the same noise contract that :func:`fuse` weights by, via the
    weighted-mean variance ``1 / sum w``. Observed spread is deliberately not
    added to the band: at the edge of robust breakdown (two symmetric
    extremists around a tight core) a data-driven MAD widens enough to hide
    the very outliers it is meant to expose. The cost is that a source whose
    declared variance is much tighter than its real disagreement looks
    deviant -- which is accurate, and surfaces separately as low agreement.
    """
    values = np.array([v for v, _, _ in items], dtype=float)
    weights = np.array([1.0 / variance for _, variance, _ in items], dtype=float)
    if half_life is not None:
        weights *= np.array([math.pow(0.5, age / half_life) for _, _, age in items])
    median = _weighted_median(values, weights)
    return median, max(math.sqrt(1.0 / float(weights.sum())), 1e-12)


class SourceHealthGuard:
    """Stateful fusion wrapper that detects, down-weights and quarantines drift.

    Parameters
    ----------
    config:
        :class:`SourceHealthConfig`; sensible defaults when omitted.
    half_life, max_age:
        Forwarded to the freshness/staleness logic of
        :func:`aegisrover.sensors.fusion.fuse`.
    audit:
        Optional hash-chained audit log. Automatic transitions are recorded
        with actor ``source-health-guard``; manual actions carry the caller's
        actor name.
    stream:
        Identifier used as the audit *subject* prefix when several guards share
        one audit log.
    clock:
        Injectable wall clock for audit timestamps.
    """

    AUTO_ACTOR = 'source-health-guard'

    def __init__(self, config: SourceHealthConfig | None = None, *,
                 half_life: float | None = None, max_age: float | None = None,
                 audit: AuditLog | None = None, stream: str = 'fusion/default',
                 clock=time.time):
        self.config = config or SourceHealthConfig()
        self._half_life = half_life
        self._max_age = max_age
        self._audit = audit
        self._stream = stream
        self._clock = clock
        self._tracks: dict[str, _Track] = {}
        self._transitions: list[Transition] = []
        self._next_seq = 1
        self._tick = 0

    # -- public introspection --------------------------------------------------
    def status_of(self, source: str) -> str:
        return self._tracks[source].status if source in self._tracks else WARMING

    def trust_of(self, source: str) -> float:
        track = self._tracks.get(source)
        return self._trust(track) if track is not None else self.config.probation_trust

    def tracks(self) -> dict[str, dict]:
        return {name: track.snapshot() for name, track in sorted(self._tracks.items())}

    def history(self, source: str | None = None) -> tuple[Transition, ...]:
        if source is None:
            return tuple(self._transitions)
        return tuple(t for t in self._transitions if t.source == source)

    # -- manual override -------------------------------------------------------
    def isolate(self, source: str, *, actor: str, reason: str = '') -> Transition:
        """Manually quarantine a source (e.g. scheduled recalibration)."""
        if not actor:
            raise ValueError('actor is required')
        track = self._tracks.setdefault(source, _Track(source))
        return self._transition(track, MUTED, reason or 'manual isolate',
                                actor, z=None, cusum=None, reference=None,
                                sigma=None, supporters=0)

    def reinstate(self, source: str, *, actor: str, reason: str = '') -> Transition:
        """Lift a manual mute; the source must still re-earn full trust."""
        if not actor:
            raise ValueError('actor is required')
        track = self._tracks.get(source)
        if track is None:
            track = _Track(source)
            self._tracks[source] = track
        track.cusum_pos = track.cusum_neg = 0.0
        track.deviate_streak = 0
        track.clean_streak = 0
        return self._transition(track, RECOVERING, reason or 'manual reinstate',
                                actor, z=None, cusum=None, reference=None,
                                sigma=None, supporters=0)

    # -- the fused tick --------------------------------------------------------
    def fuse(self, measurements: Sequence[Measurement]) -> GuardedFusion:
        if not measurements:
            raise ValueError('at least one measurement is required')
        self._tick += 1
        stale: list[str] = []
        active: list[Measurement] = []
        for item in measurements:
            if self._max_age is not None and item.age > self._max_age:
                stale.append(item.source)
                continue
            active.append(item)
        if not active:
            raise ValueError('all measurements are stale')

        for item in active:
            self._tracks.setdefault(item.source, _Track(item.source)).seen += 1

        # Verdict panel is fixed before any status changes this tick.
        prior_status = {m.source: self._tracks[m.source].status for m in active}
        prior_trust = {m.source: self._trust(self._tracks[m.source]) for m in active}

        transitions: list[Transition] = []
        for item in active:
            event = self._evaluate(self._tracks[item.source], item, active,
                                   prior_status, prior_trust)
            if event is not None:
                transitions.append(event)

        # Everything below uses the *post-update* status, so a source convicted
        # on this tick carries zero weight on this tick.
        accepted: list[Measurement] = []
        trust: dict[str, float] = {}
        isolated, muted, suspect = [], [], []
        for item in active:
            track = self._tracks[item.source]
            factor = self._trust(track)
            trust[item.source] = factor
            if track.status == ISOLATED:
                isolated.append(item.source)
            elif track.status == MUTED:
                muted.append(item.source)
            elif track.status == SUSPECT:
                suspect.append(item.source)
            if factor > 0.0:
                accepted.append(Measurement(item.source, item.value,
                                            item.variance / factor, item.age))

        spike_rejected: list[str] = []
        # The plain Hampel pass is a backstop for instantaneous spikes on top
        # of the stateful guard. With only three contributors a single unlucky
        # draw can look like the outlier of the triplet, so leave small panels
        # entirely to the persistence detector (which needs a streak).
        if len(accepted) >= 4:
            flags = hampel([m.value for m in accepted])
            survivors = [m for m, bad in zip(accepted, flags) if not bad]
            if survivors:
                spike_rejected = [m.source for m, bad in zip(accepted, flags) if bad]
                accepted = survivors

        weights: dict[str, float] = {item.source: 0.0 for item in active}
        total = 0.0
        for item in accepted:
            weight = 1.0 / item.variance
            if self._half_life is not None:
                weight *= math.pow(0.5, item.age / self._half_life)
            weights[item.source] = weight
            total += weight
        if total <= 0.0:
            raise ValueError('fused weight is zero')
        value = sum(weights[m.source] * m.value for m in accepted) / total
        normalised = {k: v / total for k, v in weights.items()}
        rejected = tuple(sorted(set(isolated) | set(muted) | set(spike_rejected)))
        status = {item.source: self._tracks[item.source].status for item in active}
        return GuardedFusion(
            value=value, weights=normalised,
            agreement=agreement([m.value for m in accepted]),
            rejected=rejected, stale=tuple(stale),
            isolated=tuple(sorted(isolated)), muted=tuple(sorted(muted)),
            suspect=tuple(sorted(suspect)), status=status, trust=trust,
            transitions=tuple(transitions))

    # -- internals -------------------------------------------------------------
    def _evaluate(self, track: _Track, item: Measurement,
                  active: Sequence[Measurement],
                  prior_status: dict[str, str],
                  prior_trust: dict[str, float]) -> Transition | None:
        cfg = self.config
        if track.status == MUTED:
            return None
        if track.seen <= cfg.warmup:
            return None  # not enough history on this source to judge it
        if track.status == WARMING:
            track.status = HEALTHY  # first judged tick

        # The consensus is built from statuses snapshotted at the start of the
        # tick, so the verdict cannot depend on list order and a source
        # convicted this tick cannot change the panel it was judged by. Healthy
        # sources are judged by the whole healthy panel (weighted median plus
        # the panel's declared noise); a healthy majority is required, so a
        # single contradicting source cannot raise suspicion. A source already
        # on probation is judged against whatever trusted witnesses remain, so
        # one good source can test and clear a quarantined sensor even when the
        # others are already gone.
        def effective(m: Measurement) -> tuple[float, float, float]:
            return (m.value, m.variance / prior_trust[m.source], m.age)

        full_pool = [m for m in active if prior_status[m.source] == HEALTHY]
        trusted_pool = [m for m in active if prior_trust[m.source] > 0.0]

        if track.status == HEALTHY:
            # Need a healthy majority: with fewer than three healthy sources no
            # outlier can be identified robustly, so nobody is convicted.
            if len(full_pool) <= cfg.min_support:
                return None
            reference, ref_sigma = _consensus(
                [effective(m) for m in full_pool], self._half_life)
            supporters = len(full_pool) - 1
        else:
            other_trusted = [m for m in trusted_pool if m.source != item.source]
            if not other_trusted:
                return None
            panel = full_pool if len(full_pool) >= 2 else trusted_pool
            reference, ref_sigma = _consensus(
                [effective(m) for m in panel], self._half_life)
            supporters = len(other_trusted)

        sigma = math.sqrt(ref_sigma * ref_sigma + item.variance)
        z = (item.value - reference) / sigma
        track.last_z = z
        track.max_abs_z = max(track.max_abs_z, abs(z))
        deviates = abs(z) > cfg.k_sigma

        old = track.status
        if deviates:
            track.cusum_pos = max(0.0, track.cusum_pos + z - cfg.cusum_k)
            track.cusum_neg = max(0.0, track.cusum_neg - z - cfg.cusum_k)
            track.deviate_streak += 1
            track.clean_streak = 0
        else:
            track.cusum_pos = track.cusum_neg = 0.0
            track.deviate_streak = 0
            track.clean_streak += 1
        cusum = max(track.cusum_pos, track.cusum_neg)

        new_status, reason = self._next_status(track, deviates, cusum)
        if new_status is None or new_status == old:
            return None
        if new_status == ISOLATED:
            # Isolation requires at least one other source that still carries
            # usable weight, so a correlated majority failure can degrade the
            # fusion to low-trust suspicion but can never remove every source
            # and produce "no estimate at all".
            if not any(m.source != item.source and prior_trust[m.source] > 0.0
                       for m in active):
                return None
        return self._transition(track, new_status, reason, self.AUTO_ACTOR,
                                z=z, cusum=cusum, reference=reference,
                                sigma=sigma, supporters=supporters)

    def _next_status(self, track: _Track, deviates: bool,
                     cusum: float) -> tuple[str | None, str]:
        cfg = self.config
        status = track.status

        if status == HEALTHY:
            if deviates and track.deviate_streak >= cfg.suspect_streak:
                # The ramp that earned suspicion is not evidence of how long
                # the source has been on probation: start the clock fresh so
                # isolation reflects persistence *after* suspicion.
                track.cusum_pos = track.cusum_neg = 0.0
                track.suspect_age = 0
                return SUSPECT, f'{track.deviate_streak} consecutive deviations'
            return None, ''

        if status == SUSPECT:
            track.suspect_age += 1
            dwelled = track.suspect_age >= cfg.min_suspect
            if dwelled and cusum >= cfg.cusum_h:
                track.isolated_count += 1
                track.clean_streak = 0
                return ISOLATED, (f'cusum {cusum:.2f} >= {cfg.cusum_h:.2f} '
                                  f'after {track.suspect_age} suspect samples')
            if not deviates and track.clean_streak >= cfg.clear_streak:
                return HEALTHY, f'{track.clean_streak} clean samples'
            return None, ''

        # isolated or recovering: cool-down at zero weight, then ramp back.
        required_cooldown = min(
            cfg.cooldown_streak << min(max(track.isolated_count - 1, 0), 16),
            cfg.cooldown_cap)
        if deviates:
            if status == RECOVERING:
                track.isolated_count += 1
                return ISOLATED, 'bias returned during probation'
            return None, ''
        if status == ISOLATED and track.clean_streak >= required_cooldown:
            return RECOVERING, f'{required_cooldown} clean samples in quarantine'
        if track.clean_streak >= cfg.recovery_streak:
            return HEALTHY, f'{track.clean_streak} clean samples, trust restored'
        return None, ''

    def _trust(self, track: _Track) -> float:
        cfg = self.config
        if track.status in (ISOLATED, MUTED):
            return 0.0
        if track.status == WARMING:
            return cfg.probation_trust
        if track.status == SUSPECT:
            return cfg.suspect_trust
        if track.status == HEALTHY:
            return 1.0
        # recovering: linear ramp from recovering_trust up to full trust
        progress = min(1.0, track.clean_streak / cfg.recovery_streak)
        return cfg.recovering_trust + (1.0 - cfg.recovering_trust) * progress

    def _transition(self, track: _Track, new: str, reason: str, actor: str, *,
                    z: float | None, cusum: float | None, reference: float | None,
                    sigma: float | None, supporters: int) -> Transition:
        old = track.status
        track.status = new
        entry = Transition(
            seq=self._next_seq, tick=self._tick, at=self._clock(),
            source=track.source, old=old, new=new, reason=reason, actor=actor,
            z=z, cusum=cusum, reference=reference, sigma=sigma,
            supporters=supporters, isolated_count=track.isolated_count)
        self._next_seq += 1
        self._transitions.append(entry)
        if self._audit is not None:
            action = _AUDIT_ACTIONS.get(new, 'sensor.transition')
            self._audit.append(actor, action, f'{self._stream}/{track.source}',
                               entry.to_dict())
        return entry
