"""Persistent-drift supervision for sensor fusion.

:func:`aegisrover.sensors.fusion.fuse` is stateless: its Hampel rejection drops a
single-cycle spike, but a source that drifts *persistently* stays inside the
instantaneous bounds — and with only two sources there is no rejection at all —
so the fused value is dragged along until a human notices.

:class:`FusionGuard` wraps ``fuse`` with per-source health memory. Every cycle
each source is compared against a robust reference (the median of the usable
sources, so the suspect cannot drag its own reference), the normalised residual
is folded into an EWMA score, and the score drives a state machine::

    TRUSTED ── score ≥ degrade_score ──▶ DEGRADED ── score ≥ isolate_score ──▶ ISOLATED
       ▲                                    │                                    │
       │                                    └─ score back below threshold ──────┤
       │                                                                       │ residual < recover_z
       │                                                                       │ for recover_cycles
       │                                                                       ▼
       └──────────── clean for probation_cycles ◀──────────────────────── PROBATION

Isolation is fast and re-admission is slow on purpose, and a re-offence during
probation drops the source straight back to ISOLATED. Degraded sources keep
fusing with a weight that ramps down as the score grows; isolated sources are
excluded but keep being scored, which is what makes automatic recovery possible.

Every transition is kept in memory and, when an
:class:`aegisrover.storage.audit.AuditLog` is supplied, appended to the
hash-chained audit trail, so the whole episode — who drifted, when, how far,
when it came back — can be reconstructed and verified afterwards.

Limits: arbitration needs at least two usable sources. With exactly two, the
data alone cannot say which one is wrong, so both are penalised symmetrically
unless ``GuardConfig.anchor`` names the source to trust in a dispute. A single
lone source cannot be scored at all.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from aegisrover.sensors.fusion import FusionResult, Measurement, fuse

if TYPE_CHECKING:  # avoid a hard dependency cycle; any object with .append works
    from aegisrover.storage.audit import AuditLog

__all__ = ('GuardConfig', 'GuardedResult', 'FusionGuard', 'SourceReport', 'Transition',
           'TRUSTED', 'DEGRADED', 'ISOLATED', 'PROBATION')

TRUSTED = 'trusted'
DEGRADED = 'degraded'
ISOLATED = 'isolated'
PROBATION = 'probation'

ACTOR = 'fusion-guard'


@dataclass(frozen=True)
class GuardConfig:
    degrade_score: float = 1.5      # EWMA score where down-weighting starts
    isolate_score: float = 2.5      # EWMA score where the source is excluded
    recover_z: float = 1.0          # normalised residual counted as a clean cycle
    recover_cycles: int = 5         # clean cycles before re-admission to probation
    probation_cycles: int = 5       # clean probation cycles before full trust
    probation_weight: float = 0.25  # weight multiplier on re-entry
    min_weight: float = 0.05        # floor of the degraded weight ramp
    score_half_life: float = 4.0    # EWMA memory, in fusion cycles
    hysteresis: float = 0.5         # degraded exits below degrade_score - hysteresis
    hard_z: float | None = 6.0      # instant-isolation residual, None disables
    anchor: str | None = None       # source trusted to arbitrate two-source disputes

    def __post_init__(self):
        if self.degrade_score <= 0:
            raise ValueError('degrade_score must be positive')
        if self.isolate_score <= self.degrade_score:
            raise ValueError('isolate_score must exceed degrade_score')
        if self.recover_z <= 0:
            raise ValueError('recover_z must be positive')
        if self.recover_cycles < 1 or self.probation_cycles < 1:
            raise ValueError('recovery cycle counts must be at least 1')
        if not 0.0 < self.min_weight <= 1.0:
            raise ValueError('min_weight must be in (0, 1]')
        if not 0.0 < self.probation_weight <= 1.0:
            raise ValueError('probation_weight must be in (0, 1]')
        if self.score_half_life <= 0:
            raise ValueError('score_half_life must be positive')
        if not 0.0 <= self.hysteresis < self.degrade_score:
            raise ValueError('hysteresis must be in [0, degrade_score)')
        if self.hard_z is not None and self.hard_z <= 0:
            raise ValueError('hard_z must be positive')


@dataclass(frozen=True)
class SourceReport:
    """Per-cycle view of one source: what the guard saw and decided."""
    source: str
    status: str
    score: float
    residual: float | None      # normalised residual this cycle, None if not scorable
    multiplier: float           # weight multiplier applied to this cycle's fusion
    reference: float | None     # robust reference the residual was measured against
    clean_cycles: int


@dataclass(frozen=True)
class Transition:
    source: str
    previous: str
    status: str
    score: float
    reason: str


@dataclass(frozen=True)
class GuardedResult:
    fusion: FusionResult
    sources: tuple[SourceReport, ...]
    transitions: tuple[Transition, ...]
    fallback: bool              # True when every source was isolated and the
                                # least-bad one was used as a last resort

    @property
    def value(self) -> float:
        return self.fusion.value

    def to_dict(self) -> dict:
        return {'fusion': self.fusion.to_dict(),
                'fallback': self.fallback,
                'sources': {s.source: {'status': s.status, 'score': s.score,
                                       'residual': s.residual, 'multiplier': s.multiplier,
                                       'clean_cycles': s.clean_cycles}
                            for s in self.sources},
                'transitions': [t.__dict__ for t in self.transitions]}


@dataclass
class _SourceState:
    status: str = TRUSTED
    score: float = 0.0
    multiplier: float = 1.0
    clean_cycles: int = 0


class FusionGuard:
    """Stateful supervisor that down-weights, isolates and re-admits sources.

    Parameters
    ----------
    config:
        Thresholds and timings; see :class:`GuardConfig`.
    audit:
        Optional :class:`aegisrover.storage.audit.AuditLog`. Every status
        transition and every fallback activation is appended to it.
    """

    def __init__(self, config: GuardConfig | None = None, audit: 'AuditLog | None' = None):
        self._config = config or GuardConfig()
        self._audit = audit
        self._states: dict[str, _SourceState] = {}
        self._transitions: list[Transition] = []
        self._cycle = 0
        self._fallback_active = False

    # -- main entry ------------------------------------------------------------
    def update(self, measurements: Sequence[Measurement], *,
               half_life: float | None = None, now: float | None = None,
               max_age: float | None = None) -> GuardedResult:
        """Score one fusion cycle and return the guarded fusion result."""
        items = list(measurements)
        if not items:
            raise ValueError('at least one measurement is required')
        cfg = self._config
        self._cycle += 1
        for m in items:
            self._states.setdefault(m.source, _SourceState())

        usable = [m for m in items if self._states[m.source].multiplier > 0.0]
        # When nothing is usable, the least-bad source still feeds the fusion and
        # acts as the reference, so the others can earn their way back.
        fallback_source = None if usable else min(items, key=lambda m: self._states[m.source].score)

        references, reference_vars = self._references(items, usable, fallback_source)
        residuals = self._score(items, references, reference_vars)
        transitions = self._transit(items, residuals)
        self._update_multipliers(items)

        fused_in = [Measurement(m.source, m.value,
                                m.variance / self._states[m.source].multiplier, m.age)
                    for m in items if self._states[m.source].multiplier > 0.0]
        fallback = False
        if not fused_in:
            fallback = True
            best = fallback_source if fallback_source is not None else \
                min(items, key=lambda m: self._states[m.source].score)
            fused_in = [Measurement(best.source, best.value, best.variance, best.age)]
        fusion = fuse(fused_in, half_life=half_life, now=now, max_age=max_age)

        if fallback != self._fallback_active:
            self._fallback_active = fallback
            self._record('fusion.fallback', 'fusion',
                         {'active': fallback, 'cycle': self._cycle,
                          'sources': [m.source for m in items]})
        for t in transitions:
            self._transitions.append(t)
            self._record(f'fusion.source.{t.status}', t.source,
                         {'cycle': self._cycle, 'previous': t.previous, 'status': t.status,
                          'score': t.score, 'reason': t.reason,
                          'residual': residuals[t.source],
                          'multiplier': round(self._states[t.source].multiplier, 6)})

        reports = tuple(SourceReport(
            source=m.source,
            status=self._states[m.source].status,
            score=round(self._states[m.source].score, 6),
            residual=None if residuals[m.source] is None else round(residuals[m.source], 6),
            multiplier=round(self._states[m.source].multiplier, 6),
            reference=references[m.source],
            clean_cycles=self._states[m.source].clean_cycles,
        ) for m in items)
        return GuardedResult(fusion=fusion, sources=reports,
                             transitions=tuple(transitions), fallback=fallback)

    # -- inspection --------------------------------------------------------------
    def status(self) -> dict[str, dict]:
        """Snapshot of every tracked source, for monitoring and alerting."""
        return {name: {'status': st.status, 'score': round(st.score, 6),
                       'multiplier': round(st.multiplier, 6),
                       'clean_cycles': st.clean_cycles}
                for name, st in self._states.items()}

    def history(self) -> tuple[Transition, ...]:
        """Every status transition since the guard was created."""
        return tuple(self._transitions)

    def reset(self, source: str) -> None:
        """Operator action after maintenance: forget a source's history."""
        if source not in self._states:
            raise KeyError(f'unknown source {source!r}')
        self._states[source] = _SourceState()
        self._record('fusion.source.reset', source, {'cycle': self._cycle})

    # -- internals ---------------------------------------------------------------
    def _references(self, items, usable, fallback_source):
        """Robust per-source reference values.

        The median of the usable sources (including the suspect) is robust to a
        minority of drifters. With exactly two usable sources the median cannot
        arbitrate, so each is referenced against the other — both carry the full
        dispute residual — unless ``config.anchor`` names the one to trust.
        Isolated sources are referenced against the usable set so recovery can be
        seen; a lone usable source has no reference and is simply not scored.
        """
        cfg = self._config
        references: dict[str, float | None] = {}
        reference_vars: dict[str, float] = {}
        if usable:
            values = [m.value for m in usable]
            median = statistics.median(values)
            median_var = statistics.median(m.variance for m in usable)
            anchor = next((m for m in usable if m.source == cfg.anchor), None)
            pair = usable if len(usable) == 2 else None
            for m in items:
                if self._states[m.source].multiplier > 0.0 and pair is not None:
                    other = anchor or (pair[0] if pair[1].source == m.source else pair[1])
                    references[m.source] = other.value
                    reference_vars[m.source] = other.variance
                else:
                    references[m.source] = median
                    reference_vars[m.source] = median_var
        elif fallback_source is not None:
            for m in items:
                if m is fallback_source:
                    references[m.source] = None
                    reference_vars[m.source] = 0.0
                else:
                    references[m.source] = fallback_source.value
                    reference_vars[m.source] = fallback_source.variance
        else:
            for m in items:
                references[m.source] = None
                reference_vars[m.source] = 0.0
        return references, reference_vars

    def _score(self, items, references, reference_vars):
        """Fold this cycle's normalised residual into the EWMA score."""
        cfg = self._config
        alpha = 1.0 - math.pow(0.5, 1.0 / cfg.score_half_life)
        residuals: dict[str, float | None] = {}
        for m in items:
            st = self._states[m.source]
            ref = references[m.source]
            if ref is None:
                residuals[m.source] = None
                continue
            residual = abs(m.value - ref) / math.sqrt(m.variance + reference_vars[m.source])
            residuals[m.source] = residual
            st.score += alpha * (residual - st.score)
            if residual < cfg.recover_z:
                st.clean_cycles += 1
            else:
                st.clean_cycles = 0
        return residuals

    def _transit(self, items, residuals):
        cfg = self._config
        transitions: list[Transition] = []
        for m in items:
            st = self._states[m.source]
            r = residuals[m.source]
            hard = r is not None and cfg.hard_z is not None and r > cfg.hard_z
            if st.status in (TRUSTED, DEGRADED):
                if hard:
                    self._move(transitions, m.source, ISOLATED,
                               f'residual {r:.1f} exceeds hard limit {cfg.hard_z}')
                elif st.score >= cfg.isolate_score:
                    self._move(transitions, m.source, ISOLATED,
                               f'score {st.score:.2f} reached isolate threshold {cfg.isolate_score}')
                elif st.score >= cfg.degrade_score:
                    self._move(transitions, m.source, DEGRADED,
                               f'score {st.score:.2f} reached degrade threshold {cfg.degrade_score}')
                elif st.status == DEGRADED and st.score <= cfg.degrade_score - cfg.hysteresis:
                    self._move(transitions, m.source, TRUSTED,
                               f'score {st.score:.2f} back below {cfg.degrade_score - cfg.hysteresis:.2f}')
            elif st.status == ISOLATED:
                if st.clean_cycles >= cfg.recover_cycles:
                    self._move(transitions, m.source, PROBATION,
                               f'clean for {st.clean_cycles} cycles')
                    st.clean_cycles = 0
                    st.score = 0.0
            elif st.status == PROBATION:
                if hard:
                    self._move(transitions, m.source, ISOLATED,
                               f'residual {r:.1f} exceeds hard limit during probation')
                elif st.score >= cfg.degrade_score:
                    self._move(transitions, m.source, ISOLATED,
                               f're-offence during probation (score {st.score:.2f})')
                elif st.clean_cycles >= cfg.probation_cycles:
                    self._move(transitions, m.source, TRUSTED,
                               f'probation clean for {st.clean_cycles} cycles')
                    st.clean_cycles = 0
                    st.score = 0.0
        return transitions

    def _move(self, transitions, source, new_status, reason):
        st = self._states[source]
        if st.status == new_status:
            return
        if new_status == ISOLATED:
            st.clean_cycles = 0
        transitions.append(Transition(source, st.status, new_status,
                                      round(st.score, 6), reason))
        st.status = new_status

    def _update_multipliers(self, items):
        cfg = self._config
        span = cfg.isolate_score - cfg.degrade_score
        for m in items:
            st = self._states[m.source]
            if st.status == TRUSTED:
                st.multiplier = 1.0
            elif st.status == DEGRADED:
                frac = min(1.0, max(0.0, (cfg.isolate_score - st.score) / span))
                st.multiplier = cfg.min_weight + (1.0 - cfg.min_weight) * frac
            elif st.status == PROBATION:
                ramp = min(1.0, st.clean_cycles / cfg.probation_cycles)
                st.multiplier = cfg.probation_weight + (1.0 - cfg.probation_weight) * ramp
            else:
                st.multiplier = 0.0

    def _record(self, action, subject, payload):
        if self._audit is not None:
            self._audit.append(ACTOR, action, subject, payload)
