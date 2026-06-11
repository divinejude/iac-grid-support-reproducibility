"""Inverter capability limits and P/Q allocation."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PQPoint:
    p_kw: float
    q_kvar: float
    apparent_kva: float
    active_was_limited: bool
    reactive_was_limited: bool


class InverterCapability:
    """Enforces compressor mechanical limits and apparent-power circle."""

    def __init__(self, s_rated_kva: float, p_min_kw: float, p_max_kw: float, q_max_kvar: float):
        self.s_rated_kva = s_rated_kva
        self.p_min_kw = p_min_kw
        self.p_max_kw = p_max_kw
        self.q_max_kvar = q_max_kvar

    def active_bounds_for_q(self, q_kvar: float) -> tuple[float, float]:
        q_limited = min(abs(q_kvar), self.s_rated_kva)
        p_circle = math.sqrt(max(self.s_rated_kva**2 - q_limited**2, 0.0))
        return self.p_min_kw, min(self.p_max_kw, p_circle)

    def max_reactive_for_p(self, p_kw: float) -> float:
        p_limited = min(abs(p_kw), self.s_rated_kva)
        circle_q = math.sqrt(max(self.s_rated_kva**2 - p_limited**2, 0.0))
        return min(self.q_max_kvar, circle_q)

    def allocate(self, p_desired_kw: float, q_desired_kvar: float, reactive_priority: bool = True) -> PQPoint:
        if reactive_priority:
            q_limit_at_min_p = self.max_reactive_for_p(self.p_min_kw)
            q_after_cap = max(-q_limit_at_min_p, min(q_limit_at_min_p, q_desired_kvar))
            p_low, p_high = self.active_bounds_for_q(q_after_cap)
            p_after_cap = max(p_low, min(p_high, p_desired_kw))
        else:
            p_after_cap = max(self.p_min_kw, min(self.p_max_kw, p_desired_kw))
            q_limit = self.max_reactive_for_p(p_after_cap)
            q_after_cap = max(-q_limit, min(q_limit, q_desired_kvar))

        apparent = math.hypot(p_after_cap, q_after_cap)
        return PQPoint(
            p_kw=p_after_cap,
            q_kvar=q_after_cap,
            apparent_kva=apparent,
            active_was_limited=not math.isclose(p_after_cap, p_desired_kw, rel_tol=0.0, abs_tol=1e-9),
            reactive_was_limited=not math.isclose(q_after_cap, q_desired_kvar, rel_tol=0.0, abs_tol=1e-9),
        )
