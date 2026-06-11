"""Inverter inner-loop dynamics, current limits, and anti-windup."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .capability import InverterCapability, PQPoint
from .parameters import InverterInnerLoopParameters, InverterParameters


@dataclass
class InverterInnerLoopState:
    id_a: float = 0.0
    iq_a: float = 0.0
    id_integrator_a: float = 0.0
    iq_integrator_a: float = 0.0


@dataclass(frozen=True)
class InverterStepResult:
    state: InverterInnerLoopState
    p_kw: float
    q_kvar: float
    id_ref_a: float
    iq_ref_a: float
    saturated: bool
    dc_link_limited: bool
    pq_limited: PQPoint


class InverterInnerLoop:
    """First-order dq current-loop approximation with saturation and anti-windup."""

    def __init__(self, inverter: InverterParameters, inner: InverterInnerLoopParameters):
        self.inverter = inverter
        self.inner = inner
        self.capability = InverterCapability(
            inverter.s_rated_kva,
            inverter.p_min_kw,
            inverter.p_max_kw,
            inverter.q_max_kvar,
        )

    def step(
        self,
        state: InverterInnerLoopState,
        p_ref_kw: float,
        q_ref_kvar: float,
        voltage_rms: float,
        sample_time_s: float,
    ) -> InverterStepResult:
        pq = self.capability.allocate(p_ref_kw, q_ref_kvar, reactive_priority=True)
        id_ref = pq.p_kw * 1000.0 / max(voltage_rms, 1e-6)
        iq_ref = pq.q_kvar * 1000.0 / max(voltage_rms, 1e-6)

        dc_current_limit = self.inner.dc_link_voltage_v * self.inner.dc_link_current_limit_a / max(voltage_rms, 1e-6)
        current_limit = min(self.inner.rms_current_limit_a, dc_current_limit)
        magnitude = math.hypot(id_ref, iq_ref)
        saturated = magnitude > current_limit
        if saturated:
            scale = current_limit / max(magnitude, 1e-9)
            id_ref_sat = id_ref * scale
            iq_ref_sat = iq_ref * scale
        else:
            id_ref_sat = id_ref
            iq_ref_sat = iq_ref

        bandwidth = 2.0 * math.pi * self.inner.current_controller_bandwidth_hz
        alpha = 1.0 - math.exp(-bandwidth * sample_time_s)
        id_error = id_ref_sat - state.id_a
        iq_error = iq_ref_sat - state.iq_a
        id_aw = self.inner.anti_windup_gain * (id_ref_sat - id_ref)
        iq_aw = self.inner.anti_windup_gain * (iq_ref_sat - iq_ref)
        next_state = InverterInnerLoopState(
            id_a=state.id_a + alpha * id_error,
            iq_a=state.iq_a + alpha * iq_error,
            id_integrator_a=state.id_integrator_a + sample_time_s * (id_error + id_aw),
            iq_integrator_a=state.iq_integrator_a + sample_time_s * (iq_error + iq_aw),
        )

        p_kw = voltage_rms * next_state.id_a / 1000.0
        q_kvar = voltage_rms * next_state.iq_a / 1000.0
        return InverterStepResult(
            state=next_state,
            p_kw=p_kw,
            q_kvar=q_kvar,
            id_ref_a=id_ref_sat,
            iq_ref_a=iq_ref_sat,
            saturated=saturated,
            dc_link_limited=dc_current_limit < self.inner.rms_current_limit_a,
            pq_limited=pq,
        )
