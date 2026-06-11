"""Single-phase SOGI-PLL and dq current decomposition."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class DQComponents:
    d: float
    q: float
    magnitude: float


class SOGIFilter:
    """Second-order generalized integrator for single-phase alpha/beta signals.

    For input u, the SOGI states implement:
        alpha(s)/u(s) = k*w*s / (s^2 + k*w*s + w^2)
        beta(s)/u(s)  = k*w^2 / (s^2 + k*w*s + w^2)

    The beta output is an orthogonal virtual phase used with synchronous-frame
    transformations.
    """

    def __init__(self, sample_time_s: float, nominal_omega_rad_s: float, gain: float):
        self.dt = sample_time_s
        self.nominal_omega = nominal_omega_rad_s
        self.gain = gain
        self.alpha = 0.0
        self.beta = 0.0

    def reset(self) -> None:
        self.alpha = 0.0
        self.beta = 0.0

    def update(self, u: float, omega_rad_s: float | None = None) -> tuple[float, float]:
        omega = omega_rad_s or self.nominal_omega

        def f(alpha: float, beta: float) -> tuple[float, float]:
            error = u - alpha
            return self.gain * omega * error - omega * beta, omega * alpha

        k1a, k1b = f(self.alpha, self.beta)
        k2a, k2b = f(self.alpha + 0.5 * self.dt * k1a, self.beta + 0.5 * self.dt * k1b)
        k3a, k3b = f(self.alpha + 0.5 * self.dt * k2a, self.beta + 0.5 * self.dt * k2b)
        k4a, k4b = f(self.alpha + self.dt * k3a, self.beta + self.dt * k3b)
        self.alpha += self.dt * (k1a + 2.0 * k2a + 2.0 * k3a + k4a) / 6.0
        self.beta += self.dt * (k1b + 2.0 * k2b + 2.0 * k3b + k4b) / 6.0
        return self.alpha, self.beta


class SRFPLL:
    """Synchronous-reference-frame PLL using alpha/beta voltage signals."""

    def __init__(
        self,
        sample_time_s: float,
        nominal_omega_rad_s: float,
        kp: float,
        ki: float,
        initial_theta_rad: float = 0.0,
    ):
        self.dt = sample_time_s
        self.nominal_omega = nominal_omega_rad_s
        self.kp = kp
        self.ki = ki
        self.theta = initial_theta_rad
        self.integrator = 0.0
        self.omega = nominal_omega_rad_s

    def reset(self, theta_rad: float = 0.0) -> None:
        self.theta = theta_rad
        self.integrator = 0.0
        self.omega = self.nominal_omega

    def update(self, v_alpha: float, v_beta: float) -> tuple[float, float]:
        sin_t = math.sin(self.theta)
        cos_t = math.cos(self.theta)
        v_q = -v_alpha * sin_t + v_beta * cos_t
        v_mag = max(math.hypot(v_alpha, v_beta), 1e-6)
        error = v_q / v_mag

        self.integrator += self.ki * error * self.dt
        self.omega = self.nominal_omega + self.kp * error + self.integrator
        self.theta = (self.theta + self.omega * self.dt) % (2.0 * math.pi)
        return self.theta, self.omega


class DQTransform:
    """Park transform helpers for alpha/beta to dq quantities."""

    @staticmethod
    def alpha_beta_to_dq(alpha: float, beta: float, theta_rad: float) -> DQComponents:
        sin_t = math.sin(theta_rad)
        cos_t = math.cos(theta_rad)
        d = alpha * cos_t + beta * sin_t
        q = -alpha * sin_t + beta * cos_t
        return DQComponents(d=d, q=q, magnitude=float(np.hypot(d, q)))


class SinglePhaseCurrentDecoupler:
    """Combines current SOGI, voltage SOGI, PLL, and dq decomposition."""

    def __init__(
        self,
        sample_time_s: float,
        nominal_frequency_hz: float,
        sogi_gain: float,
        pll_kp: float,
        pll_ki: float,
    ):
        omega = 2.0 * math.pi * nominal_frequency_hz
        self.current_sogi = SOGIFilter(sample_time_s, omega, sogi_gain)
        self.voltage_sogi = SOGIFilter(sample_time_s, omega, sogi_gain)
        self.pll = SRFPLL(sample_time_s, omega, pll_kp, pll_ki)

    def reset(self) -> None:
        self.current_sogi.reset()
        self.voltage_sogi.reset()
        self.pll.reset()

    def update(self, current_a: float, voltage_v: float) -> tuple[DQComponents, float, float]:
        v_alpha, v_beta = self.voltage_sogi.update(voltage_v, self.pll.omega)
        theta, omega = self.pll.update(v_alpha, v_beta)
        i_alpha, i_beta = self.current_sogi.update(current_a, omega)
        theta_for_current = (theta - omega * self.current_sogi.dt) % (2.0 * math.pi)
        return DQTransform.alpha_beta_to_dq(i_alpha, i_beta, theta_for_current), theta, omega
