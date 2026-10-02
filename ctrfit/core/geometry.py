"""Diffraction geometry: incident and exit wave vectors for a reflection (H K L).

Sample frame (same as the structure factors and the 3D viewer): x || [001] (a1), y || [1-10]
(a2), z || [110] (surface normal, out of the sample). q = 2 pi (H / a1, K / a2, L / c3).

Fixed incidence angle alpha (grazing, onto the surface): k_in = k (cos a cos phi, cos a sin phi,
-sin a), with k = 2 pi / lambda. The sample azimuth phi is chosen so that |k_in + q| = k (elastic
scattering); of the two solutions the one with the larger exit angle is used. On the specular rod
(H = K = 0) the azimuth is free and the incidence angle follows from L (alpha = exit angle = theta).
"""
import math

import numpy as np


def q_vector(m, H, K, L):
    """q (1/Å) in the sample frame for a CTRModel-like object with A1, A2, C3."""
    return np.array([2 * np.pi * H / m.A1, 2 * np.pi * K / m.A2, 2 * np.pi * L / m.C3])


def beam_geometry(m, H, K, L, alpha_deg=0.5, phi0_deg=0.0):
    """Incident and exit wave vectors for (H K L) at incidence angle alpha.

    Returns a dict: ok (reachable with the exit beam above the surface), reason (text when not),
    k_in, k_out, q (1/Å, sample frame), alpha, beta (exit angle above the surface), two_theta,
    phi (sample azimuth of the incident beam), all angles in degrees, and lam (Å).
    phi0_deg is the beam azimuth used on the specular rod, where any azimuth works.
    """
    lam = m.lam
    k = 2 * np.pi / lam
    q = q_vector(m, H, K, L)
    qn = float(np.linalg.norm(q))
    out = dict(ok=False, reason="", q=q, lam=lam, alpha=alpha_deg, beta=float("nan"),
               two_theta=float("nan"), phi=float("nan"), k_in=None, k_out=None)
    if qn == 0:
        out["reason"] = "q = 0"
        return out
    if qn > 2 * k:
        out["reason"] = f"|q| = {qn:.2f} 1/Å is beyond 2k = {2 * k:.2f} 1/Å at this energy"
        return out
    two_theta = 2 * math.degrees(math.asin(qn / (2 * k)))
    qpar = math.hypot(q[0], q[1])
    if qpar < 1e-12:                        # specular: alpha = beta = theta
        a = math.radians(two_theta / 2)
        ph = math.radians(phi0_deg)
        alpha_deg = two_theta / 2
    else:
        a = math.radians(alpha_deg)
        psi = math.atan2(q[1], q[0])
        # 2 k_in . q + q^2 = 0  ->  cos(phi - psi) = (2 k sin(a) qz - q^2) / (2 k cos(a) qpar)
        c = (2 * k * math.sin(a) * q[2] - qn ** 2) / (2 * k * math.cos(a) * qpar)
        if abs(c) > 1:
            out.update(reason=f"not reachable at alpha = {alpha_deg:g}° (change the incidence angle)",
                       two_theta=two_theta)
            return out
        best = None
        for sgn in (1, -1):
            ph_ = psi + sgn * math.acos(c)
            kin = k * np.array([math.cos(a) * math.cos(ph_), math.cos(a) * math.sin(ph_), -math.sin(a)])
            kout = kin + q
            if best is None or kout[2] > best[1][2]:
                best = (ph_, kout)
        ph = best[0]
    k_in = k * np.array([math.cos(a) * math.cos(ph), math.cos(a) * math.sin(ph), -math.sin(a)])
    k_out = k_in + q
    beta = math.degrees(math.asin(np.clip(k_out[2] / k, -1, 1)))
    out.update(k_in=k_in, k_out=k_out, alpha=alpha_deg, beta=beta, two_theta=two_theta,
               phi=math.degrees(ph) % 360)
    if beta <= 0:
        out["reason"] = f"exit beam goes into the sample (exit angle {beta:.2f}°); raise L or alpha"
        return out
    out["ok"] = True
    return out
