"""Water balance, monotonicity and boundary behaviour."""
import numpy as np
import pytest
import torch

import dcrest
from dcrest.model import CrestConfig


def _params(synthetic, **over):
    vals = dict(lf=0.0)          # closure only holds without the loss term
    vals.update(over)
    return dcrest.default_parameters(synthetic["n"], synthetic["area_km2"],
                                      dtype=torch.float64, values=vals)


def _run(synthetic, p, cfg=None):
    f = [synthetic[k].to(torch.float64) for k in ("P", "TA", "PET")]
    return dcrest.simulate(*f, p, cfg or CrestConfig(), return_states=True)


def test_water_balance_closes(synthetic):
    """P = runoff + ET + storage change, per unit, to round-off."""
    p = _params(synthetic)
    q, aux = _run(synthetic, p)
    n = synthetic["n"]
    init = dcrest.initial_states(p, n)
    stored = sum(aux[k] - init[k] for k in ("swe", "canopy", "sm", "gw", "fast"))
    residual = (synthetic["P"].to(torch.float64).sum(0)
                - aux["runoff"].sum(0) - aux["et"].sum(0) - stored)
    rel = (residual.abs() / synthetic["P"].to(torch.float64).sum(0)).max().item()
    assert rel < 1e-12, f"max relative water-balance residual {rel:.3e}"


def test_loss_factor_removes_exactly_its_share(synthetic):
    """`lf` scales the routed output and nothing else."""
    q0, _ = _run(synthetic, _params(synthetic, lf=0.0))
    q1, _ = _run(synthetic, _params(synthetic, lf=0.25))
    assert torch.allclose(q1, 0.75 * q0, rtol=1e-12, atol=1e-12)


def test_no_rain_no_runoff_beyond_storage(synthetic):
    """With zero precipitation, discharge only ever recedes."""
    p = _params(synthetic)
    Z = torch.zeros_like(synthetic["P"], dtype=torch.float64)
    TA = synthetic["TA"].to(torch.float64)
    PET = synthetic["PET"].to(torch.float64)
    q = dcrest.simulate(Z, TA, PET, p)
    assert (q >= 0).all()
    later = q[400:]
    assert (later.diff(dim=0) <= 1e-9).all(), "recession is not monotonic"


def test_discharge_is_nonnegative_and_finite(synthetic):
    p = _params(synthetic)
    q, _ = _run(synthetic, p)
    assert torch.isfinite(q).all()
    assert (q >= 0).all()


def test_more_rain_gives_more_discharge(synthetic):
    """Monotone response to forcing — a basic sanity property of the chain."""
    p = _params(synthetic)
    f = [synthetic[k].to(torch.float64) for k in ("P", "TA", "PET")]
    q_lo = dcrest.simulate(*f, p)
    q_hi = dcrest.simulate(f[0] * 1.1, f[1], f[2], p)
    assert q_hi.sum(0).gt(q_lo.sum(0)).all()


def test_snow_delays_water(synthetic):
    """Below the threshold, precipitation is stored rather than routed."""
    n = synthetic["n"]
    p = dcrest.default_parameters(n, synthetic["area_km2"], dtype=torch.float64,
                                   values=dict(lf=0.0, tsnow=0.0, ddf=3.0))
    T = 400
    P = torch.zeros(T, n, dtype=torch.float64)
    P[10] = 50.0
    cold = torch.full((T, n), -5.0, dtype=torch.float64)
    warm = torch.full((T, n), 10.0, dtype=torch.float64)
    PET = torch.full((T, n), 0.0, dtype=torch.float64)
    q_cold, aux_cold = dcrest.simulate(P, cold, PET, p, return_states=True)
    q_warm = dcrest.simulate(P, warm, PET, p)
    # Baseline: the same run with no precipitation at all. Discharge is not zero
    # in either case because the initial soil and aquifer storage drains.
    q_dry = dcrest.simulate(torch.zeros_like(P), cold, PET, p)
    assert torch.allclose(aux_cold["swe"], torch.full((n,), 50.0,
                                                      dtype=torch.float64))
    assert torch.allclose(q_cold, q_dry, atol=1e-12), \
        "frozen precipitation reached the channel"
    assert (q_warm.sum(0) > q_dry.sum(0)).all()


def test_uh_weights_sum_to_one(synthetic):
    w = dcrest.gamma_uh(torch.tensor([1.0, 2.0, 4.0, 8.0]),
                         torch.tensor([2.0, 24.0, 120.0, 400.0]),
                         nstep=60, dt_hours=24.0, kmin_h=1.0)
    assert torch.allclose(w.sum(0), torch.ones(4), atol=1e-6)
    assert (w >= 0).all()


def test_uh_conserves_mass(synthetic):
    """Routing moves water in time without creating or destroying it."""
    T, N = 800, 4
    q = torch.zeros(T, N, dtype=torch.float64)
    q[100] = 10.0
    w = dcrest.gamma_uh(torch.tensor([2.0, 3.0, 1.5, 5.0], dtype=torch.float64),
                         torch.tensor([12.0, 40.0, 90.0, 200.0], dtype=torch.float64),
                         nstep=60, dt_hours=24.0, kmin_h=1.0)
    for method in ("loop", "conv"):
        routed = dcrest.routing.convolve_uh(q, w, method=method)
        assert torch.allclose(routed.sum(0), q.sum(0), rtol=1e-10)


def test_geomorphic_lag_increases_with_area():
    a = torch.tensor([10.0, 100.0, 1000.0, 10000.0, 100000.0])
    lag = dcrest.geomorphic_lag_hours(a, dtype=torch.float64)
    assert (lag.diff() > 0).all()
    assert lag.min() > 0


# --- optional Sacramento-style tension water --------------------------------

def test_tension_defaults_are_exact_noops(synthetic):
    """ptens=0 and plp=1 must reproduce the 16-parameter model bit-for-bit."""
    p = _params(synthetic)
    q_base, _ = _run(synthetic, p)
    p_ext = dict(p)
    p_ext["ptens"] = torch.zeros_like(p["wm"])
    p_ext["plp"] = torch.ones_like(p["wm"])
    f = [synthetic[k].to(torch.float64) for k in ("P", "TA", "PET")]
    q_ext = dcrest.simulate(*f, p_ext, CrestConfig(), validate=False)
    assert torch.equal(q_base, q_ext)


def test_tension_water_cannot_drain(synthetic):
    """With a tension fraction, soil water below it never reaches the aquifer."""
    n = synthetic["n"]
    f = [synthetic[k].to(torch.float64) for k in ("P", "TA", "PET")]
    out = {}
    for pt in (0.0, 0.6):
        p = _params(synthetic, ksoil=0.3)
        p["ptens"] = torch.full_like(p["wm"], pt)
        _, aux = dcrest.simulate(*f, p, CrestConfig(), return_states=True,
                                  validate=False)
        out[pt] = aux
    assert out[0.6]["sm"].gt(out[0.0]["sm"]).all(), "tension did not retain water"
    assert out[0.6]["gw"].le(out[0.0]["gw"] + 1e-9).all()


def test_plp_raises_evapotranspiration(synthetic):
    """Lowering plp holds ET at the full demand further into a dry-down."""
    f = [synthetic[k].to(torch.float64) for k in ("P", "TA", "PET")]
    et = {}
    for lp in (1.0, 0.3):
        p = _params(synthetic)
        p["plp"] = torch.full_like(p["wm"], lp)
        _, aux = dcrest.simulate(*f, p, CrestConfig(), return_states=True,
                                  validate=False)
        et[lp] = aux["et"].sum(0)
    assert et[0.3].ge(et[1.0] - 1e-9).all()


def test_water_balance_closes_with_tension(synthetic):
    p = _params(synthetic)
    p["ptens"] = torch.full_like(p["wm"], 0.35)
    p["plp"] = torch.full_like(p["wm"], 0.5)
    f = [synthetic[k].to(torch.float64) for k in ("P", "TA", "PET")]
    q, aux = dcrest.simulate(*f, p, CrestConfig(), return_states=True,
                              validate=False)
    init = dcrest.initial_states(p, synthetic["n"])
    stored = sum(aux[k] - init[k] for k in dcrest.STATE_NAMES)
    resid = f[0].sum(0) - aux["runoff"].sum(0) - aux["et"].sum(0) - stored
    assert (resid.abs() / f[0].sum(0)).max().item() < 1e-12


def test_tension_parameters_receive_gradient(synthetic):
    f = [synthetic[k].to(torch.float64) for k in ("P", "TA", "PET")]
    p = _params(synthetic, ksoil=0.2)
    for k, v in (("ptens", 0.3), ("plp", 0.6)):
        p[k] = torch.full_like(p["wm"], v).requires_grad_(True)
    dcrest.simulate(*f, p, CrestConfig(), validate=False).pow(2).sum().backward()
    for k in ("ptens", "plp"):
        g = p[k].grad
        assert torch.isfinite(g).all() and (g != 0).any(), f"{k} has no gradient"
