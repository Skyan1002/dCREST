"""Analytic gradients against central finite differences.

The model is only useful as a decoder if its gradients are right, so this is
checked against finite differences in double precision for every parameter and
for the forcing, on inputs that exercise both the wet and the dry branch.
"""
import numpy as np
import pytest
import torch

import dcrest
from dcrest.model import CrestConfig
from dcrest.params import BOUNDS

CFG = CrestConfig(uh_kmin_h=1.0, uh_method="loop")


def _setup(n=4, T=240, seed=3):
    torch.manual_seed(seed)
    t = torch.arange(T, dtype=torch.float64)
    season = torch.sin(2 * np.pi * t / 365.25)[:, None]
    P = torch.clamp(torch.randn(T, n, dtype=torch.float64) * 4 + 2, min=0.0)
    TA = 8 + 12 * season.expand(T, n) + torch.randn(T, n, dtype=torch.float64)
    PET = torch.full((T, n), 3.0, dtype=torch.float64)
    area = torch.logspace(1.7, 4.4, n, dtype=torch.float64)   # 50 .. 25000 km2
    p = dcrest.default_parameters(n, area, dtype=torch.float64)
    return P, TA, PET, p


def _objective(P, TA, PET, p):
    """A scalar with contributions from every timestep and unit."""
    q = dcrest.simulate(P, TA, PET, p, CFG, validate=False)
    w = torch.linspace(0.5, 1.5, q.shape[0], dtype=q.dtype)[:, None]
    return (w * q ** 2).sum()


@pytest.mark.parametrize("name", list(BOUNDS))
def test_parameter_gradient_matches_finite_difference(name):
    P, TA, PET, p = _setup()
    p = {k: (v.clone().requires_grad_(k == name) if torch.is_tensor(v) else v)
         for k, v in p.items()}
    _objective(P, TA, PET, p).backward()
    analytic = p[name].grad.clone()

    lo, hi = BOUNDS[name]
    h = 1e-6 * (hi - lo)
    numeric = torch.empty_like(analytic)
    with torch.no_grad():
        for i in range(analytic.numel()):
            base = p[name].detach().clone()
            up, dn = base.clone(), base.clone()
            up[i] += h
            dn[i] -= h
            pu = dict(p, **{name: up})
            pd = dict(p, **{name: dn})
            numeric[i] = (_objective(P, TA, PET, pu)
                          - _objective(P, TA, PET, pd)) / (2 * h)

    scale = max(analytic.abs().max().item(), numeric.abs().max().item(), 1e-12)
    rel = (analytic - numeric).abs().max().item() / scale
    assert rel < 2e-5, (f"{name}: max relative error {rel:.3e} "
                        f"(analytic {analytic.tolist()}, numeric {numeric.tolist()})")


def test_forcing_gradient_matches_finite_difference():
    """dQ/dP — the derivative the inverse problem is solved with."""
    P, TA, PET, p = _setup(n=3, T=120)
    Pg = P.clone().requires_grad_(True)
    _objective(Pg, TA, PET, p).backward()
    analytic = Pg.grad.clone()

    rng = np.random.default_rng(0)
    idx = [(int(rng.integers(0, P.shape[0])), int(rng.integers(0, P.shape[1])))
           for _ in range(40)]
    h = 1e-6
    worst = 0.0
    with torch.no_grad():
        for t, b in idx:
            up, dn = P.clone(), P.clone()
            up[t, b] += h
            dn[t, b] -= h
            num = (_objective(up, TA, PET, p) - _objective(dn, TA, PET, p)) / (2 * h)
            denom = max(abs(num.item()), abs(analytic[t, b].item()), 1e-9)
            worst = max(worst, abs(num.item() - analytic[t, b].item()) / denom)
    assert worst < 2e-5, f"max relative error {worst:.3e}"


def test_gradients_are_finite_at_saturation():
    """Saturated columns are where the VIC power terms are singular."""
    n, T = 3, 150
    p = dcrest.default_parameters(n, [100.0, 1000.0, 9000.0],
                                   dtype=torch.float64, values=dict(wm=5.0))
    P = torch.full((T, n), 300.0, dtype=torch.float64).requires_grad_(True)
    TA = torch.full((T, n), 15.0, dtype=torch.float64)
    PET = torch.full((T, n), 1.0, dtype=torch.float64)
    dcrest.simulate(P, TA, PET, p, CFG).sum().backward()
    assert torch.isfinite(P.grad).all()


def test_gradients_are_finite_when_dry():
    n, T = 3, 200
    p = dcrest.default_parameters(n, [100.0, 1000.0, 9000.0], dtype=torch.float64)
    P = torch.zeros(T, n, dtype=torch.float64).requires_grad_(True)
    TA = torch.full((T, n), 25.0, dtype=torch.float64)
    PET = torch.full((T, n), 12.0, dtype=torch.float64)
    dcrest.simulate(P, TA, PET, p, CFG).sum().backward()
    assert torch.isfinite(P.grad).all()


def test_gradient_activity_flags_a_frozen_parameter():
    """The diagnostic must actually fire when a bound silences a parameter.

    With `fc` on its legacy (0, 150) mm/h range the interflow cap never binds,
    so `fc` receives exactly zero gradient — the failure mode this package
    documents in `dcrest.params`.
    """
    n, T = 6, 300
    torch.manual_seed(5)
    area = torch.tensor([50.0, 200.0, 900.0, 3000.0, 9000.0, 30000.0])
    model = dcrest.CrestLumped(n, area, CFG, bounds=dict(BOUNDS, fc=(0.0, 150.0)),
                                init=dict(fc=10.0), dtype=torch.float64)
    P = torch.clamp(torch.randn(T, n, dtype=torch.float64) * 4 + 2, min=0.0)
    TA = torch.full((T, n), 12.0, dtype=torch.float64)
    PET = torch.full((T, n), 3.0, dtype=torch.float64)
    model(P, TA, PET, validate=False).pow(2).sum().backward()
    activity = model.gradient_activity()
    assert activity["fc"] == 1.0, "legacy fc bound should be fully inactive"
    assert activity["wm"] < 0.5 and activity["b"] < 0.5
