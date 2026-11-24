"""Parameter ranges are taken from one place; metrics and losses behave at their edges.

Each test pins a defect found in the public-release audit (2026-11-19): ranges given through a
``CrestConfig`` were ignored by ``save_parameters``, ``calibrate`` and ``CrestLumped``; the
metrics scored units without observations as perfect; the square-root loss had an infinite
derivative at zero flow; the reference-loop convolution failed for series shorter than the
unit hydrograph.
"""
import dataclasses

import numpy as np
import pytest
import torch

import dcrest
from dcrest.calibrate import CalibrationSchedule, calibrate, nse_loss
from dcrest.metrics import kge, nse
from dcrest.params import BOUNDS, BOUNDS_TENSION, resolve_bounds
from dcrest.routing import convolve_uh

WIDE = dict(BOUNDS, ft=(0.0, 0.95), b=(0.01, 10.0), hmaxaq=(5.0, 6000.0), gwe=(0.5, 9.0))


def test_resolve_bounds_order_and_conflict():
    cfg = dcrest.CrestConfig(bounds=WIDE)
    assert resolve_bounds(None, None) == dict(BOUNDS)
    assert resolve_bounds(None, cfg) == WIDE
    assert resolve_bounds(WIDE, None) == WIDE
    assert resolve_bounds(dict(WIDE), cfg) == WIDE
    with pytest.raises(ValueError):
        resolve_bounds(BOUNDS, cfg)


def test_save_parameters_keeps_the_config_ranges(tmp_path, synthetic):
    n, area = synthetic["n"], synthetic["area_km2"]
    model = dcrest.CrestLumped(n, area, config=dcrest.CrestConfig(bounds=BOUNDS_TENSION))
    p = {k: v.detach() for k, v in model.physical.items()}
    assert "ptens" in p
    path = dcrest.save_parameters(str(tmp_path / "t.npz"), p, [str(i) for i in range(n)],
                                  config=model.config)
    q, meta = dcrest.load_parameters(path)
    assert "ptens" in q and meta["bounds"]["ptens"] == BOUNDS_TENSION["ptens"]
    cfg = dcrest.config_from_meta(meta)
    a = dcrest.simulate(synthetic["P"], synthetic["TA"], synthetic["PET"], p, model.config)
    b = dcrest.simulate(synthetic["P"], synthetic["TA"], synthetic["PET"], q, cfg)
    torch.testing.assert_close(a, b, rtol=0, atol=1e-5)


def test_crestlumped_uses_config_ranges(synthetic):
    n, area = synthetic["n"], synthetic["area_km2"]
    model = dcrest.CrestLumped(n, area, config=dcrest.CrestConfig(bounds=WIDE))
    assert model.bounds == WIDE and model.config.bounds == WIDE
    with torch.no_grad():
        model.raw["ft"].fill_(6.0)          # sigmoid(6) * 0.95 = 0.944, above the default 0.7
    q = model(synthetic["P"], synthetic["TA"], synthetic["PET"])   # validates against WIDE
    assert torch.isfinite(q).all()


@pytest.mark.filterwarnings("ignore:parameters receive no gradient:RuntimeWarning")
def test_calibrate_uses_config_ranges(synthetic):
    P, TA, PET, area, n = (synthetic[k] for k in ("P", "TA", "PET", "area_km2", "n"))
    wide_fc = dict(BOUNDS, fc=(0.0, 150.0))
    obs = dcrest.simulate(P, TA, PET, dict(dcrest.default_parameters(n, area))).detach()
    res = calibrate(P, TA, PET, obs, area, loss_fn=nse_loss(min_obs=1),
                    config=dcrest.CrestConfig(bounds=wide_fc),
                    schedule=CalibrationSchedule(iterations=2, phases=1),
                    init={"fc": 10.0})
    back = dcrest.params.to_physical(res.raw, wide_fc)
    torch.testing.assert_close(back["fc"], res.params["fc"])
    assert float(res.params["fc"].max()) > 5.0   # the default range would cap it at 5


def test_metrics_are_nan_without_observations():
    sim = torch.rand(50, 3)
    obs = torch.rand(50, 3)
    obs[:, 1] = float("nan")
    for f in (nse, kge):
        v = f(sim, obs)
        assert torch.isnan(v[1]) and torch.isfinite(v[[0, 2]]).all()


def test_sqrt_loss_has_finite_gradient_at_zero_flow():
    sim = torch.zeros(40, 2, requires_grad=True)
    obs = torch.rand(40, 2)
    obs[5:9, 0] = float("nan")
    per_unit, _ = nse_loss(min_obs=1, transform="sqrt")(sim, obs)
    per_unit.mean().backward()
    assert torch.isfinite(sim.grad).all()


@pytest.mark.parametrize("T", [1, 5, 29, 30, 31])
def test_loop_and_conv_routing_agree_for_short_series(T):
    torch.manual_seed(0)
    q = torch.rand(T, 4, dtype=torch.float64)
    w = torch.softmax(torch.rand(30, 4, dtype=torch.float64), 0)
    torch.testing.assert_close(convolve_uh(q, w, method="loop"), convolve_uh(q, w, method="conv"))


def test_crestlumped_default_matches_previous_behaviour(synthetic):
    n, area = synthetic["n"], synthetic["area_km2"]
    model = dcrest.CrestLumped(n, area)
    assert model.bounds == dict(BOUNDS)
    assert dataclasses.replace(model.config, bounds=None) == dcrest.CrestConfig()
