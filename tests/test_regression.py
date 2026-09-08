"""Regression against the reference implementation.

Every published result of the amortized-inversion study was produced by
``crest_blackbox.py``. If this package does not reproduce it, the package is
wrong — not the reference. These tests pin that agreement on real CAMELS
forcing with a real calibrated parameter set, which is the only configuration
that matters for reproducibility.
"""
import numpy as np
import pytest
import torch

import dcrest
from dcrest.model import CrestConfig


def _legacy_config(camels):
    # The released set was calibrated with a 1 h UH clamp; older sets used the
    # dt/2 default. Honour whatever the file records.
    return CrestConfig(uh_kmin_h=camels["uh_kmin_h"], uh_method="loop")


def _to_torch(camels, dtype=torch.float64, device="cpu"):
    p = {k: torch.tensor(v, dtype=dtype, device=device)
         for k, v in camels["params"].items()}
    p["area_km2"] = torch.tensor(camels["area_km2"], dtype=dtype, device=device)
    f = [torch.tensor(camels[k], dtype=dtype, device=device)
         for k in ("P", "TA", "PET")]
    return f, p


def test_matches_reference_float64(reference, camels):
    """Bit-level agreement in double precision: the algorithms are identical."""
    (P, TA, PET), p = _to_torch(camels, torch.float64)
    reference.UH_KMIN_H = camels["uh_kmin_h"]
    expected = reference.simulate(P, TA, PET, p)
    got = dcrest.simulate(P, TA, PET, p, _legacy_config(camels))
    diff = (got - expected).abs().max().item()
    scale = expected.abs().max().item()
    assert diff <= 1e-12 * max(scale, 1.0), (
        f"max abs diff {diff:.3e} on peak {scale:.3f} mm/day")


def test_matches_reference_float32(reference, camels):
    """Agreement in the single precision actually used for calibration."""
    (P, TA, PET), p = _to_torch(camels, torch.float32)
    reference.UH_KMIN_H = camels["uh_kmin_h"]
    expected = reference.simulate(P, TA, PET, p)
    got = dcrest.simulate(P, TA, PET, p, _legacy_config(camels))
    diff = (got - expected).abs().max().item()
    assert diff <= 1e-4, f"max abs diff {diff:.3e} mm/day"
    assert torch.allclose(got, expected, rtol=1e-4, atol=1e-5)


def test_conv_and_loop_routing_agree(reference, camels):
    """The fast grouped-convolution router matches the reference loop."""
    (P, TA, PET), p = _to_torch(camels, torch.float64)
    base = _legacy_config(camels)
    q_loop = dcrest.simulate(P, TA, PET, p, base)
    q_conv = dcrest.simulate(P, TA, PET, p,
                              CrestConfig(uh_kmin_h=base.uh_kmin_h,
                                          uh_method="conv"))
    assert (q_loop - q_conv).abs().max().item() <= 1e-10


def test_reference_gradient_agreement(reference, camels):
    """Gradients w.r.t. precipitation agree — this is the inverse-problem path."""
    (P, TA, PET), p = _to_torch(camels, torch.float64)
    reference.UH_KMIN_H = camels["uh_kmin_h"]
    cfg = _legacy_config(camels)

    Pa = P.clone().requires_grad_(True)
    reference.simulate(Pa, TA, PET, p).pow(2).sum().backward()
    Pb = P.clone().requires_grad_(True)
    dcrest.simulate(Pb, TA, PET, p, cfg).pow(2).sum().backward()

    denom = Pa.grad.abs().max().item()
    assert (Pa.grad - Pb.grad).abs().max().item() <= 1e-10 * max(denom, 1.0)


def test_checkpointing_is_transparent(camels):
    """Checkpointed and plain runs agree in value and in gradient."""
    (P, TA, PET), p = _to_torch(camels, torch.float64)
    cfg = _legacy_config(camels)
    ckpt = CrestConfig(uh_kmin_h=cfg.uh_kmin_h, uh_method=cfg.uh_method,
                       checkpoint_every=97)

    Pa = P.clone().requires_grad_(True)
    qa = dcrest.simulate(Pa, TA, PET, p, cfg)
    qa.pow(2).sum().backward()
    Pb = P.clone().requires_grad_(True)
    qb = dcrest.simulate(Pb, TA, PET, p, ckpt)
    qb.pow(2).sum().backward()

    assert (qa - qb).abs().max().item() <= 1e-12
    assert (Pa.grad - Pb.grad).abs().max().item() <= 1e-10


def test_metrics_match_reference(reference, camels):
    """The metric implementations agree with the study's."""
    (P, TA, PET), p = _to_torch(camels, torch.float32)
    q = dcrest.simulate(P, TA, PET, p, _legacy_config(camels))
    obs = torch.tensor(camels["obs"])
    ref = reference.metrics(q, obs)
    got = dcrest.metrics.summary(q, obs)
    for key in ("nse", "kge", "r", "alpha", "beta"):
        a, b = ref[key], got[key]
        ok = np.isfinite(a) & np.isfinite(b)
        assert ok.sum() > 0
        assert np.abs(a[ok] - b[ok]).max() < 2e-4, key
