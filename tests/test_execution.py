"""Device, dtype, determinism, memory and API-contract behaviour."""
import numpy as np
import pytest
import torch

import dcrest
from dcrest.model import CrestConfig

CFG = CrestConfig(uh_kmin_h=1.0)


def _fixture_to(synthetic, device, dtype):
    f = [synthetic[k].to(device=device, dtype=dtype) for k in ("P", "TA", "PET")]
    p = dcrest.default_parameters(synthetic["n"], synthetic["area_km2"],
                                   dtype=dtype, device=device)
    return f, p


def test_cpu_gpu_parity(synthetic, cuda_or_skip):
    """Same answer on CPU and CUDA, to single-precision round-off."""
    (Pc, Tc, Ec), pc = _fixture_to(synthetic, "cpu", torch.float32)
    (Pg, Tg, Eg), pg = _fixture_to(synthetic, "cuda", torch.float32)
    qc = dcrest.simulate(Pc, Tc, Ec, pc, CFG)
    qg = dcrest.simulate(Pg, Tg, Eg, pg, CFG).cpu()
    rel = ((qc - qg).abs().max() / qc.abs().max()).item()
    assert rel < 1e-5, f"CPU/GPU relative difference {rel:.3e}"


def test_cpu_gpu_gradient_parity(synthetic, cuda_or_skip):
    grads = {}
    for dev in ("cpu", "cuda"):
        (P, TA, PET), p = _fixture_to(synthetic, dev, torch.float64)
        P = P.clone().requires_grad_(True)
        dcrest.simulate(P, TA, PET, p, CFG).pow(2).sum().backward()
        grads[dev] = P.grad.cpu()
    rel = ((grads["cpu"] - grads["cuda"]).abs().max()
           / grads["cpu"].abs().max()).item()
    assert rel < 1e-9, f"gradient CPU/GPU relative difference {rel:.3e}"


def test_deterministic_across_runs(synthetic):
    (P, TA, PET), p = _fixture_to(synthetic, "cpu", torch.float32)
    a = dcrest.simulate(P, TA, PET, p, CFG)
    b = dcrest.simulate(P, TA, PET, p, CFG)
    assert torch.equal(a, b)


def test_float32_tracks_float64(synthetic):
    (P64, T64, E64), p64 = _fixture_to(synthetic, "cpu", torch.float64)
    (P32, T32, E32), p32 = _fixture_to(synthetic, "cpu", torch.float32)
    q64 = dcrest.simulate(P64, T64, E64, p64, CFG)
    q32 = dcrest.simulate(P32, T32, E32, p32, CFG)
    rel = ((q64.float() - q32).abs().max() / q64.abs().max()).item()
    assert rel < 1e-4, f"float32 drift {rel:.3e}"


def test_checkpointing_reduces_peak_memory(synthetic, cuda_or_skip):
    """The documented memory knob actually works."""
    (P, TA, PET), p = _fixture_to(synthetic, "cuda", torch.float32)
    P = P.clone().requires_grad_(True)
    peaks = {}
    for seg in (None, 48):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        cfg = CrestConfig(uh_kmin_h=1.0, checkpoint_every=seg)
        q = dcrest.simulate(P, TA, PET, p, cfg)
        q.pow(2).sum().backward()
        P.grad = None
        peaks[seg] = torch.cuda.max_memory_allocated()
    assert peaks[48] < 0.6 * peaks[None], (
        f"checkpointing saved too little: {peaks[48]} vs {peaks[None]} bytes")


def test_batch_independence(synthetic):
    """Units do not interact: a subset run alone gives identical results."""
    (P, TA, PET), p = _fixture_to(synthetic, "cpu", torch.float64)
    full = dcrest.simulate(P, TA, PET, p, CFG)
    sel = [0, 3, 5]
    psub = {k: (v[sel] if torch.is_tensor(v) and v.ndim else v)
            for k, v in p.items()}
    sub = dcrest.simulate(P[:, sel], TA[:, sel], PET[:, sel], psub, CFG)
    assert torch.allclose(full[:, sel], sub, atol=1e-12)


def test_compiled_step_matches(synthetic):
    """`compile_step` changes speed, not the algorithm.

    In float64 the two paths agree to ~1e-14, which shows the fused kernels are
    doing the same arithmetic. In float32 they differ by a few times 1e-6
    relative after 720 recursive steps: operator fusion reassociates the
    additions, and single precision remembers the difference. That is the
    expected cost of compilation, not a discrepancy.
    """
    for dtype, tol in ((torch.float64, 1e-12), (torch.float32, 1e-5)):
        (P, TA, PET), p = _fixture_to(synthetic, "cpu", dtype)
        plain = dcrest.simulate(P, TA, PET, p, CFG)
        comp = dcrest.simulate(P, TA, PET, p,
                                CrestConfig(uh_kmin_h=1.0, compile_step=True))
        rel = ((plain - comp).abs().max() / plain.abs().max()).item()
        assert rel < tol, f"{dtype}: relative difference {rel:.3e}"


def test_rejects_shape_mismatch(synthetic):
    (P, TA, PET), p = _fixture_to(synthetic, "cpu", torch.float32)
    with pytest.raises(ValueError, match="same shape"):
        dcrest.simulate(P, TA[:, :2], PET, p, CFG)
    with pytest.raises(ValueError, match="units"):
        dcrest.simulate(P[:, :2], TA[:, :2], PET[:, :2], p, CFG)
    with pytest.raises(ValueError, match=r"\(T, N\)"):
        dcrest.simulate(P[:, 0], TA[:, 0], PET[:, 0], p, CFG)


def test_rejects_bad_parameters(synthetic):
    (P, TA, PET), p = _fixture_to(synthetic, "cpu", torch.float32)
    bad = dict(p, wm=p["wm"] * 0 + 1e6)
    with pytest.raises(ValueError, match="outside bounds"):
        dcrest.simulate(P, TA, PET, bad, CFG)
    missing = {k: v for k, v in p.items() if k != "b"}
    with pytest.raises(KeyError, match="missing parameters"):
        dcrest.simulate(P, TA, PET, missing, CFG)
    no_area = {k: v for k, v in p.items() if k != "area_km2"}
    with pytest.raises(KeyError, match="area_km2"):
        dcrest.simulate(P, TA, PET, no_area, CFG)


def test_roundtrip_parameter_io(synthetic, tmp_path):
    (P, TA, PET), p = _fixture_to(synthetic, "cpu", torch.float32)
    ids = [f"basin{i:02d}" for i in range(synthetic["n"])]
    path = str(tmp_path / "params.npz")
    dcrest.save_parameters(path, p, ids, config=CFG,
                            provenance={"source": "unit test"})
    q, meta = dcrest.load_parameters(path)
    assert list(meta["ids"]) == ids
    assert meta["uh_kmin_h"] == 1.0
    cfg = dcrest.config_from_meta(meta)
    assert cfg.uh_kmin_h == CFG.uh_kmin_h and cfg.dt_hours == CFG.dt_hours
    a = dcrest.simulate(P, TA, PET, p, CFG)
    b = dcrest.simulate(P, TA, PET, q, cfg)
    assert torch.allclose(a, b, atol=1e-6)

    sub = dcrest.load_parameters(path, subset=ids[::2])[0]
    assert sub["wm"].shape[0] == len(ids[::2])


def test_raw_physical_roundtrip():
    raw = dcrest.params.default_raw(7, values=dict(wm=345.0, b=2.5))
    phys = dcrest.params.to_physical(raw)
    assert torch.allclose(phys["wm"], torch.full((7,), 345.0), atol=1e-3)
    back = dcrest.params.to_raw(phys)
    again = dcrest.params.to_physical(back)
    for k in phys:
        assert torch.allclose(phys[k], again[k], atol=1e-4), k


def test_relaxed_bounds_roundtrip(synthetic, tmp_path):
    """A set calibrated under widened bounds must load and validate cleanly."""
    relaxed = dict(dcrest.BOUNDS, ft=(0.0, 0.95), hmaxaq=(5.0, 6000.0))
    (P, TA, PET), p = _fixture_to(synthetic, "cpu", torch.float32)
    p = dict(p)
    p["ft"] = torch.full_like(p["ft"], 0.9)          # legal only under relaxed
    path = str(tmp_path / "relaxed.npz")
    dcrest.save_parameters(path, p, [f"b{i}" for i in range(synthetic["n"])],
                            config=CFG, bounds=relaxed)
    q, meta = dcrest.load_parameters(path)
    cfg = dcrest.config_from_meta(meta)

    # config_from_meta carries the calibrated bounds, so the safe path is also
    # the short one: no extra argument is needed to run a released set.
    assert cfg.bounds is not None and cfg.bounds["ft"] == (0.0, 0.95)
    out = dcrest.simulate(P, TA, PET, q, cfg)
    assert torch.isfinite(out).all()

    # A deliberately default config still rejects it: the guard is intact, it is
    # only the *stored* config that knows this set is legal.
    with pytest.raises(ValueError, match="outside bounds"):
        dcrest.simulate(P, TA, PET, q, dcrest.CrestConfig())

    # An explicit bounds= argument still overrides the config.
    out2 = dcrest.simulate(P, TA, PET, q, cfg, bounds=meta["bounds"])
    assert torch.equal(out, out2)


def test_metrics_follow_the_subset(synthetic, tmp_path):
    """Per-unit metrics stored in a file must be subset with the parameters."""
    (P, TA, PET), p = _fixture_to(synthetic, "cpu", torch.float32)
    n = synthetic["n"]
    ids = [f"b{i}" for i in range(n)]
    path = str(tmp_path / "metrics.npz")
    dcrest.save_parameters(path, p, ids, config=CFG,
                           metrics={"test_nse": [0.1 * i for i in range(n)]})
    want = ids[::2]
    q, meta = dcrest.load_parameters(path, subset=want)
    assert list(meta["ids"]) == want
    assert len(meta["metric_test_nse"]) == len(want)
    assert q["wm"].shape[0] == len(want)
    for j, b in enumerate(want):
        assert meta["metric_test_nse"][j] == pytest.approx(0.1 * ids.index(b))
