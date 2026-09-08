"""Shared fixtures.

The regression tests need the reference implementation (``crest_blackbox.py``)
and real CAMELS forcing from the study repository. Both are optional: the suite
degrades to synthetic data and skips the regression tests when they are absent,
so the package can be tested standalone.
"""
import os
import sys

import numpy as np
import pytest
import torch

STUDY_ROOT = os.environ.get("DCREST_STUDY_ROOT", "/media/scratch/skyan/RL_Precipitation")
FORCING = os.path.join(STUDY_ROOT, "data", "forcing_daily.npz")
PARAMS = os.path.join(STUDY_ROOT, "data", "params_v2A.npz")


@pytest.fixture(scope="session")
def reference():
    """The original ``crest_blackbox`` module, or skip."""
    if not os.path.exists(os.path.join(STUDY_ROOT, "crest_blackbox.py")):
        pytest.skip("reference crest_blackbox.py not available")
    if STUDY_ROOT not in sys.path:
        sys.path.insert(0, STUDY_ROOT)
    os.environ.pop("UH_KMIN_H", None)
    import crest_blackbox
    return crest_blackbox


@pytest.fixture(scope="session")
def camels():
    """Real forcing and a calibrated parameter set for 24 CAMELS basins."""
    if not (os.path.exists(FORCING) and os.path.exists(PARAMS)):
        pytest.skip("CAMELS forcing / parameters not available")
    d = np.load(FORCING, allow_pickle=True)
    z = np.load(PARAMS, allow_pickle=True)
    ids_f = np.array([str(i) for i in d["ids"]])
    ids_p = np.array([str(i) for i in z["ids"]])
    common = [b for b in ids_p if b in set(ids_f)][:24]
    cf = np.array([int(np.nonzero(ids_f == b)[0][0]) for b in common])
    cp = np.array([int(np.nonzero(ids_p == b)[0][0]) for b in common])
    T = slice(0, 2000)
    keys = [str(k) for k in z["pkeys"]] if "pkeys" in z.files else None
    from dcrest.params import BOUNDS
    keys = keys or list(BOUNDS)
    return {
        "ids": common,
        "P": np.nan_to_num(d["P_daymet"][T][:, cf]).astype(np.float32),
        "TA": np.nan_to_num(d["tmean"][T][:, cf]).astype(np.float32),
        "PET": np.nan_to_num(d["pet"][T][:, cf]).astype(np.float32),
        "obs": d["obs"][T][:, cf].astype(np.float32),
        "params": {k: z[k][cp].astype(np.float32) for k in keys},
        "area_km2": z["area_km2"][cp].astype(np.float32),
        "uh_kmin_h": float(z["uh_kmin_h"]) if "uh_kmin_h" in z.files else None,
    }


@pytest.fixture
def synthetic():
    """Deterministic synthetic forcing that exercises snow, storms and dry spells."""
    torch.manual_seed(11)
    T, N = 720, 6
    t = torch.arange(T, dtype=torch.float32)
    season = torch.sin(2 * np.pi * t / 365.25)[:, None]
    P = torch.clamp(torch.randn(T, N) * 4 + 1.5 + 2 * season, min=0.0)
    TA = 10 + 14 * season + torch.randn(T, N) * 2.5
    PET = torch.clamp(2.5 + 2.0 * season + 0.2 * torch.randn(T, N), min=0.1)
    area = torch.tensor([35.0, 180.0, 900.0, 3400.0, 12000.0, 40000.0])
    return {"P": P, "TA": TA, "PET": PET, "area_km2": area, "n": N, "T": T}


@pytest.fixture
def cuda_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return "cuda"
