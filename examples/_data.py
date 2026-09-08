"""Loader for the bundled 8-basin CAMELS sample (examples/data/sample_camels.npz).

The sample is a slice of the full dCREST-CAMELS forcing: 8 catchments spanning
16 to 1775 km2, daily 1980-01-01 .. 2014-12-31. It ships with the repository so
every example runs offline. For all 665 catchments see `scripts/fetch_camels.py`.

Every array is (T, N) with T = 12784 days and N = 8 basins, in mm/day except
`tmean` (degC). `obs` is USGS discharge with NaN where a day is missing.
"""
from __future__ import annotations

import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(HERE, "data", "sample_camels.npz")

# The community benchmark protocol (Newman et al. 2017; Kratzert et al. 2019).
TRAIN = ("1999-10-01", "2008-09-30")
TEST = ("1989-10-01", "1999-09-30")
SPINUP_DAYS = 1095  # 3 years, discarded before scoring


def load(forcing: str = "P_daymet", device: str = "cpu", dtype=torch.float32):
    """Return (data, dates) where data holds torch tensors on `device`.

    `forcing` selects the precipitation product: "P_daymet" (what v1.0 was
    calibrated on), "P_maurer", or "P_imerg".
    """
    z = np.load(SAMPLE, allow_pickle=True)
    dates = np.array([np.datetime64(str(d)[:10]) for d in z["dates"]])
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype, device=device)  # noqa: E731
    data = {
        "P": t(z[forcing]),
        "T": t(z["tmean"]),
        "E": t(z["pet"]),
        "obs": t(z["obs"]),
        "area_km2": t(z["area_km2"]),
        "ids": [str(x) for x in z["ids"]],
    }
    return data, dates


def window(dates: np.ndarray, start: str, end: str) -> slice:
    """Index slice for a date range, inclusive of both ends."""
    i0 = int(np.searchsorted(dates, np.datetime64(start)))
    i1 = int(np.searchsorted(dates, np.datetime64(end), side="right"))
    return slice(i0, i1)


def scoring_mask(dates: np.ndarray, obs: torch.Tensor, start: str, end: str,
                 spinup_days: int = SPINUP_DAYS) -> torch.Tensor:
    """Boolean (T, N) mask: True on days inside [start, end] with a valid obs.

    Simulation always starts `spinup_days` before `start` so states are warm;
    those days are excluded from scoring by this mask, never by truncating the
    forcing.
    """
    m = torch.zeros_like(obs, dtype=torch.bool)
    w = window(dates, start, end)
    m[w] = True
    return m & torch.isfinite(obs)


def run_window(dates: np.ndarray, start: str, end: str,
               spinup_days: int = SPINUP_DAYS) -> slice:
    """Index slice covering [start, end] plus the spin-up that precedes it."""
    w = window(dates, start, end)
    return slice(max(0, w.start - spinup_days), w.stop)
