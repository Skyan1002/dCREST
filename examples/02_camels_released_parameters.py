"""02 — Reproduce released CAMELS skill on real catchments.

    python examples/02_camels_released_parameters.py

Loads dCREST-CAMELS v1.0, drives it with Daymet forcing over the community
benchmark test window (1989-10-01 .. 1999-09-30 — a period the calibration
never saw), and scores it. The numbers printed here should match the
`metric_test_nse` stored inside the parameter file to ~1e-3.

The one rule worth remembering: **load the config from the file's metadata**.
A parameter set is only valid under the unit-hydrograph clamp it was calibrated
with, so `load_parameters` returns that clamp and `config_from_meta` turns it
into the `CrestConfig` to simulate with. Using a bare `CrestConfig()` here would
silently give different discharge.
"""
import os

import torch

import dcrest
from _data import TEST, load, run_window, scoring_mask

HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS = os.path.join(HERE, "..", "camels", "dCREST_CAMELS_v1.0_daymet665.npz")

data, dates = load("P_daymet")

params, meta = dcrest.load_parameters(PARAMS, subset=data["ids"])   # 8 of the 665
cfg = dcrest.config_from_meta(meta)                              # NOT CrestConfig()
print(f"protocol : {meta['prov_protocol']}")
print(f"forcing  : {meta['prov_forcing']}")
print(f"UH clamp : {cfg.uh_kmin_h} h\n")

run = run_window(dates, *TEST)
Q = dcrest.simulate(data["P"][run], data["T"][run], data["E"][run], params, cfg)

mask = scoring_mask(dates, data["obs"], *TEST)[run]
nse = dcrest.metrics.nse(Q, data["obs"][run], mask=mask)
kge = dcrest.metrics.kge(Q, data["obs"][run], mask=mask)

stored = meta["metric_test_nse"]
print(f"{'gauge':>10}  {'area km2':>8}  {'NSE':>6}  {'stored':>7}  {'KGE':>6}")
for i, gid in enumerate(data["ids"]):
    print(f"{gid:>10}  {data['area_km2'][i]:8.0f}  {nse[i]:6.3f}  "
          f"{stored[i]:7.3f}  {kge[i]:6.3f}")

err = (nse - torch.as_tensor(stored)).abs().max()
print(f"\nmedian NSE {nse.median():.3f} over {len(nse)} catchments")
print(f"max |reproduced - stored| = {err:.2e}")
assert err < 5e-3, "released metrics not reproduced"
print("OK — the released parameter set reproduces its stored skill.")
