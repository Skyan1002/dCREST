"""03 — Calibrate parameters by gradient descent.

    python examples/03_calibrate.py           # 1-3 min, see the speed note below

Calibrates all 8 sample catchments *simultaneously*. Units do not interact, so
calibrating 8 costs about the same as calibrating 1 — the same code calibrates
665 on a GPU. `CrestLumped` holds unconstrained parameters that map into their
physical ranges through a sigmoid, so any `torch.optim` optimiser stays in
bounds by construction: no clipping, no penalty terms, no rejected proposals.

This is a deliberately small demonstration: a 1-year window, one seed, 80
iterations. **It will not match the released set**, which used the full
9-year window, 2048 iterations in four restart phases, and the best of six seeds
per catchment. For that, see `scripts/calibrate_camels.py`.

Speed note. The model is a long recursion over short steps, so it is bound by
kernel-launch overhead rather than arithmetic. `compile_step=True` fuses the
per-step function and is worth roughly 5x here, after a one-off compilation of
about 30 s. That single setting matters more than the device.
"""
import time

import torch

import dcrest
from _data import TEST, TRAIN, load, run_window, scoring_mask

torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ITERS = 80
TRAIN_SHORT = ("2007-10-01", "2008-09-30")   # 1 year of the 9-year train window
SPINUP = 365                                  # short spin-up to keep the demo quick

data, dates = load("P_daymet", device=DEV)
obs, area = data["obs"], data["area_km2"]

tr, te = run_window(dates, *TRAIN_SHORT, SPINUP), run_window(dates, *TEST)
m_tr = scoring_mask(dates, obs, *TRAIN_SHORT)[tr]
m_te = scoring_mask(dates, obs, *TEST)[te]


def build(compiled):
    cfg = dcrest.CrestConfig(compile_step=compiled)
    m = dcrest.CrestLumped(n_units=len(data["ids"]), area_km2=area,
                           config=cfg, device=DEV)
    return m, torch.optim.Adam(m.parameters(), lr=0.05)


def step():
    opt.zero_grad()
    Q = model(data["P"][tr], data["T"][tr], data["E"][tr])
    loss = (1.0 - dcrest.metrics.nse(Q, obs[tr], mask=m_tr)).mean()
    loss.backward()
    opt.step()
    return loss.item()


# compile_step fuses the per-step kernels and is worth ~5x here, at a one-off
# cost of ~30 s. Not every PyTorch build can compile this graph, so fall back.
model, opt = build(True)
try:
    step()
    print("compiled step: on (one-off compilation done)")
except Exception as exc:                      # noqa: BLE001
    print(f"compiled step unavailable ({type(exc).__name__}); running eager")
    model, opt = build(False)

print(f"calibrating {len(data['ids'])} catchments on {DEV}, "
      f"{tr.stop - tr.start} days, {ITERS} iterations")
t0 = time.time()
for i in range(ITERS):
    val = step()
    if i % 20 == 0 or i == ITERS - 1:
        print(f"  iter {i:4d}   train NSE {1 - val:6.3f}   "
              f"[{time.time() - t0:5.1f} s]")

with torch.no_grad():
    nse_te = dcrest.metrics.nse(model(data["P"][te], data["T"][te], data["E"][te]),
                                obs[te], mask=m_te)
print(f"\nmedian NSE on the held-out test window: {nse_te.median():.3f}")
print("released v1.0, full protocol, same catchments: 0.790")

# Every parameter should receive gradient. A value near 1.0 means that
# parameter is inert: the optimiser wastes a search dimension on it, and no
# loss curve would ever tell you. A derivative-free calibration cannot see this.
dead = {k: v for k, v in model.gradient_activity().items() if v > 0.5}
print("\ngradient health:", "all 16 parameters active" if not dead
      else f"DEAD (fraction of catchments with zero gradient) {dead}")
