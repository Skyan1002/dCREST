"""04 — Invert the forcing: hold the model fixed, solve for precipitation.

    python examples/04_invert_the_forcing.py          # ~3 min

This is what distinguishes dcrest from a conventional rainfall-runoff code.
Discharge is differentiable with respect to the *input*, not only the
parameters, so with the parameters frozen you can ask: what precipitation would
have produced the observed hydrograph?

Setup here: the released v1.0 parameters are frozen, satellite precipitation
(IMERG) is the starting point, and a bounded daily correction is fitted against
observed discharge alone — no reference precipitation enters the objective. The
correction is a multiplier in [1/3, 3] parameterised as exp(ln3 * tanh(u)), so
it stays in range by construction.

We then check the corrected precipitation against Daymet, which never entered
the optimisation. Agreement usually improves, but read the caveat at the end:
fitting the hydrograph is not the same as recovering the rain.
"""
import os

import torch

import dcrest
from _data import load, run_window, scoring_mask

HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS = os.path.join(HERE, "..", "camels", "dCREST_CAMELS_v1.0_daymet665.npz")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ITERS = 150
WINDOW = ("2012-01-01", "2014-12-31")   # IMERG V07 starts in 2001, so the
SPINUP = 365                            # benchmark test window cannot be used

sat, dates = load("P_imerg", device=DEV)
ref, _ = load("P_daymet", device=DEV)          # reference, for scoring only

params, meta = dcrest.load_parameters(PARAMS, subset=sat["ids"], device=DEV)
# compile_step fuses the per-step kernels: ~5x here. Falls back if unavailable.
try:
    cfg = dcrest.config_from_meta(meta, compile_step=True)
    dcrest.simulate(sat["P"][:400], sat["T"][:400], sat["E"][:400], params, cfg)
    print("compiled step: on")
except Exception as exc:                        # noqa: BLE001
    cfg = dcrest.config_from_meta(meta)
    print(f"compiled step unavailable ({type(exc).__name__}); running eager")
for v in params.values():                       # frozen: no parameter gradients
    v.requires_grad_(False)

w = run_window(dates, *WINDOW, SPINUP)
P0, TA, E = sat["P"][w], sat["T"][w], sat["E"][w]
obs = sat["obs"][w]
mask = scoring_mask(dates, sat["obs"], *WINDOW)[w]

def skill(P):
    with torch.no_grad():
        return dcrest.metrics.kge(dcrest.simulate(P, TA, E, params, cfg),
                                  obs, mask=mask).median().item()

def agreement(P):                               # daily correlation with Daymet
    a, b = P.detach(), ref["P"][w]
    ok = torch.isfinite(a).all(1) & torch.isfinite(b).all(1)
    a, b = a[ok], b[ok]
    a, b = a - a.mean(0), b - b.mean(0)
    r = (a * b).mean(0) / (a.std(0) * b.std(0) + 1e-9)
    return r.median().item()

print(f"uncorrected satellite : KGE {skill(P0):.3f}   corr with Daymet {agreement(P0):.3f}")

u = torch.zeros_like(P0, requires_grad=True)    # one free variable per day, per basin
opt = torch.optim.Adam([u], lr=0.05)
LN3 = torch.log(torch.tensor(3.0, device=DEV))

for i in range(ITERS):
    opt.zero_grad()
    P = P0 * torch.exp(LN3 * torch.tanh(u))     # bounded multiplier, [1/3, 3]
    Q = dcrest.simulate(P, TA, E, params, cfg)
    loss = (1.0 - dcrest.metrics.kge(Q, obs, mask=mask)).mean() \
        + 0.01 * u.abs().mean()                 # keep the correction small
    loss.backward()
    opt.step()
    if i % 50 == 0:
        print(f"  iter {i:4d}   loss {loss.item():.4f}")

with torch.no_grad():
    P_corr = P0 * torch.exp(LN3 * torch.tanh(u))
print(f"corrected             : KGE {skill(P_corr):.3f}   corr with Daymet {agreement(P_corr):.3f}")

print("""
Read this result carefully. The discharge fit improves a great deal because the
observed discharge is both the constraint and the score — that measures
constraint satisfaction, not predictive skill. The agreement with Daymet is the
honest number here, since Daymet never entered the objective, and it moves much
less. What the hydrograph mostly constrains is the *depth* of storms the
satellite already detected; it says far less about storms it missed entirely.""")
