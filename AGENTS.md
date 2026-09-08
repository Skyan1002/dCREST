# AGENTS.md — orientation for coding agents

Read this before touching the repository. It states what the code is, the five
rules that prevent silently-wrong results, and the shortest path to a working
run. Humans should read `README.md` instead; this file is terser and assumes you
will verify claims by running things.

## What this is

`dcrest` is the CREST rainfall–runoff model — the hydrologic core of the EF5
operational forecasting system — reimplemented in PyTorch so that discharge is
differentiable with respect to **both** the parameters and the meteorological
forcing.

Two consequences, and they are the reason the package exists:

1. Parameters can be fitted by gradient descent over thousands of catchments at
   once, instead of one catchment at a time by a derivative-free search.
2. The forcing sits on the same footing as the parameters, so the model can be
   **inverted**: frozen, and solved for the input that reproduces an observed
   hydrograph. That is what makes it usable as a decoder in forcing-inversion
   and data-assimilation problems.

The repository also ships **dCREST-CAMELS**: parameter sets for 665 CAMELS-US
catchments, calibrated and scored under the community benchmark protocol, in
`camels/`.

## Run something in 60 seconds

```bash
pip install -e ".[test]"
python examples/01_run_the_model.py                 # synthetic, instant
python examples/02_camels_released_parameters.py    # real catchments, ~15 s
pytest -q                                           # 50+ tests, ~90 s
```

`examples/` is ordered; each file is standalone, prints what it did, and asserts
its own result where there is a number to check. Start at 01 and stop when you
have what you need.

| example | shows | runtime |
|---|---|---|
| `01_run_the_model.py` | shapes, the forward call | instant |
| `02_camels_released_parameters.py` | load a released set, reproduce its published skill | ~15 s |
| `03_calibrate.py` | gradient-descent calibration of 8 catchments at once | ~60 s |
| `04_invert_the_forcing.py` | freeze the model, solve for precipitation | ~60 s |
| `05_gradient_health.py` | find parameters that receive no gradient | ~5 s |

## The five rules

**1. Forcing is `(T, N)`, parameters are `(N,)`, discharge is `(T, N)`.**
`N` is the number of modelling units (catchments). Units never interact, so the
whole batch runs in parallel — calibrating 600 catchments costs little more than
calibrating one. Units are mm per step throughout, degC for temperature. A
`(T,)` array broadcast against `(T, N)` is rejected, not silently expanded.

**2. Load a released parameter set with its own config. Never a bare `CrestConfig()`.**

```python
params, meta = dcrest.load_parameters("camels/dCREST_CAMELS_v1.0_daymet665.npz")
cfg = dcrest.config_from_meta(meta)          # carries the UH clamp AND the bounds
Q = dcrest.simulate(P, T, E, params, cfg)
```

A set is only valid under the unit-hydrograph clamp and the parameter ranges it
was fitted under. `config_from_meta` reads both from the file. Constructing
`CrestConfig()` by hand gives different discharge, or a spurious out-of-bounds
error, with no warning.

**3. Spin up, then mask. Do not truncate.**
Start the simulation ~3 years before the window you score, and exclude the
warm-up with a boolean mask rather than by slicing the forcing. Initial states
are seeds, not calibrated quantities. `examples/_data.py` has `run_window()` and
`scoring_mask()` for exactly this.

**4. Check `gradient_activity()` after any change to bounds or structure.**
A parameter whose gradient is identically zero is not being calibrated, and no
loss curve will tell you. This is not hypothetical: in the first calibration of
this model two of sixteen parameters were inert (`fc` bounded so its cap never
bound; `kmult` clamped by the UH floor). A value near 1.0 means dead.

**5. `1 - NSE` and `1 - KGE` are the losses; both are differentiable.**
`dcrest.metrics` returns per-unit `(N,)` tensors. Take `.mean()` over units for
a scalar loss and `.median()` for the number you report — the CAMELS literature
reports medians across catchments.

## Layout

```
src/dcrest/
  model.py       CrestConfig, simulate(), CrestLumped   ← start here
  params.py      bounds, defaults, raw↔physical transforms, gradient diagnostics
  metrics.py     NSE / KGE
  io.py          parameter sets: save/load with provenance
  kernels/       snow, canopy, soil (VIC curve + aquifer), reservoir
  routing/       Gamma unit hydrograph
camels/          released parameter sets (npz + csv), calibration seeds, metrics
examples/        numbered, runnable, self-checking
  data/          8-basin CAMELS sample so examples work offline
scripts/
  verify.py            the correctness table in the README
  calibrate_camels.py  full 665-basin calibration driver
  select_camels.py     best-of-N selection and the SAC-SMA paired comparison
  fetch_camels.py      obtain the full 665-basin forcing (not in git: 96 MB)
tests/           50+ tests: physics, gradients, execution, regression
docs/
  model.md       the process chain, equation by equation
  parameters.md  all 16 parameters: meaning, units, ranges, what they trade off
  camels.md      how the released sets were produced and how they place
```

## Parameter sets in `camels/`

| file | basins | forcing | params | median test NSE |
|---|---|---|---|---|
| `dCREST_CAMELS_v1.0_daymet665` | 665 | Daymet | 16 | 0.662 |
| `dCREST_CAMELS_v1.0_maurer528` | 528 | Maurer | 16 | 0.659 |
| `dCREST_CAMELS_v1.0t_tension665` | 665 | Daymet | 17 | 0.669 |
| `dCREST_CAMELS_v2A_inversion_decoder` | 665 | Daymet | 16 | — (different protocol) |

**v1.0 is the default choice.** Test window is 1989-10-01..1999-09-30, which the
calibration never touched.

**v2A is a trap if you use it wrongly.** It was calibrated 1983–2000 for a study
needing a frozen model over 2003–2014, so its calibration period *contains* the
benchmark test window. Never score it on the benchmark test window.

## Things that will bite you

- **Contended GPU.** The model is a long recursion over short steps, so it is
  kernel-launch bound, not compute bound: expect under 10% utilisation. Set
  `compile_step=True` for an ~8× speed-up on long runs, and run several jobs
  concurrently rather than trying to make one job saturate the card.
- **Memory in long back-propagation.** Every step of the recursion stores its
  intermediates. Set `checkpoint_every=96` for multi-decade daily runs;
  gradients are bit-identical, cost is one extra forward pass.
- **`sanitize=True` (default) hides pathological units** by replacing non-finite
  intermediates. That is deliberate — one bad unit would otherwise poison the
  whole batch's gradient — but set it `False` when you are debugging physics.
- **float32 vs float64.** The compiled and eager paths agree to ~1e-14 in
  float64 but differ by a few times 1e-6 in float32 after several hundred
  recursive steps, because fusion reassociates additions. Both are correct.
- **NaN in observations is normal.** USGS discharge has gaps. Always pass a
  mask; the metrics take one.

## Verifying a change

```bash
pytest -q                    # must stay green
python scripts/verify.py     # numerical identity table: bit-exactness,
                             # finite differences, water balance, CPU vs CUDA
```

`scripts/verify.py` is the authority on whether a change altered the model.
Several entries are exact zeros, so any drift shows immediately.

## Scope

Lumped only: one modelling unit per tensor column, no lateral coupling, response
translated by a per-unit Gamma unit hydrograph. Cell-to-cell channel routing —
the *coupled routing* half of the full CREST — is **not** implemented. Snow is
degree-day, not Snow-17. The kernels are grid-independent and take `dt_hours`,
so distributed and sub-daily configurations are an addition rather than a
rewrite, but they do not exist today. Do not claim otherwise in documentation.
