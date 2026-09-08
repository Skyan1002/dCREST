# dcrest

A differentiable, GPU-parallel implementation of **CREST** (Coupled Routing and
Excess Storage), the hydrologic model behind the [EF5](https://github.com/HyDROSLab/EF5)
operational forecasting system — with **dCREST-CAMELS**, calibrated parameter
sets for 665 CAMELS-US catchments.

Discharge is differentiable with respect to both the parameters **and the
meteorological forcing**. Two things follow. Parameters can be calibrated by
gradient descent over thousands of catchments simultaneously. And because the
forcing sits on the same footing as the parameters, the model can be *inverted*:
held fixed while the input is solved for — which is what makes it usable as a
decoder in forcing-inversion and data-assimilation problems.

```bash
pip install dcrest                # released version
pip install -e ".[test]"          # from a checkout, with pytest
```

Python ≥ 3.9, PyTorch ≥ 2.0. No compiled extensions, no Fortran, no
model-specific file formats.

> Using an agent on this repository? Point it at [`AGENTS.md`](AGENTS.md) — it
> is the short version, with the rules that prevent silently-wrong results.

## Quick start

```python
import torch, dcrest

params = dcrest.default_parameters(n_units=3, area_km2=[120., 800., 4300.])
P = torch.rand(365, 3) * 10          # precipitation, mm/day
T = 10 + 8 * torch.randn(365, 3)     # air temperature, degC
E = torch.full((365, 3), 3.0)        # potential ET, mm/day

Q = dcrest.simulate(P, T, E, params)   # (365, 3) discharge, mm/day
```

Forcing is `(T, N)`, parameters are `(N,)`, discharge is `(T, N)`. `N` is the
number of modelling units — catchments here — and units never interact, so the
whole batch runs in parallel on one GPU.

### Real catchments, released parameters

```python
params, meta = dcrest.load_parameters("camels/dCREST_CAMELS_v1.0_daymet665.npz")
cfg = dcrest.config_from_meta(meta)     # carries the UH clamp and the bounds
Q = dcrest.simulate(P, T, E, params, cfg)
```

Always build the config from the file's metadata. A set is only valid under the
unit-hydrograph clamp and parameter ranges it was fitted under, so
`load_parameters` records them and `config_from_meta` applies them rather than
guessing. `examples/02_camels_released_parameters.py` reproduces the released
skill from the shipped 8-catchment sample.

### Calibrating

`CrestLumped` holds unconstrained parameters that map into their physical ranges
through a sigmoid, so any `torch.optim` optimiser stays in bounds by
construction.

```python
model = dcrest.CrestLumped(n_units=N, area_km2=area, device="cuda")
opt = torch.optim.Adam(model.parameters(), lr=0.05)

for _ in range(2000):
    opt.zero_grad()
    loss = (1 - dcrest.metrics.kge(model(P, T, E), obs, mask=train_days)).mean()
    loss.backward()
    opt.step()

print(model.gradient_activity())   # see "Dead parameters" below
```

### Inverting the forcing

```python
P = P_satellite.clone().requires_grad_(True)
Q = dcrest.simulate(P, T, E, frozen_params)      # parameters frozen
(1 - dcrest.metrics.kge(Q, obs)).mean().backward()
P.grad                                            # dJ/dP, per day, per catchment
```

## Examples

Numbered, standalone, and each one checks its own result.

```bash
python examples/01_run_the_model.py                 # shapes and the forward call
python examples/02_camels_released_parameters.py    # reproduce published skill
python examples/03_calibrate.py                     # gradient-descent calibration
python examples/04_invert_the_forcing.py            # solve for precipitation
python examples/05_gradient_health.py               # find inert parameters
```

They run on an 8-catchment CAMELS sample shipped in `examples/data/` (1.2 MB),
so nothing needs downloading. `scripts/fetch_camels.py` obtains the full
665-catchment forcing.

## dCREST-CAMELS parameter sets

In `camels/`, as `.npz` (with provenance) and `.csv` (one row per gauge).

| file | catchments | forcing | params | median test NSE |
|---|---|---|---|---|
| `dCREST_CAMELS_v1.0_daymet665` | 665 | Daymet | 16 | 0.662 |
| `dCREST_CAMELS_v1.0_maurer528` | 528 | Maurer | 16 | 0.659 |
| `dCREST_CAMELS_v1.0t_tension665` | 665 | Daymet | 17 | 0.669 |
| `dCREST_CAMELS_v2A_inversion_decoder` | 665 | Daymet | 16 | different protocol |

Calibrated and scored under the community benchmark protocol (Newman et al.
2017; Kratzert et al. 2019): train 1999-10-01 to 2008-09-30, **test 1989-10-01
to 1999-09-30**, six Adam seeds, per catchment the seed with the best *training*
NSE — the same best-of-N-on-calibration rule behind the published SAC-SMA
benchmark, so the numbers are directly comparable.

| model | median test NSE |
|---|---|
| LSTM ensemble (Kratzert et al. 2019) | 0.74 |
| HBV upper benchmark (calibrated ensemble, Seibert et al. 2018) | 0.68 |
| dCREST 6-member ensemble mean | 0.67–0.68 |
| SAC-SMA + Snow-17 (Newman et al. 2015) | 0.664 |
| **dCREST best-of-6 (this release)** | **0.659–0.662** |
| mHM | 0.66 |
| VIC | 0.55 |

Paired against SAC-SMA on the same 528 catchments and the same forcing: median
ΔNSE −0.005, dCREST better in 47% of catchments, Wilcoxon p = 0.009. Slightly
but genuinely below SAC-SMA, at mHM level, with better volume bias (β 0.955 vs
0.936) and better low-flow log-NSE (+0.044). See [`docs/camels.md`](docs/camels.md).

**v2A uses a different protocol** and must not be scored on the benchmark test
window, which its calibration period contains.

## The model

```
P, T_air  ── degree-day snow (tsnow, ddf) ───────────────► liquid input
          ── interception store (ft·wm, evaporates first) ► input to soil
          ── CRESTPhys ──────────────────────────────────► overland
                VIC infiltration curve (wm, b, im)         interflow
                ET, then aquifer-supported residual ET     baseflow
                soil → aquifer drainage (ksoil)
                fill-and-spill aquifer (hmaxaq)
                exponential baseflow (gwc, gwe)
  overland ── linear reservoir (qc) ──► release
  (release + interflow + baseflow) · (1 − lf)
          ── Gamma unit hydrograph (nshape, kmult) ───────► Q
```

Two features distinguish CREST from the HBV/GR family:

* **Runoff generation is a VIC variable-infiltration-capacity curve.** Storage
  capacity is distributed across the unit with shape exponent `b`, and runoff is
  saturation excess integrated over that distribution — not a pointwise power
  law. It represents sub-grid heterogeneity in infiltration capacity, which is
  what keeps the formulation meaningful at sub-daily steps and high rainfall
  intensity.
* **Baseflow is exponential from a fill-and-spill aquifer** of finite capacity,
  `base = gwc · expm1(gwe · GW / hmaxaq)`, rather than a linear reservoir. It
  goes smoothly to zero as the aquifer empties, so a dry aquifer needs no dead
  threshold and the gradient survives everywhere.

[`docs/model.md`](docs/model.md) has the chain equation by equation;
[`docs/parameters.md`](docs/parameters.md) has every parameter with units,
ranges and what it trades off against.

## Correctness

Reproduce with `python scripts/verify.py`; `pytest` runs the same ground plus
API contracts.

| check | result | note |
|---|---|---|
| discharge vs reference implementation (float64) | **0.00e+00** mm/day | 32 CAMELS catchments × 3000 days |
| discharge vs reference implementation (float32) | **0.00e+00** mm/day | 32 CAMELS catchments × 3000 days |
| dQ/dP vs reference implementation | **0.00e+00** | peak gradient 1.4e+02 |
| autodiff vs central finite differences | 6.99e-08 | worst of all 16 parameters |
| water balance closure | 4.63e-16 | relative residual |
| unit-hydrograph mass conservation | 2.22e-16 | Σ weights − 1 |
| grouped-convolution router vs reference loop | 0.00e+00 | |
| gradient with vs without checkpointing | 0.00e+00 | |
| CPU vs CUDA (float32) | 1.65e-05 | relative |

**Provenance.** The NumPy sibling of this water-balance kernel was verified
element-wise against the verbatim EF5 C++ (`CRESTModel.cpp`, `Snow17Model.cpp`)
on randomised year-long multi-cell input, agreeing to ~1e-9 mm/s — single-precision
round-off in EF5's own state storage.

## Performance

Kernel-launch overhead dominates: the model is a long recursion over short
steps, so the GPU idles unless the step is fused. `compile_step=True` is the
single most valuable setting.

| configuration | 512 units × 4000 steps, forward + backward |
|---|---|
| default | 63.8 s |
| `compile_step=True` | **7.6 s** (8.4× faster) |

Measured on an RTX 4500 Ada with `checkpoint_every=96`; peak memory 0.07 GiB.

* **`compile_step=True`** fuses the per-step kernels, at a one-off compilation
  cost of a few seconds. In float64 compiled and eager agree to ~1e-14; in
  float32 they differ by a few times 1e-6 relative after ~700 recursive steps,
  because fusion reassociates additions.
* **`checkpoint_every=96`** makes activation memory O(segment) instead of O(T),
  for one extra forward pass. Gradients are bit-identical. Without it,
  back-propagating through 4000 daily steps for a few hundred catchments will
  exhaust a 12 GB card.

If one job leaves the GPU underused (it will), run several concurrently; they
scale almost linearly until memory runs out.

## Dead parameters

A parameter whose gradient is identically zero is not being calibrated, and a
derivative-free optimiser cannot tell you — it just wastes a search dimension.
`gradient_activity()` reports, per parameter, the fraction of units with zero
gradient after `backward()`.

Not hypothetical. In the first calibration of this model over 665 CAMELS
catchments, two of sixteen parameters were inert:

| parameter | cause | catchments affected |
|---|---|---|
| `fc` | bounded (0, 150) mm/h, so the interflow cap `min(R, temX)` never bound | 100% |
| `kmult` | UH scale clamped at `dt/2` = 12 h, truncating the geomorphic scale | 99.2% |

Re-bounding `fc` to (0, 5) mm/h and lowering the UH clamp to 1 h activated both.
The package defaults to the corrected settings; `CrestConfig.legacy()` and
`dcrest.params.LEGACY_BOUNDS` reproduce the old behaviour when needed. See
`examples/05_gradient_health.py`.

## Scope and limitations

* **Lumped only.** One modelling unit per tensor column, no lateral coupling;
  response is translated by a per-unit Gamma unit hydrograph. Cell-to-cell
  channel routing — the *coupled routing* half of the full model — is not
  implemented here. The kernels are grid-independent and take `dt_hours`, so
  distributed and sub-daily configurations are an addition rather than a
  rewrite, but today they are absent.
* **Degree-day snow, not Snow-17.** Adequate where discharge is the only
  constraint; use a fuller snow model if snow observations are the target.
* **Initial states are spin-up seeds**, not calibrated. Discard a warm-up period
  rather than tuning them.
* `sanitize=True` (default) replaces non-finite intermediates and clamps states
  at zero each step, so one pathological unit cannot poison a batch's gradient.
  Set it `False` to have such cases fail loudly.

## Layout

```
src/dcrest/       the package
camels/           released parameter sets, calibration seeds, benchmark results
examples/         numbered, runnable, self-checking; 8-catchment sample included
scripts/          verify.py, the CAMELS calibration driver, data fetch
tests/            physics, gradients, execution, regression
docs/             model.md, parameters.md, camels.md
AGENTS.md         orientation for coding agents
```

## Citation

Parameter sets and the inversion study they were built for: *[manuscript in
preparation, Journal of Hydrology]*. Model lineage: Wang et al. (2011), CREST;
EF5 (HyDROS Lab / NOAA National Weather Service).

## Licence

MIT. Forcing and discharge in `examples/data/` are derived from public datasets
(CAMELS, Newman et al. 2015; IMERG V07, NASA GES DISC; USGS NWIS) and remain
subject to their providers' terms.
