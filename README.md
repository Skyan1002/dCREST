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
pip install -e ".[test]"          # from a checkout, with pytest
```

(Not yet on PyPI; install from a checkout.)

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
# subset= selects and orders the catchments to match your forcing columns
params, meta = dcrest.load_parameters("camels/dCREST_CAMELS_v1.0_daymet665.npz",
                                      subset=["01031500", "02361000"])
cfg = dcrest.config_from_meta(meta)     # carries the UH clamp and the bounds
Q = dcrest.simulate(P, T, E, params, cfg)      # P, T, E are (T, 2) here
```

Parameter columns must line up with forcing columns: the file holds 665
catchments, so pass `subset=` unless your forcing has all 665 in the file's own
order. Paths above are relative to the repository root.

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

They run on an 8-catchment CAMELS sample shipped in `examples/data/` (1.1 MB),
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

Test window 1989–1999, Maurer precipitation, the 528 shared catchments of the
531-catchment comparison set (Kratzert, 2019, HydroShare collection):

| model | median test NSE |
|---|---|
| LSTM, regional (Kratzert et al. 2019) | 0.760 |
| HBV upper benchmark (mean of 100 calibrated runs) | 0.679 |
| dCREST, mean of the 6 Adam runs | 0.672 |
| mHM (489 catchments) | 0.665 |
| SAC-SMA + Snow-17, calibrated 1980–1995 (overlaps the test window) | 0.664 |
| **dCREST best-of-6 (this release)** | **0.659** (0.662 on all 665 with Daymet) |
| SAC-SMA + Snow-17, calibrated 1999–2008 (period-matched) | 0.608 |
| VIC | 0.554 |

Against the period-matched SAC-SMA (same calibration and test periods), the paired
median ΔNSE is +0.042 and dCREST is better in 71% of catchments. The other
SAC-SMA set was calibrated on 1980–1995 and so has six of the ten test years in
its calibration; on 1995–1999, outside it, dCREST leads by +0.036. See
[`docs/camels.md`](docs/camels.md).

**v2A uses a different protocol** and must not be scored on the benchmark test
window, which its calibration period contains.

## The model

```
P, T_air  ── degree-day snow (tsnow, ddf) ───────────────► liquid input
          ── interception store (ft·wm, evaporates first) ► input to soil
          ── CREST soil column + aquifer ────────────────► overland
                VIC infiltration curve (wm, b, im)         interflow
                ET, then aquifer-supported residual ET     baseflow
                soil → aquifer drainage (ksoil)
                fill-and-spill aquifer (hmaxaq)
                exponential baseflow (gwc, gwe)
  overland ── linear reservoir (qc) ──► release
  (release + interflow + baseflow + spill) · (1 − lf)
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
| grouped-convolution router vs reference loop | **0.00e+00** | exact by construction |
| gradient with vs without checkpointing | **0.00e+00** | exact by construction |
| water balance closure | 4.63e-16 | relative residual |
| unit-hydrograph mass conservation | 2.22e-16 | Σ weights − 1 |
| discharge vs reference implementation (float64) | 2.71e-14 mm/day | 32 CAMELS catchments × 3000 days |
| dQ/dP vs reference implementation (float64) | 1.60e-14 | relative to peak gradient 1.4e+02 |
| autodiff vs central finite differences | 6.99e-08 | worst of all 16 parameters |
| CPU vs CUDA (float32) | 1.65e-05 | relative |
| discharge vs reference implementation (float32) | 1.91e-05 mm/day | 32 CAMELS catchments × 3000 days |

Measured with PyTorch 2.13 on an RTX 4500 Ada. **Only the first two rows are
exact by construction**; the rest are tolerances. The reference comparisons were
exactly zero under an earlier PyTorch and are float round-off under this one,
because reduction order and kernel fusion differ between builds — 3000 recursive
steps accumulate that. Treat a change of order-of-magnitude as a regression, not
a change in the last digits.

The three "vs reference implementation" rows need an independent implementation
that is **not part of this repository**; it is archived with the paper's data
record (folder `verification/study_root`), together with the EF5 C++ check.
Point `DCREST_STUDY_ROOT` at that folder. Without it these rows print `SKIPPED`
and the other rows still run; `--require-reference` makes their absence an error.

**Provenance.** The soil column (VIC infiltration curve, evaporation, interflow/
overland split) follows EF5's CREST water balance (`CRESTModel.cpp`); its NumPy
sibling was verified element-wise against the verbatim C++ on randomised year-long
multi-cell input, agreeing to ~1e-9 mm/s — single-precision round-off in EF5's own
state storage. The aquifer (ksoil drainage, fill-and-spill store, exponential
baseflow, aquifer-supported ET) follows CRESTPhys, a groundwater option in a
development fork of EF5 (github.com/chrimerss/EF5, commit 780420d), which codes the
CRESTv3.0 bucket (Chen et al. 2023); unlike CRESTPhys, the excess capped by `fc`
becomes interflow, as in CREST, rather than aquifer recharge. The aquifer,
interception store, fast reservoir, degree-day snow and unit hydrograph have no
counterpart in `CRESTModel.cpp` and are verified against the reference
implementation.

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
The package defaults to the corrected settings; `CrestConfig().legacy()` (or
`dcrest.model.LEGACY_CONFIG`) and
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
  rather than tuning them. The released scores assume a three-year spin-up from
  the default states; in 32 dry catchments with large, slowly draining aquifers a
  longer spin-up changes test NSE by more than 0.01 (see `docs/camels.md`).
* **The interception store is an effective evaporating store.** Its capacity
  `ft · wm` is calibrated (median 39 mm in the Daymet set) and supplies most of
  the evapotranspiration; it is not a leaf-interception capacity.
* `sanitize=True` (default) replaces non-finite intermediates and clamps states
  at zero each step, so one pathological unit cannot poison a batch's gradient.
  Set it `False` to have such cases fail loudly.

## Layout

```
src/dcrest/       the package
camels/           released parameter sets, calibration seeds, benchmark results
examples/         numbered, runnable, self-checking; 8-catchment sample included
scripts/
  selfcheck.py         eight checks in ~20 s; the fastest "does this work?"
  verify.py            numerical identity against a reference implementation
  gen_parameter_table.py  regenerates docs/parameters.md from the code
  calibrate_camels.py / select_camels.py   the CAMELS calibration pipeline
  fetch_camels.py      the full 665-catchment forcing (not in git)
tests/            physics, gradients, execution, regression
docs/             model.md, parameters.md, camels.md
AGENTS.md         orientation for coding agents
```

## Release 1.0.0

This is the release that accompanies the paper. The parameter sets were
calibrated with the calibration driver as first published (commit b1fc089), which
stored each catchment's best parameter vector one Adam step after the iterate that
earned the best score. Every stored metric, the best-of-six selection and the
paper's figures are computed from the stored vectors, so they are consistent
with the files. This release records the scored iterate instead; rerunning the
default-start Daymet calibration with it gives median test NSE 0.6618 against
0.6617, about as large a change as repeating the same GPU calibration. The
parameter sets were not recalibrated. The release also takes parameter ranges
from one place (`bounds=` or `config.bounds`), returns NaN instead of a score for
units without observations, and keeps the square-root loss differentiable at zero
flow.

## Citation

Model, benchmark and parameter sets: *dCREST-CAMELS: Numerical verification and
benchmarking of a differentiable lumped CREST model across the contiguous United
States* (manuscript in preparation for Environmental Modelling & Software). Code,
parameter sets and analysis archive of this release: Yan, S., Li, Z., Hong, Y.
(2026), Zenodo, https://doi.org/10.5281/zenodo.24492417.
Model lineage: Wang et al. (2011), CREST; Chen et al. (2023), the CRESTv3.0
groundwater bucket; EF5 (Flamig et al., 2020; HyDROS Lab / NOAA National
Weather Service).

## Licence

MIT. Forcing and discharge in `examples/data/` are derived from public datasets
(CAMELS, Newman et al. 2015; IMERG V07, NASA GES DISC; USGS NWIS) and remain
subject to their providers' terms.
