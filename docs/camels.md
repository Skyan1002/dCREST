# dCREST-CAMELS

Calibrated parameter sets for the CAMELS-US catchments, produced under the
community benchmark protocol so the numbers sit directly beside published
CAMELS model comparisons.

## Protocol

Newman et al. (2017) and Kratzert et al. (2019), followed exactly:

* train `1999-10-01 .. 2008-09-30`, spin-up from `1996-10-01`
* **test `1989-10-01 .. 1999-09-30`**, spin-up from `1986-10-01`, never touched
  during calibration or seed selection
* the 531-catchment comparison subset (area discrepancy < 10%); 528 of them are
  held here, and the 665-catchment sets additionally cover every CAMELS gauge
* median NSE across catchments; loss `mean_b(1 − NSE_b)` with a variance floor

Six Adam seeds per configuration (2048 iterations in four restart phases, with
per-basin best-iterate tracking). The released set takes, per catchment, the
seed with the best **training** NSE — the same best-of-N-on-calibration rule
behind the published SAC-SMA benchmark, so the comparison is like for like. Test
NSE is flat from 512 to 2048 iterations (0.662 / 0.664 / 0.663 / 0.664), so the
optimisation budget is not the limiter.

## The four sets

| file | catchments | forcing | params | median test NSE |
|---|---|---|---|---|
| `dCREST_CAMELS_v1.0_daymet665` | 665 | Daymet | 16 | 0.662 (n = 665) |
| `dCREST_CAMELS_v1.0_maurer528` | 528 | Maurer | 16 | 0.659 |
| `dCREST_CAMELS_v1.0t_tension665` | 665 | Daymet | 17 | 0.669 |
| `dCREST_CAMELS_v2A_inversion_decoder` | 665 | Daymet | 16 | see below |

**v1.0 daymet665 is the default choice.**

**v2A is not a benchmark set.** It was calibrated over 1983–2000 for a study that
needed a model frozen over 2003–2014, so its calibration period *contains* the
benchmark test window. Scoring it there measures nothing. It is released because
the inversion results depend on it, not as a competitive product.

The three benchmark sets ship as `.npz` with full provenance (bounds, UH clamp,
protocol, member list, per-catchment metrics) and `.csv` with one row per gauge.
**v2A carries only `ids`, `bounds` and `uh_kmin_h`** — it predates the provenance
format, so it has no stored metrics and `config_from_meta` supplies the default
`dt_hours` and `uh_nstep` for it. A loop over all four sets that reads
`meta["metric_test_nse"]` will raise `KeyError` on v2A. `members/` holds the individual seeds behind the
ensemble rows: six each for `daymet665` and `maurer528`, four for the
`tension665` variant, whose ensemble row is therefore a four-member mean.

## Placement

Matched forcing, same metric code, medians across catchments.

| model | median test NSE |
|---|---|
| LSTM ensemble (Kratzert et al. 2019) | 0.74 |
| HBV upper benchmark (calibrated ensemble, Seibert et al. 2018) | 0.68 |
| dCREST ensemble mean (6 seeds) | 0.672 (Maurer), 0.684 (Daymet) |
| SAC-SMA + Snow-17 (Newman et al. 2015 simulations) | 0.664 |
| **dCREST best-of-6 (this release)** | **0.659–0.662** |
| mHM | 0.66 |
| VIC | 0.55 |

Paired against Newman's published SAC-SMA simulations, re-scored with the same
code on the same 528 catchments and the same forcing (Maurer):

| | dCREST v1.0 | SAC-SMA |
|---|---|---|
| median NSE | 0.659 | 0.664 |
| median KGE | 0.728 | 0.719 |
| r | 0.825 | 0.827 |
| α (variability ratio) | 0.844 | 0.823 |
| β (volume bias) | 0.953 | 0.937 |
| log-NSE (low flows) | 0.684 | 0.607 |
| FHV (%) | −27.4 | −28.0 |
| FLV (%) | +43.6 | +73.0 |
| catchments with NSE < 0 | 0.9% | 0.2% |

Paired ΔNSE −0.005, dCREST better in 47% of catchments, Wilcoxon p = 0.009:
slightly but genuinely below SAC-SMA, at mHM level, with better volume bias and
markedly better low-flow behaviour. SAC-SMA has the more robust tail.

Decomposing NSE, the gap is one of **variability**, not timing or volume:
dCREST's α (0.844) exceeds its r (0.825), whereas SAC-SMA's α (0.823) sits at
its r (0.827), and NSE is maximised at α = r. dCREST is slightly over-dispersed.

## The tension-store experiment

CREST has no tension-water store: soil water drains to the aquifer in proportion
to the whole storage. Importing the Sacramento construct — a calibrated fraction
of soil storage that can evaporate but not drain — as a seventeenth parameter
yields a small but extremely robust paired gain: ΔNSE +0.0025, seed-matched over
six seeds and both forcings, p = 8e-32, with a dose-response in the fitted
fraction (+0.010 in the tercile of catchments with the largest `ptens`, zero in
the smallest). An HBV-style ET-efficiency threshold gives nothing (−0.0002,
p = 0.26).

So part of the remaining gap to SAC-SMA is structural rather than a calibration
artefact. The 16-parameter model is kept as the primary product for continuity
with EF5.

## Gradient diagnostics

After one backward pass, the fraction of catchments in which each parameter
receives exactly zero gradient. In an earlier calibration two of sixteen were
inert — `fc`, whose range (0–150 mm/h) kept its capacity term from ever binding,
and `kmult`, clamped by a UH floor of half the time step in 99.2% of catchments.
Re-bounding `fc` to 0–5 mm/h and lowering the clamp to 1 h activated both. **The
released set has no inert parameters.** See `examples/05_gradient_health.py`.

## Reproducing

```bash
pip install -e .
python scripts/fetch_camels.py                                   # full forcing, ~96 MB
python scripts/calibrate_camels.py --tag repro --basins all --seed 0    # ~5 h on one GPU
python scripts/select_camels.py --tags repro --basins all --out repro_sel
```

The SAC-SMA paired comparison additionally needs Newman's published model output
(`basin_timeseries_v1p2_modelOutput_maurer.zip`, NCAR GDEX), parsed to an npz
and passed through the `SACSMA_NPZ` environment variable.

`camels/results/` holds the summary JSON behind every number on this page.

## Upstream data and licences

Forcing and discharge are derived from public datasets: CAMELS basin forcing
(Daymet, Maurer; Newman et al. 2015, NCAR), IMERG V07 (NASA GES DISC), USGS NWIS
discharge. What is redistributed here is basin-mean daily aggregates, for
reproducibility; the originals remain subject to their providers' terms.
Parameter sets and code are MIT.
