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
| LSTM ensemble (regional; Kratzert et al. 2019) | 0.760 |
| HBV upper benchmark (mean of 100 calibrated runs; Seibert et al. 2018) | 0.679 |
| dCREST mean of 6 runs | 0.672 (Maurer), 0.684 (Daymet) |
| SAC-SMA + Snow-17 "A" (Newman et al. 2015; calibrated 1980–1995, so 6 of the 10 test years are in-sample) | 0.664 |
| mHM (Mizukami et al. 2019) | 0.665 |
| **dCREST best-of-6 (this release)** | **0.659 (Maurer), 0.668 (Daymet)** |
| SAC-SMA + Snow-17 "B" (Kratzert et al. 2019 collection; calibrated 1999–2008) | 0.608 |
| VIC | 0.554 |

528 matched catchments of the 531-catchment comparison set; all Maurer-forced
except dCREST-Daymet. Out of sample for both (1995–1999), dCREST is ahead of
SAC-SMA "A" by +0.036 (paired median, p = 5e-9); against the protocol-matched "B"
it is ahead by +0.042 overall (p = 1.5e-21).

Paired against Newman's published SAC-SMA simulations ("A"), re-scored with the
same code on the same 528 catchments and the same forcing (Maurer); note that six
of the ten test years lie inside A's calibration period:

| | dCREST v1.0 | SAC-SMA |
|---|---|---|
| median NSE | 0.659 | 0.664 |
| median KGE | 0.728 | 0.719 |
| r | 0.825 | 0.827 |
| α (variability ratio) | 0.844 | 0.823 |
| β (volume bias) | 0.953 | 0.936 |
| log-NSE (low flows; log(Q + 0.01)) | 0.66 | 0.56 |
| FHV (%) | −27.4 | −28.0 |
| FLV (%) | +43.6 | +73.0 |
| catchments with NSE < 0 | 0.9% | 0.4% |

Paired ΔNSE −0.005, dCREST better in 47% of catchments, Wilcoxon p = 0.009 over
the full test window; A's lead comes from its in-sample years (1989–1995: −0.034;
1995–1999: +0.036). dCREST is at mHM level, with better volume bias and
markedly better low-flow behaviour. SAC-SMA has the more robust tail.

Decomposing NSE, dCREST's α (0.844) exceeds its r (0.825), whereas SAC-SMA's α
(0.823) sits at its r (0.827): dCREST is slightly over-dispersed. Most of the small
full-window gap to SAC-SMA "A" is, however, A's in-sample skill (see above).

## The tension-store experiment

CREST has no tension-water store: soil water drains to the aquifer in proportion
to the whole storage. Importing the Sacramento construct — a calibrated fraction
of soil storage that can evaporate but not drain — as a seventeenth parameter
yields a small but consistent gain. Pairing each of the four released tension runs
with the 16-parameter run of the same seed (Daymet), the median change in test NSE
is +0.0016 to +0.0022 per seed (each p < 1e-4); averaged over the four seeds per
catchment it is +0.0035 (positive in 61% of catchments, p = 2e-10). The gain grows
with the fitted fraction (about zero in the third of catchments with the smallest
`ptens`, +0.008 in the third with the largest) and does not grow with aridity.

The 16-parameter model is kept as the primary product for continuity with EF5.

## Gradient diagnostics

After one backward pass, the fraction of catchments in which each parameter
receives exactly zero gradient. In an earlier calibration two of sixteen were
inert — `fc`, whose range (0–150 mm/h) kept its capacity term from ever binding,
and `kmult`, clamped by a UH floor of half the time step in 99.2% of catchments.
Re-bounding `fc` to 0–5 mm/h and lowering the clamp to 1 h activated both: at
the starting values of the released calibration every parameter receives gradient
in nearly all catchments. At the released optimum the derivative of the training
loss is exactly zero for `fc` in 36% of catchments (there the interflow cap never
binds; removing it leaves the training discharge bitwise unchanged), for the two
snow parameters in 10% (catchments with almost no snow) and for `kmult` in 5%.
See `examples/05_gradient_health.py`.

## Spin-up

Scores are computed after a three-year spin-up from half-full soil and an
aquifer at 30% of `hmaxaq`. For most catchments the choice does not matter, but
in 32 (mostly dry, with a large and slowly draining calibrated aquifer) a
nine-year spin-up changes test NSE by more than 0.01: their parameters partly fit
the drainage of the initial aquifer storage. Run the released sets with the
same initial states and spin-up to reproduce the stored scores.

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
