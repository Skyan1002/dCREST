# Examples

Numbered, standalone, self-checking. Each prints what it did and asserts any
number it claims. They run on the 8-catchment CAMELS sample in `data/`
(1.1 MB), so nothing needs downloading.

| file | shows | runtime (CPU) |
|---|---|---|
| `01_run_the_model.py` | shapes, the forward call | 4 s |
| `02_camels_released_parameters.py` | load a released set, reproduce its published skill | 8 s |
| `03_calibrate.py` | gradient-descent calibration of 8 catchments at once | ~2 min |
| `04_invert_the_forcing.py` | freeze the model, solve for precipitation | ~3 min |
| `05_gradient_health.py` | find parameters that receive no gradient | 17 s |

`_data.py` is the shared loader. It also holds the benchmark windows and the
`run_window` / `scoring_mask` helpers that implement the spin-up-then-mask rule:
start the simulation about three years early, and exclude the warm-up with a
mask rather than by truncating the forcing.

To check the whole checkout in one command instead: `python scripts/selfcheck.py`.

## The sample

Eight CAMELS catchments spanning 17 to 1776 km2 and test NSE 0.58 to 0.86, daily
1980-01-01 to 2014-12-31. Chosen to be diverse rather than favourable, and all
present in both the Daymet and Maurer released sets.

Product coverage differs, which matters when choosing a window:

| product | usable period |
|---|---|
| `P_daymet` | 1980–2014 |
| `P_maurer` | 1980–2008 |
| `P_imerg` | 2001–2014 |

The benchmark test window (1989–1999) predates IMERG, which is why
`04_invert_the_forcing.py` uses 2011–2014 instead.
