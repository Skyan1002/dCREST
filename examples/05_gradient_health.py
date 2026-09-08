"""05 — Find parameters that receive no gradient.

    python examples/05_gradient_health.py          # ~5 s

A parameter whose gradient is identically zero is not being calibrated. The loss
still falls, the calibration still "succeeds", and nothing warns you — the
optimiser is simply wasting a search dimension. A derivative-free method
(SCE-UA, DDS, a genetic algorithm) cannot detect this at all. Having gradients
is what makes the check possible.

This is not a hypothetical failure mode. In the first calibration of this model
over 665 CAMELS catchments, two of sixteen parameters were inert:

  fc     bounded (0, 150) mm/h, so the interflow cap min(R, temX) never bound
         -> zero gradient in 100% of catchments
  kmult  the UH scale was clamped at dt/2 = 12 h, truncating the geomorphic
         scale -> zero gradient in 99.2% of catchments

Re-bounding `fc` to (0, 5) mm/h and lowering the UH clamp to 1 h activated both,
and that is what the package now defaults to. Below, the legacy configuration is
run first, then the current one.

A caveat on reading the output. `kmult` reproduces here immediately, because the
UH clamp kills its gradient whatever the parameter values are. `fc` will often
look alive in this small run: its inertness arose at the *calibrated* values
across 665 catchments, where the interflow cap never bound, and a fresh
default-initialised 8-catchment run need not land in that regime. The lesson is
the procedure, not this particular table — run the check on your own calibrated
set, not on defaults.
"""
import torch

import dcrest
from _data import load

torch.manual_seed(0)

data, dates = load("P_daymet")
N = len(data["ids"])
P, TA, E = (x[:1500] for x in (data["P"], data["T"], data["E"]))
obs = data["obs"][:1500]
mask = torch.isfinite(obs)


def dead_fraction(config, bounds=None):
    model = dcrest.CrestLumped(n_units=N, area_km2=data["area_km2"],
                               config=config, bounds=bounds)
    (1.0 - dcrest.metrics.nse(model(P, TA, E), obs, mask=mask)).mean().backward()
    return model.gradient_activity()


print("fraction of catchments in which a parameter receives ZERO gradient\n")
legacy = dead_fraction(dcrest.LEGACY_CONFIG, bounds=dcrest.params.LEGACY_BOUNDS)
current = dead_fraction(dcrest.CrestConfig())

print(f"{'parameter':>10}  {'legacy':>8}  {'current':>8}")
for k in dcrest.PARAM_NAMES:
    flag = "  <-- was dead" if legacy[k] > 0.5 and current[k] <= 0.5 else ""
    print(f"{k:>10}  {legacy[k]:8.2f}  {current[k]:8.2f}{flag}")

worst_name = max(current, key=current.get)
print(f"\nworst current value: {current[worst_name]:.2f} ({worst_name})")
assert current[worst_name] <= 0.5, "a parameter is inert under the current configuration"
print("OK — every parameter receives gradient under the current defaults.")
print("""
Run this after any change to bounds, structure or the UH clamp, and on your own
calibrated set rather than on defaults: whether a bound binds depends on where
the parameters actually sit.""")
