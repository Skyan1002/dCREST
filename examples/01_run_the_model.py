"""01 — Run the model. The smallest thing that works.

    python examples/01_run_the_model.py

Shows the shape contract: forcing is (T, N), parameters are (N,), discharge is
(T, N). N is the number of modelling units (catchments); units never interact,
so the batch runs in parallel.
"""
import torch

import dcrest

torch.manual_seed(0)

N, T = 3, 730
params = dcrest.default_parameters(n_units=N, area_km2=[120.0, 800.0, 4300.0])

P = (torch.rand(T, N) < 0.3) * torch.rand(T, N) * 30.0   # mm/day, ~30% wet days
TA = (12.0 + 10.0 * torch.sin(torch.arange(T) / 58.0))[:, None].expand(T, N).contiguous()
E = torch.full((T, N), 3.0)                               # potential ET, mm/day

Q = dcrest.simulate(P, TA, E, params)

print(f"forcing {tuple(P.shape)} -> discharge {tuple(Q.shape)}")
print(f"mean precipitation {P.mean():.3f} mm/day")
print(f"mean discharge     {Q.mean():.3f} mm/day")
print(f"runoff ratio       {Q.mean() / P.mean():.3f}")

# Parameters are a plain dict of (N,) tensors. Nothing is hidden in an object.
print("\nparameters:", ", ".join(dcrest.PARAM_NAMES))
print("`python -c \"import dcrest; print(dcrest.params.describe())\"` for units and ranges")

# States are available if you need them; five per unit.
print("states:", ", ".join(dcrest.STATE_NAMES))
