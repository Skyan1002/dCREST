"""The calibration driver returns the parameters that earned the reported score."""
import pytest
import torch

import dcrest
from dcrest.calibrate import CalibrationSchedule, calibrate, nse_loss


def test_returned_parameters_earn_the_returned_score(synthetic):
    P, TA, PET = synthetic["P"], synthetic["TA"], synthetic["PET"]
    area, n = synthetic["area_km2"], synthetic["n"]
    truth = {k: torch.full((n,), float(v)) for k, v in dcrest.DEFAULTS.items() if k in dcrest.BOUNDS}
    truth.update(wm=torch.full((n,), 320.0), b=torch.full((n,), 0.4),
                 ksoil=torch.full((n,), 0.12), kmult=torch.full((n,), 1.5))
    truth["area_km2"] = area
    obs = dcrest.simulate(P, TA, PET, truth).detach()
    loss = nse_loss(min_obs=1)

    res = calibrate(P, TA, PET, obs, area, loss_fn=loss,
                    schedule=CalibrationSchedule(iterations=12, phases=2))
    sim = dcrest.simulate(P, TA, PET, res.params, validate=False)
    per_unit, _ = loss(sim, obs)
    # best-iterate tracking must keep the iterate that was scored, not the one
    # produced by the optimiser step taken after scoring it
    torch.testing.assert_close(-per_unit, res.score, rtol=0.0, atol=1e-6)


@pytest.mark.filterwarnings("ignore:parameters receive no gradient:RuntimeWarning")
def test_schedule_runs_exactly_the_requested_iterations(synthetic):
    """`iterations` is the total number of optimiser steps, whatever `phases` is."""
    P, TA, PET = synthetic["P"][:90], synthetic["TA"][:90], synthetic["PET"][:90]
    area = synthetic["area_km2"]
    obs = dcrest.simulate(P, TA, PET,
                          dcrest.default_parameters(synthetic["n"], area)).detach()
    base = nse_loss(min_obs=1)
    for iterations, phases in ((1, 1), (5, 2), (7, 4), (8, 4)):
        calls = []
        def counting(sim, o):
            calls.append(1)
            return base(sim, o)
        calibrate(P, TA, PET, obs, area, loss_fn=counting,
                  schedule=CalibrationSchedule(iterations=iterations, phases=phases))
        assert len(calls) == iterations, (iterations, phases, len(calls))
    assert CalibrationSchedule(iterations=7, phases=4).phase_lengths() == [2, 2, 2, 1]
    with pytest.raises(ValueError, match="at least one iteration"):
        calibrate(P, TA, PET, obs, area, loss_fn=base,
                  schedule=CalibrationSchedule(iterations=1, phases=4))


def test_calibrate_rejects_bad_inputs(synthetic):
    P, TA, PET = synthetic["P"][:30], synthetic["TA"][:30], synthetic["PET"][:30]
    area = synthetic["area_km2"]
    obs = dcrest.simulate(P, TA, PET,
                          dcrest.default_parameters(synthetic["n"], area)).detach()
    neg = P.clone(); neg[0, 0] = -1.0
    with pytest.raises(ValueError, match="precip must be non-negative"):
        calibrate(neg, TA, PET, obs, area, loss_fn=nse_loss(min_obs=1),
                  schedule=CalibrationSchedule(iterations=1, phases=1))
    with pytest.raises(ValueError, match="area_km2"):
        calibrate(P, TA, PET, obs, area * 0.0, loss_fn=nse_loss(min_obs=1),
                  schedule=CalibrationSchedule(iterations=1, phases=1))
