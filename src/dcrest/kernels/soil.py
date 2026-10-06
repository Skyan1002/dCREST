"""CRESTPhys soil column and aquifer.

Port of ``CRESTPHYSModel::WaterBalanceInt`` from the EF5 development fork,
reduced to one modelling unit per tensor element, with ``torch.where`` in place
of the C++ branches and guarded powers, so it is subdifferentiable everywhere.

Runoff partition
----------------
The soil excess ``R`` is split by the interflow capacity
``temX = (SM + SM') / (2 wm) * fc * dt``. Two splits exist, selected by
``partition``:

``"ef5"`` (default)
    EF5's CRESTPhys (``CRESTPhysModel.cpp``): ``min(R, temX)`` **recharges the
    aquifer**, ``R - min(R, temX)`` leaves as interflow, and impervious plus
    above-capacity water is overland flow, in both the wet and the dry branch.
    With this split the kernel reproduces the verbatim C++ to float32 round-off
    on randomised multi-year input (``scripts/verify_ef5_partition.py``).
``"classic"``
    The split of EF5's classic CREST (``CRESTModel.cpp``): ``min(R, temX)`` is
    interflow, the rest plus impervious water is overland flow, above-capacity
    water is interflow, and nothing recharges the aquifer from ``R``. Versions
    up to 1.0.0 used this split together with the CRESTPhys aquifer -- a
    combination that is in no single EF5 model -- while describing it as a port
    of CRESTPhys; their element-wise check was against ``CRESTModel.cpp``. The
    released dCREST-CAMELS parameter sets were calibrated with it, so stored
    sets without a recorded partition load as ``"classic"``.

The choice matters for what ``fc`` means. Under ``"classic"`` it only splits
fast runoff between two fast paths and is weakly identified; under ``"ef5"`` it
sets how much event water recharges the aquifer, i.e. the fast/slow split.

Two structural features distinguish CREST from the HBV/GR family and are worth
stating because they are what the parameters mean:

* **Runoff generation is a VIC variable-infiltration-capacity curve.** Storage
  capacity is distributed across the unit with shape exponent ``b``; runoff is
  saturation excess integrated over that distribution rather than a pointwise
  power law. Setting ``b`` near zero collapses it to a bucket.
* **Baseflow is exponential from a fill-and-spill aquifer**, not a linear
  reservoir: ``base = gwc * expm1(gwe * GW / hmaxaq)``. It tends smoothly to
  zero as the aquifer empties, so a dry aquifer is representable without a dead
  threshold and the gradient survives everywhere.

Sequence within a step: partition by wet/dry, VIC infiltration and interflow
split, soil drainage to the aquifer, spill, baseflow release.

Optional tension water
----------------------
Two extension parameters, both exact no-ops at their defaults, import the part
of the Sacramento (SAC-SMA) structure that CREST otherwise lacks: a fraction of
soil storage that can be evaporated but not drained.

``ptens`` (default 0)
    Tension fraction of ``wm``. Only ``(SM - ptens*wm)+`` — the free water — can
    drain to the aquifer. Below that, water leaves only by evapotranspiration.
    In CREST as originally formulated, ``soil2gw = ksoil * SM`` drains the whole
    store, so there is no protected reservoir and no wetting-up threshold after
    a dry season.
``plp`` (default 1)
    Fraction of ``wm`` above which evapotranspiration proceeds at the full
    atmospheric demand, in the manner of HBV's ``lp``. At 1 the demand scales
    linearly with wetness from the top of the store, which is the original
    behaviour.

Neither is part of :data:`dcrest.params.BOUNDS`; use
:data:`dcrest.params.BOUNDS_TENSION` to calibrate them.
"""
from __future__ import annotations

from typing import Mapping

import torch

Tensor = torch.Tensor


def _safe_pow(base: Tensor, exponent: Tensor) -> Tensor:
    """``base ** exponent`` with the base held off zero.

    The VIC curve evaluates ``x ** (1/(1+b))`` at ``x`` that reaches exactly 0
    when the column saturates. ``pow`` is finite there but its derivative is
    not; the offset keeps the backward pass finite without measurably moving
    the forward value.
    """
    return torch.pow(torch.clamp(base, min=0.0) + 1e-9, exponent)


def crestphys_step(p: Mapping[str, Tensor], sm: Tensor, gw: Tensor,
                   precip_mm: Tensor, pet_mm: Tensor,
                   dt_hours: float = 24.0, return_et: bool = False,
                   partition: str = "ef5"):
    """One CRESTPhys step. All fluxes in **mm per step**.

    Parameters
    ----------
    p
        Mapping providing ``wm, b, im, ke, fc, ksoil, hmaxaq, gwc, gwe`` as
        tensors broadcastable against the states.
    sm, gw
        Soil moisture and aquifer storage [mm].
    precip_mm, pet_mm
        Liquid input reaching the soil and potential ET for this step [mm].
        ``pet_mm`` is multiplied by ``ke`` inside.
    dt_hours
        Step length. Rate parameters (``fc`` per hour, ``ksoil``/``gwc`` per
        day) are scaled to the step; at ``dt_hours=24`` the scaling is the
        identity, which is the daily reference configuration.
    partition
        ``"ef5"`` (EF5 CRESTPhys) or ``"classic"`` (classic CREST split); see
        the module docstring.

    Returns
    -------
    ``(overland, interflow, baseflow, sm_new, gw_new)``, and additionally
    actual ET if ``return_et``.
    """
    if partition not in ("ef5", "classic"):
        raise ValueError(f"partition must be 'ef5' or 'classic', got {partition!r}")
    kd = dt_hours / 24.0
    WM, B, IM, KE, FC = p["wm"], p["b"], p["im"], p["ke"], p["fc"]
    KSOIL, HMAX, GWC, GWE = p["ksoil"], p["hmaxaq"], p["gwc"], p["gwe"]
    PTENS = p.get("ptens")            # tension fraction of wm; None -> 0
    PLP = p.get("plp")                # ET-efficiency fraction; None -> 1

    precip = precip_mm
    adjPET = pet_mm * KE
    SM0 = sm

    # ---- wet branch: input exceeds demand, VIC partitioning applies --------
    precipSoil = (precip - adjPET) * (1.0 - IM)
    precipImperv = precip - adjPET - precipSoil
    interflowExcess_w = torch.clamp(SM0 - WM, min=0.0)
    SM_c = torch.minimum(SM0, WM)
    sm_lt = SM_c < WM

    Wmaxm = WM * (1.0 + B)
    A = Wmaxm * (1.0 - _safe_pow(torch.clamp(1.0 - SM_c / WM, min=0.0),
                                 1.0 / (1.0 + B)))
    saturated = (precipSoil + A) >= Wmaxm
    R_sat = torch.clamp(precipSoil - (WM - SM_c), min=0.0)
    b1 = torch.clamp(1.0 - A / Wmaxm, min=0.0)
    b2 = torch.clamp(1.0 - (A + precipSoil) / Wmaxm, min=0.0)
    infiltration = torch.minimum(
        WM * (_safe_pow(b1, 1.0 + B) - _safe_pow(b2, 1.0 + B)), precipSoil)
    R_inf = torch.clamp(precipSoil - infiltration, min=0.0)
    Wo_inf = SM_c + infiltration
    R_lt = torch.where(saturated, R_sat, R_inf)
    Wo_lt = torch.where(saturated, torch.broadcast_to(WM, R_sat.shape), Wo_inf)
    R_w = torch.where(sm_lt, R_lt, precipSoil)
    Wo_w = torch.where(sm_lt, Wo_lt, torch.broadcast_to(WM, R_w.shape))

    # interflow capacity: mean wetness over the step times the interflow rate
    temX = (SM_c + Wo_w) / WM / 2.0 * (FC * dt_hours)
    capped = torch.minimum(R_w, temX)
    if partition == "ef5":
        # CRESTPhysModel.cpp: the capped part recharges the aquifer
        recharge_w = capped
        inter_w = R_w - capped
        over_w = precipImperv + interflowExcess_w
    else:
        # CRESTModel.cpp: the capped part is interflow, nothing recharges
        recharge_w = torch.zeros_like(R_w)
        inter_w = capped + interflowExcess_w
        over_w = R_w - capped + precipImperv

    # ---- dry branch: soil ET, then aquifer-supported residual ET -----------
    interflowExcess_d = torch.clamp(SM0 - WM, min=0.0)
    SM_cd = torch.minimum(SM0, WM)
    wetness = SM_cd / WM if PLP is None else torch.clamp(
        SM_cd / (torch.clamp(PLP, min=1e-3) * WM), max=1.0)
    ExcessET = (adjPET - precip) * wetness
    ExcessET_act = torch.minimum(ExcessET, SM_cd)
    Wo_d = SM_cd - ExcessET_act
    unmet = torch.clamp(adjPET - precip - ExcessET_act, min=0.0)
    resET_d = torch.minimum(unmet * gw / HMAX, gw)      # riparian / deep-rooted

    wet = precip > adjPET
    if partition == "ef5":            # above-capacity water is overland flow
        over = torch.where(wet, over_w, interflowExcess_d)
        inter = torch.where(wet, inter_w, torch.zeros_like(SM0))
    else:                             # above-capacity water is interflow
        over = torch.where(wet, over_w, torch.zeros_like(SM0))
        inter = torch.where(wet, inter_w, interflowExcess_d)
    Wo = torch.where(wet, Wo_w, Wo_d)
    gw = (gw - torch.where(wet, torch.zeros_like(SM0), resET_d)
          + torch.where(wet, recharge_w, torch.zeros_like(SM0)))

    # ---- soil drainage to the aquifer, from free water only ----------------
    drainable = Wo if PTENS is None else torch.clamp(Wo - PTENS * WM, min=0.0)
    soil2gw = torch.minimum(KSOIL * kd * drainable, drainable)
    Wo = Wo - soil2gw
    gw = gw + soil2gw

    # ---- fill-and-spill aquifer, exponential baseflow ----------------------
    spill = torch.clamp(gw - HMAX, min=0.0)
    gw = torch.minimum(gw, HMAX)
    base = torch.minimum(
        GWC * kd * torch.expm1(GWE * torch.clamp(gw, min=0.0) / HMAX), gw)
    gw = gw - base

    if return_et:
        et = torch.where(wet, adjPET, precip + ExcessET_act + resET_d)
        return over, inter, base + spill, Wo, gw, et
    return over, inter, base + spill, Wo, gw


__all__ = ["crestphys_step", "_safe_pow"]
