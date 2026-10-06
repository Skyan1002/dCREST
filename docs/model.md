# Model description

Notation: all fluxes are **mm per step**; states are **mm**. `Δt` is the step in
hours, `k_d = Δt / 24` converts per-day rate parameters to the step. At the
daily step `k_d = 1`, which is the configuration every published result of this
model used.

Every equation below acts element-wise over modelling units; units never
exchange water, so the whole domain is one batched tensor operation.

## 1. Degree-day snow

    snowfall  = P            if T_air < tsnow, else 0
    rain      = P − snowfall
    melt      = min(SWE + snowfall,  ddf · k_d · (T_air − tsnow)⁺)
    SWE      ← SWE + snowfall − melt
    L         = rain + melt

The rain/snow split is a hard threshold. `tsnow` still receives gradient through
the melt term, which is where its discharge-identifiable signal is;
`kernels.snow.degree_day_step_soft` offers a sigmoid partition of width `τ` when
`tsnow` must be inferred from snow observations instead.

## 2. Interception

Capacity `C = ft · wm`. The store fills before the soil sees water and is
evaporated at the full atmospheric demand before the soil is asked.

    fill      = min(L, (C − W)⁺);      W ← W + fill;      L ← L − fill
    D         = ke · PET
    ET_canopy = min(D, W);             W ← min((W − ET_canopy)⁺, C)
    PET_soil  = (D − ET_canopy)⁺ / ke

The division by `ke` is deliberate: the soil kernel re-applies `ke` internally,
so passing the *unmultiplied* residual demand keeps the total atmospheric demand
consistent across the two stores.

## 3. CRESTPhys soil column

With `adjPET = ke · PET_soil`, the step takes one of two branches.

### Wet branch (`L > adjPET`)

Demand is met in full, and the surplus is partitioned by the VIC curve.

    P_soil    = (L − adjPET)(1 − im)
    P_imperv  = (L − adjPET) im
    SM_c      = min(SM, wm)
    W_max     = wm (1 + b)
    A         = W_max [1 − (1 − SM_c/wm)^(1/(1+b))]

`A` is the equivalent depth on the infiltration-capacity distribution
corresponding to the current storage. If `P_soil + A ≥ W_max` the column
saturates and everything above capacity runs off; otherwise infiltration is the
integral of the curve between `A` and `A + P_soil`:

    infiltration = min( wm [ (1 − A/W_max)^(1+b) − (1 − (A + P_soil)/W_max)^(1+b) ],
                        P_soil )
    R            = P_soil − infiltration          (saturation excess)
    SM'          = SM_c + infiltration

Setting `b → 0` collapses the distribution to a single capacity, i.e. a bucket.

The excess is then split by an interflow capacity proportional to mean wetness
over the step, in one of two ways (`CrestConfig.runoff_partition`):

    temX      = (SM_c + SM') / (2 wm) · fc · Δt

    "ef5"     (EF5 CRESTPhys, CRESTPhysModel.cpp; default)
    recharge  = min(R, temX)                  → aquifer, added with drainage below
    interflow = R − min(R, temX)
    overland  = P_imperv + (SM − wm)⁺

    "classic" (classic CREST, CRESTModel.cpp; releases ≤ 1.0.0)
    interflow = min(R, temX) + (SM − wm)⁺
    overland  = R − min(R, temX) + P_imperv

`fc` is in mm/h; `temX` is therefore a depth per step. Under `"classic"` it only
divides fast runoff between two fast paths (see "Dead parameters" in the README);
under `"ef5"` it sets how much of each storm recharges the aquifer.

### Dry branch (`L ≤ adjPET`)

Soil water evaporates in proportion to relative wetness; unmet demand is drawn
from the aquifer in proportion to its filling, representing riparian and
deep-rooted uptake.

    ET_soil   = min( (adjPET − L) SM_c / wm,  SM_c );      SM' = SM_c − ET_soil
    unmet     = (adjPET − L − ET_soil)⁺
    ET_gw     = min(unmet · GW / hmaxaq,  GW);             GW ← GW − ET_gw
    overland  = 0,          interflow = (SM − wm)⁺     ("classic")
    overland  = (SM − wm)⁺, interflow = 0              ("ef5")

### Aquifer (both branches)

    drainage  = min(ksoil · k_d · SM',  SM');   SM' ← SM' − drainage
    GW       ← GW + drainage + recharge          (recharge = 0 under "classic")
    spill     = (GW − hmaxaq)⁺;                 GW ← min(GW, hmaxaq)
    baseflow  = min( gwc · k_d · [exp(gwe · GW / hmaxaq) − 1],  GW )
    GW       ← GW − baseflow
    slow      = baseflow + spill

`expm1` is used rather than `exp(...) − 1` so baseflow is accurate — and its
gradient finite — as the aquifer empties. A linear reservoir would need a
threshold to stop producing flow from a dry store; this formulation does not.

## 4. Response and routing

Overland flow passes through a linear reservoir; interflow and baseflow do not.

    S        ← S + overland
    release   = min(qc · k_d, 1) · S;      S ← S − release
    runoff    = (release + interflow + slow) · (1 − lf)

`lf` is a deep-loss / abstraction fraction applied to the total: it scales the
routed output and nothing else, so it is exactly the term to hold at zero when
checking water balance.

Routing is a per-unit Gamma unit hydrograph evaluated at step midpoints and
normalised, so truncation at `L` steps conserves mass by construction:

    k_geo   = 0.35 (A_km² / 1000)^0.38 · 24        [hours]
    k       = max(kmult · k_geo,  k_min)
    w_i    ∝ ((i + ½)Δt / k)^(n−1) exp(−(i + ½)Δt / k) / Γ(n),   i = 0 … L−1
    w      ← w / Σ w
    Q_t     = Σ_i w_i · runoff_{t−i}

Anchoring the scale to an area-based prior makes `kmult` a dimensionless
multiplier that is comparable across catchment sizes, instead of a free time
constant that means something different in every basin.

`k_min` defaults to 1 h. The legacy value `Δt/2` truncated the scale in 99.2% of
CAMELS catchments and silenced the `kmult` gradient entirely; it is retained
only for reproducing older parameter sets.

## 5. Initial states

    SWE = 0,   W = 0,   SM = 0.5 · wm,   GW = 0.3 · hmaxaq,   S = 0

These are spin-up seeds, not calibrated quantities. Discard a warm-up period
(three years is ample at the daily step) rather than tuning them.

## 6. Water balance

With `lf = 0`, over any window,

    Σ P  =  Σ runoff  +  Σ ET  +  ΔSWE + ΔW + ΔSM + ΔGW + ΔS

closes to floating-point round-off (measured relative residual 4.6e-16 in
float64). `simulate(..., return_states=True)` returns `runoff` and `et` as
`(T, N)` series together with the final states, which is enough to audit the
budget of any run.

Note that `Q` (routed) does not close the budget on its own: at the end of a
run, up to `L` steps of water are still in the unit hydrograph. Use `runoff`
for balance checks and `Q` for comparison with observations.
