#!/usr/bin/env python3
"""Print the correctness evidence table quoted in the README.

    python scripts/verify.py
    python scripts/verify.py --quick               # skip the GPU timing rows
    python scripts/verify.py --require-reference   # fail if rows must be skipped

Three rows compare against an independent reference implementation that is NOT
part of this repository (`crest_blackbox.py` and the CAMELS arrays from the
study the model was built for). Without it those rows print SKIPPED and the rest
still run; point DCREST_STUDY_ROOT at a checkout that has them to enable them.

The reference comparisons are tolerances, not guarantees of bit-exactness: they
were exactly zero under an earlier PyTorch and are round-off (~1e-14 in float64,
~2e-5 in float32 after 3000 recursive steps) under others, because reduction
order and kernel fusion differ between builds. The rows that should be exactly
zero are the internal equivalences — router and checkpointing.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

import dcrest
from dcrest.model import CrestConfig

ROOT = os.environ.get("DCREST_STUDY_ROOT", "/media/scratch/skyan/RL_Precipitation")
rows = []


def add(check, value, note=""):
    rows.append((check, value, note))


def _camels(n_basins=32, n_days=3000):
    f = np.load(f"{ROOT}/data/forcing_daily.npz", allow_pickle=True)
    z = np.load(f"{ROOT}/data/params_v2A.npz", allow_pickle=True)
    idf = np.array([str(i) for i in f["ids"]])
    idp = np.array([str(i) for i in z["ids"]])
    common = [b for b in idp if b in set(idf)][:n_basins]
    cf = np.array([int(np.nonzero(idf == b)[0][0]) for b in common])
    cp = np.array([int(np.nonzero(idp == b)[0][0]) for b in common])
    T = slice(0, n_days)
    keys = [str(k) for k in z["pkeys"]]
    return common, cf, cp, T, f, z, keys


REFERENCE_ROWS = ("discharge vs reference implementation (float64)",
                  "discharge vs reference implementation (float32)",
                  "dQ/dP vs reference implementation (float64)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--require-reference", action="store_true",
                    help="exit non-zero if the external reference is missing")
    ap.add_argument("--quick", action="store_true",
                    help="skip the GPU timing rows (they dominate the runtime)")
    args = ap.parse_args()

    sys.path.insert(0, ROOT)
    torch.manual_seed(0)
    have_ref = (os.path.exists(f"{ROOT}/crest_blackbox.py")
                and os.path.exists(f"{ROOT}/data/forcing_daily.npz")
                and os.path.exists(f"{ROOT}/data/params_v2A.npz"))

    if have_ref:
        common, cf, cp, T, f, z, keys = _camels()
        kmin = float(z["uh_kmin_h"]) if "uh_kmin_h" in z.files else None
        cfg = CrestConfig(uh_kmin_h=kmin, uh_method="loop")

        def pack(dt):
            p = {k: torch.tensor(z[k][cp], dtype=dt) for k in keys}
            p["area_km2"] = torch.tensor(z["area_km2"][cp], dtype=dt)
            g = [torch.tensor(np.nan_to_num(f[k][T][:, cf]), dtype=dt)
                 for k in ("P_daymet", "tmean", "pet")]
            return g, p

        import crest_blackbox as REF
        REF.UH_KMIN_H = kmin
        for dt, label in ((torch.float64, "float64"), (torch.float32, "float32")):
            (P, TA, PET), p = pack(dt)
            a = REF.simulate(P, TA, PET, p)
            b = dcrest.simulate(P, TA, PET, p, cfg)
            add(f"discharge vs reference implementation ({label})",
                f"{(a - b).abs().max().item():.2e} mm/day",
                f"{len(common)} CAMELS basins x {P.shape[0]} days")

        (P, TA, PET), p = pack(torch.float64)
        ga, gb = [], []
        for mod, out in ((REF, ga), (dcrest, gb)):
            Px = P.clone().requires_grad_(True)
            (mod.simulate(Px, TA, PET, p) if mod is REF
             else mod.simulate(Px, TA, PET, p, cfg)).pow(2).sum().backward()
            out.append(Px.grad)
        add("dQ/dP vs reference implementation (float64)",
            f"{(ga[0] - gb[0]).abs().max().item():.2e}", "relative to peak "
            f"{ga[0].abs().max().item():.2e}")

    # --- gradient check against central finite differences -----------------
    n, Td = 4, 240
    area = torch.logspace(1.7, 4.4, n, dtype=torch.float64)
    Pf = torch.clamp(torch.randn(Td, n, dtype=torch.float64) * 4 + 2, min=0.0)
    TAf = torch.full((Td, n), 11.0, dtype=torch.float64)
    Ef = torch.full((Td, n), 3.0, dtype=torch.float64)
    base = dcrest.default_parameters(n, area, dtype=torch.float64)
    fdcfg = CrestConfig(uh_kmin_h=1.0, uh_method="loop")

    def obj(pp):
        q = dcrest.simulate(Pf, TAf, Ef, pp, fdcfg, validate=False)
        w = torch.linspace(0.5, 1.5, q.shape[0], dtype=q.dtype)[:, None]
        return (w * q ** 2).sum()

    worst, worst_name = 0.0, ""
    for name, (lo, hi) in dcrest.BOUNDS.items():
        p = {k: (v.clone().requires_grad_(k == name) if torch.is_tensor(v) else v)
             for k, v in base.items()}
        obj(p).backward()
        an = p[name].grad.clone()
        h = 1e-6 * (hi - lo)
        num = torch.empty_like(an)
        with torch.no_grad():
            for i in range(n):
                up, dn = p[name].detach().clone(), p[name].detach().clone()
                up[i] += h
                dn[i] -= h
                num[i] = (obj(dict(p, **{name: up}))
                          - obj(dict(p, **{name: dn}))) / (2 * h)
        scale = max(an.abs().max().item(), num.abs().max().item(), 1e-12)
        rel = (an - num).abs().max().item() / scale
        if rel > worst:
            worst, worst_name = rel, name
    add("autodiff vs central finite differences",
        f"{worst:.2e}", f"worst of 16 parameters ({worst_name})")

    # --- water balance ------------------------------------------------------
    pw = dcrest.default_parameters(n, area, dtype=torch.float64,
                                    values=dict(lf=0.0))
    q, aux = dcrest.simulate(Pf, TAf, Ef, pw, fdcfg, return_states=True)
    init = dcrest.initial_states(pw, n)
    stored = sum(aux[k] - init[k] for k in dcrest.STATE_NAMES)
    resid = Pf.sum(0) - aux["runoff"].sum(0) - aux["et"].sum(0) - stored
    add("water balance closure",
        f"{(resid.abs() / Pf.sum(0)).max().item():.2e}", "relative residual")

    # --- routing ------------------------------------------------------------
    w = dcrest.gamma_uh(torch.tensor([1.5, 3.0, 5.0, 7.5], dtype=torch.float64),
                         torch.tensor([6.0, 40.0, 150.0, 500.0], dtype=torch.float64),
                         nstep=60, kmin_h=1.0)
    add("unit-hydrograph mass conservation",
        f"{(w.sum(0) - 1).abs().max().item():.2e}", "sum of weights minus one")
    qs = torch.zeros(600, 4, dtype=torch.float64)
    qs[50] = 10.0
    add("grouped-convolution router vs reference loop",
        f"{(dcrest.routing.convolve_uh(qs, w, 'conv') - dcrest.routing.convolve_uh(qs, w, 'loop')).abs().max().item():.2e}")

    # --- checkpointing, devices, throughput ---------------------------------
    Pc = Pf.clone().requires_grad_(True)
    dcrest.simulate(Pc, TAf, Ef, base, fdcfg).pow(2).sum().backward()
    Pd = Pf.clone().requires_grad_(True)
    dcrest.simulate(Pd, TAf, Ef, base,
                     CrestConfig(uh_kmin_h=1.0, uh_method="loop",
                                 checkpoint_every=37)).pow(2).sum().backward()
    add("gradient with vs without checkpointing",
        f"{(Pc.grad - Pd.grad).abs().max().item():.2e}")

    if torch.cuda.is_available() and not args.quick:
        N, Tg = 512, 4000
        Pg = torch.rand(Tg, N, device="cuda") * 8
        TAg = torch.full((Tg, N), 11.0, device="cuda")
        Eg = torch.full((Tg, N), 3.0, device="cuda")
        pg = dcrest.default_parameters(N, torch.rand(N, device="cuda") * 5000 + 50,
                                        device="cuda")
        pc = {k: v.cpu() for k, v in pg.items()}
        a = dcrest.simulate(Pg.cpu(), TAg.cpu(), Eg.cpu(), pc, fdcfg)
        b = dcrest.simulate(Pg, TAg, Eg, pg, fdcfg).cpu()
        add("CPU vs CUDA (float32)",
            f"{((a - b).abs().max() / a.abs().max()).item():.2e}", "relative")

        for compiled in (False, True):
            cfg_gpu = CrestConfig(uh_kmin_h=1.0, checkpoint_every=96,
                                  compile_step=compiled)
            Pv = Pg.clone().requires_grad_(True)
            if compiled:      # exclude one-off compilation from the timing
                dcrest.simulate(Pv, TAg, Eg, pg, cfg_gpu).sum().backward()
                Pv.grad = None
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            dcrest.simulate(Pv, TAg, Eg, pg, cfg_gpu).pow(2).sum().backward()
            torch.cuda.synchronize()
            add(f"GPU forward+backward, compile_step={compiled}",
                f"{time.time() - t0:.2f} s",
                f"{N} units x {Tg} steps, peak "
                f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB "
                f"({torch.cuda.get_device_name(0)})")

    if not have_ref:
        for name in REFERENCE_ROWS:
            add(name, "SKIPPED", f"reference not found under DCREST_STUDY_ROOT={ROOT}")

    w1 = max(len(r[0]) for r in rows)
    w2 = max(len(r[1]) for r in rows)
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu only"
    print(f"dcrest {dcrest.__version__} — verification")
    print(f"torch {torch.__version__}, {dev}\n")
    print(f"{'check'.ljust(w1)}  {'result'.ljust(w2)}  note")
    print("-" * (w1 + w2 + 40))
    for c, v, note in rows:
        print(f"{c.ljust(w1)}  {v.ljust(w2)}  {note}")

    if not have_ref:
        print(f"\n{len(REFERENCE_ROWS)} rows SKIPPED: the external reference "
              f"implementation was not found.\nSet DCREST_STUDY_ROOT to a "
              f"checkout containing crest_blackbox.py to run them.")
        if args.require_reference:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
