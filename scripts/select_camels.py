#!/usr/bin/env python3
"""Combine multi-seed CAMELS calibrations into the released parameter set.

Two products, reported separately because they are not the same claim:

**best-of-N** — for each basin independently, keep the seed with the highest
*training*-window NSE. This is exactly the protocol behind the published
SAC-SMA benchmark (Newman et al. ran ~10 SCE seeds per basin and selected on
the calibration window), so the resulting number is directly comparable. The
test window is never consulted, so there is no leakage.

**ensemble** — the mean of the member simulations. This is the protocol behind
Seibert's HBV "upper benchmark", and it is a different object: a set of models,
not a parameter set. Reported alongside, never instead.

Also runs the matched-forcing paired comparison against SAC-SMA.

Usage:  python select_camels.py --tags p2_s0,p2_s1,... [--forcing P_daymet]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

import dcrest as pycrest
from calibrate_camels import TEST, TRAIN, evaluate, load, summarize, _report

HERE = os.path.dirname(os.path.abspath(__file__))
# Newman et al. (2015) published SAC-SMA simulations, re-scored for the paired
# benchmark. Not redistributed here; download basin_timeseries_v1p2_modelOutput
# from NCAR/UCAR and point this at the parsed npz (see README).
SACSMA = os.environ.get("SACSMA_NPZ", "")


def sacsma_nse(ids):
    """Newman's published SAC-SMA simulations, re-scored with our metric code."""
    if not os.path.exists(SACSMA):
        return None
    S = np.load(SACSMA, allow_pickle=True)
    sd = S["dates"].astype("datetime64[D]")
    sid = np.array([str(x).zfill(8) for x in S["ids"]])
    m = (sd >= TEST[0]) & (sd <= TEST[1])
    sim, obs = S["sim"][:, m], S["obs"][:, m]
    out = np.full(len(ids), np.nan)
    pos = {b: i for i, b in enumerate(sid)}
    for j, b in enumerate(ids):
        i = pos.get(b)
        if i is None:
            continue
        ok = np.isfinite(sim[i]) & np.isfinite(obs[i])
        if ok.sum() < 730:
            continue
        s, o = sim[i][ok], obs[i][ok]
        out[j] = 1 - ((s - o) ** 2).sum() / ((o - o.mean()) ** 2).sum()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--forcing", default="P_daymet")
    ap.add_argument("--basins", default="531")
    ap.add_argument("--out", default="camels_release")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--compile", action="store_true", default=True)
    args = ap.parse_args()
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    data = load(args)
    ids = data["ids"]
    print(f"combining {len(tags)} members over {len(ids)} basins", flush=True)

    members, sims = [], {"train": [], "test": []}
    for tag in tags:
        p, meta = pycrest.load_parameters(f"{HERE}/params/camels_{tag}.npz",
                                          device=args.device)
        cfg = pycrest.config_from_meta(meta, compile_step=args.compile)
        ev = evaluate(p, data, cfg)
        members.append({"tag": tag, "params": p, "meta": meta, "cfg": cfg,
                        "train_nse": ev["train"]["nse"], "ev": ev})
        for split in ("train", "test"):
            P, TA, PET, _ = data[split]
            with torch.no_grad():
                sims[split].append(
                    pycrest.simulate(P, TA, PET, p, cfg, validate=False))
        print("  " + _report(ev, tag), flush=True)

    # ---- best-of-N, selected on the TRAINING window only -------------------
    train_scores = np.vstack([m["train_nse"] for m in members])
    pick = np.nanargmax(np.nan_to_num(train_scores, nan=-9e9), axis=0)
    bounds = members[0]["meta"].get("bounds")
    keys = list(bounds) if bounds else list(pycrest.BOUNDS)
    best = {}
    for k in keys + ["area_km2"]:
        stack = torch.stack([m["params"][k] for m in members])
        best[k] = stack[torch.as_tensor(pick, device=stack.device), 
                        torch.arange(stack.shape[1], device=stack.device)]
    cfg0 = members[0]["cfg"]
    ev_best = evaluate(best, data, cfg0)
    print("\n" + _report(ev_best, "BEST-OF-N (train-selected)"), flush=True)
    counts = np.bincount(pick, minlength=len(tags))
    print("  seed usage: " + ", ".join(f"{t}={c}" for t, c in zip(tags, counts)),
          flush=True)

    # ---- ensemble mean simulation ------------------------------------------
    ev_ens = {s: pycrest.metrics.summary(torch.stack(sims[s]).mean(0),
                                         data[s][3], data[f"mask_{s}"],
                                         min_days=730)
              for s in ("train", "test")}
    print(_report(ev_ens, f"ENSEMBLE MEAN ({len(tags)} members)"), flush=True)

    # ---- paired comparison against SAC-SMA ---------------------------------
    comparison = None
    sac = sacsma_nse(ids)
    if sac is not None:
        from scipy.stats import wilcoxon
        ours = ev_best["test"]["nse"]
        v = np.isfinite(ours) & np.isfinite(sac)
        d = ours[v] - sac[v]
        stat, pval = wilcoxon(d)
        comparison = {"n": int(v.sum()), "ours_median": float(np.median(ours[v])),
                      "sacsma_median": float(np.median(sac[v])),
                      "paired_diff_median": float(np.median(d)),
                      "ours_better_pct": float(100 * (d > 0).mean()),
                      "wilcoxon_p": float(pval),
                      "note": "SAC-SMA is Maurer-forced; compare like with like"}
        print(f"\nvs SAC-SMA (n={int(v.sum())}): ours {np.median(ours[v]):+.4f} "
              f"| SAC-SMA {np.median(sac[v]):+.4f} | paired diff "
              f"{np.median(d):+.4f} | ours better {100*(d>0).mean():.1f}% "
              f"| Wilcoxon p={pval:.3g}", flush=True)

    out = {"protocol": "Newman2017/Kratzert2019 train 1999-10-01..2008-09-30, "
                       "test 1989-10-01..1999-09-30, 531-subset, median NSE",
           "forcing": args.forcing, "members": tags,
           "per_member": {m["tag"]: summarize(m["ev"]) for m in members},
           "best_of_n": summarize(ev_best),
           "ensemble_mean": summarize(ev_ens),
           "seed_usage": {t: int(c) for t, c in zip(tags, counts)},
           "vs_sacsma": comparison}
    json.dump(out, open(f"{HERE}/results/{args.out}.json", "w"), indent=1)
    pycrest.save_parameters(
        f"{HERE}/params/{args.out}.npz", best, ids, config=cfg0, bounds=bounds,
        metrics={"test_nse": ev_best["test"]["nse"],
                 "test_kge": ev_best["test"]["kge"],
                 "train_nse": ev_best["train"]["nse"]},
        provenance={"protocol": "Newman2017/Kratzert2019",
                    "selection": "per-basin best training NSE of "
                                 f"{len(tags)} seeds",
                    "members": ",".join(tags), "forcing": args.forcing,
                    "pycrest": pycrest.__version__})
    print(f"\nwrote params/{args.out}.npz and results/{args.out}.json", flush=True)


if __name__ == "__main__":
    main()
