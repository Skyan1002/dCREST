#!/usr/bin/env python3
"""Calibrate a CAMELS parameter set with pycrest under the community protocol.

Protocol (Newman et al. 2017 / Kratzert et al. 2019 — the split every CAMELS
model-comparison table uses):

    train   1999-10-01 .. 2008-09-30   (spin-up from 1996-10-01)
    TEST    1989-10-01 .. 1999-09-30   (spin-up from 1986-10-01)
    basins  the 531-subset (area discrepancy < 10%)
    metric  median NSE across basins

Train and test are run as two independent trajectories so that only the
training window carries an autograd graph; the test window is evaluated under
``no_grad`` and costs no activation memory. Parameters are selected on the
training window only.

Usage
-----
    python calibrate_camels.py --tag base
    python calibrate_camels.py --tag relaxed --relax-bounds
    python calibrate_camels.py --tag sqrt --transform sqrt
    python calibrate_camels.py --tag ens3 --seed 3 --init-jitter 0.5

Every run writes ``results/camels_<tag>.json`` and
``params/camels_<tag>.npz``; ``--ensemble tag1,tag2,...`` scores the mean
simulation of several completed runs without re-calibrating.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

import dcrest as pycrest
from dcrest.calibrate import CalibrationSchedule, calibrate, kge_loss, nse_loss
from dcrest.model import CrestConfig

HERE = os.path.dirname(os.path.abspath(__file__))
FORCING = f"{os.path.dirname(HERE)}/data/forcing_daily.npz"
BASIN_531 = f"{os.path.dirname(HERE)}/data/531_basin_list.txt"

TRAIN = (np.datetime64("1999-10-01"), np.datetime64("2008-09-30"))
TEST = (np.datetime64("1989-10-01"), np.datetime64("1999-09-30"))
TRAIN_SPINUP = np.datetime64("1996-10-01")
TEST_SPINUP = np.datetime64("1986-10-01")

#: Bounds the released set railed against, widened. `hmaxaq` at its 50 mm floor
#: in 22% of catchments, `ft` at its 0.7 ceiling in 16%, `gwe` at 6 in 6%.
RELAXED = dict(pycrest.BOUNDS, hmaxaq=(5.0, 6000.0), ft=(0.0, 0.95),
               gwe=(0.5, 9.0), b=(0.01, 10.0))

#: Sacramento-style tension water on top of the relaxed ranges. `ptens` protects
#: a fraction of soil storage from draining to the aquifer; `plp` holds ET at the
#: full atmospheric demand until storage drops below that fraction. Both reduce
#: to the base model at ptens=0 / plp=1, so these bounds strictly contain it.
TENSION = {"ptens": (0.0, 0.9), "plp": (0.05, 1.0)}


def load(args):
    d = np.load(FORCING, allow_pickle=True)
    dates = d["dates"].astype("datetime64[D]")
    ids = np.array([str(i) for i in d["ids"]])
    keep = np.ones(len(ids), bool)
    if args.basins == "531":
        want = {l.strip().zfill(8) for l in open(BASIN_531) if l.strip()}
        keep = np.isin(ids, np.array(sorted(want)))
    cols = np.nonzero(keep)[0]

    def window(start, end):
        m = (dates >= start) & (dates <= end)
        pack = [torch.tensor(np.nan_to_num(d[k][m][:, cols]), dtype=torch.float32,
                             device=args.device)
                for k in (args.forcing, "tmean", "pet")]
        pack.append(torch.tensor(d["obs"][m][:, cols], dtype=torch.float32,
                                 device=args.device))
        return pack, dates[m]

    tr, dtr = window(TRAIN_SPINUP, TRAIN[1])
    te, dte = window(TEST_SPINUP, TEST[1])
    score_tr = torch.tensor((dtr >= TRAIN[0]) & (dtr <= TRAIN[1]), device=args.device)
    score_te = torch.tensor((dte >= TEST[0]) & (dte <= TEST[1]), device=args.device)
    return {"ids": ids[cols], "area_km2": d["area_km2"][cols],
            "train": tr, "test": te, "mask_train": score_tr, "mask_test": score_te,
            "n_train": int(score_tr.sum()), "n_test": int(score_te.sum())}


def evaluate(params, data, cfg):
    out = {}
    for split in ("train", "test"):
        P, TA, PET, obs = data[split]
        with torch.no_grad():
            q = pycrest.simulate(P, TA, PET, params, cfg, validate=False)
        s = pycrest.metrics.summary(q, obs, data[f"mask_{split}"], min_days=730)
        out[split] = s
    return out


def _report(ev, label, extra=""):
    tr, te = ev["train"], ev["test"]
    v = np.isfinite(te["nse"]) & np.isfinite(tr["nse"])
    return (f"[{label}] n={int(v.sum())} train NSE {np.median(tr['nse'][v]):+.4f} "
            f"| TEST NSE med {np.median(te['nse'][v]):+.4f} "
            f"mean {np.mean(te['nse'][v]):+.4f} KGE {np.median(te['kge'][v]):+.4f} "
            f"beta {np.median(te['beta'][v]):.3f} "
            f"%<0 {100 * np.mean(te['nse'][v] < 0):.1f} "
            f"%>0.5 {100 * np.mean(te['nse'][v] > 0.5):.1f}{extra}")


def summarize(ev):
    tr, te = ev["train"], ev["test"]
    v = np.isfinite(te["nse"]) & np.isfinite(tr["nse"])
    return {"n": int(v.sum()),
            "train_nse": float(np.median(tr["nse"][v])),
            "train_kge": float(np.median(tr["kge"][v])),
            "test_nse": float(np.median(te["nse"][v])),
            "test_nse_mean": float(np.mean(te["nse"][v])),
            "test_kge": float(np.median(te["kge"][v])),
            "test_beta": float(np.median(te["beta"][v])),
            "test_frac_neg": float(np.mean(te["nse"][v] < 0)),
            "test_frac_gt05": float(np.mean(te["nse"][v] > 0.5))}


def run_ensemble(args, data, cfg):
    """Score the mean simulation of several completed runs."""
    tags = [t.strip() for t in args.ensemble.split(",") if t.strip()]
    sims = {"train": [], "test": []}
    for tag in tags:
        p, meta = pycrest.load_parameters(f"{HERE}/params/camels_{tag}.npz",
                                          device=args.device)
        c = pycrest.config_from_meta(meta, compile_step=args.compile)
        for split in ("train", "test"):
            P, TA, PET, _ = data[split]
            with torch.no_grad():
                sims[split].append(
                    pycrest.simulate(P, TA, PET, p, c, validate=False))
    ev = {}
    for split in ("train", "test"):
        mean = torch.stack(sims[split]).mean(0)
        ev[split] = pycrest.metrics.summary(mean, data[split][3],
                                            data[f"mask_{split}"], min_days=730)
    print(_report(ev, f"ensemble({len(tags)})"), flush=True)
    res = summarize(ev)
    res["members"] = tags
    json.dump(res, open(f"{HERE}/results/camels_ensemble_"
                        f"{'_'.join(tags)}.json", "w"), indent=1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="base")
    ap.add_argument("--forcing", default="P_daymet",
                    choices=["P_daymet", "P_maurer", "P_imerg"])
    ap.add_argument("--basins", default="531", choices=["531", "all"])
    ap.add_argument("--loss", default="nse", choices=["nse", "kge"])
    ap.add_argument("--transform", default=None, choices=[None, "sqrt", "log"])
    ap.add_argument("--iterations", type=int, default=2048)
    ap.add_argument("--phases", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-jitter", type=float, default=0.0)
    ap.add_argument("--relax-bounds", action="store_true")
    ap.add_argument("--tension", default="", choices=["", "ptens", "plp", "both"],
                    help="add the optional Sacramento-style tension parameters")
    ap.add_argument("--var-floor", type=float, default=0.10)
    ap.add_argument("--checkpoint-every", type=int, default=96)
    ap.add_argument("--uh-kmin-h", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--compile", action="store_true", default=True)
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.add_argument("--ensemble", default=None,
                    help="comma-separated tags to score as an ensemble mean")
    args = ap.parse_args()

    os.makedirs(f"{HERE}/results", exist_ok=True)
    os.makedirs(f"{HERE}/params", exist_ok=True)
    cfg = CrestConfig(uh_kmin_h=args.uh_kmin_h,
                      checkpoint_every=args.checkpoint_every,
                      compile_step=args.compile)
    data = load(args)
    print(f"[{args.tag}] {len(data['ids'])} basins | train {TRAIN[0]}..{TRAIN[1]} "
          f"({data['n_train']} d) test {TEST[0]}..{TEST[1]} ({data['n_test']} d) "
          f"| {args.forcing} loss={args.loss}{'/' + args.transform if args.transform else ''} "
          f"seed={args.seed} bounds={'relaxed' if args.relax_bounds else 'default'} "
          f"| {args.device}", flush=True)

    if args.ensemble:
        run_ensemble(args, data, cfg)
        return

    bounds = dict(RELAXED if args.relax_bounds else pycrest.BOUNDS)
    if args.tension:
        keys = ["ptens", "plp"] if args.tension == "both" else [args.tension]
        bounds.update({k: TENSION[k] for k in keys})
    P, TA, PET, obs = data["train"]
    loss = (nse_loss(data["mask_train"], var_floor=args.var_floor,
                     transform=args.transform) if args.loss == "nse"
            else kge_loss(data["mask_train"], transform=args.transform))

    def on_phase(i, res):
        ev = evaluate(res.params, data, cfg)
        print("  " + _report(ev, f"{args.tag} phase{i}",
                             f" ({res.history[-1]['seconds']:.0f}s)"), flush=True)

    res = calibrate(P, TA, PET, obs, data["area_km2"], loss_fn=loss, config=cfg,
                    schedule=CalibrationSchedule(iterations=args.iterations,
                                                 phases=args.phases),
                    bounds=bounds, seed=args.seed, init_jitter=args.init_jitter,
                    on_phase=on_phase, log_every=64)

    ev = evaluate(res.params, data, cfg)
    summary = summarize(ev)
    summary.update(tag=args.tag, forcing=args.forcing, loss=args.loss,
                   transform=args.transform, seed=args.seed,
                   iterations=args.iterations, relax_bounds=args.relax_bounds,
                   init_jitter=args.init_jitter, seconds=res.seconds,
                   tension=args.tension, n_params=len(bounds),
                   gradient_activity=res.gradient_activity,
                   protocol="newman2017/kratzert2019")
    print(_report(ev, f"{args.tag} FINAL"), flush=True)
    dead = {k: v for k, v in res.gradient_activity.items() if v > 0.5}
    if dead:
        print(f"  parameters with >50% zero gradient at iteration 0: {dead}",
              flush=True)

    pycrest.save_parameters(
        f"{HERE}/params/camels_{args.tag}.npz", res.params, data["ids"],
        config=cfg, bounds=bounds,
        metrics={"test_nse": ev["test"]["nse"], "test_kge": ev["test"]["kge"],
                 "train_nse": ev["train"]["nse"]},
        provenance={"protocol": "Newman2017/Kratzert2019 train1999-2008 test1989-1999",
                    "forcing": args.forcing, "loss": args.loss,
                    "pycrest": pycrest.__version__, "tag": args.tag})
    json.dump(summary, open(f"{HERE}/results/camels_{args.tag}.json", "w"), indent=1)
    print(f"  wrote params/camels_{args.tag}.npz and results/camels_{args.tag}.json",
          flush=True)


if __name__ == "__main__":
    main()
