#!/usr/bin/env python3
"""One command that tells you whether this checkout is working.

    python scripts/selfcheck.py            # ~30 s, no GPU needed
    python scripts/selfcheck.py --full     # also runs pytest and verify.py

Written for agents: every check prints PASS or FAIL with the number it checked,
and the exit code is non-zero if anything failed. Nothing here needs a download.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "examples"))

results: list[tuple[str, bool, str]] = []


def check(name):
    def deco(fn):
        try:
            ok, note = fn()
        except Exception as exc:  # noqa: BLE001
            ok, note = False, f"{type(exc).__name__}: {exc}"
        results.append((name, ok, note))
        print(f"{'PASS' if ok else 'FAIL'}  {name:38s}  {note}")
        return fn
    return deco


@check("import and version")
def _import():
    import dcrest
    return True, f"dcrest {dcrest.__version__}, torch {__import__('torch').__version__}"


@check("forward run, synthetic")
def _forward():
    import torch

    import dcrest
    p = dcrest.default_parameters(4, area_km2=[100.0, 500.0, 1000.0, 5000.0])
    P = torch.rand(400, 4) * 8
    Q = dcrest.simulate(P, torch.full((400, 4), 10.0), torch.full((400, 4), 3.0), p)
    ok = Q.shape == (400, 4) and torch.isfinite(Q).all() and (Q >= 0).all()
    return ok, f"Q {tuple(Q.shape)}, mean {Q.mean():.3f} mm/day"


@check("gradient flows to the forcing")
def _grad():
    import torch

    import dcrest
    p = dcrest.default_parameters(2, area_km2=[300.0, 900.0])
    P = (torch.rand(300, 2) * 6).requires_grad_(True)
    dcrest.simulate(P, torch.full((300, 2), 12.0), torch.full((300, 2), 3.0), p).sum().backward()
    n = int((P.grad.abs() > 0).sum())
    return n > 0 and torch.isfinite(P.grad).all(), f"{n} of {P.grad.numel()} days carry dJ/dP"


@check("every parameter receives gradient")
def _activity():
    # On real forcing, not synthetic: snow parameters need temperatures that
    # actually cross freezing, and a capacity bound only shows as inert where
    # the flux reaches it. A constant-temperature toy would report false deaths.
    import dcrest
    from _data import load
    data, _ = load("P_daymet")
    n = len(data["ids"])
    m = dcrest.CrestLumped(n_units=n, area_km2=data["area_km2"])
    sl = slice(0, 1500)
    q = m(data["P"][sl], data["T"][sl], data["E"][sl])
    (1.0 - dcrest.metrics.nse(q, data["obs"][sl],
                              mask=torch.isfinite(data["obs"][sl]))).mean().backward()
    act = m.gradient_activity()
    dead = [k for k, v in act.items() if v > 0.5]
    return not dead, ("all 16 active over %d catchments" % n) if not dead \
        else f"DEAD: {dead}"


@check("sample data present")
def _sample():
    from _data import SAMPLE, load
    data, dates = load()
    return os.path.exists(SAMPLE), (f"{len(data['ids'])} catchments, "
                                    f"{len(dates)} days, "
                                    f"{os.path.getsize(SAMPLE)/1e6:.1f} MB")


@check("released set reproduces its stored skill")
def _released():
    import torch

    import dcrest
    from _data import TEST, load, run_window, scoring_mask
    data, dates = load("P_daymet")
    path = os.path.join(ROOT, "camels", "dCREST_CAMELS_v1.0_daymet665.npz")
    params, meta = dcrest.load_parameters(path, subset=data["ids"])
    cfg = dcrest.config_from_meta(meta)
    w = run_window(dates, *TEST)
    Q = dcrest.simulate(data["P"][w], data["T"][w], data["E"][w], params, cfg)
    nse = dcrest.metrics.nse(Q, data["obs"][w],
                             mask=scoring_mask(dates, data["obs"], *TEST)[w])
    err = float((nse - torch.as_tensor(meta["metric_test_nse"])).abs().max())
    return err < 5e-3, f"median NSE {nse.median():.3f}, max error vs stored {err:.1e}"


@check("all four parameter sets load")
def _sets():
    import dcrest
    names = []
    for f in sorted(os.listdir(os.path.join(ROOT, "camels"))):
        if f.endswith(".npz"):
            p, m = dcrest.load_parameters(os.path.join(ROOT, "camels", f))
            names.append(f"{f.split('_v')[-1][:12]}:{len(m['ids'])}")
    return len(names) == 4, ", ".join(names)


@check("parameter docs match the code")
def _docs():
    r = subprocess.run([sys.executable,
                        os.path.join(ROOT, "scripts", "gen_parameter_table.py"),
                        "--check"], capture_output=True, text=True)
    return r.returncode == 0, r.stdout.strip() or r.stderr.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="also run pytest and scripts/verify.py")
    args = ap.parse_args()

    if args.full:
        for cmd, label in (([sys.executable, "-m", "pytest", "-q"], "pytest"),
                           ([sys.executable, os.path.join(ROOT, "scripts", "verify.py")],
                            "verify.py")):
            print(f"\n--- {label} ---")
            r = subprocess.run(cmd, cwd=ROOT)
            results.append((label, r.returncode == 0, f"exit {r.returncode}"))

    bad = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(bad)}/{len(results)} checks passed")
    if bad:
        print("FAILED:", ", ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
