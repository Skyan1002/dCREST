#!/usr/bin/env python3
"""Parity check: dcrest's CRESTPhys kernel against EF5's own C++.

Extracts ``CRESTPHYSModel::WaterBalanceInt`` verbatim from the EF5 development
fork's ``src/CRESTPhysModel.cpp`` (by brace matching, never edited), compiles it
into ``scripts/ef5_ref/crestphys_ref.cpp``, runs 200 random units for 3 years
at the daily step, and compares :func:`dcrest.kernels.soil.crestphys_step`
step by step under both runoff partitions.

Expected: ``partition="ef5"`` agrees to float32 round-off (EF5 keeps its states
in single precision); ``partition="classic"`` does not -- it is classic CREST's
split, which releases up to 1.0.0 used.

Usage:
    python scripts/verify_ef5_partition.py --ef5-src /path/to/EF5/src/CRESTPhysModel.cpp
    (or set EF5_CRESTPHYS=/path/to/CRESTPhysModel.cpp)
Exit code 0 when the ef5 split matches and the classic split does not.
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "ef5_ref")
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from dcrest.kernels.soil import crestphys_step                       # noqa: E402


def extract_body(cpp_path):
    """The whole WaterBalanceInt definition, by brace matching."""
    text = open(cpp_path).read()
    start = text.index("void CRESTPHYSModel::WaterBalanceInt")
    depth, i = 0, text.index("{", start)
    while True:
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        i += 1
        if depth == 0:
            return text[start:i]


def build(cpp_path):
    open(os.path.join(REF, "body.inc"), "w").write(extract_body(cpp_path) + "\n")
    subprocess.run(["g++", "-O2", "-o", "crestphys_ref", "crestphys_ref.cpp"], cwd=REF, check=True)


def case(seed=7, n=200, t=3 * 365):
    rng = np.random.default_rng(seed)
    pr = {"wm": rng.uniform(50, 600, n), "b": rng.uniform(0.5, 6.0, n), "im": rng.uniform(0, 0.1, n),
          "ke": rng.uniform(0.6, 1.4, n), "fc": np.exp(rng.uniform(np.log(0.05), np.log(5.0), n)),
          "ksoil": np.exp(rng.uniform(np.log(0.005), np.log(0.3), n)),
          "hmaxaq": rng.uniform(100, 3000, n), "gwc": np.exp(rng.uniform(np.log(0.02), np.log(3.0), n)),
          "gwe": rng.uniform(0.5, 6.0, n)}
    pr = {k: v.astype(np.float32) for k, v in pr.items()}
    sm0 = (0.5 * pr["wm"]).astype(np.float32); gw0 = (0.3 * pr["hmaxaq"]).astype(np.float32)
    doy = np.arange(t) % 365
    wet = rng.random((t, n)) < 0.35
    P = (np.where(wet, rng.gamma(0.7, 12.0, (t, n)), 0.0) / 24.0).astype(np.float32)        # mm/h
    PET = ((2.5 + 2.0 * np.sin(2 * np.pi * (doy - 100) / 365))[:, None]
           * np.ones((1, n)) / 24.0).astype(np.float32)
    return pr, sm0, gw0, P, PET


def run_cpp(pr, sm0, gw0, P, PET):
    t, n = P.shape
    order = ["wm", "b", "im", "ke", "fc", "ksoil", "iwu", "igw", "hmaxaq", "gwc", "gwe"]
    cf, of = os.path.join(REF, "case.txt"), os.path.join(REF, "out.txt")
    with open(cf, "w") as f:
        f.write(f"{n} {t} 24.0\n")
        for i in range(n):
            row = [pr.get(k, np.zeros(n, np.float32))[i] for k in order] + [sm0[i], gw0[i]]
            f.write(" ".join(f"{x:.9e}" for x in row) + "\n")
        for k in range(t):
            f.write("".join(f"{P[k, i]:.9e} {PET[k, i]:.9e}\n" for i in range(n)))
    subprocess.run([os.path.join(REF, "crestphys_ref"), cf, of], check=True)
    out = np.loadtxt(of).reshape(t, n, 5)
    out[..., :3] *= 86400.0                                # mm/s -> mm/day
    return out


def run_dcrest(partition, pr, sm0, gw0, P, PET):
    p = {k: torch.tensor(v, dtype=torch.float64) for k, v in pr.items()}
    sm = torch.tensor(sm0, dtype=torch.float64); gw = torch.tensor(gw0, dtype=torch.float64)
    out = np.zeros(P.shape + (5,))
    for k in range(P.shape[0]):
        o, i_, b_, sm, gw = crestphys_step(p, sm, gw, torch.tensor(P[k] * 24.0, dtype=torch.float64),
                                           torch.tensor(PET[k] * 24.0, dtype=torch.float64), 24.0,
                                           partition=partition)
        out[k] = np.stack([o.numpy(), i_.numpy(), b_.numpy(), sm.numpy(), gw.numpy()], 1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ef5-src", default=os.environ.get("EF5_CRESTPHYS"),
                    help="path to the EF5 development fork's src/CRESTPhysModel.cpp")
    a = ap.parse_args()
    if not a.ef5_src or not os.path.exists(a.ef5_src):
        sys.exit("give --ef5-src /path/to/CRESTPhysModel.cpp (EF5 development fork)")
    build(a.ef5_src)
    args = case()
    ref = run_cpp(*args)
    shares = lambda x: np.round(x[..., :3].sum((0, 1)) / x[..., :3].sum(), 3)
    ok = {}
    for part in ("ef5", "classic"):
        out = run_dcrest(part, *args)
        dflux = np.abs(ref[..., :3] - out[..., :3]).max((0, 1))
        dstore = np.abs(ref[..., 3:] - out[..., 3:]).max((0, 1))
        # float32 state storage in EF5: allow 1e-2 mm/day on fluxes, and on
        # stores 1e-2 mm or 1e-5 of the largest store (aquifers reach ~3000 mm)
        tol_store = max(1e-2, 1e-5 * np.abs(ref[..., 3:]).max())
        ok[part] = bool(dflux.max() < 1e-2 and dstore.max() < tol_store)
        print(f"partition={part:7s} max|d| overland/interflow/baseflow "
              f"{dflux[0]:.1e}/{dflux[1]:.1e}/{dflux[2]:.1e} mm/day, SM/GW {dstore[0]:.1e}/{dstore[1]:.1e} mm; "
              f"shares {shares(out)} (EF5 C++ {shares(ref)}) -> "
              f"{'matches EF5' if ok[part] else 'differs from EF5'}")
    sys.exit(0 if (ok["ef5"] and not ok["classic"]) else 1)
