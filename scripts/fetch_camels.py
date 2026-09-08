#!/usr/bin/env python3
"""Fetch the full 665-catchment CAMELS forcing used to build the released sets.

The repository ships only an 8-catchment sample (`examples/data/`) so that the
examples run offline. The full daily file is ~96 MB and lives outside git.

    python scripts/fetch_camels.py                 # to camels/forcing_daily.npz
    python scripts/fetch_camels.py --out /tmp/f.npz

Contents: daily basin means for 665 CAMELS-US catchments, 1980-01-01 to
2014-12-31 (12784 days).

    P_daymet, P_maurer, P_imerg   precipitation, mm/day     (T, N) float32
    tmean                         air temperature, degC     (T, N)
    pet                           potential ET (Oudin), mm/day
    obs                           USGS discharge, mm/day, NaN where missing
    ids, lat, lon, elev_m, area_km2, dates

If you already have CAMELS locally, you do not need this file: the arrays are
plain basin-mean aggregates of the public CAMELS forcing (Newman et al. 2015),
IMERG V07 (NASA GES DISC) and USGS NWIS discharge. It is redistributed only so
the calibration is reproducible in one step.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

# Set when the archive is published; until then the script tells you what to do.
URL = os.environ.get("DCREST_FORCING_URL", "")
SHA256 = ""

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(HERE, "camels", "forcing_daily.npz")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    if not URL:
        print(__doc__)
        print("No download URL is configured in this checkout.\n"
              "Set DCREST_FORCING_URL, or obtain the file from the archive "
              "listed in the paper's data-availability statement.\n"
              "The 8-catchment sample in examples/data/sample_camels.npz needs "
              "no download and is enough to run every example.")
        return 1

    import urllib.request
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    print(f"downloading {URL}\n  -> {args.out}")
    urllib.request.urlretrieve(URL, args.out)

    if SHA256:
        h = hashlib.sha256()
        with open(args.out, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != SHA256:
            print(f"CHECKSUM MISMATCH\n  expected {SHA256}\n  got      {h.hexdigest()}")
            return 2
        print("checksum OK")

    size_mb = os.path.getsize(args.out) / 1e6
    print(f"done, {size_mb:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
