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

# The download URL is filled in when the archive is published (Zenodo record
# in the paper's data-availability statement); until then pass --url or set
# DCREST_FORCING_URL. The checksum is that of the released file (100,208,652
# bytes) and is verified after every download.
URL = os.environ.get("DCREST_FORCING_URL", "")
SHA256 = "eda60dcef1796f8a23523afb1906fdb19173a8add209826ea07928740dd04891"

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(HERE, "camels", "forcing_daily.npz")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--url", default=URL, help="download URL (default: built in)")
    ap.add_argument("--force", action="store_true",
                    help="download even if --out already exists with the right checksum")
    args = ap.parse_args()
    url = args.url

    if os.path.exists(args.out) and not args.force and _sha256(args.out) == SHA256:
        print(f"{args.out} already present, checksum OK")
        return 0

    if not url:
        print(__doc__)
        print("No download URL is configured in this checkout.\n"
              "Set DCREST_FORCING_URL, or obtain the file from the archive "
              "listed in the paper's data-availability statement.\n"
              "The 8-catchment sample in examples/data/sample_camels.npz needs "
              "no download and is enough to run every example.")
        return 1

    import urllib.request
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    print(f"downloading {url}\n  -> {args.out}")
    urllib.request.urlretrieve(url, args.out)

    got = _sha256(args.out)
    if got != SHA256:
        print(f"CHECKSUM MISMATCH\n  expected {SHA256}\n  got      {got}")
        return 2
    print("checksum OK")

    size_mb = os.path.getsize(args.out) / 1e6
    print(f"done, {size_mb:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
