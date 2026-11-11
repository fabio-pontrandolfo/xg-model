"""Build data/processed/shots.csv.gz from StatsBomb open data.

Usage:
    python scripts/build_dataset.py                      # read from GitHub (cached in data/raw)
    python scripts/build_dataset.py --source C:/path/to/open-data/data   # local clone
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xg.features import add_features            # noqa: E402
from xg.load import COMPETITIONS, GITHUB_ROOT, load_competition   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=GITHUB_ROOT)
    args = ap.parse_args()

    cache = None if not args.source.startswith("http") else ROOT / "data" / "raw"
    rows = []
    for name in COMPETITIONS:
        comp_rows = load_competition(name, args.source, cache)
        print(f"{name:<25} {len(comp_rows):>6} shots")
        rows += comp_rows

    df = pd.DataFrame(rows)
    n_all = len(df)
    df = df[df.shot_type != "Penalty"]      # near-constant conversion; modelled separately in practice
    print(f"dropped {n_all - len(df)} penalties")

    df = add_features(df)
    out = ROOT / "data" / "processed" / "shots.csv.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} shots -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
