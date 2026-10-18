"""Step 1: download the CRSP data (once) and build the panel.

Usage:
    python scripts/01_build_data.py                # download missing files, then build
    python scripts/01_build_data.py --no-download  # build from the local files only
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data import build_panel, check_panel, download_crsp

parser = argparse.ArgumentParser()
parser.add_argument("--no-download", action="store_true")
parser.add_argument("--force", action="store_true", help="download all files again")
args = parser.parse_args()

cfg = load_config()
if not args.no_download:
    download_crsp(cfg, force=args.force)

panel = build_panel(cfg)
out = cfg["paths"]["processed"] / "panel.parquet"
out.parent.mkdir(parents=True, exist_ok=True)
panel.to_parquet(out)

coverage = check_panel(panel)
cfg["paths"]["results"].mkdir(exist_ok=True)
coverage.to_csv(cfg["paths"]["results"] / "data_coverage.csv", float_format="%.4f")
print("\nCoverage by year:")
print(coverage.round(3).to_string())
