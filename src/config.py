"""Load the project configuration from config.yaml."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Read config.yaml and turn all entries under `paths` into absolute Paths."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {key: ROOT / value for key, value in cfg["paths"].items()}
    return cfg
