#!/usr/bin/env python3
"""Run the fixed Monthly source training, observer, and target evaluation."""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fllm.cache import build
from fllm.common import load_config, ensure_output
from fllm.evaluate import evaluate
from fllm.train_observer import train as train_observer
from fllm.train_plant import train as train_plant


def main():
    parser = argparse.ArgumentParser(description="Fixed Monthly source-to-target H18 run")
    parser.add_argument("--config", default="config/monthly_m3_to_m4_h18.json")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--stage", choices=["all", "plant", "observer", "test"], default="all")
    args = parser.parse_args()
    config = load_config(args.config)
    output = ensure_output(config, args.output_dir)
    plant_dir = output / "plant"
    cache_dir = output / "source_cache"
    observer_dir = output / "observer"
    test_dir = output / "test"
    plant_checkpoint = plant_dir / "plant_checkpoints" / "monthly_source_plant" / "checkpoint.pth"
    observer_checkpoint = observer_dir / "observer_epoch05.pt"
    if args.stage in ("all", "plant"):
        train_plant(args.config, plant_dir)
    if args.stage in ("all", "observer"):
        if not plant_checkpoint.is_file():
            raise FileNotFoundError(plant_checkpoint)
        build(args.config, plant_checkpoint, cache_dir)
        train_observer(cache_dir, observer_dir, observer_seed=2027)
    if args.stage in ("all", "test"):
        if not plant_checkpoint.is_file() or not observer_checkpoint.is_file():
            raise FileNotFoundError("run plant and observer stages first")
        result = evaluate(args.config, plant_checkpoint, observer_checkpoint, test_dir)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

