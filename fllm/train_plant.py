"""Train the fixed AutoTimes plant on source Monthly rows."""

import json
import os
from pathlib import Path

from fllm.common import load_config, model_args, seed_everything


def train(config_path, output_dir):
    import torch
    from autotimes.exp.exp_zero_shot_forecasting import Exp_Zero_Shot_Forecast

    config = load_config(config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    os.environ["RUN_RECORD_DIR"] = str(output)
    os.environ["RUN_SEED"] = str(config["plant_seed"])
    args = model_args(config, output)
    seed_everything(config["plant_seed"])
    experiment = Exp_Zero_Shot_Forecast(args)
    experiment.train("monthly_source_plant")
    checkpoint = Path(args.checkpoints) / "monthly_source_plant" / "checkpoint.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    state = torch.load(checkpoint, map_location="cpu")
    manifest = {
        "status": "COMPLETE",
        "kind": "plant",
        "checkpoint": str(checkpoint.resolve()),
        "plant_seed": config["plant_seed"],
        "config": config,
        "trainable_keys": sorted(state),
    }
    (output / "plant_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return checkpoint

