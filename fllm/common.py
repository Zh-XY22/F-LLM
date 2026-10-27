"""Shared fixed configuration and path handling."""

import json
import os
from pathlib import Path
from types import SimpleNamespace


def _resolve(value):
    if not isinstance(value, str):
        return value
    if value.startswith("${") and value.endswith("}"):
        name = value[2:-1]
        resolved = os.environ.get(name)
        if not resolved:
            raise RuntimeError(f"Set environment variable {name}")
        return resolved
    return value


def load_config(path):
    raw = json.loads(Path(path).read_text())
    raw["source_tsf"] = _resolve(raw["source_tsf"])
    raw["target_tsf"] = _resolve(raw["target_tsf"])
    raw["llama_checkpoint"] = _resolve(raw["llama_checkpoint"])
    raw["output_dir"] = str(Path(_resolve(raw["output_dir"])).expanduser())
    return raw


def model_args(config, output_dir):
    args = dict(config["plant"])
    args.update(
        task_name="zero_shot_forecast",
        is_training=1,
        model_id="m3_Monthly",
        model="AutoTimes_Llama",
        data="tsf",
        root_path=str(Path(config["source_tsf"]).parent),
        data_path=Path(config["source_tsf"]).name,
        test_data_path=Path(config["target_tsf"]).name,
        checkpoints=str(Path(output_dir) / "plant_checkpoints"),
        gpu=0,
        use_multi_gpu=False,
        local_rank=0,
        test_dir=".",
        test_file_name="checkpoint.pth",
        frequency_map=12,
        llm_ckp_dir=config["llama_checkpoint"],
    )
    return SimpleNamespace(**args)


def seed_everything(seed):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_output(config, output_dir=None):
    root = Path(output_dir or config["output_dir"]).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root

