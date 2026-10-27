"""Build the source residual cache for the fixed observer."""

import csv
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from autotimes.models.AutoTimes_Llama import Model
from fllm.common import load_config, model_args
from fllm.source_rows import SourceRows


def build(config_path, plant_checkpoint, output_dir):
    config = load_config(config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    args = model_args(config, output)
    model = Model(SimpleNamespace(**vars(args))).to("cuda:0")
    state = torch.load(plant_checkpoint, map_location="cpu")
    expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(state) != expected:
        raise ValueError("plant checkpoint trainable keys do not match model")
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys or any(not name.startswith("llama.") for name in result.missing_keys):
        raise ValueError("plant checkpoint contains an unexpected or missing trainable key")
    model.eval()
    model.requires_grad_(False)
    dataset = SourceRows(config)
    shapes = {"residual": [len(dataset), 24], "target": [len(dataset), 24],
              "future_mask": [len(dataset), 24], "residual_mask": [len(dataset), 24]}
    arrays = {name: np.lib.format.open_memmap(output / (name + ".partial.npy"), mode="w+",
                                               dtype=np.float32, shape=tuple(shape))
              for name, shape in shapes.items()}
    start = time.time()
    count = 0
    with (output / "sample_keys.partial.tsv").open("w") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["series_id", "series_index", "cut_point"])
        for batch in DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0, drop_last=False):
            x = batch["history"].cuda()
            mask = batch["history_mask"].cuda()
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=config["plant"]["use_amp"]):
                raw = model(x.unsqueeze(-1), None, None, None).squeeze(-1).float()
            valid = mask.sum(1, keepdim=True)
            mean = (x * mask).sum(1, keepdim=True) / valid.clamp_min(1)
            variance = ((x - mean).square() * mask).sum(1, keepdim=True) / valid.clamp_min(1)
            sigma = (variance + 1e-5).sqrt()
            residual_mask = mask[:, 24:]
            residual = (x[:, 24:] - raw[:, :24]) * residual_mask / sigma
            target = (batch["future"].cuda() - raw[:, -24:]) / sigma
            future_mask = batch["future_mask"].cuda()
            target = target * future_mask
            end = count + len(x)
            values = {"residual": residual, "target": target,
                      "future_mask": future_mask, "residual_mask": residual_mask}
            for name, value in values.items():
                if not torch.isfinite(value).all():
                    raise FloatingPointError("non-finite cache value: " + name)
                arrays[name][count:end] = value.cpu().numpy()
            for i in range(len(x)):
                writer.writerow([batch["series_id"][i], int(batch["series_index"][i]),
                                 int(batch["cut_point"][i])])
            count = end
    for name, array in arrays.items():
        array.flush()
        del array
        (output / (name + ".partial.npy")).replace(output / (name + ".npy"))
    (output / "sample_keys.partial.tsv").replace(output / "sample_keys.tsv")
    (output / "cache_manifest.json").write_text(json.dumps({
        "status": "COMPLETE", "plant_checkpoint": str(Path(plant_checkpoint).resolve()),
        "samples": count, "shapes": shapes, "dtype": "float32", "patch": 24,
        "context": 48, "source_only": True, "wall_seconds": time.time() - start
    }, indent=2) + "\n")

