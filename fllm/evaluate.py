"""Run RAW and observer-corrected forecasts and score exactly 18 points."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from autotimes.models.AutoTimes_Llama import Model
from autotimes.utils.tools import convert_tsf_to_dataframe
from fllm.common import load_config, model_args


def target_windows(path):
    frame, _, _, _, _ = convert_tsf_to_dataframe(str(path))
    if len(frame) != 48000:
        raise ValueError(f"expected 48000 Monthly series, got {len(frame)}")
    histories = []
    masks = []
    targets = []
    keys = []
    for index, row in frame.iterrows():
        values = np.asarray(row.series_value, dtype=np.float32)
        values = values[~np.isnan(values)]
        if len(values) < 18 or not np.isfinite(values).all():
            raise ValueError(f"invalid target series at index {index}")
        origin = len(values) - 18
        history = values[max(0, origin - 48):origin]
        x = np.zeros(48, dtype=np.float32)
        mask = np.zeros(48, dtype=np.float32)
        x[-len(history):] = history
        mask[-len(history):] = 1
        histories.append(x)
        masks.append(mask)
        targets.append(values[origin:origin + 18])
        keys.append({"series_id": str(row.series_name), "series_index": int(index),
                     "forecast_origin": int(origin), "history_valid": int(len(history))})
    if len({row["series_id"] for row in keys}) != len(keys):
        raise ValueError("duplicate Monthly series id")
    return (np.stack(histories), np.stack(masks), np.stack(targets), keys)


def correction(observer, context, history_mask, raw):
    valid = history_mask.sum(1, keepdim=True)
    mean = (context * history_mask).sum(1, keepdim=True) / valid.clamp_min(1)
    variance = ((context - mean).square() * history_mask).sum(1, keepdim=True) / valid.clamp_min(1)
    sigma = (variance + 1e-5).sqrt()
    residual = (context[:, 24:] - raw[:, :24]) * history_mask[:, 24:] / sigma
    usable = (history_mask[:, 24:].sum(1, keepdim=True) > 0).to(dtype=torch.float32)
    return sigma * observer(residual) * usable


@torch.no_grad()
def predict_batch(model, observer, history, history_mask, amp):
    state = history.clone()
    mask = history_mask.clone()
    raw_parts = []
    corrected_parts = []
    for _ in range(2):
        with torch.cuda.amp.autocast(enabled=amp):
            raw_all = model(state.unsqueeze(-1), None, None, None).squeeze(-1).float()
        raw = raw_all[:, -24:]
        corr = correction(observer, state, mask, raw_all)
        corrected = raw + corr
        if not torch.isfinite(raw).all() or not torch.isfinite(corrected).all():
            raise FloatingPointError("non-finite Monthly prediction")
        raw_parts.append(raw)
        corrected_parts.append(corrected)
        state = torch.cat([state[:, 24:], corrected], dim=1)
        mask = torch.cat([mask[:, 24:], torch.ones_like(corrected)], dim=1)
    return torch.cat(raw_parts, dim=1)[:, :18], torch.cat(corrected_parts, dim=1)[:, :18]


def metric(prediction, target):
    delta = prediction.astype(np.float64) - target.astype(np.float64)
    squared = np.square(delta)
    absolute = np.abs(delta)
    point_smape = 200.0 * absolute / (np.abs(prediction.astype(np.float64))
                                      + np.abs(target.astype(np.float64)) + 1e-8)
    return {
        "sse": float(squared.sum()),
        "sae": float(absolute.sum()),
        "smape": float(point_smape.mean(axis=1).mean()),
        "mse": float(squared.mean()),
        "mae": float(absolute.mean()),
    }


def evaluate(config_path, plant_checkpoint, observer_checkpoint, output_dir):
    config = load_config(config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    args = model_args(config, output)
    model = Model(SimpleNamespace(**vars(args))).to("cuda:0")
    plant_state = torch.load(plant_checkpoint, map_location="cpu")
    expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(plant_state) != expected:
        raise ValueError("plant checkpoint trainable keys do not match model")
    result = model.load_state_dict(plant_state, strict=False)
    if result.unexpected_keys or any(not name.startswith("llama.") for name in result.missing_keys):
        raise ValueError("plant checkpoint load mismatch")
    model.eval()
    model.requires_grad_(False)
    observer = nn.Sequential(nn.Linear(24, 192), nn.Tanh(), nn.Dropout(0.1), nn.Linear(192, 24)).cuda()
    observer_state = torch.load(observer_checkpoint, map_location="cpu")["model"]
    observer.load_state_dict(observer_state, strict=True)
    observer.eval()
    observer.requires_grad_(False)
    history, history_mask, target, keys = target_windows(config["target_tsf"])
    raw_prediction = np.empty((len(target), 18), dtype=np.float32)
    corrected_prediction = np.empty_like(raw_prediction)
    for begin in range(0, len(target), 16):
        end = min(begin + 16, len(target))
        raw, corrected = predict_batch(
            model,
            observer,
            torch.from_numpy(history[begin:end]).cuda(),
            torch.from_numpy(history_mask[begin:end]).cuda(),
            config["plant"]["use_amp"],
        )
        raw_prediction[begin:end] = raw.cpu().numpy()
        corrected_prediction[begin:end] = corrected.cpu().numpy()
    np.save(output / "raw_prediction.npy", raw_prediction, allow_pickle=False)
    np.save(output / "corrected_prediction.npy", corrected_prediction, allow_pickle=False)
    (output / "series_keys.json").write_text(json.dumps(keys) + "\n")
    raw_metrics = metric(raw_prediction, target)
    corrected_metrics = metric(corrected_prediction, target)
    result = {
        "status": "COMPLETE",
        "source": "M3 Monthly",
        "target": "M4 Monthly",
        "scored_points": 18,
        "generated_points_per_step": 24,
        "feedback_factor": 1.0,
        "series_count": len(target),
        "valid_points": int(target.size),
        "raw": raw_metrics,
        "observer_corrected": corrected_metrics,
        "plant_checkpoint": str(Path(plant_checkpoint).resolve()),
        "observer_checkpoint": str(Path(observer_checkpoint).resolve()),
        "target_tsf": str(Path(config["target_tsf"]).resolve()),
        "smape": "equal mean of per-series point means; denominator abs(pred)+abs(target)+1e-8",
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result

