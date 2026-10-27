#!/usr/bin/env python3
"""Minimal AutoTimes-based F-LLM Monthly M3→M4 H18 runner."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import LlamaForCausalLM


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_tsf(path: Path) -> List[np.ndarray]:
    """Read the standard AutoTimes TSF format without optional dependencies."""
    attributes: List[str] = []
    types: List[str] = []
    rows: List[np.ndarray] = []
    in_data = False
    for raw in path.read_text(encoding="cp1252").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower() == "@data":
            in_data = True
            continue
        if line.startswith("@"):
            fields = line.split()
            if fields[0].lower() == "@attribute":
                if len(fields) != 3:
                    raise ValueError(f"invalid attribute line: {line}")
                attributes.append(fields[1])
                types.append(fields[2].lower())
            continue
        if not in_data or not attributes:
            raise ValueError(f"data row appears before @data: {line[:80]}")
        fields = line.split(":")
        if len(fields) != len(attributes) + 1:
            raise ValueError(f"invalid TSF row with {len(fields)} fields: {line[:80]}")
        values = []
        for value in fields[-1].split(","):
            values.append(np.nan if value == "?" else float(value))
        rows.append(np.asarray(values, dtype=np.float32))
    if not rows:
        raise ValueError(f"no series found in {path}")
    return [x[np.isfinite(x)] for x in rows]


class SourceWindows(Dataset):
    """Deterministic source windows for plant and observer training."""

    def __init__(self, series: Sequence[np.ndarray], context: int, horizon: int,
                 windows_per_series: int = 8):
        self.items = []
        for series_id, values in enumerate(series):
            if len(values) < context + horizon:
                continue
            max_start = len(values) - context - horizon
            starts = np.linspace(0, max_start, windows_per_series, dtype=int)
            for start in np.unique(starts):
                history = values[start:start + context]
                future = values[start + context:start + context + horizon]
                self.items.append((series_id, history.astype(np.float32), future.astype(np.float32)))
        if not self.items:
            raise ValueError("M3 file has no usable 48+24 windows")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        _, history, future = self.items[index]
        return torch.from_numpy(history[:, None]), torch.from_numpy(future[:, None])


class TargetSeries(Dataset):
    """Use the last 18 values of every M4 series as the benchmark target."""

    def __init__(self, series: Sequence[np.ndarray], context: int, horizon: int):
        self.items = []
        for series_id, values in enumerate(series):
            if len(values) < context + horizon:
                continue
            self.items.append((series_id, values[:-horizon], values[-horizon:]))
        if not self.items:
            raise ValueError("M4 file has no usable target series")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        series_id, history, target = self.items[index]
        history = history[-48:].astype(np.float32)
        target = target.astype(np.float32)
        return series_id, torch.from_numpy(history[:, None]), torch.from_numpy(target[:, None])


class ProjectionMLP(nn.Module):
    def __init__(self, f_in: int, f_out: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        blocks: List[nn.Module] = [nn.Linear(f_in, hidden), nn.Tanh(), nn.Dropout(dropout)]
        for _ in range(max(0, layers - 2)):
            blocks += [nn.Linear(hidden, hidden), nn.Tanh(), nn.Dropout(dropout)]
        blocks.append(nn.Linear(hidden, f_out))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AutoTimesPlant(nn.Module):
    """AutoTimes numeric projections around a frozen LLaMA decoder."""

    def __init__(self, llm_ckp_dir: str, patch: int, hidden: int,
                 layers: int, dropout: float, device: torch.device):
        super().__init__()
        self.patch = patch
        self.device_name = device
        self.llama = LlamaForCausalLM.from_pretrained(
            llm_ckp_dir,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        )
        self.llama.to(device)
        for parameter in self.llama.parameters():
            parameter.requires_grad_(False)
        width = int(self.llama.config.hidden_size)
        self.encoder = ProjectionMLP(patch, width, hidden, layers, dropout).to(device)
        self.decoder = ProjectionMLP(width, patch, hidden, layers, dropout).to(device)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        history = history.to(self.device_name).float()
        mean = history.mean(dim=1, keepdim=True).detach()
        centered = history - mean
        scale = torch.sqrt(centered.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
        centered = centered / scale
        batch = centered.shape[0]
        tokens = centered.transpose(1, 2).reshape(batch, -1)
        tokens = tokens.unfold(-1, self.patch, self.patch)
        embeds = self.encoder(tokens)
        llama_dtype = next(self.llama.parameters()).dtype
        hidden = self.llama.model(inputs_embeds=embeds.to(llama_dtype)).last_hidden_state
        output = self.decoder(hidden.float()).reshape(batch, 1, -1).transpose(1, 2)
        return output * scale + mean


class Observer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(24, 192), nn.Tanh(), nn.Dropout(0.1), nn.Linear(192, 24))

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        return self.net(residual.reshape(residual.shape[0], -1)).reshape(-1, 24, 1)


@dataclass
class TrainConfig:
    context: int = 48
    patch: int = 24
    forecast: int = 24
    score_horizon: int = 18
    hidden: int = 512
    layers: int = 2
    dropout: float = 0.1
    plant_epochs: int = 10
    observer_epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 1e-5
    observer_learning_rate: float = 1e-3
    feedback: float = 1.0


def plant_output(plant: AutoTimesPlant, history: torch.Tensor) -> torch.Tensor:
    output = plant(history)
    if output.shape[1] < 24:
        raise RuntimeError("AutoTimes output is shorter than one forecast patch")
    return output[:, -24:]


def train_plant(plant: AutoTimesPlant, loader: DataLoader, cfg: TrainConfig, device: torch.device) -> None:
    optimizer = torch.optim.Adam(list(plant.encoder.parameters()) + list(plant.decoder.parameters()), lr=cfg.learning_rate)
    plant.train()
    for epoch in range(cfg.plant_epochs):
        losses = []
        for history, future in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = plant_output(plant, history)
            loss = torch.mean((prediction - future.to(device)) ** 2)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"plant epoch {epoch + 1}: mse={np.mean(losses):.6f}", flush=True)


def train_observer(plant: AutoTimesPlant, observer: Observer, loader: DataLoader,
                   cfg: TrainConfig, device: torch.device) -> None:
    optimizer = torch.optim.Adam(observer.parameters(), lr=cfg.observer_learning_rate)
    plant.eval()
    observer.train()
    for parameter in plant.parameters():
        parameter.requires_grad_(False)
    for epoch in range(cfg.observer_epochs):
        losses = []
        for history, future in loader:
            history = history.to(device)
            future = future.to(device)
            with torch.no_grad():
                raw_all = plant(history)
            residual = history[:, 24:] - raw_all[:, :24]
            correction = observer(residual)
            prediction = raw_all[:, -24:] + correction
            loss = torch.mean((prediction - future) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"observer epoch {epoch + 1}: mse={np.mean(losses):.6f}", flush=True)


def smape(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(200.0 * np.abs(prediction - target) /
                         (np.abs(prediction) + np.abs(target) + 1e-8)))


def evaluate(plant: AutoTimesPlant, observer: Observer, loader: DataLoader,
             cfg: TrainConfig, device: torch.device):
    plant.eval()
    observer.eval()
    raw_values, corrected_values, target_values = [], [], []
    with torch.no_grad():
        for _, history, target in loader:
            history = history.to(device)
            raw_all = plant(history)
            raw = raw_all[:, -24:]
            residual = history[:, 24:] - raw_all[:, :24]
            corrected = raw + cfg.feedback * observer(residual)
            raw_values.append(raw[:, :cfg.score_horizon, 0].cpu().numpy())
            corrected_values.append(corrected[:, :cfg.score_horizon, 0].cpu().numpy())
            target_values.append(target[:, :cfg.score_horizon, 0].numpy())
    raw = np.concatenate(raw_values)
    corrected = np.concatenate(corrected_values)
    target = np.concatenate(target_values)
    return {"raw_smape": smape(raw, target), "corrected_smape": smape(corrected, target),
            "series": int(target.shape[0]), "points": int(target.size)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monthly M3→M4 H18 AutoTimes-based F-LLM")
    parser.add_argument("--m3", type=Path, required=True)
    parser.add_argument("--m4", type=Path, required=True)
    parser.add_argument("--llm-ckp-dir", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--plant-epochs", type=int, default=10)
    parser.add_argument("--observer-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--windows-per-series", type=int, default=8)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(plant_epochs=args.plant_epochs, observer_epochs=args.observer_epochs,
                      batch_size=args.batch_size)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.smoke_test:
        observer = Observer().eval()
        test = observer(torch.zeros(2, 24, 1))
        assert tuple(test.shape) == (2, 24, 1)
        print("smoke test passed")
        return

    seed_everything(2022)
    source = read_tsf(args.m3)
    target = read_tsf(args.m4)
    source_data = SourceWindows(source, cfg.context, cfg.forecast, args.windows_per_series)
    target_data = TargetSeries(target, cfg.context, cfg.forecast)
    source_loader = DataLoader(source_data, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    target_loader = DataLoader(target_data, batch_size=cfg.batch_size, shuffle=False)
    plant = AutoTimesPlant(args.llm_ckp_dir, cfg.patch, cfg.hidden, cfg.layers, cfg.dropout, device)
    train_plant(plant, source_loader, cfg, device)
    plant_path = args.output_dir / "plant.pt"
    torch.save({"encoder": plant.encoder.state_dict(), "decoder": plant.decoder.state_dict()}, plant_path)

    seed_everything(2027)
    observer = Observer().to(device)
    train_observer(plant, observer, source_loader, cfg, device)
    observer_path = args.output_dir / "observer.pt"
    torch.save(observer.state_dict(), observer_path)
    metrics = evaluate(plant, observer, target_loader, cfg, device)
    metrics.update({"setting": "M3_Monthly_to_M4_Monthly_H18", "context": cfg.context,
                    "patch": cfg.patch, "feedback": cfg.feedback})
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

