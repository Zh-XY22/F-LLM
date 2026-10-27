"""Train the fixed residual observer from one plant's source cache."""

import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class CacheRows(Dataset):
    def __init__(self, cache_dir):
        root = Path(cache_dir)
        self.arrays = {name: np.load(root / (name + ".npy"), mmap_mode="r")
                       for name in ["residual", "target", "future_mask", "residual_mask"]}

    def __len__(self):
        return len(self.arrays["residual"])

    def __getitem__(self, index):
        return tuple(np.array(self.arrays[name][index], copy=True)
                     for name in ["residual", "target", "future_mask", "residual_mask"])


def train(cache_dir, output_dir, observer_seed=2027):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    random.seed(observer_seed)
    np.random.seed(observer_seed)
    torch.manual_seed(observer_seed)
    model = nn.Sequential(nn.Linear(24, 192), nn.Tanh(), nn.Dropout(0.1), nn.Linear(192, 24)).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-8)
    generator = torch.Generator().manual_seed(observer_seed)
    dataset = CacheRows(cache_dir)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0,
                        drop_last=False, generator=generator)
    metrics = []
    start = time.time()
    for epoch in range(5):
        model.train()
        total = 0.0
        samples = 0
        for residual, target, mask, residual_mask in loader:
            residual = residual.cuda()
            target = target.cuda()
            mask = mask.cuda()
            usable = (residual_mask.sum(1) > 0).to(dtype=torch.float32, device="cuda").unsqueeze(1)
            prediction = model(residual) * usable
            per_sample = ((prediction - target).square() * mask).sum(1) / mask.sum(1).clamp_min(1)
            loss = per_sample.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite observer loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("non-finite observer gradient")
            optimizer.step()
            total += float(loss.detach()) * len(residual)
            samples += len(residual)
        scheduler.step()
        metrics.append({"epoch": epoch + 1, "learning_rate": optimizer.param_groups[0]["lr"],
                        "train_normalized_residual_mse": total / samples,
                        "samples": samples, "updates": len(loader), "amp_skips": 0})
    checkpoint = output / "observer_epoch05.pt"
    torch.save({"model": {n: v.detach().cpu() for n, v in model.state_dict().items()},
                "observer_seed": observer_seed, "epoch": 5, "metrics": metrics}, checkpoint)
    with (output / "training_metrics.tsv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(metrics)
    (output / "observer_manifest.json").write_text(json.dumps({
        "status": "COMPLETE", "checkpoint": str(checkpoint.resolve()),
        "observer_seed": observer_seed, "epochs": 5, "samples": len(dataset),
        "updates": sum(row["updates"] for row in metrics),
        "wall_seconds": time.time() - start, "cache": str(Path(cache_dir).resolve())
    }, indent=2) + "\n")
    return checkpoint

