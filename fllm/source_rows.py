"""The fixed source TRAIN rows used to build the observer cache."""

from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from autotimes.data_provider.data_loader import Dataset_TSF
from autotimes.utils.tools import convert_tsf_to_dataframe


class SourceRows(Dataset):
    def __init__(self, config):
        args = config["plant"]
        self.context = args["seq_len"]
        self.patch = args["token_len"]
        self.label = args["label_len"]
        root = str(Path(config["source_tsf"]).parent)
        name = Path(config["source_tsf"]).name
        self.native = Dataset_TSF(
            root_path=root,
            flag="train",
            size=[self.context, self.label, self.patch],
            data_path=name,
            drop_short=False,
        )
        frame, _, _, _, _ = convert_tsf_to_dataframe(str(Path(root) / name))
        self.ids = frame.series_name.tolist()
        self.count = len(self.native)
        self.series_count = len(self.ids)
        if self.count <= 0 or len(set(self.ids)) != self.series_count:
            raise ValueError("Monthly source rows are empty or have duplicate IDs")

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        series = int(self.native.series_ids[index])
        local = index - int(self.native.length_offsets[index])
        values = self.native.timeseries[series]
        cut = local + self.context
        history, target, _, _ = self.native[index]
        history = history[:, 0].astype(np.float32)
        target = target[:, 0].astype(np.float32)
        history_mask = np.ones(self.context, dtype=np.float32)
        target_mask = np.ones_like(target)
        future = target[-self.patch:].copy()
        future_mask = np.ones_like(future)
        if cut + self.patch > len(values) - 2 * self.patch:
            raise ValueError("Source row crosses the fixed training boundary")
        if not (np.isfinite(history).all() and np.isfinite(target).all()):
            raise ValueError("Non-finite source row")
        return {
            "history": history,
            "history_mask": history_mask,
            "native_target": target,
            "native_mask": target_mask,
            "future": future,
            "future_mask": future_mask,
            "series_id": str(self.ids[series]),
            "series_index": series,
            "cut_point": cut,
        }

