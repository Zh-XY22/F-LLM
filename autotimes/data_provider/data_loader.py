"""Real TSF loader used by the fixed Monthly source training path."""

import os

import numpy as np
from torch.utils.data import Dataset

from autotimes.utils.tools import convert_tsf_to_dataframe


class Dataset_TSF(Dataset):
    def __init__(self, root_path, flag="train", size=None, data_path=None,
                 scale=True, seasonal_patterns=None, drop_short=False):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.token_len = self.pred_len
        self.context_len = 4 * self.token_len
        self.set_type = {"train": 0, "val": 1, "test": 2}[flag]
        self.root_path = root_path
        self.data_path = data_path
        self.drop_short = drop_short
        self.timeseries = self._read_data()

    def _read_data(self):
        frame, _, _, _, _ = convert_tsf_to_dataframe(
            os.path.join(self.root_path, self.data_path)
        )
        timeseries = [np.asarray(values,dtype=np.float32) for values in frame.series_value]
        timeseries = [values[~np.isnan(values)] for values in timeseries]
        if self.drop_short:
            timeseries = [values for values in timeseries
                          if values.shape[0] > self.context_len]

        self.total_length = 0
        self.length_offsets = []
        self.series_ids = []
        for series_index, values in enumerate(timeseries):
            pad = max(self.pred_len + self.seq_len - values.shape[0], 0)
            if pad:
                values = np.hstack([np.zeros(pad, dtype=np.float32), values])
                timeseries[series_index] = values
            train_length = values.shape[0] - self.pred_len
            starts = [0, train_length - self.seq_len - self.pred_len,
                      train_length - self.seq_len]
            ends = [train_length - self.pred_len, train_length, len(values)]
            count = max(0, ends[self.set_type] - max(starts[self.set_type], 0)
                        - self.pred_len - self.seq_len + 1)
            self.length_offsets.append(np.zeros(count) + self.total_length)
            self.series_ids.append(np.zeros(count) + series_index)
            self.total_length += count
        self.length_offsets = np.hstack(self.length_offsets)
        self.series_ids = np.hstack(self.series_ids)
        return timeseries

    def __getitem__(self, index):
        offset = self.length_offsets[index]
        series_index = int(self.series_ids[index])
        local_index = index - int(offset)
        values = self.timeseries[series_index]
        train_length = len(values) - self.pred_len
        starts = [0, train_length - self.seq_len - self.pred_len,
                  train_length - self.seq_len]
        start = local_index + starts[self.set_type]
        end = start + self.seq_len
        label_start = end - self.label_len
        label_end = end + self.pred_len
        history = values[start:end, None]
        target = values[label_start:label_end, None]
        return history, target, history, target

    def __len__(self):
        return self.total_length
