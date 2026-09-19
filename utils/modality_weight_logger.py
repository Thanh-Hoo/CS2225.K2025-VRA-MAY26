"""CSV logging helper for AW-IDEA's per-epoch modality weight statistics.

No-op friendly by design: when adaptive weighting is disabled
(cfg.MODEL.ADAPTIVE_WEIGHTING.ENABLED = False), model.last_modality_weights stays
None, ModalityWeightAccumulator.update() ignores None, has_data() stays False, and
append_modality_weights_csv() writes nothing -- so plain IDEA runs never create or
touch modality_weights.csv.
"""
import csv
import os

_MODALITIES = ('rgb', 'nir', 'tir')
_CSV_HEADER = ['epoch', 'split', 'rgb_mean', 'rgb_std', 'nir_mean', 'nir_std', 'tir_mean', 'tir_std']


class ModalityWeightAccumulator:
    """Running mean/std of alpha_rgb, alpha_nir, alpha_tir over many batches."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._sum = {m: 0.0 for m in _MODALITIES}
        self._sum_sq = {m: 0.0 for m in _MODALITIES}
        self._count = 0

    def update(self, weights):
        """weights: dict with 'rgb'/'nir'/'tir' -> Tensor[B], or None (no-op)."""
        if weights is None:
            return
        for modality in _MODALITIES:
            values = weights[modality].detach().float().reshape(-1)
            self._sum[modality] += values.sum().item()
            self._sum_sq[modality] += values.pow(2).sum().item()
        self._count += weights[_MODALITIES[0]].numel()

    def has_data(self):
        return self._count > 0

    def mean_std(self, modality):
        if self._count == 0:
            return 0.0, 0.0
        mean = self._sum[modality] / self._count
        variance = max(self._sum_sq[modality] / self._count - mean ** 2, 0.0)
        return mean, variance ** 0.5


def append_modality_weights_csv(csv_path, epoch, split, accumulator):
    """Appends one summary row for this epoch/split; creates the file+header on first use."""
    if not accumulator.has_data():
        return
    out_dir = os.path.dirname(csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(_CSV_HEADER)
        row = [epoch, split]
        for modality in _MODALITIES:
            mean, std = accumulator.mean_std(modality)
            row.extend([mean, std])
        writer.writerow(row)
