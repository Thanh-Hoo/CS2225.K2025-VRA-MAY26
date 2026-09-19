import csv
import os
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.modality_weight_logger import ModalityWeightAccumulator, append_modality_weights_csv


class TestModalityWeightLogger(unittest.TestCase):
    def test_disabled_run_writes_nothing(self):
        acc = ModalityWeightAccumulator()
        acc.update(None)  # what model.last_modality_weights is when AW is disabled
        self.assertFalse(acc.has_data())
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, 'modality_weights.csv')
            append_modality_weights_csv(csv_path, epoch=1, split='train', accumulator=acc)
            self.assertFalse(os.path.exists(csv_path))

    def test_mean_std_and_csv_roundtrip(self):
        acc = ModalityWeightAccumulator()
        acc.update({'rgb': torch.tensor([1.0, 1.5]), 'nir': torch.tensor([0.8, 0.9]), 'tir': torch.tensor([1.2, 0.6])})
        acc.update({'rgb': torch.tensor([1.0]), 'nir': torch.tensor([1.0]), 'tir': torch.tensor([1.0])})
        rgb_mean, rgb_std = acc.mean_std('rgb')
        self.assertAlmostEqual(rgb_mean, (1.0 + 1.5 + 1.0) / 3, places=5)
        self.assertGreaterEqual(rgb_std, 0.0)

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, 'nested', 'modality_weights.csv')
            append_modality_weights_csv(csv_path, epoch=1, split='train', accumulator=acc)
            append_modality_weights_csv(csv_path, epoch=1, split='val', accumulator=acc)
            with open(csv_path, newline='') as f:
                rows = list(csv.reader(f))
            self.assertEqual(rows[0], ['epoch', 'split', 'rgb_mean', 'rgb_std', 'nir_mean', 'nir_std', 'tir_mean', 'tir_std'])
            self.assertEqual(len(rows), 3)  # header + train + val
            self.assertEqual(rows[1][1], 'train')
            self.assertEqual(rows[2][1], 'val')


if __name__ == '__main__':
    unittest.main()
