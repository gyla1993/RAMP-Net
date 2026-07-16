"""CPU smoke test for a complete RAMP-Net forward pass on synthetic data."""

import argparse
import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch


REPOSITORY_ROOT = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(REPOSITORY_ROOT, "code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from models.RAMP_Net import Model  # noqa: E402


def run_smoke_test(seed: int = 2024) -> None:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    batch_size = 2
    station_count = 5
    grid_count = 9
    sequence_length = 8
    neighbor_count = 4

    with tempfile.TemporaryDirectory(prefix="ramp_net_test_") as temp_dir:
        feature_dir = os.path.join(temp_dir, "synthetic")
        scaler_dir = os.path.join(temp_dir, "scaler")
        os.makedirs(feature_dir)
        os.makedirs(scaler_dir)

        station_locations = np.column_stack(
            [rng.uniform(22.0, 24.0, station_count), rng.uniform(112.0, 114.0, station_count)]
        ).astype(np.float32)
        grid_locations = np.column_stack(
            [rng.uniform(21.5, 24.5, grid_count), rng.uniform(111.5, 114.5, grid_count)]
        ).astype(np.float32)
        np.save(os.path.join(feature_dir, "locations.npy"), station_locations)
        np.save(os.path.join(feature_dir, "locations_e.npy"), grid_locations)
        np.save(os.path.join(scaler_dir, "u10_mean.npy"), np.zeros(station_count, dtype=np.float32))
        np.save(os.path.join(scaler_dir, "u10_std.npy"), np.ones(station_count, dtype=np.float32))

        args = SimpleNamespace(
            k=neighbor_count,
            feature="u10",
            feature_dir=feature_dir,
            scaler_dir=scaler_dir,
            enc_in=1,
            seq_len=sequence_length,
            factor=3,
            device=torch.device("cpu"),
        )
        model = Model(args).eval()

        observations = torch.randn(batch_size, station_count, sequence_length, 1)
        reanalysis = torch.randn(batch_size, grid_count, sequence_length, 1)
        mask = (torch.rand(batch_size, station_count, sequence_length, 1) > 0.4).float()
        mask[:, :, 0, :] = 1.0
        dates = torch.zeros(batch_size, sequence_length, 4)
        dates[:, :, 0] = 2018
        dates[:, :, 1] = 1
        dates[:, :, 2] = torch.arange(sequence_length) % 7
        dates[:, :, 3] = torch.arange(sequence_length) % 24

        with torch.no_grad():
            output, attention = model(
                observations,
                reanalysis,
                torch.from_numpy(station_locations),
                torch.from_numpy(grid_locations),
                mask,
                dates,
            )

        expected_output_shape = (batch_size, station_count, sequence_length, 1)
        if tuple(output.shape) != expected_output_shape:
            raise AssertionError(f"unexpected output shape: {tuple(output.shape)}")
        if not torch.isfinite(output).all() or not torch.isfinite(attention).all():
            raise AssertionError("model returned non-finite values")

        print("RAMP-Net quick test passed")
        print(f"output shape: {tuple(output.shape)}")
        print(f"attention shape: {tuple(attention.shape)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2024)
    run_smoke_test(parser.parse_args().seed)
