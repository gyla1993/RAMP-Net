import hashlib
import os

import numpy as np


VARIABLES = ("TMP", "msl", "u10", "v10")
REGION_STATIONS = {"California": 103, "Guangdong": 33}
SPLIT_FILES = {
    "train": "X_train_all.npy",
    "val": "X_val_all.npy",
    "test": "X_test_all.npy",
}


def configure_data_paths(args):
    """Populate region-aware paths used throughout the experiment code."""
    data_root = os.path.abspath(getattr(args, "data_root", ".."))
    region_dir = os.path.join(data_root, args.region)
    args.data_root = data_root
    args.region_dir = region_dir
    args.feature_dir = os.path.join(region_dir, args.feature)
    args.station_csv_dir = os.path.join(region_dir, f"all_data_{args.feature}")
    args.metadata_path = os.path.join(region_dir, "metadata_5K.json")
    args.mask_dir = os.path.abspath(
        getattr(args, "mask_dir", None)
        or os.path.join("./mask_rate", args.region)
    )
    if getattr(args, "scaler_dir", None) is None:
        args.scaler_dir = os.path.abspath(os.path.join("./scaler", args.region))
    else:
        args.scaler_dir = os.path.abspath(args.scaler_dir)
    return args


def require_feature_directory(args):
    required = (
        "X_train_all.npy",
        "X_val_all.npy",
        "X_test_all.npy",
        "X_train_all_e.npy",
        "X_val_all_e.npy",
        "X_test_all_e.npy",
        "locations.npy",
        "locations_e.npy",
    )
    missing = [name for name in required if not os.path.isfile(os.path.join(args.feature_dir, name))]
    if missing:
        raise FileNotFoundError(
            f"Incomplete {args.region}/{args.feature} data at {args.feature_dir}. "
            f"Missing: {', '.join(missing)}. Download and extract the regional archive as described in README.md."
        )


def mask_path(args, split):
    rate = f"{float(args.mask_rate):g}"
    return os.path.join(
        args.mask_dir,
        rate,
        f"{split}_mask_{args.feature}_{args.iitr}.npy",
    )


def _mask_seed(args, split):
    material = f"RAMP-Net|{args.region}|{args.feature}|{float(args.mask_rate):g}|{args.iitr}|{split}"
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "little")


def load_or_create_mask(args, split, data_shape, repetitions=10):
    """Load a region-specific mask, or deterministically create an MCAR mask."""
    expected = (repetitions,) + tuple(int(value) for value in data_shape)
    path = mask_path(args, split)
    if os.path.isfile(path):
        mask = np.load(path)
        if tuple(mask.shape) == expected:
            return mask.astype(np.uint8, copy=False)
        print(f"Ignoring incompatible mask {path}: found {mask.shape}, expected {expected}")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    rng = np.random.default_rng(_mask_seed(args, split))
    mask = (rng.random(expected) >= float(args.mask_rate)).astype(np.uint8)
    np.save(path, mask)
    print(f"Created deterministic MCAR mask: {path} {expected}")
    return mask


def regional_mask_shape(args, time_steps, channels=1):
    return (REGION_STATIONS[args.region], int(time_steps), int(channels))
