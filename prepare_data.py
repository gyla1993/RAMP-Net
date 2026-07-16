"""Extract and validate a Google Drive regional data archive."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np


VARIABLES = ("TMP", "msl", "u10", "v10")
CORE_ARRAYS = (
    "X_train_all.npy",
    "X_val_all.npy",
    "X_test_all.npy",
    "X_train_all_e.npy",
    "X_val_all_e.npy",
    "X_test_all_e.npy",
    "locations.npy",
    "locations_e.npy",
)


def find_7zip(explicit=None):
    candidates = [explicit, shutil.which("7z"), shutil.which("7zz"), shutil.which("7za")]
    if os.name == "nt":
        candidates.extend(
            [
                r"C:\Program Files\7-Zip\7z.exe",
                r"D:\Program Files\7-Zip\7z.exe",
            ]
        )
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError("7-Zip was not found. Install 7-Zip or extract the RAR archive manually.")


def validate_region(root, region):
    region_dir = root / region
    failures = []
    summary = {}
    for variable in VARIABLES:
        variable_dir = region_dir / variable
        missing = [name for name in CORE_ARRAYS if not (variable_dir / name).is_file()]
        if missing:
            failures.append(f"{variable}: missing {', '.join(missing)}")
            continue
        arrays = {name: np.load(variable_dir / name, mmap_mode="r") for name in CORE_ARRAYS}
        station_shape = arrays["X_train_all.npy"].shape
        grid_shape = arrays["X_train_all_e.npy"].shape
        if station_shape[1:] != (12000, 1):
            failures.append(f"{variable}: unexpected training station shape {station_shape}")
        if arrays["X_val_all.npy"].shape[1:] != (1680, 1):
            failures.append(f"{variable}: unexpected validation shape {arrays['X_val_all.npy'].shape}")
        if arrays["X_test_all.npy"].shape[1:] != (3840, 1):
            failures.append(f"{variable}: unexpected test shape {arrays['X_test_all.npy'].shape}")
        if arrays["locations.npy"].shape[0] != station_shape[0]:
            failures.append(f"{variable}: station coordinate count does not match station data")
        if arrays["locations_e.npy"].shape[0] != grid_shape[0]:
            failures.append(f"{variable}: ERA5 coordinate count does not match reanalysis data")
        summary[variable] = {
            "stations": station_shape[0],
            "era5_grids": grid_shape[0],
            "train": station_shape[1],
            "val": arrays["X_val_all.npy"].shape[1],
            "test": arrays["X_test_all.npy"].shape[1],
        }
    for required in ("metadata_5K.json", "metadata_5K_sorted.json"):
        if not (region_dir / required).is_file():
            failures.append(f"missing {required}")
    for variable in VARIABLES:
        if not (region_dir / f"all_data_{variable}").is_dir():
            failures.append(f"missing all_data_{variable}/")
    if failures:
        raise RuntimeError("Data validation failed:\n- " + "\n- ".join(failures))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", nargs="?", help="California.rar or Guangdong.rar")
    parser.add_argument("--region", choices=("California", "Guangdong"))
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--seven-zip", dest="seven_zip")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    output = Path(args.output).resolve()
    if args.archive:
        archive = Path(args.archive).resolve()
        region = args.region or archive.stem
    elif args.region:
        archive = None
        region = args.region
    else:
        parser.error("provide an archive or --region for --check-only")

    if region not in ("California", "Guangdong"):
        parser.error("archive name must be California.rar or Guangdong.rar, or pass --region")
    if not args.check_only:
        if archive is None or not archive.is_file():
            parser.error("the archive file does not exist")
        seven_zip = find_7zip(args.seven_zip)
        subprocess.run([seven_zip, "x", str(archive), f"-o{output}", "-y"], check=True)

    summary = validate_region(output, region)
    print(f"{region} data are ready at {output / region}")
    for variable, values in summary.items():
        print(
            f"  {variable}: {values['stations']} stations, {values['era5_grids']} ERA5 grids, "
            f"splits={values['train']}/{values['val']}/{values['test']}"
        )


if __name__ == "__main__":
    main()
