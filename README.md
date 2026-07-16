# RAMP-Net

Official implementation of **A Reanalysis-Infused Attentive Message Passing Network for Multi-Station Meteorological Imputation**.

RAMP-Net combines ERA5 reanalysis fields with station observations. A grid-to-station attention stage provides a coarse physical prior, and temporal plus inter-station attention refines missing values using dynamic spatial dependencies.

## Repository contents

```text
RAMP-Net-main/
|-- code/
|   |-- data_provider/   Data loading and preprocessing
|   |-- exp/             Training and evaluation pipelines
|   |-- layers/          Neural network layers
|   |-- models/          RAMP-Net, ablations, and baselines
|   |-- scaler/          Published normalization statistics
|   |-- scripts/         Reproduction scripts by model and variable
|   |-- utils/           Metrics and utilities
|   |-- run.py           Multi-station entry point
|   `-- run_s.py         Single-station entry point
|-- data/                Local location for processed regional data (not committed)
|-- quick_test.py        Synthetic CPU smoke test
|-- requirements.txt    Python dependencies
`-- LICENSE             MIT License
```

## Requirements

- Python 3.10 or 3.11
- Linux is recommended for the published shell scripts
- An NVIDIA GPU is recommended for full experiments; the quick test runs on CPU
- Sufficient local storage for the WEATHER-5K and ERA5 source data

Create an isolated environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For a CUDA build of PyTorch, follow the selector at <https://pytorch.org/get-started/locally/> before installing the remaining requirements. `torch-geometric` installation details are available at <https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html>.

## Quick test

The quick test requires no WEATHER-5K or ERA5 download. It creates synthetic station and grid coordinates, performs a complete RAMP-Net forward pass on CPU, and verifies the output and attention tensors.

From the repository root, run:

```bash
python quick_test.py
```

Expected output begins with:

```text
RAMP-Net quick test passed
output shape: (2, 5, 8, 1)
```

## Data preparation

The experiments use hourly WEATHER-5K station observations and ERA5 single-level reanalysis for 2018--2019 in Guangdong, China and California, USA.

1. Obtain the original WEATHER-5K data from <https://github.com/thanadol-git/WEATHER-5K> or its official download link.
2. Obtain ERA5 single-level data from <https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels>.
3. Prepare one regional experiment at a time. Place the processed variable folders (`u10`, `v10`, `TMP`, and `msl`) in the repository root, next to `code/`.
4. Place the matching train, validation, and test mask arrays under `code/mask_rate/<missing-rate>/`, using the filenames expected by `code/data_provider/data_loader.py`.

Large processed data and mask archives are intentionally not committed to GitHub. All source code is stored as individual files in the repository. WEATHER-5K and ERA5 remain subject to their original terms; the MIT license in this repository applies only to the RAMP-Net source code and documentation.

## Reproducing experiments

Run commands from `code/`, because the original experiment paths are relative to that directory. Extract one regional data/mask archive at a time, then set `REGION` to the same region so the correct normalization statistics are selected:

```bash
cd code
REGION=California bash scripts/RAMP-Net/RAMP-Net_u10.sh
REGION=California bash scripts/RAMP-Net/RAMP-Net_v10.sh
REGION=California bash scripts/RAMP-Net/RAMP-Net_TMP.sh
REGION=California bash scripts/RAMP-Net/RAMP-Net_msl.sh
```

For Guangdong, replace `REGION=California` with `REGION=Guangdong` after extracting the Guangdong data and masks. The default is California.

The scripts train five seeds (`2024`--`2028`) at missing rates 0.125, 0.25, 0.50, and 0.75. They use a 24-hour input window, four neighboring ERA5 grid points, a batch size of 6, up to 50 epochs, and early stopping with patience 3.

To evaluate a saved checkpoint, use the same arguments as the corresponding script and change `--is_training 1` to `--is_training 0`. For CPU execution, pass `--no-use_gpu` and do not pass `--use_multi_gpu`.

Baseline scripts are organized under `code/scripts/Autoformer`, `Kriging`, `LR`, `MPNN`, `Non-stationary Transformer`, and `TimesNet`.

## Outputs and reproducibility

- Checkpoints are written below `code/checkpoints/`.
- Test arrays and metrics are written below `code/test_results/` and `code/results/` by the experiment classes.
- Random seeds are set in the entry points and the published scripts run five independent seeds.
- Normalization statistics included in `code/scaler/` are separated by region and variable.

The full experiments require substantial disk space and GPU time. The quick test is intended only to confirm installation and core-model execution; it does not reproduce the manuscript accuracy tables.

## License and citation

The RAMP-Net source code is released under the [MIT License](LICENSE). Dataset and third-party model licenses are not changed by this repository.

When using this code, cite the accompanying *Computers & Geosciences* manuscript. A final bibliographic entry can be added after publication.
