<div align="center">

# RAMP-Net

### Reanalysis-Infused Attentive Message Passing for Multi-Station Meteorological Imputation

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-2ea44f.svg)](LICENSE)
[![Data](https://img.shields.io/badge/Processed%20Data-Google%20Drive-4285F4?logo=googledrive&logoColor=white)](https://drive.google.com/drive/folders/1Lm8lddu4-yTe54fitGt2sFfIIb3k2sMv?usp=sharing)

Official implementation accompanying the *Computers & Geosciences* manuscript.

[Processed data](https://drive.google.com/drive/folders/1Lm8lddu4-yTe54fitGt2sFfIIb3k2sMv?usp=sharing) · [Quick test](#quick-test) · [Full experiments](#reproducing-the-experiments) · [License](#license-and-citation)

</div>

RAMP-Net reconstructs missing meteorological station observations by combining the spatially complete physical background of ERA5 reanalysis with dynamic dependencies learned across stations. The model first transfers information from nearby ERA5 grid cells to each station, then refines the coarse estimate through temporal and inter-station attention.

## Highlights

- **Reanalysis-guided reconstruction:** uses ERA5 fields as physically meaningful priors instead of treating them as ordinary auxiliary features.
- **Dynamic message passing:** learns time-varying station relationships conditioned on meteorological features and geographic coordinates.
- **Coarse-to-fine imputation:** aligns gridded reanalysis with point observations before station-scale refinement.
- **Two distinct study regions:** evaluates 33 stations in Guangdong and 103 stations in California across four meteorological variables and four missing rates.
- **Reproducible release:** provides model and baseline code, regional normalization statistics, experiment scripts, a dependency specification, and a data-free CPU quick test.

## Method Overview

<p align="center">
  <img src="code/docs/schematic%20diagram.png" width="92%" alt="RAMP-Net imputation workflow">
</p>
<p align="center"><em>Coarse-to-fine reconstruction using ERA5 background fields and station observations.</em></p>

<p align="center">
  <img src="code/docs/Overall%20Framework.png" width="92%" alt="Detailed RAMP-Net architecture">
</p>
<p align="center"><em>RAMP-Net architecture: reanalysis prior infusion, temporal attention, and attentive inter-station message passing.</em></p>

## Study Regions and Data Characteristics

The experiments use hourly station observations from WEATHER-5K and ERA5 single-level reanalysis for 2018--2019. The target variables are 2-m temperature (`TMP`), mean sea-level pressure (`msl`), and the 10-m zonal and meridional wind components (`u10` and `v10`).

<table>
  <tr>
    <td align="center" width="50%"><img src="code/docs/Gdong_terrain_map.png" width="95%" alt="Guangdong station network and terrain"></td>
    <td align="center" width="50%"><img src="code/docs/California_terrain_map.png" width="95%" alt="California station network and terrain"></td>
  </tr>
  <tr>
    <td align="center"><strong>Guangdong, China</strong><br>33 stations; monsoon, coastal, plain, hill, and mountain influences.</td>
    <td align="center"><strong>California, USA</strong><br>103 stations; coastal, valley, desert, and strong orographic contrasts.</td>
  </tr>
</table>

<table>
  <tr>
    <td align="center" width="50%"><img src="code/docs/Gd_scatter_v10_TMP_aligned.png" width="98%" alt="Guangdong ERA5 and station comparison"></td>
    <td align="center" width="50%"><img src="code/docs/Ca_scatter_v10_TMP_aligned.png" width="98%" alt="California ERA5 and station comparison"></td>
  </tr>
  <tr>
    <td align="center">Guangdong station--ERA5 comparison</td>
    <td align="center">California station--ERA5 comparison</td>
  </tr>
</table>

Temperature follows the ERA5 background relatively closely, whereas station wind observations show stronger local variability. This scale mismatch motivates explicit grid-to-station alignment rather than direct feature concatenation.

## Processed Experimental Data

The complete processed experimental data are available in the following public Google Drive folder:

> **Download:** [RAMP-Net complete processed data and masks](https://drive.google.com/drive/folders/1Lm8lddu4-yTe54fitGt2sFfIIb3k2sMv?usp=sharing)

Download the complete contents of the shared folder and preserve the directory structure supplied there. Prepare one region at a time because the training code expects the active variable folders and mask set at fixed relative paths.

After extraction, the relevant layout should be:

```text
RAMP-Net/
|-- code/
|   |-- mask_rate/
|   |   |-- 0.125/
|   |   |-- 0.25/
|   |   |-- 0.5/
|   |   `-- 0.75/
|   `-- scaler/
|       |-- California/
|       `-- Guangdong/
|-- TMP/
|-- msl/
|-- u10/
`-- v10/
```

Each variable directory must contain the processed training, validation, test, reanalysis, and coordinate arrays used by `code/data_provider/data_factory_2.py`, including:

```text
X_train_all.npy       X_train_all_e.npy
X_val_all.npy         X_val_all_e.npy
X_test_all.npy        X_test_all_e.npy
locations.npy         locations_e.npy
```

Mask files must be placed below `code/mask_rate/<missing-rate>/` with the `train_mask_*`, `val_mask_*`, and `test_mask_*` names expected by `code/data_provider/data_loader.py`.

The original upstream data sources remain available separately:

- [WEATHER-5K](https://github.com/thanadol-git/WEATHER-5K)
- [ERA5 single-level reanalysis](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels)

The Google Drive data are distributed separately from GitHub because of their size. WEATHER-5K and ERA5 remain subject to their original terms.

## Installation

Python 3.10 or 3.11 is recommended. Linux is recommended for the published shell scripts, and an NVIDIA GPU is recommended for full experiments.

```bash
git clone https://github.com/gyla1993/RAMP-Net.git
cd RAMP-Net

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the CUDA-specific PyTorch build using the [official selector](https://pytorch.org/get-started/locally/) when required. See the [PyTorch Geometric installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) if a platform-specific wheel is needed.

## Quick Test

The quick test does not require WEATHER-5K, ERA5, or the Google Drive data. It creates a small synthetic station/grid problem, performs a complete RAMP-Net forward pass on CPU, and verifies that the outputs are finite and correctly shaped.

```bash
python quick_test.py
```

Expected output:

```text
RAMP-Net quick test passed
output shape: (2, 5, 8, 1)
attention shape: (2, 5, 5)
```

## Reproducing the Experiments

Run experiment commands from `code/` because the original data, mask, checkpoint, and output paths are relative to that directory. Set `REGION` to match the processed regional data currently extracted at the repository root.

### California

```bash
cd code
REGION=California bash scripts/RAMP-Net/RAMP-Net_u10.sh
REGION=California bash scripts/RAMP-Net/RAMP-Net_v10.sh
REGION=California bash scripts/RAMP-Net/RAMP-Net_TMP.sh
REGION=California bash scripts/RAMP-Net/RAMP-Net_msl.sh
```

### Guangdong

```bash
cd code
REGION=Guangdong bash scripts/RAMP-Net/RAMP-Net_u10.sh
REGION=Guangdong bash scripts/RAMP-Net/RAMP-Net_v10.sh
REGION=Guangdong bash scripts/RAMP-Net/RAMP-Net_TMP.sh
REGION=Guangdong bash scripts/RAMP-Net/RAMP-Net_msl.sh
```

The scripts run seeds `2024`--`2028` at missing rates `0.125`, `0.25`, `0.50`, and `0.75`. The published RAMP-Net configuration uses a 24-hour input window, four neighboring ERA5 grid points, batch size 6, up to 50 epochs, and early stopping with patience 3.

For inference, keep the corresponding script arguments and change `--is_training 1` to `--is_training 0`. For CPU execution, pass `--no-use_gpu` and omit `--use_multi_gpu`.

## Implemented Methods

| Category | Methods |
|---|---|
| Proposed | **RAMP-Net**, RAMP-Net ablations |
| Geostatistical | Kriging |
| Single-station | Linear Regression, Autoformer, TimesNet, Non-stationary Transformer |
| Multi-station | MPNN |

Baseline scripts are available under `code/scripts/Autoformer`, `Kriging`, `LR`, `MPNN`, `Non-stationary Transformer`, and `TimesNet`.

## Results and Analysis

### Robustness to Missingness

<p align="center">
  <img src="code/docs/robustness.png" width="88%" alt="Robustness across missing rates">
</p>
<p align="center"><em>MAE trends for the U and V wind components across increasing missing rates.</em></p>

### Station-Level Model Distribution

<p align="center">
  <img src="code/docs/best_model_distribution.png" width="88%" alt="Best-performing method at each station">
</p>
<p align="center"><em>Spatial distribution of the best-performing imputation method at each station.</em></p>

### Dynamic Spatial Dependencies

<table>
  <tr>
    <td align="center" width="43%"><img src="code/docs/Gdong_Terrain_Result.png" width="98%" alt="Guangdong stations indexed by Hilbert order"></td>
    <td align="center" width="57%"><img src="code/docs/pearson_attention_single.png" width="98%" alt="Learned inter-station attention matrix"></td>
  </tr>
  <tr>
    <td align="center">Station geography and Hilbert ordering</td>
    <td align="center">Learned station-to-station attention</td>
  </tr>
</table>

The attention pattern captures both geographically local station groups and longer-range relationships associated with similar terrain and atmospheric forcing.

### Temporal Reconstruction

<p align="center">
  <img src="code/docs/temporal_reconstruction.png" width="96%" alt="Temporal imputation examples">
</p>
<p align="center"><em>Example reconstructions for stable and strongly fluctuating station series under different missing rates.</em></p>

### Neighbor Sensitivity

<p align="center">
  <img src="code/docs/K_sensitivity_final.png" width="88%" alt="Sensitivity to the number of neighboring ERA5 grid points">
</p>
<p align="center"><em>Sensitivity to the number of neighboring ERA5 grid cells. K=4 provides the selected accuracy--efficiency trade-off.</em></p>

## Repository Structure

```text
RAMP-Net/
|-- code/
|   |-- data_provider/   Data loading and preprocessing
|   |-- docs/            Architecture and experimental figures
|   |-- exp/             Training and evaluation pipelines
|   |-- layers/          Neural network building blocks
|   |-- mask_rate/       Local missing-value masks (download separately)
|   |-- models/          RAMP-Net, ablations, and baselines
|   |-- scaler/          Regional normalization statistics
|   |-- scripts/         Reproduction scripts by method and variable
|   |-- utils/           Metrics and utilities
|   |-- run.py           Multi-station entry point
|   `-- run_s.py         Single-station entry point
|-- quick_test.py        Synthetic CPU smoke test
|-- requirements.txt    Python dependencies
|-- LICENSE             MIT License
`-- README.md
```

## Outputs and Reproducibility

- Checkpoints are written below `code/checkpoints/`.
- Test arrays and metrics are written below `code/test_results/` and `code/results/`.
- Entry points set deterministic seeds; published scripts run five independent seeds.
- Regional normalization statistics are included under `code/scaler/California` and `code/scaler/Guangdong`.
- The quick test verifies installation and core-model execution; it is not a substitute for reproducing the manuscript tables.

## License and Citation

The RAMP-Net source code and documentation are released under the [MIT License](LICENSE). Dataset and third-party model licenses are unchanged.

Please cite the accompanying *Computers & Geosciences* manuscript when using this repository. The final bibliographic entry will be added after publication.
