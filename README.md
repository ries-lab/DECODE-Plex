# DECODE-Plex

DECODE-Plex is a deep-learning-based framework for high-density single-molecule localization microscopy (SMLM) in multi-channel imaging experiments. It localizes dense and overlapping emitters across multiple channels while modeling the photophysical and optical properties of multi-channel systems.

This repository accompanies the DECODE-Plex manuscript and provides code, trained weights, configuration files, point spread functions (PSFs), and raw inference data required to reproduce the results reported in the paper.

DECODE-Plex is built on [DECODE](https://doi.org/10.1038/s41592-021-01236-x), a DEep COntext DEpendent neural network for sub-pixel emitter localization, and uses experimentally calibrated PSF models such as those obtained with [SMAP](https://doi.org/10.1038/s41592-020-0938-1) or [uiPSF](https://doi.org/10.1038/s41592-024-02282-x).

## Repository Contents

The repository contains scripts and notebooks for the main stages of the DECODE-Plex workflow:

1. PSF calibration and preparation.
2. Model training with experiment-specific configuration files.
3. Inference/localization using trained models.
4. Channel assignment for dual-color datasets.

Example workflows are provided in the `./notebook` directory. Please update the file paths in each notebook according to your local data layout before running the examples.

## Data and Reproducibility

We provide the materials needed to reproduce all figures in the manuscript, including:

- trained model weights;
- training configuration files;
- calibrated point spread functions;
- raw data used for inference;
- example scripts/notebooks for running localization and downstream analysis.

Data access links will be added here: xxx

After downloading the repository, create the following folders in the project root:

```bash
mkdir -p ./data ./calibration ./outputs
```

Place the downloaded files into the corresponding folders:

- `./data`: raw inference data and example datasets;
- `./calibration`: calibrated PSFs and calibration-related files;
- `./outputs`: trained models, localization results, intermediate outputs, and figure-reproduction results.

## Installation

Only the GPU version of DECODE-Plex has been tested. For model training and high-density inference, we recommend a workstation equipped with a modern NVIDIA GPU, such as an RTX 3090 or RTX 4090. A CUDA-compatible GPU with sufficient memory is required for practical training times.

### System Requirements

- GPU: NVIDIA GPU with CUDA support and at least 8 GB GPU memory
- RAM: at least 16 GB
- CPU: multi-core CPU recommended
- OS: Linux or Windows
- Package manager: conda or miniconda

### Verified Environment

| Component | Version |
|----------|---------|
| Python   | 3.10.19 |
| PyTorch  | 2.1.2 |
| CUDA     | 12.9 |

### Setup

Clone the repository:

```bash
git clone git@github.com:ries-lab/DECODE-Plex.git
cd DECODE-Plex/
```

Create the local data directories:

```bash
mkdir -p ./data ./calibration ./outputs
```

Create and activate the conda environment:

```bash
conda config --set channel_priority flexible
conda env create -n decode_plex -f environment.yaml
conda activate decode_plex
```

The cubic-spline PSF package is pre-compiled and installed automatically as part of the environment setup.

## Usage

The typical DECODE-Plex workflow consists of the following steps.

### 1. Prepare PSF Calibration Files

Obtain or download the calibrated multi-channel PSFs and place them in `./calibration`. These PSFs are used to simulate training data and to model the optical response of the imaging system.

### 2. Train or Load a Model

Use the provided configuration files to train DECODE-Plex models for the corresponding experimental conditions. For reproducing the manuscript figures, use the supplied trained weights and matching configuration files.

### 3. Run Localization

Run inference on the provided raw data or on your own SMLM movies. Raw input data should be placed in `./data`, and localization results should be written to `./results`.

### 4. Perform Channel Assignment

For dual-color experiments, run the channel-assignment workflow after localization to separate emitters by channel.

## Paper

TODO

## Contact

For questions about this repository or the manuscript reproduction workflow, please contact:

- Hao Sha (shahao@hit.edu.cn)
- Lucas-Raphael Mueller (lrm@lrm.dev)
