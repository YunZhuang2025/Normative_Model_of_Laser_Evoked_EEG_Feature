# Normative Model of Laser-Evoked EEG Features

A command-line tool for predicting laser-evoked brain responses (LEP features) using normative models trained with PCNtoolkit.

## Overview

This tool predicts ten brain electrical features from laser-evoked potentials based on covariates (laser power, sex, age, height):

- **ERP amplitudes**: N1, N2, P2 (μV)
- **ERP latencies**: N1, N2, P2 (ms)
- **Time-frequency features**: LEP, alpha, beta, gamma magnitudes (μV^2/Hz)

Each feature uses an independent hierarchical Bayesian regression (HBR) normative model with B-spline basis functions. The tool can generate normative predictions with confidence intervals and calculate Z-scores for observed values.

## Requirements

```bash
numpy
pandas
scipy
```

## Installation

1. Ensure Python 3.x is installed
2. Install dependencies: `pip install numpy pandas scipy`
3. Place `extracted_model_params.json` in the same directory as the script

## Usage

### Interactive Mode (Default)

```bash
python normative_model_LEP_EN.py
```

Select from:
- **Prediction mode**: Get normative predictions with confidence intervals
- **Z-score mode**: Calculate deviations from normative values

### Quick Prediction

```bash
python normative_model_LEP_EN.py -q 3.5 1 21 170
```

Arguments: `laserpower gender age height`

### CSV Batch Prediction (Recommended)

```bash
python normative_model_LEP_EN.py -c input.csv -o output.csv
```

**Input CSV format**: Only `laserpower` is required. Columns `gender`, `age`, `height` are optional.

**Missing value handling**:
- `age` or `height` = NaN → imputed with training set mean
- `gender` = NaN → population-weighted average of male and female predictions
- Column entirely absent → treated as all NaN

**Output**: Predictions include mean, std, 95% CI, imputation flags, and Z-scores if observation columns are provided.

### Legacy Batch Processing

```bash
python normative_model_LEP_EN.py -b input.csv -o output.csv
```

**Input CSV format**: Must contain columns `laserpower`, `gender`, `age`, `height`. Optional feature columns for Z-score calculation.

### Debug Mode

```bash
python normative_model_LEP_EN.py -d -q 3.5 1 65 170
```

Shows intermediate values including standardization parameters and B-spline basis outputs.

## Input Parameters

| Parameter  | Range     | Recommended | Description              |
|------------|-----------|-------------|--------------------------|
| laserpower | 1.0–4.5   | 2.5–4.0     | Laser stimulation power  |
| gender     | 1 or 2   | –           | 1 = male, 2 = female     |
| age        | 16–86    | 18–50       | Years                    |
| height     | 150–196  | –           | cm                       |

## Output

For each feature, predictions include:

- **mean**: Predicted normative value
- **std**: Uncertainty (heteroscedastic variance)
- **lower_95 / upper_95**: 95% confidence interval
- **z_score**: (observed − mean) / std (when observations provided)

## Citation

If you use this tool in published research, please cite:

> Zhuang Y., Zhang L.B., Wang X.Q., Geng X.Y., & Hu L. (in preparation). From Normative Features to Multidimensional Estimation of Pain: A Large-Scale Study of Laser-Evoked Brain Responses.

## Author

Yun Zhuang
Version: 4.0
Date: 2026-02-16
