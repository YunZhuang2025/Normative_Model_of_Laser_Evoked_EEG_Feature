# Normative_Model_of_Laser_Evoked_EEG_Feature

A command-line tool for predicting laser-evoked brain responses (LEP features) using normative models trained with PCNtoolkit.

## Overview

This tool predicts seven brain electrical features from laser-evoked potentials based on covariates (laser power, gender, age, height):
- **ERP components**: N1, N2, P2 amplitudes and latencies
- **Time-frequency features**: ERP, alpha, beta, gamma magnitudes

Each feature uses an independent normative model with B-spline basis functions and can calculate Z-scores for observed values.

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
```
Select from:
- Prediction mode: Get normative predictions with confidence intervals
- Z-score mode: Calculate deviations from normative values

### Quick Prediction
```bash
python predictor_by_feature.py -q 3.5 1 21 170
```
Arguments: laser_power gender age height

### Batch Processing
```bash
python predictor_by_feature.py -b input.csv -o output.csv
```

**Input CSV format**: Must contain columns `laserpower`, `gender`, `age`, `height`. Optional feature columns for Z-score calculation.

**Output**: Predictions include mean, std, 95% CI (lower/upper bounds), and Z-scores if observations provided.

## Input Parameters

| Parameter | Range | Recommended | Description |
|-----------|-------|-------------|-------------|
| laserpower | 1.0-4.5 | 2.5-4.0 | Laser stimulation power |
| gender | 1 or 2 | - | 1=male, 2=female |
| age | 16-50 | 18-25 | Years |
| height | 150-190 | - | cm |

## Output

For each feature, predictions include:
- **mean**: Predicted normative value
- **std**: Uncertainty (heteroscedastic variance)
- **lower_95/upper_95**: 95% confidence interval
- **z_score**: (observed - mean) / std (when observations provided)

## Citation

If you use this tool in published research, please cite:

Zhuang Y., Zhang L.B., Wang X.Q., Geng X.Y., & Hu L., (in preparation) From Normative Features to Multidimensional Estimation of Pain: A Large-Scale Study of Laser-Evoked Brain Responses.

## Author

Yun Zhuang  
Version: 1.0  
Date: 2026-02-16
