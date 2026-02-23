#!/usr/bin/env python3
"""
Normative model for laser-evoked brain electrical features based on PCNtoolkit

Author: Yun Zhuang
Date: 2025-01-16
Version: v5.0.0

If you use this tool in published research, please cite:
Zhuang Y., Zhang L.B., Wang X.Q., Geng X.Y., & Hu L., (in preparation) 
From Normative Features to Multidimensional Estimation of Pain: 
A Large-Scale Study of Laser-Evoked Brain Responses.

"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import BSpline
import sys
import argparse
import warnings

warnings.filterwarnings('ignore')


def softplus(x, params=(0.0, 3.0)):
    """
    Softplus mapping function.

    2-param form (shift, scale):
        out = scale * log(1 + exp((x - shift) / scale))

    3-param form (shift, scale, offset):
        out = offset + scale * log(1 + exp((x - shift) / scale))
        The offset ensures the output is always >= offset (used for delta
        in SHASHb to keep it bounded away from 0).
    """
    if len(params) == 3:
        shift, scale, offset = params
    else:
        shift, scale = params
        offset = 0.0
    x_scaled = (x - shift) / scale
    x_clipped = np.clip(x_scaled, -20, 20)
    return offset + scale * np.log1p(np.exp(x_clipped))


def apply_mapping(x, mapping_config):
    """
    Apply a named mapping function to scalar or array x.

    Supported types:
        'identity' - no transformation
        'softplus'  - 2-param or 3-param softplus (see softplus())
    """
    mtype = mapping_config.get('type', 'identity')
    params = mapping_config.get('params', [0.0, 1.0])
    if mtype == 'identity':
        return x
    elif mtype == 'softplus':
        return softplus(x, params)
    else:
        warnings.warn(f"Unknown mapping type '{mtype}', applying identity.")
        return x


def shash_quantile(mu, sigma, epsilon, delta, p):
    """
    Quantile function of the SHASHb distribution.

    For Y ~ SHASHb(mu, sigma, epsilon, delta):
        Q(p) = mu + sigma * sinh((arcsinh(z_p) + epsilon) / delta)
    where z_p = standard normal quantile at p.

    This is the correct formula for computing prediction intervals under
    the Sinh-Arcsinh (Jones & Pewsey 2009) likelihood used in SHASHb.
    """
    from scipy.stats import norm
    z_p = norm.ppf(p)
    return mu + sigma * np.sinh((np.arcsinh(z_p) + epsilon) / delta)


def shash_zscore(y, mu, sigma, epsilon, delta):
    """
    Compute the normative Z-score for the SHASHb distribution.

    If Y ~ SHASHb(mu, sigma, epsilon, delta), then:
        Z = delta * arcsinh((Y - mu) / sigma) - epsilon  ~  Normal(0, 1)

    This Z-score represents the signed distance of an observed value from
    the normative distribution on a standard-normal scale, correctly
    accounting for skewness (epsilon) and tail weight (delta).
    """
    return delta * np.arcsinh((y - mu) / sigma) - epsilon


def create_bspline_basis(x, knots, degree=3):
    """
    Create B-spline basis function matrix
    
    Parameters:
        x: Input values (standardized)
        knots: Knot vector
        degree: B-spline degree
    
    Returns:
        basis_matrix: shape (n_bases,) or (n_samples, n_bases)
    """
    x = np.atleast_1d(x)
    knots = np.array(knots)
    n_bases = len(knots) - degree - 1
    
    basis_values = []
    for i in range(n_bases):
        c = np.zeros(n_bases)
        c[i] = 1.0
        bspl = BSpline(knots, c, degree, extrapolate=True)
        basis_values.append(bspl(x))
    
    basis_matrix = np.column_stack(basis_values)
    
    # If input is a single value, return 1D array
    if basis_matrix.shape[0] == 1:
        return basis_matrix.flatten()
    
    return basis_matrix


class HBRPredictorByFeature:
    """
    HBR Predictor - feature-wise training version (10 features)
    
    Features:
    - Each feature has independent training data and standardization parameters
    - Automatically handles different inscalers for different features
    - Unified covariate input interface
    - Supports 10 EEG features: N1/N2/P2 amplitudes, N1/N2/P2 latencies, 
      LEP/alpha/beta/gamma magnitudes
    """
    
    def __init__(self, params_file=None, custom_inscaler=None):
        """
        Initialize predictor
        
        Parameters:
            params_file: Path to parameter file
            custom_inscaler: Custom inscaler dictionary, format {'mean': [...], 'std': [...]}
                            If provided, will override inscalers for all features
        """
        # Automatically find parameter file
        if params_file is None:
            params_file = self._find_params_file()
        
        params_path = Path(params_file)
        if not params_path.exists():
            raise FileNotFoundError(f"Parameter file not found: {params_file}")
        
        # Load model parameters
        print(f"Loading parameter file: {params_file}")
        with open(params_file, 'r', encoding='utf-8') as f:
            self.model_params = json.load(f)
        
        self.params_file = str(params_path)
        self.custom_inscaler = custom_inscaler
        
        # Feature names (in expected order) - Updated for 10 features
        feature_order = [
            'N1_amp', 'N2_amp', 'P2_amp',              # Amplitudes
            'N1_latency', 'N2_latency', 'P2_latency',  # Latencies
            'LEP_mag', 'alpha_mag', 'beta_mag', 'gamma_mag'  # Magnitudes
        ]
        self.feature_names = [f for f in feature_order if f in self.model_params]
        
        if not self.feature_names:
            raise ValueError("No available feature models found!")
        
        # Prepare inscaler for each feature
        self._prepare_inscalers()
        
        # Validate inscaler values are reasonable
        self._validate_inscalers()
        
        # Debug mode flag
        self.debug = False
        
        # Covariate ranges (based on training data statistics from inscaler)
        self.covariate_names = ['laserpower', 'gender', 'age', 'height']
        self.ranges = {
            'laserpower': {'min': 1.0, 'max': 4.5, 'rec_min': 2.5, 'rec_max': 4.0},
            'gender': {'min': 1, 'max': 2},
            'age': {'min': 16.0, 'max': 86.0, 'rec_min': 18.0, 'rec_max': 50.0},
            'height': {'min': 150.0, 'max': 196.0}
        }
    
    def _find_params_file(self):
        """Automatically find parameter file"""
        candidates = [
            'extracted_model_params.json',
            Path(__file__).parent / 'extracted_model_params.json',
            Path(__file__).parent.parent / 'extracted_model_params.json',
            Path.cwd() / 'extracted_model_params.json',
        ]
        
        for candidate in candidates:
            if Path(candidate).exists():
                return str(candidate)
        
        return 'extracted_model_params.json'
    
    def _prepare_inscalers(self):
        """
        Prepare inscaler for each feature
        
        Priority:
        1. If custom_inscaler is provided:
           - If it's a dictionary (per-feature values), use feature-specific values
           - If it's a single value (globally shared), all features use the same value
        2. Otherwise use inscaler from model parameters (per-feature independent)
        3. If none available, use global default values
        
        Important: Each feature may have different inscalers (due to different outlier removal)
        """
        self.inscalers = {}
        
        # Global default inscaler (as fallback only)
        # These values are from normative_model.json inscalers (training set statistics)
        # Corresponding order: laserpower, gender, age, height
        # WARNING: These should match the inscaler used during model training!
        default_inscaler = {
            'mean': np.array([3.2838, 1.5513, 27.3447, 169.9180]),
            'std': np.array([0.6542, 0.4974, 12.2390, 8.5932])
        }
        
        # Check custom_inscaler type
        if self.custom_inscaler is not None:
            # Check if it's a dictionary (per-feature independent)
            if isinstance(self.custom_inscaler, dict):
                # Check if it contains feature names as keys
                has_feature_keys = any(f in self.custom_inscaler 
                                      for f in self.feature_names)
                
                if has_feature_keys:
                    # Per-feature inscaler
                    print("Using feature-specific inscaler:")
                    for feature in self.feature_names:
                        if feature in self.custom_inscaler:
                            inscaler = self.custom_inscaler[feature]
                            self.inscalers[feature] = {
                                'mean': np.array(inscaler['mean']),
                                'std': np.array(inscaler['std'])
                            }
                            print(f"  ✓ {feature}: Using custom inscaler")
                        else:
                            # Not provided for this feature, use default
                            self.inscalers[feature] = default_inscaler
                            print(f"  ⚠️  {feature}: Not provided, using default inscaler")
                else:
                    # Globally shared inscaler
                    print("Using globally shared inscaler:")
                    print(f"  mean: {self.custom_inscaler['mean']}")
                    print(f"  std: {self.custom_inscaler['std']}")
                    shared_inscaler = {
                        'mean': np.array(self.custom_inscaler['mean']),
                        'std': np.array(self.custom_inscaler['std'])
                    }
                    for feature in self.feature_names:
                        self.inscalers[feature] = shared_inscaler
            else:
                raise ValueError("custom_inscaler must be a dictionary type")
        else:
            # Read from model parameters (per-feature independent)
            print("Reading inscaler from model parameters...")
            
            has_any_inscaler = False
            for feature in self.feature_names:
                params = self.model_params[feature]
                inscaler = params.get('inscaler', {})
                
                # Check if inscaler is valid
                mean = inscaler.get('mean', [])
                std = inscaler.get('std', [])
                
                if isinstance(mean, list) and len(mean) == 4 and \
                   isinstance(std, list) and len(std) == 4:
                    # Use feature-specific inscaler
                    self.inscalers[feature] = {
                        'mean': np.array(mean),
                        'std': np.array(std)
                    }
                    print(f"  ✓ {feature}: Using inscaler from model parameters")
                    has_any_inscaler = True
                else:
                    # Use default inscaler
                    self.inscalers[feature] = default_inscaler
                    print(f"  ⚠️  {feature}: No inscaler in parameters, using default")
            
            if not has_any_inscaler:
                print(f"  ⚠️  Warning: No inscalers found in model parameters, using global default")
    
    def _validate_inscalers(self):
        """Validate inscaler values are reasonable"""
        # Expected approximate ranges based on normative_model.json
        expected_ranges = {
            0: ('laserpower_std', 0.3, 1.5),   # std for laserpower
            1: ('gender_std', 0.3, 0.6),        # std for gender
            2: ('age_std', 5.0, 30.0),           # std for age - MUST be large (>5)
            3: ('height_std', 4.0, 15.0),        # std for height
        }
        
        for feature in self.feature_names:
            ins = self.inscalers[feature]
            for idx, (name, low, high) in expected_ranges.items():
                std_val = ins['std'][idx]
                if std_val < low or std_val > high:
                    print(f"  ⚠️  WARNING [{feature}]: {name}={std_val:.4f} "
                          f"outside expected range [{low}, {high}]")
                    if idx == 2 and std_val < 5.0:
                        print(f"     ❌ age_std is suspiciously small! "
                              f"This will cause severe prediction bias for older/younger subjects.")
                        print(f"     Expected ~12.24 (from normative_model.json), got {std_val:.4f}")
    
    def check_input(self, laserpower, gender, age, height, show_warnings=True):
        """
        Check input validity and return warnings
        
        Returns:
            (warnings_list, severity)
            severity: 'none', 'info', 'warning', 'error'
        """
        warnings_list = []
        severity = 'none'
        
        # Check laser power
        if laserpower < self.ranges['laserpower']['min'] or \
           laserpower > self.ranges['laserpower']['max']:
            warnings_list.append(
                f"⚠️  Laser power {laserpower} outside valid range "
                f"({self.ranges['laserpower']['min']}-{self.ranges['laserpower']['max']})"
            )
            severity = 'error'
        elif laserpower < self.ranges['laserpower']['rec_min'] or \
             laserpower > self.ranges['laserpower']['rec_max']:
            warnings_list.append(
                f"ℹ️  Laser power {laserpower} outside recommended range "
                f"({self.ranges['laserpower']['rec_min']}-{self.ranges['laserpower']['rec_max']})"
            )
            if severity == 'none':
                severity = 'info'
        
        # Check gender
        if gender not in [1, 2]:
            warnings_list.append(f"⚠️  Gender must be 1 (male) or 2 (female), got {gender}")
            severity = 'error'
        
        # Check age
        if age < self.ranges['age']['min'] or age > self.ranges['age']['max']:
            warnings_list.append(
                f"⚠️  Age {age} outside valid range "
                f"({self.ranges['age']['min']}-{self.ranges['age']['max']})"
            )
            severity = 'error'
        elif age < self.ranges['age']['rec_min'] or age > self.ranges['age']['rec_max']:
            warnings_list.append(
                f"ℹ️  Age {age} outside recommended range "
                f"({self.ranges['age']['rec_min']}-{self.ranges['age']['rec_max']}), "
                "prediction may be less accurate"
            )
            if severity not in ['error']:
                severity = 'warning'
        
        # Check height
        if height < self.ranges['height']['min'] or height > self.ranges['height']['max']:
            warnings_list.append(
                f"⚠️  Height {height} outside valid range "
                f"({self.ranges['height']['min']}-{self.ranges['height']['max']})"
            )
            severity = 'error'
        
        if show_warnings and warnings_list:
            print("\nInput validation warnings:")
            for warning in warnings_list:
                print(f"  {warning}")
        
        return warnings_list, severity
    
    def predict_single_feature(self, feature_name, laserpower, gender, age, height):
        """
        Predict a single feature.

        Supports two likelihood families:
          - 'SHASHb' (Sinh-Arcsinh): amp and mag features
                Parameters: mu, sigma, epsilon (skewness), delta (tail weight)
                epsilon and delta are global scalar posteriors (no B-spline regression).
                95% CI uses exact SHASH quantile formula.
                Z-score = delta * arcsinh((y - mu) / sigma) - epsilon
          - 'normal' (Gaussian): latency features
                Parameters: mu, sigma
                95% CI = mu +/- 1.96 * sigma
                Z-score = (y - mu) / sigma

        Returns:
            dict with keys:
                'mean'      : predicted mean (original scale)
                'std'       : sigma parameter (original scale)
                'lower_95'  : 2.5th percentile (original scale)
                'upper_95'  : 97.5th percentile (original scale)
            SHASHb only (additional keys):
                'epsilon'   : skewness parameter (after mapping)
                'delta'     : tail-weight parameter (after mapping, always > 0)
        """
        if feature_name not in self.model_params:
            raise ValueError(f"Feature {feature_name} not found in model parameters")

        params = self.model_params[feature_name]
        likelihood_cat = params.get('likelihood_category', 'normal')

        # ------------------------------------------------------------------
        # 1. Prepare and standardize input covariates
        # ------------------------------------------------------------------
        X_raw = np.array([laserpower, gender, age, height])
        inscaler = self.inscalers[feature_name]
        X_std = (X_raw - inscaler['mean']) / inscaler['std']

        if self.debug:
            print(f"\n  [DEBUG {feature_name}] [{likelihood_cat}]")
            print(f"    X_raw:         {X_raw}")
            print(f"    inscaler mean: {inscaler['mean']}")
            print(f"    inscaler std:  {inscaler['std']}")
            print(f"    X_std:         {X_std}")

        # ------------------------------------------------------------------
        # 2. Helper: build B-spline expanded covariate vector
        # ------------------------------------------------------------------
        def _expand_covariates(basis_config):
            """
            Replace the designated covariate column with its B-spline basis,
            keep all other columns as scalar values.
            Returns the expanded vector, padded to expected_dim if needed.
            """
            col = basis_config['basis_column'][0]
            knots_raw = basis_config['knots']
            knots = knots_raw[str(col)] if isinstance(knots_raw, dict) else knots_raw
            basis = create_bspline_basis(X_std[col], knots, degree=basis_config['degree'])

            expanded = []
            for i, x_val in enumerate(X_std):
                if i == col:
                    expanded.extend(basis)
                else:
                    expanded.append(x_val)
            expanded = np.array(expanded)

            expected_dim = len(params['covariate_dims'])
            if len(expanded) < expected_dim:
                expanded = np.append(expanded, [0] * (expected_dim - len(expanded)))
            return expanded

        # ------------------------------------------------------------------
        # 3. Compute mu (both likelihoods)
        # ------------------------------------------------------------------
        X_mu = _expand_covariates(params['mu_basis'])
        slope_mu     = np.array(params['posterior']['slope_mu']['mean'])
        intercept_mu = float(params['posterior']['intercept_mu']['mean'])  # key fixed from v4
        mu_linear    = np.dot(X_mu, slope_mu) + intercept_mu
        mu_std_val   = apply_mapping(mu_linear, params.get('mu_mapping', {'type': 'identity'}))

        if self.debug:
            print(f"    X_mu dim:      {len(X_mu)} (expected: {len(params['covariate_dims'])})")
            print(f"    mu_std:        {mu_std_val:.4f}")

        # ------------------------------------------------------------------
        # 4. Compute sigma (both likelihoods)
        # ------------------------------------------------------------------
        X_sigma       = _expand_covariates(params['sigma_basis'])
        slope_sigma     = np.array(params['posterior']['slope_sigma']['mean'])
        intercept_sigma = float(params['posterior']['intercept_sigma']['mean'])
        sigma_linear    = np.dot(X_sigma, slope_sigma) + intercept_sigma
        sigma_std_val   = apply_mapping(sigma_linear, params.get('sigma_mapping',
                                        {'type': 'softplus', 'params': [0.0, 3.0]}))

        if self.debug:
            print(f"    sigma_std:     {sigma_std_val:.4f}")

        # ------------------------------------------------------------------
        # 5. Inverse standardize to original scale
        # ------------------------------------------------------------------
        outscaler     = params['outscaler']
        mu_original   = mu_std_val * outscaler['std'] + outscaler['mean']
        sigma_original = sigma_std_val * outscaler['std']

        if self.debug:
            print(f"    outscaler:     mean={outscaler['mean']:.4f}, std={outscaler['std']:.4f}")
            print(f"    mu_original:   {mu_original:.4f}")
            print(f"    sigma_original:{sigma_original:.4f}")

        # ------------------------------------------------------------------
        # 6a. SHASHb: compute epsilon/delta, SHASH-exact CI
        # ------------------------------------------------------------------
        if likelihood_cat == 'SHASHb':
            # epsilon (skewness): scalar posterior, identity mapping
            epsilon_raw = float(params['posterior']['epsilon']['mean'])
            epsilon = apply_mapping(epsilon_raw,
                                    params.get('epsilon_mapping', {'type': 'identity'}))

            # delta (tail weight): scalar posterior, 3-param softplus keeps delta > offset
            delta_raw = float(params['posterior']['delta']['mean'])
            delta = apply_mapping(delta_raw,
                                  params.get('delta_mapping',
                                             {'type': 'softplus', 'params': [0.0, 3.0]}))

            if self.debug:
                print(f"    epsilon (raw/mapped): {epsilon_raw:.4f} / {epsilon:.4f}")
                print(f"    delta   (raw/mapped): {delta_raw:.4f} / {delta:.4f}")

            # Exact 95% PI via SHASH quantile function:
            #   Q(p) = mu + sigma * sinh((arcsinh(z_p) + epsilon) / delta)
            lower_95 = float(shash_quantile(mu_original, sigma_original, epsilon, delta, 0.025))
            upper_95 = float(shash_quantile(mu_original, sigma_original, epsilon, delta, 0.975))

            return {
                'mean':     float(mu_original),
                'std':      float(sigma_original),   # sigma scale parameter
                'lower_95': lower_95,
                'upper_95': upper_95,
                'epsilon':  float(epsilon),
                'delta':    float(delta),
            }

        # ------------------------------------------------------------------
        # 6b. Normal: symmetric Gaussian CI
        # ------------------------------------------------------------------
        else:
            lower_95 = mu_original - 1.96 * sigma_original
            upper_95 = mu_original + 1.96 * sigma_original

            return {
                'mean':     float(mu_original),
                'std':      float(sigma_original),
                'lower_95': float(lower_95),
                'upper_95': float(upper_95),
            }
    
    def predict(self, laserpower, gender, age, height, show_warnings=True):
        """
        Predict all features
        
        Returns:
            dict: {feature_name: {'mean': ..., 'std': ..., 'lower_95': ..., 'upper_95': ...}}
        """
        # Check input
        warnings_list, severity = self.check_input(laserpower, gender, age, height, show_warnings)
        
        if severity == 'error':
            raise ValueError("Input contains errors, cannot proceed with prediction")
        
        # Predict each feature
        results = {}
        for feature in self.feature_names:
            try:
                results[feature] = self.predict_single_feature(
                    feature, laserpower, gender, age, height
                )
            except Exception as e:
                print(f"  ❌ Error predicting {feature}: {e}")
                results[feature] = None
        
        return results
    
    def predict_with_observations(self, laserpower, gender, age, height, observations, show_warnings=True):
        """
        Predict and calculate Z-scores based on observed values
        
        Parameters:
            laserpower, gender, age, height: Input covariates
            observations: dict of observed values, e.g. {'N1_amp': -10.5, 'N2_amp': -18.2}
            show_warnings: Whether to show input validation warnings
        
        Returns:
            dict: includes prediction and Z-scores
        """
        # Get predictions
        predictions = self.predict(laserpower, gender, age, height, show_warnings=show_warnings)
        
        # Calculate Z-scores
        for feature, pred in predictions.items():
            if pred is not None and feature in observations:
                observed = observations[feature]
                params = self.model_params.get(feature, {})
                likelihood_cat = params.get('likelihood_category', 'normal')

                if likelihood_cat == 'SHASHb':
                    # SHASHb Z-score: delta * arcsinh((y - mu) / sigma) - epsilon
                    # Equivalent to the standard-normal quantile of F(y) under SHASH.
                    epsilon = pred.get('epsilon', 0.0)
                    delta   = pred.get('delta', 1.0)
                    z_score = float(shash_zscore(observed, pred['mean'], pred['std'],
                                                 epsilon, delta))
                else:
                    # Normal Z-score
                    z_score = (observed - pred['mean']) / pred['std']

                pred['observed'] = observed
                pred['z_score']  = z_score
        
        return predictions
    
    def get_feature_info(self):
        """Get information about all available features"""
        info = []
        for feature in self.feature_names:
            params = self.model_params[feature]
            info.append({
                'feature': feature,
                'model_type': params['model_type'],
                'outscaler_mean': params['outscaler']['mean'],
                'outscaler_std': params['outscaler']['std'],
                'n_covariates': len(params['covariate_names']),
                'n_batch_effects': len(params.get('batch_effects_names', []))
            })
        return pd.DataFrame(info)
    
    def get_training_means(self):
        """
        Get training set mean values for imputation.
        Uses inscaler means (which are the training set means before standardization).
        Returns average across all features' inscalers.
        """
        all_means = np.array([self.inscalers[f]['mean'] for f in self.feature_names])
        avg_mean = np.mean(all_means, axis=0)
        return {
            'laserpower': avg_mean[0],
            'gender': avg_mean[1],  # ~1.55, encodes gender ratio
            'age': avg_mean[2],
            'height': avg_mean[3]
        }
    
    def get_gender_ratio(self):
        """
        Get female proportion in training set from inscaler gender mean.
        gender coding: 1=male, 2=female
        mean = 1*p_male + 2*p_female = 1*(1-p_f) + 2*p_f = 1 + p_f
        So p_female = mean - 1
        """
        training_means = self.get_training_means()
        p_female = training_means['gender'] - 1.0
        return {'male': 1.0 - p_female, 'female': p_female}
    
    def predict_gender_averaged(self, laserpower, age, height):
        """
        Predict using weighted average of male and female predictions.
        Used when gender is missing (NaN).
        
        Weights are proportional to the gender ratio in the training set.
        For each feature, the averaged prediction is:
            mu = w_male * mu_male + w_female * mu_female
            sigma = sqrt(w_male * sigma_male^2 + w_female * sigma_female^2 
                         + w_male * w_female * (mu_male - mu_female)^2)
        (This accounts for both within-group variance and between-group mean difference)
        """
        ratio = self.get_gender_ratio()
        w_m, w_f = ratio['male'], ratio['female']
        
        results = {}
        for feature in self.feature_names:
            try:
                pred_m = self.predict_single_feature(feature, laserpower, 1, age, height)
                pred_f = self.predict_single_feature(feature, laserpower, 2, age, height)
                
                # Weighted mean
                mu = w_m * pred_m['mean'] + w_f * pred_f['mean']
                
                # Combined variance (law of total variance)
                var_within = w_m * pred_m['std']**2 + w_f * pred_f['std']**2
                var_between = w_m * w_f * (pred_m['mean'] - pred_f['mean'])**2
                sigma = np.sqrt(var_within + var_between)
                
                results[feature] = {
                    'mean': float(mu),
                    'std': float(sigma),
                    'lower_95': float(mu - 1.96 * sigma),
                    'upper_95': float(mu + 1.96 * sigma)
                }
            except Exception as e:
                results[feature] = None
        
        return results


def print_results(results, laserpower, gender, age, height):
    """Pretty print prediction results"""
    if not results:
        print("No prediction results available")
        return
    
    print("\n" + "="*70)
    print("📊 Prediction Results")
    print("="*70)
    
    print("\nInput:")
    print(f"  Laser Power: {laserpower}")
    print(f"  Gender: {gender} ({'Male' if gender == 1 else 'Female'})")
    print(f"  Age: {age} years")
    print(f"  Height: {height} cm")
    
    # Group features by category
    amplitude_features = [f for f in results.keys() if '_amp' in f]
    latency_features = [f for f in results.keys() if '_latency' in f]
    magnitude_features = [f for f in results.keys() if '_mag' in f]
    
    # Print amplitudes
    if amplitude_features:
        print("\n" + "-"*70)
        print("Amplitudes (μV):")
        print("-"*70)
        for feature in amplitude_features:
            pred = results[feature]
            if pred:
                print(f"\n{feature}:")
                print(f"  Predicted: {pred['mean']:.2f} ± {pred['std']:.2f} μV")
                print(f"  95% CI: [{pred['lower_95']:.2f}, {pred['upper_95']:.2f}] μV")
                if 'z_score' in pred:
                    print(f"  Observed: {pred['observed']:.2f} μV")
                    print(f"  Z-score: {pred['z_score']:.2f}")
    
    # Print latencies
    if latency_features:
        print("\n" + "-"*70)
        print("Latencies (ms):")
        print("-"*70)
        for feature in latency_features:
            pred = results[feature]
            if pred:
                print(f"\n{feature}:")
                print(f"  Predicted: {pred['mean']:.2f} ± {pred['std']:.2f} ms")
                print(f"  95% CI: [{pred['lower_95']:.2f}, {pred['upper_95']:.2f}] ms")
                if 'z_score' in pred:
                    print(f"  Observed: {pred['observed']:.2f} ms")
                    print(f"  Z-score: {pred['z_score']:.2f}")
    
    # Print magnitudes
    if magnitude_features:
        print("\n" + "-"*70)
        print("Magnitudes (μV²/Hz):")
        print("-"*70)
        for feature in magnitude_features:
            pred = results[feature]
            if pred:
                print(f"\n{feature}:")
                print(f"  Predicted: {pred['mean']:.2f} ± {pred['std']:.2f}")
                print(f"  95% CI: [{pred['lower_95']:.2f}, {pred['upper_95']:.2f}]")
                if 'z_score' in pred:
                    print(f"  Observed: {pred['observed']:.2f}")
                    print(f"  Z-score: {pred['z_score']:.2f}")
    
    print("\n" + "="*70)


def interactive_mode(predictor):
    """Interactive prediction mode"""
    print("\n" + "="*70)
    print("🎯 Interactive Prediction Mode")
    print("="*70)
    print("\nCommands: 'q' = quit, 'info' = feature info")
    print("Or press Enter to start a new prediction")
    
    while True:
        try:
            print("\n" + "-"*70)
            user_input = input("\nPress Enter to predict (or type command): ").strip()
            
            if user_input.lower() == 'q':
                print("Goodbye!")
                break
            
            if user_input.lower() == 'info':
                print("\n📋 Available Features:")
                info_df = predictor.get_feature_info()
                print(info_df.to_string(index=False))
                continue
            
            # Try to parse as "laserpower gender age height" on one line
            if user_input:
                parts = user_input.split()
                if len(parts) == 4:
                    try:
                        laserpower = float(parts[0])
                        gender = int(float(parts[1]))
                        age = float(parts[2])
                        height = float(parts[3])
                    except ValueError:
                        print(f"\n❌ Cannot parse '{user_input}' as 4 parameters")
                        print("   Format: laserpower gender age height (e.g., 3.5 1 25 170)")
                        print("   Or press Enter and input each parameter separately")
                        continue
                else:
                    print(f"\n❌ Expected 4 parameters, got {len(parts)}")
                    print("   Format: laserpower gender age height (e.g., 3.5 1 25 170)")
                    print("   Or press Enter and input each parameter separately")
                    continue
            else:
                # Interactive step-by-step input
                print("\nPlease enter the following parameters:")
                lp_input = input("  Laser power (1.0-4.5): ").strip()
                if not lp_input:
                    print("  Cancelled.")
                    continue
                laserpower = float(lp_input)
                gender = int(input("  Gender (1=male, 2=female): "))
                age = float(input("  Age (years): "))
                height = float(input("  Height (cm): "))
            
            # Ask if there are observations
            has_obs = input("\nDo you have observed values? (y/n): ").strip().lower() == 'y'
            
            if has_obs:
                observations = {}
                print("\nEnter observed values (press Enter to skip):")
                for feature in predictor.feature_names:
                    obs_input = input(f"  {feature}: ").strip()
                    if obs_input:
                        try:
                            observations[feature] = float(obs_input)
                        except ValueError:
                            print(f"    Invalid value, skipping {feature}")
                
                results = predictor.predict_with_observations(
                    laserpower, gender, age, height, observations
                )
            else:
                results = predictor.predict(laserpower, gender, age, height)
            
            # Print results
            print_results(results, laserpower, gender, age, height)
            
            # Explain Z-scores if observations were provided
            if has_obs:
                print("\nZ-score Interpretation:")
                print("  |Z| < 1.96: Within 95% confidence interval (normal)")
                print("  |Z| > 1.96: Outside 95% confidence interval (abnormal)")
                print("  |Z| > 2.58: Outside 99% confidence interval (highly abnormal)")
            
        except ValueError as e:
            print(f"\n❌ Input error: {e}")
            print("Please try again with valid numbers.")
        except KeyboardInterrupt:
            print("\n\nInterrupted, goodbye!")
            break
        except EOFError:
            print("\n\nEOF detected, goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again.")


def batch_mode(predictor, input_file, output_file):
    """
    Batch prediction mode
    
    Input CSV format: Must contain columns laserpower, gender, age, height
    Optional columns: Observed values for features (for Z-score calculation)
    """
    print("\n" + "="*70)
    print("📁 Batch Prediction Mode")
    print("="*70)
    
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"\n❌ Failed to read file: {e}")
        return
    
    # Check required columns
    required_cols = ['laserpower', 'gender', 'age', 'height']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"❌ Missing required columns: {missing}")
        return
    
    print(f"\n✓ Found {len(df)} samples")
    
    # Check for observation columns
    has_observations = any(f in df.columns for f in predictor.feature_names)
    if has_observations:
        print("✓ Observation columns detected, will calculate Z-scores")
    
    print("Starting prediction...\n")
    
    all_results = []
    warnings_count = 0
    error_count = 0
    error_samples = []
    
    for idx, row in df.iterrows():
        # Check input validity
        warnings_list, severity = predictor.check_input(
            row['laserpower'], row['gender'], row['age'], row['height']
        )
        
        if warnings_list:
            warnings_count += 1
        
        # Skip if severity is 'error' (invalid input)
        if severity == 'error':
            error_count += 1
            error_samples.append({
                'index': idx,
                'laserpower': row['laserpower'],
                'gender': row['gender'],
                'age': row['age'],
                'height': row['height'],
                'error': 'Invalid input parameters'
            })
            print(f"⚠️  Skipping sample {idx} due to invalid input")
            continue
        
        # Prepare observations (if available)
        observations = {}
        if has_observations:
            for feature in predictor.feature_names:
                if feature in df.columns and not pd.isna(row[feature]):
                    observations[feature] = row[feature]
        
        # Predict (with error handling)
        try:
            if observations:
                results = predictor.predict_with_observations(
                    row['laserpower'], row['gender'], row['age'], row['height'],
                    observations, show_warnings=False
                )
            else:
                results = predictor.predict(
                    row['laserpower'], row['gender'], row['age'], row['height'],
                    show_warnings=False
                )
        except Exception as e:
            error_count += 1
            error_samples.append({
                'index': idx,
                'laserpower': row['laserpower'],
                'gender': row['gender'],
                'age': row['age'],
                'height': row['height'],
                'error': str(e)
            })
            print(f"❌ Error predicting sample {idx}: {e}")
            continue
        
        if results:
            # Build output row
            flat_result = {
                'index': idx,
                'laserpower': row['laserpower'],
                'gender': row['gender'],
                'age': row['age'],
                'height': row['height'],
                'has_warnings': len(warnings_list) > 0,
                'warning_severity': severity
            }
            
            # Add prediction results
            for feature, pred in results.items():
                flat_result[f'{feature}_pred_mean'] = pred['mean']
                flat_result[f'{feature}_pred_std'] = pred['std']
                flat_result[f'{feature}_pred_lower95'] = pred['lower_95']
                flat_result[f'{feature}_pred_upper95'] = pred['upper_95']
                
                # If observed values and Z-scores available
                if 'observed' in pred:
                    flat_result[f'{feature}_observed'] = pred['observed']
                if 'z_score' in pred:
                    flat_result[f'{feature}_z_score'] = pred['z_score']
            
            all_results.append(flat_result)
        
        # Show progress
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(df)} samples...")
    
    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_file, index=False)
    
    # Summary
    print(f"\n" + "="*70)
    print("✅ Batch Prediction Complete!")
    print("="*70)
    print(f"  Input samples: {len(df)}")
    print(f"  Successful predictions: {len(all_results)}")
    print(f"  Failed/skipped samples: {error_count}")
    if warnings_count > 0:
        print(f"  Samples with warnings: {warnings_count}")
    
    if all_results:
        print(f"\n✓ Results saved to: {output_file}")
    
    # Report error samples if any
    if error_samples:
        print(f"\n⚠️  Failed/Skipped Samples ({len(error_samples)}):")
        for err in error_samples[:10]:  # Show first 10
            print(f"  - Index {err['index']}: age={err['age']}, error={err['error']}")
        if len(error_samples) > 10:
            print(f"  ... and {len(error_samples) - 10} more (check input file)")
        
        # Save error log
        error_log_file = output_file.replace('.csv', '_errors.csv')
        error_df = pd.DataFrame(error_samples)
        error_df.to_csv(error_log_file, index=False)
        print(f"\n✓ Error log saved to: {error_log_file}")
    
    print("="*70)
    
    # Statistics for abnormal Z-scores (if Z-scores were calculated)
    if all_results and has_observations:
        results_df = pd.DataFrame(all_results)
        z_cols = [col for col in results_df.columns if col.endswith('_z_score')]
        if z_cols:
            print(f"\nZ-score Statistics:")
            for z_col in z_cols:
                feature = z_col.replace('_z_score', '')
                z_values = results_df[z_col].dropna()
                if len(z_values) > 0:
                    n_abnormal = (z_values.abs() > 1.96).sum()
                    print(f"  {feature}: {n_abnormal}/{len(z_values)} samples abnormal (|Z| > 1.96)")
            print("="*70)


def csv_batch_mode(predictor, input_file, output_file):
    """
    CSV batch prediction with NaN handling.
    
    Input CSV format:
        Required column: laserpower
        Optional columns: gender, age, height
        - age/height NaN → imputed with training set mean
        - gender NaN → weighted average of male/female predictions
    
    Output CSV format:
        Input columns + predicted mean, std, 95% CI for all features
    """
    print("\n" + "="*70)
    print("📁 CSV Batch Prediction Mode")
    print("="*70)
    
    # Read input
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"\n❌ Failed to read file: {e}")
        return
    
    # Check laserpower column (required)
    if 'laserpower' not in df.columns:
        print("❌ Missing required column: 'laserpower'")
        print(f"   Found columns: {df.columns.tolist()}")
        return
    
    n_samples = len(df)
    print(f"\n✓ Loaded {n_samples} samples from: {input_file}")
    
    # Get training means for imputation
    training_means = predictor.get_training_means()
    gender_ratio = predictor.get_gender_ratio()
    
    # --- Handle missing columns and NaN values ---
    
    # If column doesn't exist at all, create it with NaN
    for col in ['gender', 'age', 'height']:
        if col not in df.columns:
            df[col] = np.nan
            print(f"  ℹ️  Column '{col}' not found, will use default values")
    
    # Track imputation statistics
    n_age_imputed = 0
    n_height_imputed = 0
    n_gender_missing = 0
    
    # Impute age NaN
    age_nan_mask = df['age'].isna()
    if age_nan_mask.any():
        n_age_imputed = age_nan_mask.sum()
        df.loc[age_nan_mask, 'age'] = training_means['age']
        print(f"  ℹ️  Age: {n_age_imputed} missing values imputed with training mean ({training_means['age']:.1f} years)")
    
    # Impute height NaN
    height_nan_mask = df['height'].isna()
    if height_nan_mask.any():
        n_height_imputed = height_nan_mask.sum()
        df.loc[height_nan_mask, 'height'] = training_means['height']
        print(f"  ℹ️  Height: {n_height_imputed} missing values imputed with training mean ({training_means['height']:.1f} cm)")
    
    # Gender NaN: mark for special handling (don't impute, predict averaged)
    gender_nan_mask = df['gender'].isna()
    n_gender_missing = gender_nan_mask.sum()
    if n_gender_missing > 0:
        print(f"  ℹ️  Gender: {n_gender_missing} missing values → will use population-weighted average")
        print(f"       (male weight: {gender_ratio['male']:.2%}, female weight: {gender_ratio['female']:.2%})")
    
    # Check for observation columns (for Z-score calculation)
    obs_features = [f for f in predictor.feature_names if f in df.columns]
    has_observations = len(obs_features) > 0
    if has_observations:
        print(f"  ✓ Observation columns detected: {obs_features}")
        print(f"    Will calculate Z-scores for these features")
    
    # Validate laserpower range
    lp_min = df['laserpower'].min()
    lp_max = df['laserpower'].max()
    lp_nan = df['laserpower'].isna().sum()
    if lp_nan > 0:
        print(f"  ⚠️  Warning: {lp_nan} rows have missing laserpower, will be skipped")
    print(f"  Laserpower range: [{lp_min:.2f}, {lp_max:.2f}]")
    
    print(f"\nStarting prediction for {n_samples} samples...\n")
    
    # --- Predict ---
    all_results = []
    error_count = 0
    error_samples = []
    
    for idx, row in df.iterrows():
        # Skip if laserpower is NaN
        if pd.isna(row['laserpower']):
            error_count += 1
            error_samples.append({'index': idx, 'error': 'laserpower is NaN'})
            continue
        
        laserpower = float(row['laserpower'])
        age = float(row['age'])
        height = float(row['height'])
        gender_is_nan = pd.isna(row['gender'])
        
        try:
            if gender_is_nan:
                # Gender missing → weighted average of male/female
                results = predictor.predict_gender_averaged(laserpower, age, height)
                gender_display = 'averaged'
            else:
                gender = int(row['gender'])
                if gender not in [1, 2]:
                    error_count += 1
                    error_samples.append({'index': idx, 'error': f'Invalid gender: {gender}'})
                    continue
                results = predictor.predict(laserpower, gender, age, height, show_warnings=False)
                gender_display = gender
            
            # Calculate Z-scores if observations available
            if has_observations and results:
                for feature in obs_features:
                    if feature in results and results[feature] is not None:
                        obs_val = row[feature]
                        if not pd.isna(obs_val):
                            pred = results[feature]
                            pred['observed'] = float(obs_val)
                            pred['z_score'] = (float(obs_val) - pred['mean']) / pred['std']
        
        except Exception as e:
            error_count += 1
            error_samples.append({'index': idx, 'error': str(e)})
            continue
        
        if results:
            # Build output row
            flat_result = {
                'index': idx,
                'laserpower': laserpower,
                'gender': row['gender'] if not gender_is_nan else np.nan,
                'gender_method': 'averaged' if gender_is_nan else 'direct',
                'age': age,
                'age_imputed': bool(age_nan_mask.iloc[idx]) if idx < len(age_nan_mask) else False,
                'height': height,
                'height_imputed': bool(height_nan_mask.iloc[idx]) if idx < len(height_nan_mask) else False,
            }
            
            # Add predictions for each feature
            for feature in predictor.feature_names:
                pred = results.get(feature)
                if pred is not None:
                    flat_result[f'{feature}_pred_mean'] = pred['mean']
                    flat_result[f'{feature}_pred_std'] = pred['std']
                    flat_result[f'{feature}_pred_lower95'] = pred['lower_95']
                    flat_result[f'{feature}_pred_upper95'] = pred['upper_95']
                    if 'observed' in pred:
                        flat_result[f'{feature}_observed'] = pred['observed']
                    if 'z_score' in pred:
                        flat_result[f'{feature}_z_score'] = pred['z_score']
                else:
                    flat_result[f'{feature}_pred_mean'] = np.nan
                    flat_result[f'{feature}_pred_std'] = np.nan
                    flat_result[f'{feature}_pred_lower95'] = np.nan
                    flat_result[f'{feature}_pred_upper95'] = np.nan
            
            all_results.append(flat_result)
        
        # Progress
        if (idx + 1) % 100 == 0 or (idx + 1) == n_samples:
            print(f"  Processed {idx + 1}/{n_samples} samples...")
    
    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_file, index=False, float_format='%.4f')
    
    # Summary
    print(f"\n" + "="*70)
    print("✅ CSV Batch Prediction Complete!")
    print("="*70)
    print(f"  Input samples:           {n_samples}")
    print(f"  Successful predictions:  {len(all_results)}")
    print(f"  Failed/skipped:          {error_count}")
    if n_age_imputed > 0:
        print(f"  Age imputed:             {n_age_imputed} (→ {training_means['age']:.1f})")
    if n_height_imputed > 0:
        print(f"  Height imputed:          {n_height_imputed} (→ {training_means['height']:.1f})")
    if n_gender_missing > 0:
        print(f"  Gender averaged:         {n_gender_missing}")
    
    if all_results:
        print(f"\n✓ Results saved to: {output_file}")
        
        # Show preview
        results_df = pd.DataFrame(all_results)
        pred_cols = [c for c in results_df.columns if c.endswith('_pred_mean')]
        if pred_cols:
            print(f"\nPrediction summary (mean values):")
            for col in pred_cols:
                feature = col.replace('_pred_mean', '')
                unit = "μV" if "_amp" in feature else "ms" if "_latency" in feature else "μV²/Hz"
                vals = results_df[col].dropna()
                print(f"  {feature:15s}: mean={vals.mean():.2f}, "
                      f"range=[{vals.min():.2f}, {vals.max():.2f}] {unit}")
        
        # Z-score statistics if available
        z_cols = [c for c in results_df.columns if c.endswith('_z_score')]
        if z_cols:
            print(f"\nZ-score statistics:")
            for col in z_cols:
                feature = col.replace('_z_score', '')
                z_vals = results_df[col].dropna()
                if len(z_vals) > 0:
                    n_abnormal = (z_vals.abs() > 1.96).sum()
                    print(f"  {feature:15s}: mean={z_vals.mean():.2f}, std={z_vals.std():.2f}, "
                          f"|Z|>1.96: {n_abnormal}/{len(z_vals)}")
    
    # Error log
    if error_samples:
        print(f"\n⚠️  Failed samples ({len(error_samples)}):")
        for err in error_samples[:5]:
            print(f"  - Index {err['index']}: {err['error']}")
        if len(error_samples) > 5:
            print(f"  ... and {len(error_samples) - 5} more")
        
        error_log = output_file.replace('.csv', '_errors.csv')
        pd.DataFrame(error_samples).to_csv(error_log, index=False)
        print(f"  Error log: {error_log}")
    
    print("="*70)


def quick_mode(predictor, laserpower, gender, age, height):
    """Quick prediction mode (command line)"""
    results = predictor.predict(laserpower, gender, age, height)
    print_results(results, laserpower, gender, age, height)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="HBR Interactive Predictor - 10-Feature Model Version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:

1. Interactive mode (default):
   python normative_model_LEP_v4.py

2. Quick prediction:
   python normative_model_LEP_v4.py -q 3.5 1 21 170

3. CSV batch prediction (with NaN handling):
   python normative_model_LEP_v4.py -c input.csv -o output.csv

4. Legacy batch prediction:
   python normative_model_LEP_v4.py -b input.csv -o output.csv

5. Debug mode (show intermediate values):
   python normative_model_LEP_v4.py -d -q 3.5 1 65 170

6. Specify parameter file:
   python normative_model_LEP_v4.py -p extracted_model_params.json

CSV Input Format:
  Required: laserpower
  Optional: gender (1=male, 2=female), age, height
  - age/height NaN → imputed with training set mean
  - gender NaN → population-weighted average prediction

Covariate Instructions:
  • laserpower: Laser power (1.0-4.5, recommended 2.5-4.0)
  • gender: Gender (1=male, 2=female)
  • age: Age (recommended 18-25 years)
  • height: Height (150-190 cm)

Features (10 total):
  • Amplitudes: N1_amp, N2_amp, P2_amp (μV)
  • Latencies: N1_latency, N2_latency, P2_latency (ms)
  • Magnitudes: LEP_mag, alpha_mag, beta_mag, gamma_mag (μV²/Hz)
        """
    )
    
    parser.add_argument('-p', '--params', help='Path to parameter file')
    parser.add_argument('-q', '--quick', nargs=4, 
                       metavar=('POWER', 'GENDER', 'AGE', 'HEIGHT'),
                       help='Quick prediction mode')
    parser.add_argument('-b', '--batch', metavar='INPUT', help='Legacy batch prediction input file')
    parser.add_argument('-c', '--csv', metavar='INPUT', help='CSV batch prediction with NaN handling')
    parser.add_argument('-o', '--output', metavar='OUTPUT', help='Batch prediction output file')
    parser.add_argument('-d', '--debug', action='store_true', help='Show debug information')
    
    args = parser.parse_args()
    
    # Print welcome message
    print("\n" + "="*70)
    print("🚀 HBR Interactive Predictor - 10-Feature Model Version (v5)")
    print("="*70)
    
    # Load predictor
    try:
        predictor = HBRPredictorByFeature(args.params)
        if args.debug:
            predictor.debug = True
            print("\n🔍 Debug mode enabled - will show intermediate values")
        print(f"\n✓ Using parameter file: {predictor.params_file}")
        print(f"✓ Successfully loaded {len(predictor.feature_names)} feature models:")
        for i, feature in enumerate(predictor.feature_names, 1):
            unit = "μV" if "_amp" in feature else "ms" if "_latency" in feature else "μV²/Hz"
            print(f"   {i}. {feature} ({unit})")
    except Exception as e:
        print(f"\n❌ Loading failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Run according to mode
    if args.quick:
        try:
            laserpower = float(args.quick[0])
            gender = int(args.quick[1])
            age = float(args.quick[2])
            height = float(args.quick[3])
            quick_mode(predictor, laserpower, gender, age, height)
        except ValueError as e:
            print(f"\n❌ Parameter error: {e}")
            sys.exit(1)
    
    elif args.batch:
        if not args.output:
            print("❌ Batch mode requires specifying output file (-o)")
            sys.exit(1)
        batch_mode(predictor, args.batch, args.output)
    
    elif args.csv:
        output = args.output if args.output else args.csv.replace('.csv', '_predictions.csv')
        csv_batch_mode(predictor, args.csv, output)
    
    else:
        interactive_mode(predictor)


if __name__ == "__main__":
    main()
