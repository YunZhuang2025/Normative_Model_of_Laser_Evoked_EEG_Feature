#!/usr/bin/env python3
"""
Normative model for laser-evoked brain electrical features based on PCNtoolkit
Author: Yun Zhuang
Date: 2025-11-27
Version: v1.0
If you use this tool in published research, please cite:
Zhuang Y., Zhang L.B., Wang X.Q., Geng X.Y., & Hu L., (in preparation) From Normative Features to Multidimensional Estimation of Pain: A Large-Scale Study of Laser-Evoked Brain Responses.
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
    Softplus mapping function
    sigma = scale * log(1 + exp((x - shift) / scale))
    """
    shift, scale = params
    x_scaled = (x - shift) / scale
    # Clip to prevent overflow
    x_clipped = np.clip(x_scaled, -20, 20)
    return scale * np.log1p(np.exp(x_clipped))


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
    HBR Predictor - feature-wise training version
    
    Features:
    - Each feature has independent training data and standardization parameters
    - Automatically handles different inscalers for different features
    - Unified covariate input interface
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
        
        # Feature names (in expected order)
        feature_order = ['N1_amp', 'N2_amp', 'P2_amp', 
                        'ERP_mag', 'alpha_mag', 'beta_mag', 'gamma_mag']
        self.feature_names = [f for f in feature_order if f in self.model_params]
        
        if not self.feature_names:
            raise ValueError("No available feature models found!")
        
        # Prepare inscaler for each feature
        self._prepare_inscalers()
        
        # Covariate ranges (based on training data statistics)
        self.covariate_names = ['laserpower', 'gender', 'age', 'height']
        self.ranges = {
            'laserpower': {'min': 1.0, 'max': 4.5, 'rec_min': 2.5, 'rec_max': 4.0},
            'gender': {'min': 1, 'max': 2},
            'age': {'min': 16.0, 'max': 50.0, 'rec_min': 18.0, 'rec_max': 25.0},
            'height': {'min': 150.0, 'max': 190.0}
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
        
        # Global default inscaler (as fallback only, may not be accurate)
        # Corresponding order: laserpower, gender, age, height
        default_inscaler = {
            'mean': np.array([3.3427, 1.6024, 24.5392, 167.5705]),
            'std': np.array([0.6794, 0.4894, 3.4240, 7.1038])
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
                    # Use global default inscaler
                    self.inscalers[feature] = default_inscaler
                    print(f"  ⚠️  {feature}: No inscaler in model parameters, using default values")
            
            if not has_any_inscaler:
                print("\n⚠️  Warning: All features lack inscaler, using default values")
                print("   Suggestion: Run compute_inscalers_matlab.m to calculate accurate inscaler")
        
        print()
    
    def check_input(self, laserpower, gender, age, height):
        """
        Check input validity
        
        Returns:
            warnings: List of warning messages
            severity: Severity level ('none', 'low', 'medium', 'high')
        """
        warnings_list = []
        severity = 'none'
        
        # Check laser power
        if not (self.ranges['laserpower']['min'] <= laserpower <= self.ranges['laserpower']['max']):
            warnings_list.append(f"⚠️  Laser power {laserpower} out of range [{self.ranges['laserpower']['min']}, {self.ranges['laserpower']['max']}]")
            severity = 'high'
        elif not (self.ranges['laserpower']['rec_min'] <= laserpower <= self.ranges['laserpower']['rec_max']):
            warnings_list.append(f"ℹ️  Laser power {laserpower} is within valid range, but recommended range is [{self.ranges['laserpower']['rec_min']}, {self.ranges['laserpower']['rec_max']}]")
            if severity == 'none':
                severity = 'low'
        
        # Check gender
        if gender not in [1, 2]:
            warnings_list.append(f"⚠️  Gender value {gender} invalid (should be 1=male or 2=female)")
            severity = 'high'
        
        # Check age
        if not (self.ranges['age']['min'] <= age <= self.ranges['age']['max']):
            warnings_list.append(f"⚠️  Age {age} out of training range [{self.ranges['age']['min']}, {self.ranges['age']['max']}]")
            severity = 'high'
        elif not (self.ranges['age']['rec_min'] <= age <= self.ranges['age']['rec_max']):
            warnings_list.append(f"⚠️  Age {age} out of recommended range [{self.ranges['age']['rec_min']}, {self.ranges['age']['rec_max']}]")
            warnings_list.append(f"   Training data mainly concentrated around age 21, with fewer samples above 25")
            if severity == 'none':
                severity = 'medium'
        
        # Check height
        if not (self.ranges['height']['min'] <= height <= self.ranges['height']['max']):
            warnings_list.append(f"⚠️  Height {height}cm out of range [{self.ranges['height']['min']}, {self.ranges['height']['max']}]")
            if severity == 'none':
                severity = 'low'
        
        return warnings_list, severity
    
    def standardize_input(self, X, feature_name):
        """
        Standardize input (using feature-specific inscaler)
        
        Parameters:
            X: Raw input [laserpower, gender, age, height]
            feature_name: Feature name
        
        Returns:
            X_std: Standardized input
        """
        X = np.atleast_1d(X)
        inscaler = self.inscalers[feature_name]
        
        return (X - inscaler['mean']) / inscaler['std']
    
    def _expand_covariates(self, X_std, params):
        """
        Expand covariates (apply B-spline transformation)
        
        Parameters:
            X_std: Standardized covariates [laserpower_std, gender_std, age_std, height_std]
            params: Feature model parameters
        
        Returns:
            X_expanded: Expanded covariates (including intercept term)
            
        Note:
            - Dimensions of slope_mu/slope_sigma are (n_expanded_covariates,)
            - n_expanded_covariates = 1(intercept) + n_bspline_bases + n_other_covariates
            - Example: 1 + 7(B-spline bases for laserpower) + 3(gender, age, height) = 11
        """
        mu_basis_config = params.get('mu_basis', {})
        basis_column = mu_basis_config.get('basis_column', [])
        
        # Check if B-spline is applied to first covariate (laserpower)
        if 0 in basis_column:
            laserpower_std = X_std[0]
            
            # Get B-spline knots
            knots_dict = mu_basis_config.get('knots', {})
            if '0' in knots_dict:
                knots = np.array(knots_dict['0'])
            else:
                # If no predefined knots, generate default knots
                nknots = mu_basis_config.get('nknots', 5)
                degree = mu_basis_config.get('degree', 3)
                interior_knots = np.linspace(-2, 2, nknots)
                knots = np.concatenate([
                    np.repeat(interior_knots[0], degree),
                    interior_knots,
                    np.repeat(interior_knots[-1], degree)
                ])
            
            # Create B-spline basis functions
            degree = mu_basis_config.get('degree', 3)
            bspline_bases = create_bspline_basis(laserpower_std, knots, degree)
            
            # Expand: [1(intercept), B-spline_bases(7), gender, age, height]
            X_expanded = np.concatenate([
                [1.0],  # Intercept term
                bspline_bases,  # B-spline basis function values
                X_std[1:]  # Other covariates (gender, age, height)
            ])
        else:
            # Not using B-spline, directly add intercept
            X_expanded = np.concatenate([[1.0], X_std])
        
        return X_expanded
    
    def predict_feature(self, feature_name, X_raw):
        """
        Predict a single feature
        
        Parameters:
            feature_name: Feature name
            X_raw: Raw covariates [laserpower, gender, age, height]
        
        Returns:
            Result dictionary containing mean, std, lower_95, upper_95, z_score
        """
        params = self.model_params[feature_name]
        
        # 1. Standardize input (using this feature's inscaler)
        X_std = self.standardize_input(X_raw, feature_name)
        
        # 2. Expand covariates (apply B-spline)
        X_expanded = self._expand_covariates(X_std, params)
        
        # 3. Get parameters from posterior
        posterior = params['posterior']
        
        # Get slope_mu (fixed effect coefficients)
        slope_mu = np.array(posterior['slope_mu']['mean'])
        
        # Get hierarchical intercept for mu
        mu_intercept_mu = posterior.get('mu_intercept_mu', {}).get('mean', 0.0)
        
        # 4. Predict mu (expected value)
        # mu = X_expanded @ slope_mu + mu_intercept_mu
        mu_pred_std = np.dot(X_expanded, slope_mu) + mu_intercept_mu
        
        # 5. Predict sigma (standard deviation)
        slope_sigma = np.array(posterior['slope_sigma']['mean'])
        intercept_sigma = posterior['intercept_sigma']['mean']
        
        # Linear prediction for sigma
        sigma_pred_linear = np.dot(X_expanded, slope_sigma) + intercept_sigma
        
        # Apply softplus mapping
        sigma_mapping_params = params['sigma_mapping']['params']
        sigma_pred_std = softplus(sigma_pred_linear, sigma_mapping_params)
        
        # 6. Inverse standardization to original scale
        outscaler = params['outscaler']
        mu_pred = mu_pred_std * outscaler['std'] + outscaler['mean']
        sigma_pred = sigma_pred_std * outscaler['std']
        
        # 7. Calculate confidence interval and Z-score
        lower_95 = mu_pred - 1.96 * sigma_pred
        upper_95 = mu_pred + 1.96 * sigma_pred
        
        # Note: Z-score requires observed value, here we return predicted distribution parameters
        # Actual Z-score calculation: z = (observed - mu_pred) / sigma_pred
        
        return {
            'mean': float(mu_pred),
            'std': float(sigma_pred),
            'lower_95': float(lower_95),
            'upper_95': float(upper_95),
            'mu_std': float(mu_pred_std),  # mu in standardized scale (for debugging)
            'sigma_std': float(sigma_pred_std)  # sigma in standardized scale (for debugging)
        }
    
    def calculate_z_score(self, feature_name, observed_value, predicted_result):
        """
        Calculate Z-score for observed value
        
        Parameters:
            feature_name: Feature name
            observed_value: Observed value
            predicted_result: Return result from predict_feature
        
        Returns:
            z_score: Z-score
        """
        z_score = (observed_value - predicted_result['mean']) / predicted_result['std']
        return float(z_score)
    
    def predict(self, laserpower, gender, age, height, show_warnings=True):
        """
        Complete prediction for all features
        
        Parameters:
            laserpower: Laser power
            gender: Gender (1=male, 2=female)
            age: Age
            height: Height (cm)
            show_warnings: Whether to display input validity warnings
        
        Returns:
            results: Dictionary with feature names as keys and prediction results as values
        """
        # 1. Check input validity
        warnings_list, severity = self.check_input(laserpower, gender, age, height)
        
        # 2. Display warnings
        if show_warnings and warnings_list:
            print("\n" + "="*70)
            print("⚠️  Input Validity Check")
            print("="*70)
            for w in warnings_list:
                print(w)
            
            if severity == 'high':
                print("\n🔴 Severe Warning: Input significantly out of training range, predictions may be unreliable!")
            elif severity == 'medium':
                print("\n🟡 Warning: Input near edge of training range, prediction uncertainty is high")
            elif severity == 'low':
                print("\n🟢 Notice: Input slightly deviates from recommended range")
            
            print("="*70)
            
            if severity == 'high':
                response = input("\nContinue with prediction? (y/n): ").strip().lower()
                if response != 'y':
                    print("❌ Prediction cancelled")
                    return None
            print()
        
        # 3. Prepare input
        X_raw = np.array([laserpower, gender, age, height])
        
        # 4. Predict all features
        results = {}
        for feature in self.feature_names:
            try:
                results[feature] = self.predict_feature(feature, X_raw)
            except Exception as e:
                print(f"⚠️  Error predicting feature {feature}: {e}")
                import traceback
                traceback.print_exc()
                results[feature] = {
                    'mean': np.nan,
                    'std': np.nan,
                    'lower_95': np.nan,
                    'upper_95': np.nan
                }
        
        return results
    
    def predict_with_observations(self, laserpower, gender, age, height, observations):
        """
        Predict and calculate Z-scores
        
        Parameters:
            laserpower, gender, age, height: Covariates
            observations: Dictionary with feature names as keys and observed values as values
        
        Returns:
            results: Dictionary containing predictions and Z-scores
        """
        # Get predictions
        predictions = self.predict(laserpower, gender, age, height, show_warnings=False)
        
        if predictions is None:
            return None
        
        # Calculate Z-scores
        for feature, pred in predictions.items():
            if feature in observations and not np.isnan(observations[feature]):
                z_score = self.calculate_z_score(feature, observations[feature], pred)
                pred['observed'] = observations[feature]
                pred['z_score'] = z_score
            else:
                pred['observed'] = np.nan
                pred['z_score'] = np.nan
        
        return predictions


def print_results(results, laserpower, gender, age, height, show_debug=False):
    """
    Print prediction results
    
    Parameters:
        results: Prediction result dictionary
        laserpower, gender, age, height: Input covariates
        show_debug: Whether to show debug information (values in standardized scale)
    """
    if results is None:
        return
    
    print("\n" + "="*70)
    print("📊 Prediction Results")
    print("="*70)
    print(f"\nInput Parameters:")
    print(f"  Laser power: {laserpower}")
    print(f"  Gender:      {gender} ({'Male' if gender == 1 else 'Female'})")
    print(f"  Age:         {age} years")
    print(f"  Height:      {height} cm")
    
    print(f"\nPredicted values (mean ± std):")
    print("-"*70)
    
    for feature, pred in results.items():
        mean = pred['mean']
        std = pred['std']
        lower = pred['lower_95']
        upper = pred['upper_95']
        
        # Basic information
        info_str = f"{feature:12s}: {mean:8.2f} ± {std:6.2f}  (95% CI: [{lower:7.2f}, {upper:7.2f}])"
        
        # If observed value and Z-score available
        if 'observed' in pred and not np.isnan(pred['observed']):
            obs = pred['observed']
            z = pred['z_score']
            info_str += f"  | Obs: {obs:7.2f}, Z: {z:6.2f}"
        
        print(info_str)
    
    if show_debug:
        print("\nDebug Information (standardized scale):")
        print("-"*70)
        for feature, pred in results.items():
            if 'mu_std' in pred and 'sigma_std' in pred:
                print(f"{feature:12s}: μ_std={pred['mu_std']:7.4f}, σ_std={pred['sigma_std']:7.4f}")
    
    print("="*70)


def interactive_mode(predictor):
    """Interactive input mode"""
    print("\n" + "="*70)
    print("🎯 Interactive Prediction Mode")
    print("="*70)
    print("\nCovariate Input Instructions:")
    print("  • Laser power: 1.0-4.5 (recommended 2.5-4.0)")
    print("  • Gender:      1=male, 2=female")
    print("  • Age:         recommended 18-25 years (training range 16-50)")
    print("  • Height:      150-190 cm")
    print("\nCommands:")
    print("  • Enter 'q' to quit")
    print("  • Enter 'b' to enter batch prediction mode")
    print("  • Enter 'z' to enter prediction mode with observations (calculate Z-scores)")
    print()
    
    while True:
        try:
            print("-"*70)
            power_input = input("Laser power (q=quit, b=batch, z=Z-score mode): ").strip()
            
            if power_input.lower() == 'q':
                print("\n👋 Goodbye!")
                break
            
            if power_input.lower() == 'b':
                print("\nSwitching to batch prediction mode...")
                input_file = input("Input file path: ").strip()
                output_file = input("Output file path (default: predictions.csv): ").strip()
                if not output_file:
                    output_file = 'predictions.csv'
                batch_mode(predictor, input_file, output_file)
                print("\nReturning to interactive mode")
                continue
            
            if power_input.lower() == 'z':
                print("\nSwitching to prediction mode with observations...")
                z_score_mode(predictor)
                print("\nReturning to interactive mode")
                continue
            
            # Get input
            laserpower = float(power_input)
            gender = int(input("Gender (1=male, 2=female): "))
            age = float(input("Age: "))
            height = float(input("Height (cm): "))
            
            # Predict
            results = predictor.predict(laserpower, gender, age, height)
            print_results(results, laserpower, gender, age, height)
            
        except ValueError as e:
            print(f"\n❌ Input error: {e}")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


def z_score_mode(predictor):
    """Prediction mode with observations (calculate Z-scores)"""
    print("\n" + "="*70)
    print("📈 Z-score Calculation Mode")
    print("="*70)
    print("\nIn this mode, you can enter observed values to calculate Z-scores")
    print("Z-score represents how many standard deviations the observed value deviates from predicted mean")
    print()
    
    try:
        # Get covariates
        laserpower = float(input("Laser power: "))
        gender = int(input("Gender (1=male, 2=female): "))
        age = float(input("Age: "))
        height = float(input("Height (cm): "))
        
        # Get observed values
        print("\nPlease enter observed values (leave blank to skip that feature):")
        observations = {}
        for feature in predictor.feature_names:
            obs_input = input(f"  {feature}: ").strip()
            if obs_input:
                try:
                    observations[feature] = float(obs_input)
                except ValueError:
                    print(f"    ⚠️  Invalid input, skipping {feature}")
        
        # Predict and calculate Z-scores
        results = predictor.predict_with_observations(
            laserpower, gender, age, height, observations
        )
        
        # Display results
        print_results(results, laserpower, gender, age, height)
        
        # Explain Z-scores
        print("\nZ-score Interpretation:")
        print("  |Z| < 1.96: Within 95% confidence interval (normal)")
        print("  |Z| > 1.96: Outside 95% confidence interval (abnormal)")
        print("  |Z| > 2.58: Outside 99% confidence interval (highly abnormal)")
        
    except ValueError as e:
        print(f"\n❌ Input error: {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")


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
    
    for idx, row in df.iterrows():
        # Check input validity
        warnings_list, severity = predictor.check_input(
            row['laserpower'], row['gender'], row['age'], row['height']
        )
        
        if warnings_list:
            warnings_count += 1
        
        # Prepare observations (if available)
        observations = {}
        if has_observations:
            for feature in predictor.feature_names:
                if feature in df.columns and not pd.isna(row[feature]):
                    observations[feature] = row[feature]
        
        # Predict
        if observations:
            results = predictor.predict_with_observations(
                row['laserpower'], row['gender'], row['age'], row['height'],
                observations
            )
        else:
            results = predictor.predict(
                row['laserpower'], row['gender'], row['age'], row['height'],
                show_warnings=False
            )
        
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
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_file, index=False)
    
    print(f"\n✅ Prediction complete!")
    print(f"  Results saved to: {output_file}")
    print(f"  Total samples: {len(results_df)}")
    print(f"  Samples with warnings: {warnings_count}")
    
    # Statistics for abnormal Z-scores (if Z-scores were calculated)
    if has_observations:
        z_cols = [col for col in results_df.columns if col.endswith('_z_score')]
        if z_cols:
            print(f"\nZ-score Statistics:")
            for z_col in z_cols:
                feature = z_col.replace('_z_score', '')
                z_values = results_df[z_col].dropna()
                if len(z_values) > 0:
                    n_abnormal = (z_values.abs() > 1.96).sum()
                    print(f"  {feature}: {n_abnormal}/{len(z_values)} samples abnormal (|Z| > 1.96)")


def quick_mode(predictor, laserpower, gender, age, height):
    """Quick prediction mode (command line)"""
    results = predictor.predict(laserpower, gender, age, height)
    print_results(results, laserpower, gender, age, height)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="HBR Interactive Predictor - Feature-wise Training Version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:

1. Interactive mode (default):
   python predictor_by_feature.py

2. Quick prediction:
   python predictor_by_feature.py -q 3.5 1 21 170

3. Batch prediction:
   python predictor_by_feature.py -b input.csv -o output.csv

4. Specify parameter file:
   python predictor_by_feature.py -p extracted_model_params.json

Covariate Instructions:
  • laserpower: Laser power (1.0-4.5, recommended 2.5-4.0)
  • gender: Gender (1=male, 2=female)
  • age: Age (recommended 18-25 years)
  • height: Height (150-190 cm)
        """
    )
    
    parser.add_argument('-p', '--params', help='Path to parameter file')
    parser.add_argument('-q', '--quick', nargs=4, 
                       metavar=('POWER', 'GENDER', 'AGE', 'HEIGHT'),
                       help='Quick prediction mode')
    parser.add_argument('-b', '--batch', metavar='INPUT', help='Batch prediction input file')
    parser.add_argument('-o', '--output', metavar='OUTPUT', help='Batch prediction output file')
    parser.add_argument('-d', '--debug', action='store_true', help='Show debug information')
    
    args = parser.parse_args()
    
    # Print welcome message
    print("\n" + "="*70)
    print("🚀 HBR Interactive Predictor - Feature-wise Training Version")
    print("="*70)
    
    # Load predictor
    try:
        predictor = HBRPredictorByFeature(args.params)
        print(f"\n✓ Using parameter file: {predictor.params_file}")
        print(f"✓ Successfully loaded {len(predictor.feature_names)} feature models:")
        for i, feature in enumerate(predictor.feature_names, 1):
            print(f"   {i}. {feature}")
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
    
    else:
        interactive_mode(predictor)


if __name__ == "__main__":
    main()
