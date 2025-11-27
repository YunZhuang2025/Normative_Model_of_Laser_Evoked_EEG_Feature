#!/usr/bin/env python3
"""
基于PCNtoolkit的激光诱发脑电特征值Normative model
作者: Yun Zhuang
日期: 2025-11-27
版本: v1.0
如果使用本工具发表论文，请引用：
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
    Softplus映射函数
    sigma = scale * log(1 + exp((x - shift) / scale))
    """
    shift, scale = params
    x_scaled = (x - shift) / scale
    # 裁剪防止溢出
    x_clipped = np.clip(x_scaled, -20, 20)
    return scale * np.log1p(np.exp(x_clipped))


def create_bspline_basis(x, knots, degree=3):
    """
    创建B-spline基函数矩阵
    
    参数:
        x: 输入值（标准化后的）
        knots: 节点向量
        degree: B-spline阶数
    
    返回:
        basis_matrix: shape (n_bases,) 或 (n_samples, n_bases)
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
    
    # 如果输入是单个值，返回1D数组
    if basis_matrix.shape[0] == 1:
        return basis_matrix.flatten()
    
    return basis_matrix


class HBRPredictorByFeature:
    """
    HBR预测器 - 按特征分别训练版本
    
    特点：
    - 每个特征有独立的训练数据和标准化参数
    - 自动处理不同特征可能有不同的inscaler
    - 统一的协变量输入接口
    """
    
    def __init__(self, params_file=None, custom_inscaler=None):
        """
        初始化预测器
        
        参数:
            params_file: 参数文件路径
            custom_inscaler: 自定义inscaler字典，格式 {'mean': [...], 'std': [...]}
                            如果提供，将覆盖所有特征的inscaler
        """
        # 自动查找参数文件
        if params_file is None:
            params_file = self._find_params_file()
        
        params_path = Path(params_file)
        if not params_path.exists():
            raise FileNotFoundError(f"找不到参数文件: {params_file}")
        
        # 加载模型参数
        print(f"正在加载参数文件: {params_file}")
        with open(params_file, 'r', encoding='utf-8') as f:
            self.model_params = json.load(f)
        
        self.params_file = str(params_path)
        self.custom_inscaler = custom_inscaler
        
        # 特征名称（按照期望的顺序）
        feature_order = ['N1_amp', 'N2_amp', 'P2_amp', 
                        'ERP_mag', 'alpha_mag', 'beta_mag', 'gamma_mag']
        self.feature_names = [f for f in feature_order if f in self.model_params]
        
        if not self.feature_names:
            raise ValueError("未找到任何可用的特征模型！")
        
        # 为每个特征准备inscaler
        self._prepare_inscalers()
        
        # 协变量范围（基于训练数据的统计）
        self.covariate_names = ['laserpower', 'gender', 'age', 'height']
        self.ranges = {
            'laserpower': {'min': 1.0, 'max': 4.5, 'rec_min': 2.5, 'rec_max': 4.0},
            'gender': {'min': 1, 'max': 2},
            'age': {'min': 16.0, 'max': 50.0, 'rec_min': 18.0, 'rec_max': 25.0},
            'height': {'min': 150.0, 'max': 190.0}
        }
    
    def _find_params_file(self):
        """自动查找参数文件"""
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
        为每个特征准备inscaler
        
        优先级:
        1. 如果提供了custom_inscaler：
           - 如果是字典（每个特征独立值），使用特征专属值
           - 如果是单一值（全局共享），所有特征使用相同值
        2. 否则使用模型参数中的inscaler（每个特征独立）
        3. 如果都没有，使用全局默认值
        
        重要：每个特征可能有不同的inscaler（因为剔除了不同的异常值）
        """
        self.inscalers = {}
        
        # 全局默认inscaler（仅作为后备，精度可能不够）
        # 对应顺序: laserpower, gender, age, height
        default_inscaler = {
            'mean': np.array([3.3427, 1.6024, 24.5392, 167.5705]),
            'std': np.array([0.6794, 0.4894, 3.4240, 7.1038])
        }
        
        # 检查 custom_inscaler 的类型
        if self.custom_inscaler is not None:
            # 检查是否是字典（每个特征独立）
            if isinstance(self.custom_inscaler, dict):
                # 检查是否包含特征名作为键
                has_feature_keys = any(f in self.custom_inscaler 
                                      for f in self.feature_names)
                
                if has_feature_keys:
                    # 每个特征独立的inscaler
                    print("使用特征专属inscaler:")
                    for feature in self.feature_names:
                        if feature in self.custom_inscaler:
                            inscaler = self.custom_inscaler[feature]
                            self.inscalers[feature] = {
                                'mean': np.array(inscaler['mean']),
                                'std': np.array(inscaler['std'])
                            }
                            print(f"  ✓ {feature}: 使用自定义inscaler")
                        else:
                            # 该特征没有提供，使用默认
                            self.inscalers[feature] = default_inscaler
                            print(f"  ⚠️  {feature}: 未提供，使用默认inscaler")
                else:
                    # 全局共享的inscaler
                    print("使用全局共享inscaler:")
                    print(f"  mean: {self.custom_inscaler['mean']}")
                    print(f"  std: {self.custom_inscaler['std']}")
                    shared_inscaler = {
                        'mean': np.array(self.custom_inscaler['mean']),
                        'std': np.array(self.custom_inscaler['std'])
                    }
                    for feature in self.feature_names:
                        self.inscalers[feature] = shared_inscaler
            else:
                raise ValueError("custom_inscaler 必须是字典类型")
        else:
            # 从模型参数中读取（每个特征独立）
            print("从模型参数中读取inscaler...")
            
            has_any_inscaler = False
            for feature in self.feature_names:
                params = self.model_params[feature]
                inscaler = params.get('inscaler', {})
                
                # 检查inscaler是否有效
                mean = inscaler.get('mean', [])
                std = inscaler.get('std', [])
                
                if isinstance(mean, list) and len(mean) == 4 and \
                   isinstance(std, list) and len(std) == 4:
                    # 使用特征专属的inscaler
                    self.inscalers[feature] = {
                        'mean': np.array(mean),
                        'std': np.array(std)
                    }
                    print(f"  ✓ {feature}: 使用模型参数中的inscaler")
                    has_any_inscaler = True
                else:
                    # 使用全局默认inscaler
                    self.inscalers[feature] = default_inscaler
                    print(f"  ⚠️  {feature}: 模型参数中无inscaler，使用默认值")
            
            if not has_any_inscaler:
                print("\n⚠️  警告: 所有特征都缺少inscaler，使用默认值")
                print("   建议: 运行 compute_inscalers_matlab.m 计算准确的inscaler")
        
        print()
    
    def check_input(self, laserpower, gender, age, height):
        """
        检查输入有效性
        
        返回:
            warnings: 警告信息列表
            severity: 严重程度 ('none', 'low', 'medium', 'high')
        """
        warnings_list = []
        severity = 'none'
        
        # 检查激光功率
        if not (self.ranges['laserpower']['min'] <= laserpower <= self.ranges['laserpower']['max']):
            warnings_list.append(f"⚠️  激光功率 {laserpower} 超出范围 [{self.ranges['laserpower']['min']}, {self.ranges['laserpower']['max']}]")
            severity = 'high'
        elif not (self.ranges['laserpower']['rec_min'] <= laserpower <= self.ranges['laserpower']['rec_max']):
            warnings_list.append(f"ℹ️  激光功率 {laserpower} 在有效范围内，但建议使用 [{self.ranges['laserpower']['rec_min']}, {self.ranges['laserpower']['rec_max']}]")
            if severity == 'none':
                severity = 'low'
        
        # 检查性别
        if gender not in [1, 2]:
            warnings_list.append(f"⚠️  性别值 {gender} 无效 (应为 1=男 或 2=女)")
            severity = 'high'
        
        # 检查年龄
        if not (self.ranges['age']['min'] <= age <= self.ranges['age']['max']):
            warnings_list.append(f"⚠️  年龄 {age} 超出训练范围 [{self.ranges['age']['min']}, {self.ranges['age']['max']}]")
            severity = 'high'
        elif not (self.ranges['age']['rec_min'] <= age <= self.ranges['age']['rec_max']):
            warnings_list.append(f"⚠️  年龄 {age} 超出建议范围 [{self.ranges['age']['rec_min']}, {self.ranges['age']['rec_max']}]")
            warnings_list.append(f"   训练数据主要集中在21岁，25岁以上样本较少")
            if severity == 'none':
                severity = 'medium'
        
        # 检查身高
        if not (self.ranges['height']['min'] <= height <= self.ranges['height']['max']):
            warnings_list.append(f"⚠️  身高 {height}cm 超出范围 [{self.ranges['height']['min']}, {self.ranges['height']['max']}]")
            if severity == 'none':
                severity = 'low'
        
        return warnings_list, severity
    
    def standardize_input(self, X, feature_name):
        """
        标准化输入（使用特征专属的inscaler）
        
        参数:
            X: 原始输入 [laserpower, gender, age, height]
            feature_name: 特征名称
        
        返回:
            X_std: 标准化后的输入
        """
        X = np.atleast_1d(X)
        inscaler = self.inscalers[feature_name]
        
        return (X - inscaler['mean']) / inscaler['std']
    
    def _expand_covariates(self, X_std, params):
        """
        展开协变量（应用B-spline变换）
        
        参数:
            X_std: 标准化后的协变量 [laserpower_std, gender_std, age_std, height_std]
            params: 特征的模型参数
        
        返回:
            X_expanded: 展开后的协变量（包含截距项）
            
        注意：
            - slope_mu/slope_sigma 的维度是 (n_expanded_covariates,)
            - n_expanded_covariates = 1(截距) + n_bspline_bases + n_other_covariates
            - 例如: 1 + 7(B-spline bases for laserpower) + 3(gender, age, height) = 11
        """
        mu_basis_config = params.get('mu_basis', {})
        basis_column = mu_basis_config.get('basis_column', [])
        
        # 检查是否对第一个协变量（laserpower）应用B-spline
        if 0 in basis_column:
            laserpower_std = X_std[0]
            
            # 获取B-spline节点
            knots_dict = mu_basis_config.get('knots', {})
            if '0' in knots_dict:
                knots = np.array(knots_dict['0'])
            else:
                # 如果没有预定义的节点，生成默认节点
                nknots = mu_basis_config.get('nknots', 5)
                degree = mu_basis_config.get('degree', 3)
                interior_knots = np.linspace(-2, 2, nknots)
                knots = np.concatenate([
                    np.repeat(interior_knots[0], degree),
                    interior_knots,
                    np.repeat(interior_knots[-1], degree)
                ])
            
            # 创建B-spline基函数
            degree = mu_basis_config.get('degree', 3)
            bspline_bases = create_bspline_basis(laserpower_std, knots, degree)
            
            # 展开: [1(截距), B-spline_bases(7个), gender, age, height]
            X_expanded = np.concatenate([
                [1.0],  # 截距项
                bspline_bases,  # B-spline基函数值
                X_std[1:]  # 其他协变量 (gender, age, height)
            ])
        else:
            # 不使用B-spline，直接添加截距
            X_expanded = np.concatenate([[1.0], X_std])
        
        return X_expanded
    
    def predict_feature(self, feature_name, X_raw):
        """
        预测单个特征
        
        参数:
            feature_name: 特征名称
            X_raw: 原始协变量 [laserpower, gender, age, height]
        
        返回:
            结果字典，包含 mean, std, lower_95, upper_95, z_score
        """
        params = self.model_params[feature_name]
        
        # 1. 标准化输入（使用该特征的inscaler）
        X_std = self.standardize_input(X_raw, feature_name)
        
        # 2. 展开协变量（应用B-spline）
        X_expanded = self._expand_covariates(X_std, params)
        
        # 3. 从posterior中获取参数
        posterior = params['posterior']
        
        # 获取slope_mu（固定效应系数）
        slope_mu = np.array(posterior['slope_mu']['mean'])
        
        # 获取mu的层级截距
        mu_intercept_mu = posterior.get('mu_intercept_mu', {}).get('mean', 0.0)
        
        # 4. 预测 mu (期望值)
        # mu = X_expanded @ slope_mu + mu_intercept_mu
        mu_pred_std = np.dot(X_expanded, slope_mu) + mu_intercept_mu
        
        # 5. 预测 sigma (标准差)
        slope_sigma = np.array(posterior['slope_sigma']['mean'])
        intercept_sigma = posterior['intercept_sigma']['mean']
        
        # sigma的线性预测
        sigma_pred_linear = np.dot(X_expanded, slope_sigma) + intercept_sigma
        
        # 应用softplus映射
        sigma_mapping_params = params['sigma_mapping']['params']
        sigma_pred_std = softplus(sigma_pred_linear, sigma_mapping_params)
        
        # 6. 反标准化到原始尺度
        outscaler = params['outscaler']
        mu_pred = mu_pred_std * outscaler['std'] + outscaler['mean']
        sigma_pred = sigma_pred_std * outscaler['std']
        
        # 7. 计算置信区间和Z分数
        lower_95 = mu_pred - 1.96 * sigma_pred
        upper_95 = mu_pred + 1.96 * sigma_pred
        
        # 注意：Z分数需要观测值，这里返回的是预测分布的参数
        # 实际Z分数计算: z = (observed - mu_pred) / sigma_pred
        
        return {
            'mean': float(mu_pred),
            'std': float(sigma_pred),
            'lower_95': float(lower_95),
            'upper_95': float(upper_95),
            'mu_std': float(mu_pred_std),  # 标准化尺度的mu（调试用）
            'sigma_std': float(sigma_pred_std)  # 标准化尺度的sigma（调试用）
        }
    
    def calculate_z_score(self, feature_name, observed_value, predicted_result):
        """
        计算观测值的Z分数
        
        参数:
            feature_name: 特征名称
            observed_value: 观测值
            predicted_result: predict_feature的返回结果
        
        返回:
            z_score: Z分数
        """
        z_score = (observed_value - predicted_result['mean']) / predicted_result['std']
        return float(z_score)
    
    def predict(self, laserpower, gender, age, height, show_warnings=True):
        """
        完整预测所有特征
        
        参数:
            laserpower: 激光功率
            gender: 性别 (1=男, 2=女)
            age: 年龄
            height: 身高 (cm)
            show_warnings: 是否显示输入有效性警告
        
        返回:
            results: 字典，键为特征名，值为预测结果
        """
        # 1. 检查输入有效性
        warnings_list, severity = self.check_input(laserpower, gender, age, height)
        
        # 2. 显示警告
        if show_warnings and warnings_list:
            print("\n" + "="*70)
            print("⚠️  输入有效性检查")
            print("="*70)
            for w in warnings_list:
                print(w)
            
            if severity == 'high':
                print("\n🔴 严重警告: 输入显著超出训练范围，预测可能不可靠！")
            elif severity == 'medium':
                print("\n🟡 警告: 输入接近训练范围边缘，预测不确定性较高")
            elif severity == 'low':
                print("\n🟢 提示: 输入略微偏离建议范围")
            
            print("="*70)
            
            if severity == 'high':
                response = input("\n是否继续预测？(y/n): ").strip().lower()
                if response != 'y':
                    print("❌ 已取消预测")
                    return None
            print()
        
        # 3. 准备输入
        X_raw = np.array([laserpower, gender, age, height])
        
        # 4. 预测所有特征
        results = {}
        for feature in self.feature_names:
            try:
                results[feature] = self.predict_feature(feature, X_raw)
            except Exception as e:
                print(f"⚠️  预测特征 {feature} 时出错: {e}")
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
        预测并计算Z分数
        
        参数:
            laserpower, gender, age, height: 协变量
            observations: 字典，键为特征名，值为观测值
        
        返回:
            results: 包含预测值和Z分数的字典
        """
        # 获取预测结果
        predictions = self.predict(laserpower, gender, age, height, show_warnings=False)
        
        if predictions is None:
            return None
        
        # 计算Z分数
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
    打印预测结果
    
    参数:
        results: 预测结果字典
        laserpower, gender, age, height: 输入协变量
        show_debug: 是否显示调试信息（标准化尺度的值）
    """
    if results is None:
        return
    
    print("\n" + "="*70)
    print("📊 预测结果")
    print("="*70)
    print(f"\n输入参数:")
    print(f"  激光功率: {laserpower}")
    print(f"  性别:     {gender} ({'男' if gender == 1 else '女'})")
    print(f"  年龄:     {age} 岁")
    print(f"  身高:     {height} cm")
    
    print(f"\n预测值 (均值 ± 标准差):")
    print("-"*70)
    
    for feature, pred in results.items():
        mean = pred['mean']
        std = pred['std']
        lower = pred['lower_95']
        upper = pred['upper_95']
        
        # 基本信息
        info_str = f"{feature:12s}: {mean:8.2f} ± {std:6.2f}  (95% CI: [{lower:7.2f}, {upper:7.2f}])"
        
        # 如果有观测值和Z分数
        if 'observed' in pred and not np.isnan(pred['observed']):
            obs = pred['observed']
            z = pred['z_score']
            info_str += f"  | Obs: {obs:7.2f}, Z: {z:6.2f}"
        
        print(info_str)
    
    if show_debug:
        print("\n调试信息 (标准化尺度):")
        print("-"*70)
        for feature, pred in results.items():
            if 'mu_std' in pred and 'sigma_std' in pred:
                print(f"{feature:12s}: μ_std={pred['mu_std']:7.4f}, σ_std={pred['sigma_std']:7.4f}")
    
    print("="*70)


def interactive_mode(predictor):
    """交互式输入模式"""
    print("\n" + "="*70)
    print("🎯 交互式预测模式")
    print("="*70)
    print("\n协变量输入说明:")
    print("  • 激光功率: 1.0-4.5 (建议 2.5-4.0)")
    print("  • 性别:     1=男, 2=女")
    print("  • 年龄:     建议 18-25岁 (训练范围 16-50)")
    print("  • 身高:     150-190 cm")
    print("\n命令:")
    print("  • 输入 'q' 退出")
    print("  • 输入 'b' 进入批量预测模式")
    print("  • 输入 'z' 进入带观测值的预测模式（计算Z分数）")
    print()
    
    while True:
        try:
            print("-"*70)
            power_input = input("激光功率 (q=退出, b=批量, z=Z分数模式): ").strip()
            
            if power_input.lower() == 'q':
                print("\n👋 再见！")
                break
            
            if power_input.lower() == 'b':
                print("\n切换到批量预测模式...")
                input_file = input("输入文件路径: ").strip()
                output_file = input("输出文件路径 (默认: predictions.csv): ").strip()
                if not output_file:
                    output_file = 'predictions.csv'
                batch_mode(predictor, input_file, output_file)
                print("\n返回交互式模式")
                continue
            
            if power_input.lower() == 'z':
                print("\n切换到带观测值的预测模式...")
                z_score_mode(predictor)
                print("\n返回交互式模式")
                continue
            
            # 获取输入
            laserpower = float(power_input)
            gender = int(input("性别 (1=男, 2=女): "))
            age = float(input("年龄: "))
            height = float(input("身高 (cm): "))
            
            # 预测
            results = predictor.predict(laserpower, gender, age, height)
            print_results(results, laserpower, gender, age, height)
            
        except ValueError as e:
            print(f"\n❌ 输入错误: {e}")
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            import traceback
            traceback.print_exc()


def z_score_mode(predictor):
    """带观测值的预测模式（计算Z分数）"""
    print("\n" + "="*70)
    print("📈 Z分数计算模式")
    print("="*70)
    print("\n在此模式下，您可以输入观测值来计算Z分数")
    print("Z分数表示观测值偏离预测均值的标准差数")
    print()
    
    try:
        # 获取协变量
        laserpower = float(input("激光功率: "))
        gender = int(input("性别 (1=男, 2=女): "))
        age = float(input("年龄: "))
        height = float(input("身高 (cm): "))
        
        # 获取观测值
        print("\n请输入观测值（留空跳过该特征）:")
        observations = {}
        for feature in predictor.feature_names:
            obs_input = input(f"  {feature}: ").strip()
            if obs_input:
                try:
                    observations[feature] = float(obs_input)
                except ValueError:
                    print(f"    ⚠️  无效输入，跳过 {feature}")
        
        # 预测并计算Z分数
        results = predictor.predict_with_observations(
            laserpower, gender, age, height, observations
        )
        
        # 显示结果
        print_results(results, laserpower, gender, age, height)
        
        # 解释Z分数
        print("\nZ分数解释:")
        print("  |Z| < 1.96: 在95%置信区间内（正常）")
        print("  |Z| > 1.96: 超出95%置信区间（异常）")
        print("  |Z| > 2.58: 超出99%置信区间（高度异常）")
        
    except ValueError as e:
        print(f"\n❌ 输入错误: {e}")
    except Exception as e:
        print(f"\n❌ 错误: {e}")


def batch_mode(predictor, input_file, output_file):
    """
    批量预测模式
    
    输入CSV格式：至少包含列 laserpower, gender, age, height
    可选列：各特征的观测值（用于计算Z分数）
    """
    print("\n" + "="*70)
    print("📁 批量预测模式")
    print("="*70)
    
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"\n❌ 读取文件失败: {e}")
        return
    
    # 检查必需列
    required_cols = ['laserpower', 'gender', 'age', 'height']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"❌ 缺少必需的列: {missing}")
        return
    
    print(f"\n✓ 找到 {len(df)} 个样本")
    
    # 检查是否有观测值列
    has_observations = any(f in df.columns for f in predictor.feature_names)
    if has_observations:
        print("✓ 检测到观测值列，将计算Z分数")
    
    print("开始预测...\n")
    
    all_results = []
    warnings_count = 0
    
    for idx, row in df.iterrows():
        # 检查输入有效性
        warnings_list, severity = predictor.check_input(
            row['laserpower'], row['gender'], row['age'], row['height']
        )
        
        if warnings_list:
            warnings_count += 1
        
        # 准备观测值（如果有）
        observations = {}
        if has_observations:
            for feature in predictor.feature_names:
                if feature in df.columns and not pd.isna(row[feature]):
                    observations[feature] = row[feature]
        
        # 预测
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
            # 构建输出行
            flat_result = {
                'index': idx,
                'laserpower': row['laserpower'],
                'gender': row['gender'],
                'age': row['age'],
                'height': row['height'],
                'has_warnings': len(warnings_list) > 0,
                'warning_severity': severity
            }
            
            # 添加预测结果
            for feature, pred in results.items():
                flat_result[f'{feature}_pred_mean'] = pred['mean']
                flat_result[f'{feature}_pred_std'] = pred['std']
                flat_result[f'{feature}_pred_lower95'] = pred['lower_95']
                flat_result[f'{feature}_pred_upper95'] = pred['upper_95']
                
                # 如果有观测值和Z分数
                if 'observed' in pred:
                    flat_result[f'{feature}_observed'] = pred['observed']
                if 'z_score' in pred:
                    flat_result[f'{feature}_z_score'] = pred['z_score']
            
            all_results.append(flat_result)
        
        # 显示进度
        if (idx + 1) % 10 == 0:
            print(f"  已处理 {idx + 1}/{len(df)} 个样本...")
    
    # 保存结果
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_file, index=False)
    
    print(f"\n✅ 预测完成！")
    print(f"  结果已保存到: {output_file}")
    print(f"  总样本数: {len(results_df)}")
    print(f"  有警告的样本: {warnings_count}")
    
    # 统计异常Z分数（如果计算了Z分数）
    if has_observations:
        z_cols = [col for col in results_df.columns if col.endswith('_z_score')]
        if z_cols:
            print(f"\nZ分数统计:")
            for z_col in z_cols:
                feature = z_col.replace('_z_score', '')
                z_values = results_df[z_col].dropna()
                if len(z_values) > 0:
                    n_abnormal = (z_values.abs() > 1.96).sum()
                    print(f"  {feature}: {n_abnormal}/{len(z_values)} 样本异常 (|Z| > 1.96)")


def quick_mode(predictor, laserpower, gender, age, height):
    """快速预测模式（命令行）"""
    results = predictor.predict(laserpower, gender, age, height)
    print_results(results, laserpower, gender, age, height)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="HBR交互式预测器 - 按特征分别训练版本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

1. 交互式模式（默认）:
   python predictor_by_feature.py

2. 快速预测:
   python predictor_by_feature.py -q 3.5 1 21 170

3. 批量预测:
   python predictor_by_feature.py -b input.csv -o output.csv

4. 指定参数文件:
   python predictor_by_feature.py -p extracted_model_params.json

协变量说明:
  • laserpower: 激光功率 (1.0-4.5, 建议2.5-4.0)
  • gender: 性别 (1=男, 2=女)
  • age: 年龄 (建议18-25岁)
  • height: 身高 (150-190 cm)
        """
    )
    
    parser.add_argument('-p', '--params', help='参数文件路径')
    parser.add_argument('-q', '--quick', nargs=4, 
                       metavar=('POWER', 'GENDER', 'AGE', 'HEIGHT'),
                       help='快速预测模式')
    parser.add_argument('-b', '--batch', metavar='INPUT', help='批量预测输入文件')
    parser.add_argument('-o', '--output', metavar='OUTPUT', help='批量预测输出文件')
    parser.add_argument('-d', '--debug', action='store_true', help='显示调试信息')
    
    args = parser.parse_args()
    
    # 打印欢迎信息
    print("\n" + "="*70)
    print("🚀 HBR交互式预测器 - 按特征分别训练版本")
    print("="*70)
    
    # 加载预测器
    try:
        predictor = HBRPredictorByFeature(args.params)
        print(f"\n✓ 使用参数文件: {predictor.params_file}")
        print(f"✓ 成功加载 {len(predictor.feature_names)} 个特征模型:")
        for i, feature in enumerate(predictor.feature_names, 1):
            print(f"   {i}. {feature}")
    except Exception as e:
        print(f"\n❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 根据模式运行
    if args.quick:
        try:
            laserpower = float(args.quick[0])
            gender = int(args.quick[1])
            age = float(args.quick[2])
            height = float(args.quick[3])
            quick_mode(predictor, laserpower, gender, age, height)
        except ValueError as e:
            print(f"\n❌ 参数错误: {e}")
            sys.exit(1)
    
    elif args.batch:
        if not args.output:
            print("❌ 批量模式需要指定输出文件 (-o)")
            sys.exit(1)
        batch_mode(predictor, args.batch, args.output)
    
    else:
        interactive_mode(predictor)


if __name__ == "__main__":
    main()
