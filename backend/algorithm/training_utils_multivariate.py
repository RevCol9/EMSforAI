"""
多变量LSTM训练工具函数

用于准备多变量LSTM训练数据，支持一个设备的所有测点同时训练。

Author: EMSforAI Team
License: MIT
"""
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple, List, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from backend.algorithm.data_service import prepare_process_series
from backend.algorithm.lstm_model import MultiVariateLSTMPredictor
from backend.algorithm.training_utils import (
    calculate_rul_targets,
    clean_telemetry_data,
    MAX_RUL_DAYS,
    print_section_header,
    print_info_box,
)

log = logging.getLogger(__name__)


def prepare_multivariate_training_data(
    data: Dict[str, pd.DataFrame],
    asset_id: str,
    metric_ids: List[str],
    sequence_length: int = 30,
    window_days: int = 365,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, MinMaxScaler], List[str]]:
    """
    准备多变量LSTM训练数据
    
    将所有测点的数据对齐到相同的时间戳，构建多变量时间序列。
    多变量模型可以同时学习多个测点之间的关系，提高RUL预测的准确性。
    
    处理流程：
    1. 验证测点定义和数据完整性
    2. 为每个测点准备时间序列数据
    3. 数据清洗和标准化（每个测点独立标准化）
    4. 对齐所有测点的时间戳（使用交集）
    5. 计算RUL目标值（使用关键测点）
    6. 构建多变量序列（序列长度×特征数量）
    7. 划分训练集和验证集
    
    Args:
        data: 数据字典，包含所有表的数据（assets、metric_definitions、telemetry_process等）
        asset_id: 设备ID，用于过滤数据
        metric_ids: 测点ID列表，要参与训练的测点（必须都属于该设备）
        sequence_length: 序列长度，即时间窗口大小（默认30）
        window_days: 时间窗口天数，用于过滤历史数据（默认365天）
    
    Returns:
        Tuple包含：
            - X_train: 训练集特征，形状为(n_samples, sequence_length, n_features)
            - y_train: 训练集标签（归一化的RUL），形状为(n_samples,)
            - X_val: 验证集特征，形状为(n_samples, sequence_length, n_features)
            - y_val: 验证集标签（归一化的RUL），形状为(n_samples,)
            - scalers: 每个测点的MinMaxScaler字典，key为metric_id，用于推理时反标准化
            - feature_names: 特征名称列表（与metric_ids相同顺序），用于标识每个特征维度
    
    Raises:
        ValueError: 当测点定义不存在、数据不足、或时间戳对齐失败时抛出
    
    Note:
        - 每个测点独立进行标准化，保持各自的分布特性
        - RUL目标值基于关键测点（优先选择温度或振动测点）计算
        - 时间戳对齐使用交集，确保所有测点在相同时间点都有数据
    """
    # 检查测点定义
    metric_defs = data.get("metric_definitions", pd.DataFrame())
    if metric_defs.empty:
        raise ValueError(f"未找到测点定义数据，请先导入metric_definitions数据")
    
    # 验证所有测点都属于该设备
    asset_metrics = metric_defs[metric_defs["asset_id"] == asset_id]
    if asset_metrics.empty:
        raise ValueError(f"设备 {asset_id} 没有测点定义")
    
    available_metric_ids = asset_metrics["metric_id"].tolist()
    invalid_metrics = [m for m in metric_ids if m not in available_metric_ids]
    if invalid_metrics:
        raise ValueError(
            f"以下测点不属于设备 {asset_id}: {', '.join(invalid_metrics)}\n"
            f"可用测点: {', '.join(available_metric_ids[:10])}"
            + (f" (共{len(available_metric_ids)}个)" if len(available_metric_ids) > 10 else "")
        )
    
    # 准备每个测点的时间序列
    metric_dataframes = {}
    metric_scalers = {}
    metric_crit_thresholds = {}
    
    print(f"  准备 {len(metric_ids)} 个测点的数据...")
    
    for metric_id in metric_ids:
        metric_def = metric_defs[metric_defs["metric_id"] == metric_id].iloc[0]
        crit_threshold = metric_def["crit_threshold"]
        
        if crit_threshold is None:
            log.warning(f"  测点 {metric_id} 没有临界阈值，跳过")
            continue
        
        metric_crit_thresholds[metric_id] = crit_threshold
        
        # 准备时间序列
        df = prepare_process_series(
            data,
            metric_id,
            asset_id=asset_id,
            window_days=window_days,
            machine_state=None,  # 训练时不过滤工况
            quality_threshold=1,
        )
        
        if df.empty or len(df) < sequence_length + 10:
            log.warning(f"  测点 {metric_id} 数据不足（需要至少{sequence_length + 10}个点，当前{len(df)}个），跳过")
            continue
        
        # 数据清洗
        df = clean_telemetry_data(
            df,
            metric_id=metric_id,
            crit_threshold=crit_threshold,
            remove_outliers=True,
            remove_duplicates=True,
            smooth_noise=False,
            min_time_interval_seconds=0.0,
        )
        
        if df.empty or len(df) < sequence_length + 10:
            log.warning(f"  测点 {metric_id} 清洗后数据不足，跳过")
            continue
        
        # 排序
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # 标准化
        values = df["value"].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        values_scaled = scaler.fit_transform(values)
        
        metric_dataframes[metric_id] = pd.DataFrame({
            "timestamp": df["timestamp"].values,
            "value": values_scaled.flatten(),
            "value_raw": df["value"].values,  # 保留原始值用于RUL计算
        })
        metric_scalers[metric_id] = scaler
        
        log.info(f"  ✓ {metric_id}: {len(df)} 个数据点")
    
    if not metric_dataframes:
        raise ValueError(f"没有足够的测点数据用于训练（至少需要1个测点）")
    
    # 对齐时间戳（使用时间戳的交集）
    print(f"  对齐时间戳...")
    all_timestamps = set(metric_dataframes[list(metric_dataframes.keys())[0]]["timestamp"])
    for metric_id, df in metric_dataframes.items():
        all_timestamps = all_timestamps.intersection(set(df["timestamp"]))
    
    if len(all_timestamps) < sequence_length + 10:
        raise ValueError(
            f"对齐后时间戳数量不足（{len(all_timestamps)}个），需要至少{sequence_length + 10}个"
        )
    
    # 转换为排序后的列表
    all_timestamps = sorted(list(all_timestamps))
    
    # 构建对齐后的数据
    aligned_data = {}
    for metric_id in metric_dataframes.keys():
        df = metric_dataframes[metric_id]
        df_aligned = df[df["timestamp"].isin(all_timestamps)].sort_values("timestamp")
        aligned_data[metric_id] = {
            "value": df_aligned["value"].values,
            "value_raw": df_aligned["value_raw"].values,
        }
    
    log.info(f"  ✓ 对齐后时间戳数量: {len(all_timestamps)}")
    
    # 计算RUL目标值（使用所有测点中最短的RUL）
    # 对于多变量模型，我们使用最关键的测点（通常是温度或振动）来计算RUL
    # 或者使用所有测点的最小RUL
    print(f"  计算RUL目标值...")
    
    # 选择关键测点（优先选择温度或振动）
    key_metric = None
    for metric_id in metric_ids:
        if "TEMP" in metric_id or "VIB" in metric_id:
            key_metric = metric_id
            break
    
    if key_metric is None:
        key_metric = metric_ids[0]  # 如果没有温度或振动，使用第一个测点
    
    crit_threshold = metric_crit_thresholds[key_metric]
    values_raw = aligned_data[key_metric]["value_raw"]
    timestamps = pd.Series(all_timestamps)
    
    # 判断趋势方向
    is_rising = True  # 默认上升趋势（值越大越差）
    
    rul_targets = calculate_rul_targets(
        values_raw,
        timestamps,
        crit_threshold,
        is_rising,
    )
    
    # 将RUL缩放到[0, 1]范围
    rul_targets_scaled = rul_targets / MAX_RUL_DAYS
    
    # 准备多变量序列
    # 构建数据字典：key为metric_id，value为标准化后的值数组
    data_dict = {metric_id: aligned_data[metric_id]["value"] for metric_id in metric_ids if metric_id in aligned_data}
    
    if not data_dict:
        raise ValueError("没有可用的测点数据")
    
    predictor = MultiVariateLSTMPredictor(
        sequence_length=sequence_length,
        n_features=len(data_dict),
    )
    
    X, y = predictor.prepare_multivariate_sequences(data_dict, rul_targets_scaled)
    
    if len(X) < 20:
        raise ValueError(f"序列数量不足，只有{len(X)}个，需要至少20个")
    
    # 划分训练集和验证集（80/20）
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    feature_names = list(data_dict.keys())
    
    log.info(f"多变量训练数据准备完成:")
    log.info(f"  测点数量: {len(feature_names)}")
    log.info(f"  总序列数: {len(X)}")
    log.info(f"  训练集: {len(X_train)} 个序列")
    log.info(f"  验证集: {len(X_val)} 个序列")
    log.info(f"  序列长度: {sequence_length}")
    log.info(f"  特征维度: {len(feature_names)}")
    log.info(f"  RUL范围: {y_train.min() * MAX_RUL_DAYS:.1f} - {y_train.max() * MAX_RUL_DAYS:.1f} 天")
    
    return X_train, y_train, X_val, y_val, metric_scalers, feature_names

