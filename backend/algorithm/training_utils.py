"""
LSTM训练工具函数

包含数据准备、RUL计算、数据清洗等训练相关的工具函数。

Author: EMSforAI Team
License: MIT
"""
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from backend.algorithm.data_service import prepare_process_series
from backend.algorithm.lstm_model import LSTMRULPredictor

log = logging.getLogger(__name__)

# 最大RUL天数，用于裁剪与缩放
MAX_RUL_DAYS = 90


def calculate_rul_targets(
    values: np.ndarray,
    timestamps: pd.Series,
    crit_threshold: float,
    is_rising: bool = True,
) -> np.ndarray:
    """
    计算RUL目标值（用于训练）
    
    对于每个时间点，计算从该点到达到临界阈值的时间（天数）
    
    Args:
        values: 时间序列值
        timestamps: 时间戳序列
        crit_threshold: 临界阈值
        is_rising: 是否为上升趋势（值越大越差）
    
    Returns:
        RUL值数组，形状与values相同
    """
    rul_targets = np.zeros(len(values))
    
    # 从后往前计算RUL
    for i in range(len(values) - 1, -1, -1):
        current_value = values[i]
        
        # 检查是否已达到临界值
        if is_rising and current_value >= crit_threshold:
            rul_targets[i] = 0.0
        elif not is_rising and current_value <= crit_threshold:
            rul_targets[i] = 0.0
        else:
            # 向前查找达到阈值的时间点
            found = False
            for j in range(i + 1, len(values)):
                if is_rising and values[j] >= crit_threshold:
                    # 计算天数差
                    days_diff = (timestamps.iloc[j] - timestamps.iloc[i]).total_seconds() / 86400.0
                    rul_targets[i] = max(0.0, days_diff)
                    found = True
                    break
                elif not is_rising and values[j] <= crit_threshold:
                    days_diff = (timestamps.iloc[j] - timestamps.iloc[i]).total_seconds() / 86400.0
                    rul_targets[i] = max(0.0, days_diff)
                    found = True
                    break
            
            if not found:
                # 如果未来没有达到阈值，使用线性外推
                # 计算趋势斜率
                if i < len(values) - 1:
                    delta_days = (timestamps.iloc[i + 1] - timestamps.iloc[i]).total_seconds() / 86400.0
                    # 避免时间差为0导致除零
                    if abs(delta_days) < 1e-6:
                        rul_targets[i] = MAX_RUL_DAYS
                    else:
                        slope = (values[i + 1] - values[i]) / delta_days
                        if abs(slope) > 1e-6:
                            if is_rising:
                                rul_targets[i] = max(0.0, (crit_threshold - current_value) / slope)
                            else:
                                rul_targets[i] = max(0.0, (current_value - crit_threshold) / abs(slope))
                        else:
                            rul_targets[i] = MAX_RUL_DAYS
                else:
                    rul_targets[i] = MAX_RUL_DAYS

    # 统一裁剪到合理范围，避免极端大值影响训练
    rul_targets = np.clip(rul_targets, 0.0, MAX_RUL_DAYS)
    return rul_targets


def clean_telemetry_data(
    df: pd.DataFrame,
    metric_id: str,
    crit_threshold: float,
    remove_outliers: bool = True,
    remove_duplicates: bool = True,
    smooth_noise: bool = False,
    min_time_interval_seconds: float = 0.0,
) -> pd.DataFrame:
    """
    清洗过程数据，提升训练数据质量
    
    Args:
        df: 原始时间序列DataFrame，包含timestamp, value, quality, machine_state
        metric_id: 测点ID（用于日志）
        crit_threshold: 临界阈值（用于范围检查）
        remove_outliers: 是否移除异常值（3-sigma规则）
        remove_duplicates: 是否移除重复时间戳
        smooth_noise: 是否进行平滑处理（移动平均）
        min_time_interval_seconds: 最小时间间隔（秒），小于此值的记录会被移除
    
    Returns:
        清洗后的DataFrame
    """
    if df.empty:
        return df
    
    original_count = len(df)
    df = df.copy()
    
    # 1. 移除重复时间戳（保留第一个）
    if remove_duplicates:
        before_dedup = len(df)
        df = df.drop_duplicates(subset=["timestamp"], keep="first")
        removed_dedup = before_dedup - len(df)
        if removed_dedup > 0:
            log.info(f"  移除重复时间戳: {removed_dedup} 条")
    
    # 2. 时间间隔检查（移除时间间隔过小的记录）
    if min_time_interval_seconds > 0 and len(df) > 1:
        df = df.sort_values("timestamp")
        time_diffs = df["timestamp"].diff().dt.total_seconds()
        mask = (time_diffs >= min_time_interval_seconds) | (time_diffs.isna())
        removed_interval = len(df) - mask.sum()
        df = df[mask]
        if removed_interval > 0:
            log.info(f"  移除时间间隔过小的记录: {removed_interval} 条")
    
    # 3. 数值范围检查（基于临界阈值）
    if "TEMP" in metric_id or "VIB" in metric_id:
        max_reasonable = crit_threshold * 1.5
        min_reasonable = 0.0
    elif "PRESSURE" in metric_id:
        max_reasonable = crit_threshold * 2.0
        min_reasonable = 0.0
    elif "FLOW" in metric_id or "LOAD" in metric_id:
        max_reasonable = crit_threshold * 2.0
        min_reasonable = 0.0
    else:
        # 默认：使用3-sigma规则
        mean_val = df["value"].mean()
        std_val = df["value"].std()
        max_reasonable = mean_val + 5 * std_val if std_val > 0 else mean_val * 2
        min_reasonable = mean_val - 5 * std_val if std_val > 0 else 0.0
    
    before_range = len(df)
    df = df[(df["value"] >= min_reasonable) & (df["value"] <= max_reasonable)]
    removed_range = before_range - len(df)
    if removed_range > 0:
        log.info(f"  移除超出合理范围的记录: {removed_range} 条 (范围: {min_reasonable:.2f} - {max_reasonable:.2f})")
    
    # 4. 异常值检测（3-sigma规则）
    if remove_outliers and len(df) > 10:
        mean_val = df["value"].mean()
        std_val = df["value"].std()
        
        if std_val > 0:
            z_scores = np.abs((df["value"] - mean_val) / std_val)
            # 使用3.5-sigma（更宽松，避免过度删除）
            before_outlier = len(df)
            df = df[z_scores <= 3.5]
            removed_outlier = before_outlier - len(df)
            if removed_outlier > 0:
                log.info(f"  移除异常值(3.5-sigma): {removed_outlier} 条")
    
    # 5. 平滑处理（移动平均，可选）
    if smooth_noise and len(df) > 5:
        window_size = min(5, len(df) // 10)
        if window_size >= 3:
            df = df.sort_values("timestamp")
            df["value"] = df["value"].rolling(window=window_size, center=True, min_periods=1).mean()
            log.info(f"  应用平滑处理: 窗口大小={window_size}")
    
    # 6. 确保时间序列连续（排序）
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    final_count = len(df)
    removed_total = original_count - final_count
    if removed_total > 0:
        log.info(f"  数据清洗完成: 原始 {original_count} 条 -> 清洗后 {final_count} 条 (移除 {removed_total} 条, {removed_total/original_count*100:.1f}%)")
    
    return df


def prepare_training_data(
    data: Dict[str, pd.DataFrame],
    asset_id: str,
    metric_id: str,
    sequence_length: int = 90,
    window_days: int = 365,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """
    准备训练数据
    
    Args:
        data: 数据字典
        asset_id: 设备ID
        metric_id: 测点ID
        sequence_length: 序列长度
        window_days: 时间窗口（天）
    
    Returns:
        (X_train, y_train, X_val, y_val, scaler) 元组
    """
    # 先检查测点定义，确保设备ID和测点ID匹配
    metric_defs = data.get("metric_definitions", pd.DataFrame())
    if metric_defs.empty:
        raise ValueError(f"未找到测点定义数据，请先导入metric_definitions数据")
    
    metric_def = metric_defs[metric_defs["metric_id"] == metric_id]
    
    if metric_def.empty:
        # 检查是否是设备ID和测点ID不匹配
        available_metrics = metric_defs[metric_defs["asset_id"] == asset_id]["metric_id"].tolist()
        if available_metrics:
            raise ValueError(
                f"✗ 测点 {metric_id} 不属于设备 {asset_id}。\n"
                f"该设备的可用测点: {', '.join(available_metrics[:10])}"
                + (f" (共{len(available_metrics)}个)" if len(available_metrics) > 10 else "")
            )
        else:
            raise ValueError(
                f"✗ 测点 {metric_id} 不存在，且设备 {asset_id} 没有任何测点定义。\n"
                f"请检查：\n"
                f"  1. 设备ID是否正确\n"
                f"  2. 是否已导入metric_definitions数据"
            )
    
    # 检查测点是否属于该设备
    metric_asset_id = metric_def.iloc[0]["asset_id"]
    if metric_asset_id != asset_id:
        raise ValueError(
            f"✗ 设备ID和测点ID不匹配！\n"
            f"  测点 {metric_id} 属于设备: {metric_asset_id}\n"
            f"  你指定的设备ID: {asset_id}\n"
            f"请使用正确的设备ID: {metric_asset_id}\n"
            f"或者使用该设备的测点ID"
        )
    
    crit_threshold = metric_def.iloc[0]["crit_threshold"]
    if crit_threshold is None:
        raise ValueError(f"测点 {metric_id} 没有临界阈值")
    
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
        raise ValueError(f"数据不足，需要至少{sequence_length + 10}个数据点，当前只有{len(df)}个")
    
    # 数据清洗
    print("  清洗数据...")
    df = clean_telemetry_data(
        df,
        metric_id=metric_id,
        crit_threshold=crit_threshold,
        remove_outliers=True,
        remove_duplicates=True,
        smooth_noise=False,  # 暂时关闭平滑，避免过度平滑真实趋势
        min_time_interval_seconds=0.0,  # 不限制最小时间间隔
    )
    
    if df.empty or len(df) < sequence_length + 10:
        raise ValueError(f"清洗后数据不足，需要至少{sequence_length + 10}个数据点，当前只有{len(df)}个")
    
    # 排序
    df = df.sort_values("timestamp")
    
    # 数据预处理
    values = df["value"].values.reshape(-1, 1)
    
    # 标准化
    scaler = MinMaxScaler()
    values_scaled = scaler.fit_transform(values)
    
    # 计算RUL目标值
    # 根据测点类型判断趋势方向
    if "LOAD" in metric_id or "PRESSURE" in metric_id or "FLOW" in metric_id or "LUBE" in metric_id:
        is_rising = True
    elif "TEMP" in metric_id or "VIB" in metric_id:
        is_rising = True
    else:
        is_rising = True
    
    rul_targets = calculate_rul_targets(
        df["value"].values,
        df["timestamp"],
        crit_threshold,
        is_rising,
    )
    
    # 诊断：检查RUL标签分布
    rul_stats = {
        "min": rul_targets.min(),
        "max": rul_targets.max(),
        "mean": rul_targets.mean(),
        "median": np.median(rul_targets),
        "std": rul_targets.std(),
        "count_0": np.sum(rul_targets == 0),
        "count_max": np.sum(rul_targets >= MAX_RUL_DAYS * 0.95),
        "count_0_to_30": np.sum((rul_targets > 0) & (rul_targets <= 30)),
        "count_30_to_90": np.sum((rul_targets > 30) & (rul_targets <= 90)),
        "count_90_plus": np.sum(rul_targets > 90),
    }
    log.info(f"  RUL标签统计:")
    log.info(f"    范围: {rul_stats['min']:.1f} - {rul_stats['max']:.1f} 天")
    log.info(f"    均值: {rul_stats['mean']:.1f} 天, 中位数: {rul_stats['median']:.1f} 天")
    log.info(f"    标准差: {rul_stats['std']:.1f} 天")
    log.info(f"    分布: 0天={rul_stats['count_0']}, 0-30天={rul_stats['count_0_to_30']}, 30-90天={rul_stats['count_30_to_90']}, 90+天={rul_stats['count_90_plus']}")
    log.info(f"    接近上限(≥{MAX_RUL_DAYS*0.95:.0f}天): {rul_stats['count_max']} ({rul_stats['count_max']/len(rul_targets)*100:.1f}%)")
    
    # 如果大部分RUL都是上限，给出警告
    if rul_stats['count_max'] / len(rul_targets) > 0.5:
        log.warning(f"  ⚠️  超过50%的RUL标签接近上限，标签信息量不足，建议:")
        log.warning(f"     1. 减小MAX_RUL_DAYS（当前{MAX_RUL_DAYS}天）")
        log.warning(f"     2. 减小window_days，只使用接近故障的数据")
        log.warning(f"     3. 检查crit_threshold是否合理（当前{crit_threshold:.2f}）")

    # 将RUL缩放到[0, 1]范围，提升训练稳定性
    rul_targets_scaled = rul_targets / MAX_RUL_DAYS

    # 准备序列
    predictor = LSTMRULPredictor(sequence_length=sequence_length, n_features=1)
    X, y = predictor.prepare_sequences(values_scaled, rul_targets_scaled)
    
    if len(X) < 20:
        raise ValueError(f"序列数量不足，只有{len(X)}个，需要至少20个")
    
    # 划分训练集和验证集（80/20）
    # 使用随机划分，确保训练集和验证集分布一致
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    log.info(f"训练数据准备完成:")
    log.info(f"  总序列数: {len(X)}")
    log.info(f"  训练集: {len(X_train)} 个序列")
    log.info(f"  验证集: {len(X_val)} 个序列")
    log.info(f"  序列长度: {sequence_length}")
    log.info(
        f"  RUL范围: {y_train.min() * MAX_RUL_DAYS:.1f} - {y_train.max() * MAX_RUL_DAYS:.1f} 天"
    )
    
    return X_train, y_train, X_val, y_val, scaler


def print_section_header(title: str, char: str = "=", width: int = 70):
    """打印美观的分节标题"""
    print()
    print(char * width)
    print(f"{title:^{width}}")
    print(char * width)
    print()


def print_info_box(title: str, items: Dict[str, Any], width: int = 70):
    """打印信息框"""
    print(f"\n{title}")
    print("─" * width)
    for key, value in items.items():
        if isinstance(value, float):
            if abs(value) < 1:
                print(f"  {key:.<30} {value:>10.4f}")
            else:
                print(f"  {key:.<30} {value:>10.2f}")
        else:
            print(f"  {key:.<30} {str(value):>10}")
    print("─" * width)

