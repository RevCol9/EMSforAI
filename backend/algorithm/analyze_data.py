"""
EMSforAI 数据分析工具脚本

本脚本用于分析指定设备和测点的数据，包括数据统计、RUL目标值计算等。
主要用于数据探索和模型训练前的数据质量检查。

使用方法：
    python -m backend.algorithm.analyze_data

Author: EMSforAI Team
License: MIT
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from backend.algorithm.data_service import load_data_from_db, prepare_process_series
from backend.algorithm.training_utils import calculate_rul_targets


def analyze_data(asset_id: str = "CNC-MAZAK-01", metric_id: str = "CNC01_LOAD"):
    """
    分析指定设备和测点的数据
    
    包括：
    - 数据点数量统计
    - 数值范围检查
    - RUL目标值统计
    - 数据分布分析
    
    Args:
        asset_id: 设备ID，默认为CNC-MAZAK-01
        metric_id: 测点ID，默认为CNC01_LOAD
    """
    print(f"正在分析 {asset_id} - {metric_id}")
    
    # 步骤1: 从数据库加载数据
    data = load_data_from_db(asset_id=asset_id)
    
    # 步骤2: 准备时间序列数据
    # 使用365天窗口，仅高质量数据，所有工况状态
    df = prepare_process_series(
        data,
        metric_id,
        asset_id=asset_id,
        window_days=365,
        machine_state=None,
        quality_threshold=1,
    )
    
    if df.empty:
        print("✗ 未找到数据！")
        return

    # 步骤3: 显示数据统计信息
    print(f"\n数据统计:")
    print(f"  数据点数量: {len(df)}")
    print(f"  数值范围: {df['value'].min():.2f} - {df['value'].max():.2f}")
    
    # 步骤4: 获取测点定义和临界阈值
    metric_defs = data.get("metric_definitions", pd.DataFrame())
    metric_def = metric_defs[metric_defs["metric_id"] == metric_id]
    
    if metric_def.empty:
        print("✗ 未找到测点定义")
        return
        
    crit_threshold = metric_def.iloc[0]["crit_threshold"]
    print(f"  临界阈值: {crit_threshold}")
    
    # 步骤5: 计算RUL目标值（用于训练）
    is_rising = True  # 假设为上升趋势（值越大越差）
    rul_targets = calculate_rul_targets(
        df["value"].values,
        df["timestamp"],
        crit_threshold,
        is_rising,
    )
    
    # 步骤6: 显示RUL目标值统计
    print(f"\nRUL目标值统计:")
    print(f"  最小值: {rul_targets.min():.2f} 天")
    print(f"  最大值: {rul_targets.max():.2f} 天")
    print(f"  平均值: {rul_targets.mean():.2f} 天")
    print(f"  标准差: {rul_targets.std():.2f} 天")
    
    # 检查RUL分布
    max_rul_days = 30  # 从training_utils导入的MAX_RUL_DAYS
    max_rul_count = np.sum(rul_targets >= max_rul_days)
    print(f"  达到最大RUL ({max_rul_days}天) 的数据点: {max_rul_count} ({max_rul_count/len(rul_targets)*100:.1f}%)")
    
    zero_rul_count = np.sum(rul_targets <= 0)
    print(f"  RUL为0的数据点: {zero_rul_count} ({zero_rul_count/len(rul_targets)*100:.1f}%)")

if __name__ == "__main__":
    analyze_data()
