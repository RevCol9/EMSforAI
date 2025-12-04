"""
EMSforAI RUL计算诊断脚本

本脚本用于诊断为什么RUL（剩余使用寿命）无法计算，帮助开发者快速定位问题。
会检查数据完整性、测点定义、数据质量、工况状态等多个方面。

使用方法：
    python -m backend.algorithm.diagnose_rul

Author: EMSforAI Team
License: MIT
"""
from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from backend.algorithm.data_service import load_data_from_db, prepare_process_series
from backend.algorithm.algorithms_engine import AlgorithmEngine
import pandas as pd


def diagnose_rul(asset_id: str = "CNC-MAZAK-01", metric_id: str = "CNC01_SPINDLE_TEMP"):
    """
    诊断RUL计算问题
    
    系统性地检查可能导致RUL无法计算的各种原因，包括：
    - 测点定义是否存在
    - 过程数据是否充足
    - 数据质量是否符合要求
    - 工况状态分布是否合理
    - 数据趋势是否明显
    
    Args:
        asset_id: 设备ID，默认为CNC-MAZAK-01
        metric_id: 测点ID，默认为CNC01_SPINDLE_TEMP
    """
    print("=" * 60)
    print(f"RUL计算诊断")
    print("=" * 60)
    print(f"\n设备ID: {asset_id}")
    print(f"测点ID: {metric_id}")
    print()
    
    # 加载数据
    print("1. 加载数据...")
    data = load_data_from_db(asset_id=asset_id)
    
    # 检查测点定义
    print("\n2. 检查测点定义...")
    metric_defs = data.get("metric_definitions", pd.DataFrame())
    if metric_defs.empty:
        print("  ✗ 没有测点定义")
        return
    
    metric_def = metric_defs[metric_defs["metric_id"] == metric_id]
    if metric_def.empty:
        print(f"  ✗ 测点 {metric_id} 不存在")
        print(f"  可用测点: {metric_defs['metric_id'].tolist()}")
        return
    
    print(f"  ✓ 测点定义存在")
    print(f"    警告阈值: {metric_def.iloc[0]['warn_threshold']}")
    print(f"    临界阈值: {metric_def.iloc[0]['crit_threshold']}")
    
    # 检查过程数据
    print("\n3. 检查过程数据...")
    process_df = data.get("telemetry_process", pd.DataFrame())
    if process_df.empty:
        print("  ✗ 没有过程数据")
        return
    
    print(f"  ✓ 总过程数据: {len(process_df)} 条")
    
    # 过滤该测点的数据
    metric_data = process_df[process_df["metric_id"] == metric_id]
    print(f"  ✓ 该测点数据: {len(metric_data)} 条")
    
    if metric_data.empty:
        print(f"  ✗ 测点 {metric_id} 没有数据")
        return
    
    # 检查数据质量
    print("\n4. 检查数据质量...")
    quality_1 = len(metric_data[metric_data["quality"] == 1])
    quality_0 = len(metric_data[metric_data["quality"] == 0])
    print(f"  高质量(quality=1): {quality_1} 条")
    print(f"  低质量(quality=0): {quality_0} 条")
    
    # 检查工况状态
    print("\n5. 检查工况状态分布...")
    if "machine_state" in metric_data.columns:
        state_counts = metric_data["machine_state"].value_counts()
        print(f"  工况状态分布:")
        for state, count in state_counts.items():
            state_name = {0: "停机", 1: "待机", 2: "加工"}.get(state, f"未知({state})")
            print(f"    {state_name}: {count} 条")
    else:
        print("  ⚠️  数据中没有machine_state列")
    
    # 测试不同过滤条件
    print("\n6. 测试不同过滤条件...")
    
    # 不过滤工况
    df_no_filter = prepare_process_series(
        data, metric_id, asset_id=asset_id, window_days=30,
        machine_state=None, quality_threshold=1
    )
    print(f"  不过滤工况: {len(df_no_filter)} 条数据")
    
    # 只过滤加工状态
    df_state_2 = prepare_process_series(
        data, metric_id, asset_id=asset_id, window_days=30,
        machine_state=2, quality_threshold=1
    )
    print(f"  只过滤加工状态(machine_state=2): {len(df_state_2)} 条数据")
    
    # 检查时间窗口
    if not metric_data.empty:
        print("\n7. 检查时间范围...")
        metric_data["timestamp"] = pd.to_datetime(metric_data["timestamp"])
        min_time = metric_data["timestamp"].min()
        max_time = metric_data["timestamp"].max()
        days_span = (max_time - min_time).days
        print(f"  最早数据: {min_time}")
        print(f"  最新数据: {max_time}")
        print(f"  时间跨度: {days_span} 天")
        
        # 检查最近30天的数据
        cutoff = max_time - pd.Timedelta(days=30)
        recent_data = metric_data[metric_data["timestamp"] >= cutoff]
        print(f"  最近30天数据: {len(recent_data)} 条")
    
    # 尝试计算RUL
    print("\n8. 尝试计算RUL...")
    engine = AlgorithmEngine(
        window_days=30,
        quality_threshold=1,
        machine_state_filter=2,  # 只分析加工状态
    )
    
    rul, slope = engine.analyze_rul(data, asset_id, metric_id)
    print(f"  结果: RUL={rul}, 斜率={slope}")
    
    if rul is None:
        print("\n9. 诊断结果:")
        print("  ✗ RUL无法计算")
        print("\n  可能原因:")
        if len(df_state_2) < 3:
            print(f"    - 加工状态数据不足（只有{len(df_state_2)}条，需要至少3条）")
            print(f"    - 建议: 尝试不过滤工况状态，或增加时间窗口")
        else:
            print(f"    - 数据量足够（{len(df_state_2)}条），可能是趋势不明显或计算错误")
    else:
        print(f"\n  ✓ RUL计算成功: {rul} 天")
    
    # 显示数据样本
    if not df_state_2.empty:
        print("\n10. 数据样本（最近10条，加工状态）:")
        print(df_state_2[["timestamp", "value", "quality", "machine_state"]].tail(10).to_string())


if __name__ == "__main__":
    diagnose_rul()

