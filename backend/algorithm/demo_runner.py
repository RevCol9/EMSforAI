"""
EMSforAI 算法引擎演示脚本

本脚本演示如何使用算法引擎进行设备健康分析。
包括数据加载、算法执行、结果展示等功能。

使用方法：
    python backend/algorithm/demo_runner.py

Author: EMSforAI Team
License: MIT
"""
from __future__ import annotations

from typing import Any
from pathlib import Path
import sys
import json

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from backend.algorithm.algorithms_engine import AlgorithmEngine
from backend.algorithm.data_service import load_data_from_db


def run_demo(asset_id: str = "CNC-MAZAK-01", window_days: int = 30) -> Any:
    """
    运行算法引擎演示
    
    Args:
        asset_id: 要分析的设备ID，默认为CNC-MAZAK-01
        window_days: 分析窗口天数，默认30天
        
    Returns:
        设备分析结果
    """
    print("=" * 60)
    print(f"EMSforAI 算法引擎演示")
    print("=" * 60)
    print(f"\n设备ID: {asset_id}")
    print(f"分析窗口: {window_days} 天")
    print()
    
    # 加载数据
    print("正在加载数据...")
    try:
        data = load_data_from_db(asset_id=asset_id)
        print(f"✓ 数据加载完成")
    
        # 显示数据统计
        print("\n数据统计:")
        assets_count = len(data.get('assets', []))
        metrics_count = len(data.get('metric_definitions', []))
        process_count = len(data.get('telemetry_process', []))
        waveform_count = len(data.get('telemetry_waveform', []))
        maintenance_count = len(data.get('maintenance_records', []))
        knowledge_count = len(data.get('knowledge_base', []))
        
        print(f"  设备数: {assets_count}")
        print(f"  测点数: {metrics_count}")
        print(f"  过程数据: {process_count} 条")
        print(f"  波形数据: {waveform_count} 条")
        print(f"  运维记录: {maintenance_count} 条")
        print(f"  知识库: {knowledge_count} 条")
        
        # 检查数据完整性
        if process_count == 0:
            print("\n⚠️  警告: 没有过程数据，无法进行完整分析")
            print("   建议: 生成示例数据或导入telemetry_process.csv")
            print("   示例: 运行数据生成脚本创建示例数据")
        
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 创建算法引擎
    print("\n创建算法引擎...")
    engine = AlgorithmEngine(
        window_days=window_days,
        quality_threshold=1,
        machine_state_filter=2,  # 只分析加工状态数据
    )
    
    # 执行分析
    print("\n执行设备分析...")
    try:
        result = engine.analyze_asset(data, asset_id)
        
        # 显示结果
        print("\n" + "=" * 60)
        print(f"设备 {asset_id} 分析结果")
        print("=" * 60)
        print(f"\n健康分: {result.health_score:.1f}/100")
        if result.rul_days is None:
            print(f"剩余寿命: 无法计算")
        elif result.rul_days == 0.0:
            print(f"剩余寿命: 0 天 ⚠️ 设备已达到临界值，需要立即维护！")
        else:
            print(f"剩余寿命: {result.rul_days:.1f} 天")
        print(f"趋势斜率: {result.trend_slope:.6f} ({'上升' if result.trend_slope > 0 else '下降' if result.trend_slope < 0 else '稳定'})")
        print(f"预测置信度: {result.prediction_confidence:.3f}")
        
        print(f"\n故障诊断结果:")
        for fault, prob in result.diagnosis_result.items():
            print(f"  {fault}: {prob:.2%}")
        
        # 显示详细诊断信息
        if result.rul_days is None:
            print(f"\n⚠️  RUL无法计算的可能原因:")
            print(f"  1. 数据量不足（需要至少3个数据点，且在同一工况状态下）")
            print(f"  2. 趋势不明显（斜率接近0，无法预测未来趋势）")
            print(f"  3. 工况过滤过严（当前只分析machine_state=2的加工状态数据）")
            print(f"  4. 数据质量过滤（当前只使用quality=1的高质量数据）")
            print(f"  5. 时间窗口过短（当前窗口={window_days}天）")
            print(f"\n  建议:")
            print(f"  - 尝试增加时间窗口（如60天）")
            print(f"  - 尝试不过滤工况状态（machine_state_filter=None）")
            print(f"  - 检查数据中是否有足够的加工状态数据")
    
        # 显示多维度评分（用于雷达图）
        if hasattr(result, 'dimension_scores') and result.dimension_scores:
            print("\n多维度健康评分（用于雷达图）:")
            for dim_score in result.dimension_scores:
                print(f"  {dim_score.dimension_name}: {dim_score.health_score:.1f}/100")
                print(f"    测点: {dim_score.metric_id}, 当前值: {dim_score.current_value:.2f}")
                print(f"    趋势: {dim_score.trend}, 告警级别: {dim_score.alert_level}")
        
        # 显示时间序列数据统计（用于曲线图）
        if hasattr(result, 'time_series_data') and result.time_series_data:
            print("\n时间序列数据（用于曲线图）:")
            for dimension_name, points in result.time_series_data.items():
                print(f"  {dimension_name}: {len(points)} 个数据点")
                if points:
                    print(f"    时间范围: {points[0].timestamp} 到 {points[-1].timestamp}")
        
        # 显示统计信息（用于统计图）
        if hasattr(result, 'statistics') and result.statistics:
            print("\n统计信息（用于统计图）:")
            print(f"  总数据点数: {result.statistics.get('total_data_points', 0)}")
            date_range = result.statistics.get('date_range', {})
            if date_range:
                print(f"  时间范围: {date_range.get('start', 'N/A')} 到 {date_range.get('end', 'N/A')}")
        
        print("\n" + "=" * 60)
        print("\n完整结果（JSON格式，包含前端所需的所有数据）:")
        
        # 构建前端友好的JSON结构
        result_dict = {
            "asset_id": asset_id,
            "health_score": result.health_score,
            "rul_days": result.rul_days,
            "trend_slope": result.trend_slope,
            "diagnosis_result": result.diagnosis_result,
            "prediction_confidence": result.prediction_confidence,
            "model_version": result.model_version,
        }
        
        # 添加多维度评分（如果存在）
        if hasattr(result, 'dimension_scores') and result.dimension_scores:
            result_dict["dimension_scores"] = [
                {
                    "dimension_name": ds.dimension_name,
                    "metric_id": ds.metric_id,
                    "health_score": ds.health_score,
                    "current_value": ds.current_value,
                    "warn_threshold": ds.warn_threshold,
                    "crit_threshold": ds.crit_threshold,
                    "trend": ds.trend,
                    "alert_level": ds.alert_level,
                }
                for ds in result.dimension_scores
            ]
        
        # 添加时间序列数据（如果存在）
        if hasattr(result, 'time_series_data') and result.time_series_data:
            result_dict["time_series_data"] = {
                dim_name: [
                    {
                        "timestamp": point.timestamp,
                        "value": point.value,
                        "smoothed_value": point.smoothed_value,
                    }
                    for point in points
                ]
                for dim_name, points in result.time_series_data.items()
            }
        
        # 添加统计信息（如果存在）
        if hasattr(result, 'statistics') and result.statistics:
            result_dict["statistics"] = result.statistics
        
        # print(json.dumps(result_dict, indent=2, ensure_ascii=False, default=str))
    
        return result
        
    except Exception as e:
        print(f"✗ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    run_demo()
