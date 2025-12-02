"""
算法引擎演示脚本

用于快速测试算法引擎功能，可直接运行查看设备分析结果
"""
from __future__ import annotations

from typing import Any
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from backend.algorithm.algorithms_engine import AlgorithmEngine
from backend.algorithm.data_service import load_data_from_db


def run_demo(device_id: int = 1, window_days: int = 50, quality_cutoff: float = 0.8) -> Any:
    """
    运行算法引擎演示
    
    Args:
        device_id: 要分析的设备ID，默认为1
        window_days: 分析窗口天数，默认50天
        quality_cutoff: 数据质量阈值，默认0.8
        
    Returns:
        设备分析结果字典
    """
    data_bundle = load_data_from_db()
    engine = AlgorithmEngine(window_days=window_days, quality_cutoff=quality_cutoff)
    result = engine.analyze_device(context=data_bundle, device_id=device_id)
    print(result)
    return result


if __name__ == "__main__":
    run_demo()

