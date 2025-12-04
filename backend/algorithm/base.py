"""
EMSforAI 算法引擎抽象基类

本模块定义了算法引擎的抽象基类，为所有算法引擎实现提供统一的接口规范。
通过继承此基类，可以轻松扩展新的算法引擎实现。

Author: EMSforAI Team
License: MIT
"""
# coding:utf-8

from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional 

log = logging.getLogger(__name__)


class Engine(ABC):
    """
    算法引擎的抽象基类
    
    定义了算法引擎的统一生命周期和接口规范。所有具体的算法引擎实现都应该继承此类。
    
    生命周期：
    1. __init__: 初始化引擎，保存配置参数
    2. load: 加载和预处理数据
    3. analyze_metric: 分析单个指标
    4. analyze_device: 分析整个设备（聚合多个指标）
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        初始化算法引擎
        
        Args:
            **kwargs: 引擎配置参数，会被保存到self.params中供子类使用
        """
        self.params = kwargs  # 保存参数，方便子类读取

    @abstractmethod
    def load(self, data_bundle: Dict[str, Any]) -> Any:
        """
        加载或预处理数据
        
        从数据包中提取和预处理所需的数据，返回处理后的上下文对象。
        子类应该在此方法中完成数据清洗、特征提取等预处理工作。
        
        Args:
            data_bundle: 数据包字典，包含所有数据表（assets、metric_definitions、telemetry_process等）
        
        Returns:
            处理后的数据上下文对象，类型由子类决定
        """
        raise NotImplementedError

    @abstractmethod
    def analyze_metric(self, context: Any, defn: Dict[str, Any], device_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        针对单个指标执行分析
        
        对指定的测点进行健康度分析、趋势分析、RUL预测等。
        
        Args:
            context: 数据上下文对象（由load方法返回）
            defn: 指标定义字典，必须包含 metric_key 或 metric_id 字段
            device_id: 设备ID，如果指定则只分析该设备的该指标
        
        Returns:
            分析结果字典，包含以下字段（如果数据充足）：
                - health_score: 健康度分数（0-100）
                - current_value: 当前值
                - trend: 趋势信息
                - rul_days: 剩余使用寿命（天数）
                - alert_level: 告警级别（normal/warning/critical）
            如果数据不足，返回 None
        """
        raise NotImplementedError

    @abstractmethod
    def analyze_device(self, context: Any, device_id: Optional[int] = None) -> Dict[str, Any]:
        """
        针对设备聚合多个指标的分析结果
        
        对设备的所有测点进行分析，并聚合得到设备的综合健康度。
        
        Args:
            context: 数据上下文对象（由load方法返回）
            device_id: 设备ID，如果为 None 则分析所有设备
        
        Returns:
            设备分析结果字典，包含以下字段：
                - device_id: 设备ID
                - device_health_score: 设备综合健康度（0-100）
                - device_alert_level: 设备告警级别
                - metrics: 各指标的分析结果列表
                - downtime_risk: 停机风险（0-1）
                - throughput_impact: 产能影响（0-1）
        """
        raise NotImplementedError
