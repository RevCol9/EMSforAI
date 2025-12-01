# coding:utf-8

from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional 


class Engine(ABC):
    """算法引擎的抽象基类，约定统一的生命周期"""

    def __init__(self, **kwargs: Any) -> None:
        self.params = kwargs  # 保存参数，方便子类读取

    @abstractmethod
    def load(self, data_bundle: Dict[str, Any]) -> Any:
        """加载或预处理数据，返回上下文对象。"""
        raise NotImplementedError

    @abstractmethod
    def analyze_metric(self, context: Any, defn: Dict[str, Any], device_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """针对单个指标执行分析
        
        Args:
            context: 数据上下文(data_bundle)，包含所有数据表的字典
            defn: 指标定义字典，必须包含 metric_key 字段
            device_id: 设备ID，如果指定则只分析该设备的该指标
            
        Returns:
            分析结果字典，包含健康度、趋势、RUL等信息；如果数据不足则返回 None
        """
        raise NotImplementedError

    @abstractmethod
    def analyze_device(self, context: Any, device_id: Optional[int] = None) -> Dict[str, Any]:
        """针对设备聚合多个指标的分析结果
        
        Args:
            context: 数据上下文(data_bundle)，包含所有数据表的字典
            device_id: 设备ID，如果为 None 则分析所有设备
            
        Returns:
            设备分析结果字典，包含设备健康度、各指标分析结果等
        """
        raise NotImplementedError
