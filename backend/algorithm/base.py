# coding:utf-8

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional 


class AlgorithmEngine(ABC):
    """算法引擎的抽象基类，约定统一的生命周期"""

    def __init__(self, **kwargs: Any) -> None:
        self.params = kwargs  # 保存参数，方便子类读取

    @abstractmethod
    def load(self, data_bundle: Dict[str, Any]) -> Any:
        """加载或预处理数据，返回上下文对象。"""
        raise NotImplementedError

    @abstractmethod
    def analyze_metric(self, context: Any, defn: Dict[str, Any], device_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """针对单个指标执行分析，返回结果字典或 None。"""
        raise NotImplementedError

    @abstractmethod
    def analyze_device(self, context: Any, device_id: Optional[int] = None) -> Dict[str, Any]:
        """针对设备聚合多个指标的分析结果。"""
        raise NotImplementedError
