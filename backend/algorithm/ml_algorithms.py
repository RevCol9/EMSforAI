# coding:utf-8

import logging  
import math  
import sys
from dataclasses import dataclass 
from pathlib import Path  
from typing import Any, Dict, Iterable, List, Optional, Tuple 

import numpy as np  
import pandas as pd  
import json  

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from backend.algorithm.base import AlgorithmEngine
from backend.algorithm.data_service import load_data_from_csv, aggregate_device_health

# 基础日志配置
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

# 设备关键程度权重
CRIT_WEIGHT = {"high": 1.2, "medium": 0.85, "low": 0.5}

# 默认数据目录
CSV_PATH = BASE_DIR / "data" / "csv"


@dataclass
class TrendResults:
    alpha: float  
    beta: float  
    R2: float  
    t_days: np.ndarray  
    y_items: np.ndarray 


class LinearAlgorithmEngine(AlgorithmEngine):

    def __init__(self, window_days: int = 30, quality_cutoff: float = 0.8) -> None:
        super().__init__(window_days=window_days, quality_cutoff=quality_cutoff)
        self.quality_cutoff = quality_cutoff  #

    # 数据加载/预处理
    def load(self, data_bundle: Dict[str, Any]) -> Dict[str, Any]:
        """透传数据，在此做缓存/索引预处理。"""
        return data_bundle

    # 数据准备：抽取目标指标
    def prepare_metric_series(
        self,
        data: Dict[str, Any],
        metric_key: str,
        device_id: Optional[int] = None,
    ) -> pd.DataFrame:
        df = data["inspection_submits"] 
        if device_id is not None:
            df = df[df["device_id"] == device_id]  # 按设备过滤

        df = df.copy()  # 避免修改原数据
        df["value"] = df["metrics"].apply(
            lambda line: float(line.get(metric_key, np.nan)) if isinstance(line, dict) else np.nan
        )  # 抽取目标指标
        df["quality"] = df["data_quality_score"].fillna(0.8) if "data_quality_score" in df else 0.8  # 质量评分
        df = df.dropna(subset=["value"])  # 去掉空值
        df["record_time"] = pd.to_datetime(df["recorded_at"])  # 转换 datetime
        cutoff = df["record_time"].max() - pd.Timedelta(days=self.window_days)  # 滑窗起点
        df = df[(df["record_time"] >= cutoff) & (df["quality"] >= self.quality_cutoff)]  # 滤窗口和质量
        return df.sort_values("record_time")[["record_time", "value", "quality"]] 

    # 线性回归
    def linear_regression(self, df: pd.DataFrame) -> Optional[TrendResults]:
        # 样本边界检查
        if len(df) < 3:  
            return None
        t0 = df["record_time"].min()  # 起始时间
        t_days = (df["record_time"] - t0).dt.total_seconds() / 86400.0  
        y = df["value"].astype(float).to_numpy()
        # 时间中心化
        t_center = t_days - t_days.mean()
        y_center = y - y.mean()  # 值中心化
        denom = np.sum(t_center ** 2)  # 方差
        if denom == 0:
            return None
        beta = float(np.sum(t_center * y_center) / denom)  
        alpha = float(y.mean() - beta * t_days.mean()) 
        y_hat = alpha + beta * t_days  # 预测值
        ss_res = float(np.sum((y - y_hat) ** 2))  
        ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1e-6  
        r2 = 1.0 - ss_res / ss_tot  
        log.info(f"R^2: {r2}")
        return TrendResults(alpha=alpha, beta=beta, R2=r2, t_days=t_days.to_numpy(), y_items=y)

    # RUL：撞严重阈值
    def compute_rul(self, defn: Dict[str, Any], trend: TrendResults, max_days: int = 365) -> Any:
        d = defn["trend_direction"]  # 趋势方向
        z_c = d * defn["crit_threshold"]  # 目标阈值
        # 趋势参数映射到统一坐标系 z
        alpha_z = d * trend.alpha  
        beta_z = d * trend.beta  
        t_now = trend.t_days[-1]  
        if beta_z <= 0: 
            return None, "状态稳定"
        t_star = (z_c - alpha_z) / beta_z  # 撞线时间
        rul = t_star - t_now  # 剩余
        if rul <= 0:
            return 0, "设备寿命已经结束"
        if rul > max_days:
            return max_days, "长期安全"
        return int(rul), "正在衰退"

    # 单指标健康度 h
    @staticmethod
    def compute_metric_health(defn: Dict[str, Any], current_val: float) -> float:
        '''健康度阈值归一化'''
        d = defn["trend_direction"]
        z = d * current_val
        z_w = d * defn["warn_threshold"]
        z_c = d * defn["crit_threshold"]
        s = (z - z_w) / (z_c - z_w)    # 线性插值
        # 归一化
        ss = np.clip(s, 0.0, 1.0)
        return round(100.0 * (1.0 - ss), 1)

    @staticmethod
    def metric_alert_level(df: Dict[str, Any], current_item: float) -> str:
        '''警告级别'''
        d = df["trend_direction"]
        w = d * current_item
        if w >= d * df["crit_threshold"]:
            return "危险级别"
        if w >= d * df["warn_threshold"]:
            return "警告级别"
        return "正常"
    
    @staticmethod
    def aggregate_device_health(metrics: List[Dict[str, Any]]) -> Any:
        '''设备多指标联合健康度'''
        # 初始化累加器和默认警告
        weighted_sum = 0.0
        weight_total = 0.0
        alert = "正常"
        for m in metrics:
            # 基础权重 * 重要性系数
            composite_Weight = m.get("weight_in_health", 1.0) * CRIT_WEIGHT.get(m.get("criticality", "medium"), 1.0)
            weighted_sum += composite_Weight * m["health_score"]
            weight_total += composite_Weight
            if m["aler_level"] == "critical":
                alert = "SOS: 危险"
            elif m["aler_level"] == "warning" and alert != "critical":
                alert = "warning: 需要注意"
        H_device = round(weighted_sum / weight_total, 1) if weight_total else 0.0   # 防止分母为0
        return H_device, alert




if __name__ == "__main__":
    data_bundle = load_data_from_csv()
    print("加载 CSV 数据表:", ", ".join(data_bundle.keys()))
    print("==" * 20)

    health_df = aggregate_device_health()
    print("设备健康度:")
    print(health_df.head())

