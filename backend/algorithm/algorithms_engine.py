# coding:utf-8
import sys
import logging  
import math  

from dataclasses import dataclass 
from pathlib import Path  
from typing import Any, Dict, Iterable, List, Optional, Tuple 

import numpy as np  
import pandas as pd  
import json  

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from backend.algorithm.base import Engine
from backend.algorithm.constants import CRIT_WEIGHT

# 基础日志配置
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

# 默认数据目录
CSV_PATH = BASE_DIR / "data" / "csv"


@dataclass
class TrendResults:
    alpha: float  
    beta: float  
    R2: float  
    t_days: np.ndarray  
    y_items: np.ndarray 


def build_metrics_for_aggregation(metrics_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract minimal fields required for device_multi_indicator_health."""
    payload: List[Dict[str, Any]] = []
    for metric in metrics_results:
        payload.append(
            {
                "health_score": metric["health_score"],
                "weight_in_health": metric.get("weight_in_health", 1.0),
                "criticality": metric.get("criticality", "medium"),
                "alert_level": metric.get("alert_level", "normal"),
            }
        )
    return payload


class AlgorithmEngine(Engine):

    def __init__(self, window_days: int = 30, quality_cutoff: float = 0.8) -> None:
        super().__init__(window_days=window_days, quality_cutoff=quality_cutoff)
        self.window_days = window_days
        self.quality_cutoff = quality_cutoff

    # 数据加载/预处理
    def load(self, data_bundle: Dict[str, Any]) -> Dict[str, Any]:
        """透传数据，在此做缓存/索引预处理。"""
        return data_bundle

    def analyze_metric(self, context: Any, defn: Dict[str, Any], device_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        
        metric_key = defn.get("metric_key")
        if not metric_key:
            log.warning("指标定义中缺少 metric_key")
            return None
        
        # 获取指标的时间序列数据
        df_series = self.prepare_metric_series(context, metric_key, device_id)
        # 边界检查
        if df_series.empty or len(df_series) < 3:
            log.warning(f"指标: {metric_key} 数据量不足,无法进行分析")
            return None
        # 获取最新记录值
        current_value = float(df_series["value"].iloc[-1])
        health_score = self.compute_metric_health(defn=defn, current_val=current_value)
        alert_level = self.metric_alert_level(defn, current_value)
        trend = self.linear_regression(df_series)

        result = {
            "metric_key": metric_key,
            "device_id": device_id,
            "current_value": current_value,
            "health_score": health_score,
            "alert_level": alert_level,
            "data_points": len(df_series),
        }
        if trend is not None:
            rul_days, rul_status = self.compute_rul(defn=defn, trend=trend)
            result["rul_days"] = rul_days
            result["rul_status"] = rul_status
            result["trend_alpha"] = trend.alpha
            result["trend_beta"] = trend.beta
            result["trend_r2"] = trend.R2
        else:
            result["trend_alpha"] = None
            result["trend_beta"] = None
            result["trend_r2"] = None
            result["rul_days"] = None
            result["rul_status"] = "无法计算趋势"
        return result


    def analyze_device(self, context: Any, device_id: Optional[int] = None) -> Dict[str, Any]:
        """针对设备聚合多个指标的分析结果
        
        Args:
            context: 数据上下文(data_bundle)，包含所有数据表的字典
            device_id: 设备ID，如果为 None 则分析所有设备
            
        Returns:
            设备分析结果字典，包含设备健康度、各指标分析结果等
        """
        defs_df = context.get("device_metric_definitions", pd.DataFrame()) 
        devices_df = context.get("devices", pd.DataFrame())
        if defs_df.empty:
            log.warning("指标定义为空，无法进行分析")
            return {
            "device_id": device_id,
            "device_health_score": 0.0,
            "device_alert_level": "数据不足",
            "metrics": [],
            }
        
        if device_id is not None:
            # 获取设备型号id
            device_row = devices_df[devices_df["id"] == device_id]
            if device_row.empty:
                log.warning(f"设备 {device_id} 不存在")
                return {
                    "device_id": device_id,
                    "device_health_score": 0.0,
                    "device_alert_level": "设备不存在",
                    "metrics": [],   
                }
            model_id = device_row.iloc[0]["model_id"]
            # 获取当前型号的指标定义
            defs_df = defs_df[defs_df["model_id"] == model_id]
        
        # 按行遍历每个指标分析
        metrics_results = []
        for _, metric_row in defs_df.iterrows():
            # 构建指标定义字典 for analyze_metric: {}
            metric_def = {
                "metric_key": metric_row["metric_key"],
                "warn_threshold": metric_row.get("warn_threshold"),
                "crit_threshold": metric_row.get("crit_threshold"),
                "trend_direction": metric_row.get("trend_direction"),
                "weight_in_health": metric_row.get("weight_in_health", 1.0),
                "criticality": metric_row.get("criticality", "medium"),
            }
            metric_data = self.analyze_metric(context=context, defn=metric_def, device_id=device_id)
            if metric_data is not None:
                metric_data["weight_in_health"] = metric_def["weight_in_health"]
                metric_data["criticality"] = metric_def["criticality"]
                metrics_results.append(metric_data)
        
        if not metrics_results:
            log.warning(f"设备 {device_id} 无可用的指标分析结果")
            return {
                "device_id": device_id,
                "device_health_score": 0.0,
                "device_alert_level": "无可用数据",
                "metrics": [],
            }

        # 聚合metric列表 for device_multi_indicator_health
        metrics_for_aggregation = build_metrics_for_aggregation(metrics_results)
        device_health_score, device_alert = self.device_multi_indicator_health(metrics_for_aggregation)

        return {
            "device_id": device_id,
            "device_health_score": device_health_score,
            "device_alert_level": device_alert,
            "metrics": metrics_results,
        }


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

    # RUL
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
            return "critical"
        if w >= d * df["warn_threshold"]:
            return "warning"
        return "normal"
    
    @staticmethod   
    def device_multi_indicator_health(metrics: List[Dict[str, Any]]) -> Any:
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
            if m["alert_level"] == "critical":
                alert = "SOS: critical"
            elif m["alert_level"] == "warning" and alert != "critical":
                alert = "warning: Note!"
        H_device = round(weighted_sum / weight_total, 1) if weight_total else 0.0   # 防止分母为0
        return H_device, alert



if __name__ == "__main__":
    from backend.algorithm.demo_runner import run_demo

    run_demo()