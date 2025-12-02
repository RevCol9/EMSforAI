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
        """
        分析单个设备指标
        
        执行指标健康度评估、趋势分析和剩余使用寿命(RUL)预测
        
        Args:
            context: 数据上下文，包含所有数据表的字典
            defn: 指标定义字典，必须包含 metric_key, warn_threshold, crit_threshold 等字段
            device_id: 设备ID，如果指定则只分析该设备的该指标
            
        Returns:
            分析结果字典，包含健康度、趋势、RUL等信息；如果数据不足则返回 None
        """
        metric_key = defn.get("metric_key")
        if not metric_key:
            log.warning("指标定义中缺少 metric_key")
            return None
        
        # 获取指标的时间序列数据
        df_series = self.prepare_metric_series(context, metric_key, device_id)
        
        # 数据量检查：至少需要3个数据点才能进行趋势分析
        if df_series.empty or len(df_series) < 3:
            log.warning(f"指标: {metric_key} 数据量不足,无法进行分析")
            return None
        
        # 获取最新记录值作为当前值
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
            # 计算预测置信度：综合考虑样本数量和趋势拟合度
            N = len(df_series)
            N0 = 30  # 样本数量阈值，超过此值认为样本充足
            prediction_confidence = min(1.0, N / N0) * trend.R2
            result["prediction_confidence"] = round(prediction_confidence, 3)
        else:
            result["trend_alpha"] = None
            result["trend_beta"] = None
            result["trend_r2"] = None
            result["rul_days"] = None
            result["rul_status"] = "无法计算趋势"
            result["prediction_confidence"] = 0.0
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

        # 计算高级功能（停机风险、产能影响、备件寿命）
        downtime_risk = self.compute_downtime_risk(context, device_id, device_health_score, metrics_results)
        throughput_impact = self.compute_throughput_impact(context, device_id, device_health_score, downtime_risk)
        spare_life_info = self.compute_spare_life(context, device_id)

        return {
            "device_id": device_id,
            "device_health_score": device_health_score,
            "device_alert_level": device_alert,
            "metrics": metrics_results,
            "downtime_risk": downtime_risk,
            "throughput_impact": throughput_impact,
            "spare_life_info": spare_life_info,
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

    def linear_regression(self, df: pd.DataFrame) -> Optional[TrendResults]:
        """
        执行线性回归分析，计算趋势参数和拟合度
        
        使用最小二乘法拟合时间序列数据，返回趋势斜率、截距和决定系数(R²)
        
        Args:
            df: 包含 record_time 和 value 列的 DataFrame
            
        Returns:
            TrendResults 对象，包含回归参数；如果数据不足则返回 None
        """
        # 样本量检查：至少需要3个数据点
        if len(df) < 3:  
            return None
        
        t0 = df["record_time"].min()  # 起始时间作为基准点
        t_days = (df["record_time"] - t0).dt.total_seconds() / 86400.0  # 转换为天数
        y = df["value"].astype(float).to_numpy()
        
        # 中心化处理：提高数值稳定性
        t_center = t_days - t_days.mean()
        y_center = y - y.mean()
        
        # 计算斜率 beta
        denom = np.sum(t_center ** 2)
        if denom == 0:
            return None
        beta = float(np.sum(t_center * y_center) / denom)  
        alpha = float(y.mean() - beta * t_days.mean())  # 截距
        
        # 计算决定系数 R²
        y_hat = alpha + beta * t_days
        ss_res = float(np.sum((y - y_hat) ** 2))  # 残差平方和
        ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1e-6  # 总平方和
        r2 = 1.0 - ss_res / ss_tot
        
        log.info(f"线性回归完成: R² = {r2:.3f}, 斜率 = {beta:.6f}")
        return TrendResults(alpha=alpha, beta=beta, R2=r2, t_days=t_days.to_numpy(), y_items=y)

    def compute_rul(self, defn: Dict[str, Any], trend: TrendResults, max_days: int = 365) -> Any:
        """
        计算剩余使用寿命 (Remaining Useful Life, RUL)
        
        基于线性趋势预测指标达到临界阈值所需的时间
        
        Args:
            defn: 指标定义，包含 trend_direction 和 crit_threshold
            trend: 线性回归趋势结果
            max_days: RUL 的最大值限制（天）
            
        Returns:
            (rul_days, status) 元组，rul_days 为剩余天数，status 为状态描述
        """
        d = defn["trend_direction"]  # 趋势方向：1 表示上升趋势，-1 表示下降趋势
        z_c = d * defn["crit_threshold"]  # 临界阈值（统一坐标系）
        
        # 将趋势参数映射到统一坐标系
        alpha_z = d * trend.alpha  
        beta_z = d * trend.beta  
        t_now = trend.t_days[-1]  # 当前时间点
        
        # 如果趋势稳定（斜率 <= 0），无法预测 RUL
        if beta_z <= 0: 
            return None, "状态稳定"
        
        # 计算达到临界阈值的时间点
        t_star = (z_c - alpha_z) / beta_z
        rul = t_star - t_now
        
        if rul <= 0:
            return 0, "设备寿命已经结束"
        if rul > max_days:
            return max_days, "长期安全"
        return int(rul), "正在衰退"

    @staticmethod
    def compute_metric_health(defn: Dict[str, Any], current_val: float) -> float:
        """
        计算单指标健康度分数 (0-100)
        
        基于当前值与警告阈值、临界阈值的线性插值计算健康度
        
        Args:
            defn: 指标定义，包含 trend_direction, warn_threshold, crit_threshold
            current_val: 当前指标值
            
        Returns:
            健康度分数，范围 0-100，100 表示完全健康
        """
        d = defn["trend_direction"]
        z = d * current_val  # 统一坐标系下的当前值
        z_w = d * defn["warn_threshold"]  # 警告阈值
        z_c = d * defn["crit_threshold"]  # 临界阈值
        
        # 线性插值：计算当前值在警告和临界阈值之间的位置
        s = (z - z_w) / (z_c - z_w)
        ss = np.clip(s, 0.0, 1.0)  # 限制在 [0, 1] 范围内
        return round(100.0 * (1.0 - ss), 1)  # 转换为 0-100 分数

    @staticmethod
    def metric_alert_level(defn: Dict[str, Any], current_value: float) -> str:
        """
        判断指标告警级别
        
        Args:
            defn: 指标定义，包含 trend_direction, warn_threshold, crit_threshold
            current_value: 当前指标值
            
        Returns:
            告警级别："normal", "warning", "critical"
        """
        d = defn["trend_direction"]
        w = d * current_value  # 统一坐标系下的当前值
        if w >= d * defn["crit_threshold"]:
            return "critical"
        if w >= d * defn["warn_threshold"]:
            return "warning"
        return "normal"
    
    @staticmethod   
    def device_multi_indicator_health(metrics: List[Dict[str, Any]]) -> Any:
        """
        计算设备多指标综合健康度
        
        基于加权平均方法聚合多个指标的健康度，权重由指标重要性和关键程度决定
        
        Args:
            metrics: 指标分析结果列表，每个元素包含 health_score, weight_in_health, criticality, alert_level
            
        Returns:
            (health_score, alert_level) 元组，health_score 为综合健康度分数，alert_level 为最高告警级别
        """
        weighted_sum = 0.0
        weight_total = 0.0
        alert = "正常"
        
        for m in metrics:
            # 复合权重 = 基础权重 × 关键程度系数
            composite_weight = m.get("weight_in_health", 1.0) * CRIT_WEIGHT.get(m.get("criticality", "medium"), 1.0)
            weighted_sum += composite_weight * m["health_score"]
            weight_total += composite_weight
            
            # 确定最高告警级别
            if m["alert_level"] == "critical":
                alert = "SOS: critical"
            elif m["alert_level"] == "warning" and alert != "critical":
                alert = "warning: Note!"
        
        # 加权平均计算综合健康度
        health_score = round(weighted_sum / weight_total, 1) if weight_total else 0.0
        return health_score, alert

    def compute_downtime_risk(
        self, 
        context: Any, 
        device_id: Optional[int], 
        device_health_score: float,
        metrics_results: List[Dict[str, Any]]
    ) -> float:
        """
        计算停机风险
        特征：
        - x1: 设备健康度（越低风险越高）
        - x2: 最小 RUL（越小风险越高）
        - x3: 维护费用比率（最近30天/历史均值）
        """
        if device_id is None:
            return 0.0
        
        # 特征1: 健康度风险（健康度越低，风险越高）
        health_risk = (100 - device_health_score) / 100.0
        
        # 特征2: RUL风险（找到最小的 RUL）
        min_rul = None
        for m in metrics_results:
            if m.get("rul_days") is not None:
                rul = m["rul_days"]
                if min_rul is None or (rul is not None and rul < min_rul):
                    min_rul = rul
        
        if min_rul is None:
            rul_risk = 0.0
        elif min_rul <= 0:
            rul_risk = 1.0
        elif min_rul < 30:
            rul_risk = 1.0 / (min_rul + 1)  # RUL越小，风险越大
        else:
            rul_risk = 0.1  # RUL较大时风险较低
        
        # 特征3: 维护费用比率
        maintenance_df = context.get("maintenance_costs", pd.DataFrame())
        if maintenance_df.empty:
            maintenance_risk = 0.0
        else:
            device_maintenance = maintenance_df[maintenance_df["device_id"] == device_id]
            if device_maintenance.empty:
                maintenance_risk = 0.0
            else:
                # 计算最近30天和历史均值的比率
                recent_cutoff = pd.Timestamp.now() - pd.Timedelta(days=30)
                recent = device_maintenance[device_maintenance["period_end"] >= recent_cutoff]["cost"].sum()
                historical_avg = device_maintenance["cost"].mean()
                if historical_avg > 0:
                    maintenance_risk = min(1.0, recent / (historical_avg * 30))  # 归一化
                else:
                    maintenance_risk = 0.0
        
        # 简化规则：加权平均（可调整权重）
        downtime_risk = 0.4 * health_risk + 0.4 * rul_risk + 0.2 * maintenance_risk
        return round(min(1.0, max(0.0, downtime_risk)), 3)

    def compute_throughput_impact(
        self,
        context: Any,
        device_id: Optional[int],
        device_health_score: float,
        downtime_risk: float
    ) -> float:
        """
        计算产能影响（基于规则）
        产能损失率 = 1 - OEE
        """
        if device_id is None:
            return 0.0
        
        oee_df = context.get("oee_stats", pd.DataFrame())
        if oee_df.empty:
            # 如果没有 OEE 数据，基于健康度和停机风险估算
            health_impact = (100 - device_health_score) / 100.0
            throughput_impact = 0.6 * health_impact + 0.4 * downtime_risk
            return round(min(1.0, max(0.0, throughput_impact)), 3)
        
        # 获取设备最近的 OEE 数据
        device_oee = oee_df[oee_df["device_id"] == device_id]
        if device_oee.empty:
            # 使用健康度和停机风险估算
            health_impact = (100 - device_health_score) / 100.0
            throughput_impact = 0.6 * health_impact + 0.4 * downtime_risk
            return round(min(1.0, max(0.0, throughput_impact)), 3)
        
        # 计算最近的 OEE（取最新记录）
        latest_oee = device_oee.sort_values("period_end").iloc[-1]
        availability = latest_oee.get("availability", 1.0)
        performance = latest_oee.get("performance", 1.0)
        quality_rate = latest_oee.get("quality_rate", 1.0)
        
        oee = availability * performance * quality_rate
        throughput_impact = 1.0 - oee
        
        # 结合健康度和停机风险进行修正
        health_factor = (100 - device_health_score) / 100.0
        throughput_impact = 0.5 * throughput_impact + 0.3 * health_factor + 0.2 * downtime_risk
        
        return round(min(1.0, max(0.0, throughput_impact)), 3)

    def compute_spare_life(
        self,
        context: Any,
        device_id: Optional[int]
    ) -> Dict[str, Any]:
        """
        计算备件寿命信息
        完整版本需要 Weibull 模型。
        这里先实现基于 usage_cycles 的简化版本。
        
        参数说明：
        - usage_cycles: 使用循环次数（单位：周期）
          * 一个周期 = 设备运行1小时（或根据实际业务定义，如：完成一个加工周期）
          * 具体定义需要根据业务场景确定，建议在业务文档中明确说明
        
        数据说明：
        - 当前数据中没有 max_cycles 字段，这里根据备件类型（part_code）使用映射表。
        - 如果未来数据中有 max_cycles 字段，优先使用数据中的值。
        """
        if device_id is None:
            return {"status": "no_device", "spare_parts": []}
        
        spare_df = context.get("spare_usage_cycles", pd.DataFrame())
        if spare_df.empty:
            return {"status": "no_data", "spare_parts": []}
        
        device_spares = spare_df[spare_df["device_id"] == device_id]
        if device_spares.empty:
            return {"status": "no_spares", "spare_parts": []}
        
        # 备件类型到最大使用周期的映射表
        # 
        # 单位说明：
        # - usage_cycles 和 max_cycles 的单位都是"使用周期"
        # - 一个周期通常定义为：设备运行1小时（或根据业务定义，如完成一个加工周期）
        # - 具体周期定义需要根据实际业务场景确定
        #
        # 映射表说明：
        # - 根据常见备件的设计寿命设定（单位：使用周期）
        # - 如果数据中有 max_cycles 字段，优先使用数据中的值
        part_max_cycles_map = {
            "BRG-6205": 5000,      # 轴承类型，设计寿命约 5000 周期（假设1周期=1小时，即5000小时）
            "BRG-6308": 5000,      # 轴承类型，设计寿命约 5000 周期
            "AIR-FILTER-01": 3000, # 空气过滤器，设计寿命约 3000 周期
            "AIR-FILTER-02": 3000, # 空气过滤器，设计寿命约 3000 周期
            "OIL-SEPARATOR-01": 4000,  # 油分离器，设计寿命约 4000 周期
            "SPINDLE-BELT-01": 8000,    # 主轴皮带，设计寿命约 8000 周期
            "COOLANT-PUMP-01": 6000,    # 冷却泵，设计寿命约 6000 周期
            "TOOL-HOLDER-01": 10000,     # 刀柄，设计寿命约 10000 周期
            "PRESSURE-VALVE-01": 5000,  # 压力阀，设计寿命约 5000 周期
        }
        default_max_cycles = 5000  # 未知备件类型的默认值（5000 周期）
        
        spare_parts_info = []
        for _, row in device_spares.iterrows():
            # 获取使用周期
            usage_cycles = row.get("usage_cycles", 0)
            
            # 获取最大使用周期：优先使用数据中的值，否则从映射表查找，最后用默认值
            if "max_cycles" in row and pd.notna(row.get("max_cycles")):
                max_cycles = float(row["max_cycles"])
            else:
                part_code = row.get("part_code", "")
                max_cycles = part_max_cycles_map.get(part_code, default_max_cycles)
            
            # 计算使用率
            if max_cycles > 0:
                usage_ratio = usage_cycles / max_cycles
            else:
                usage_ratio = 0.0
            remaining_ratio = 1.0 - usage_ratio
            
            if usage_ratio > 0.9:
                status = "critical"
            elif usage_ratio > 0.7:
                status = "warning"
            else:
                status = "normal"
            
            # 构建备件信息（如果数据中没有 spare_part_name，使用 part_code 作为名称）
            spare_parts_info.append({
                "spare_part_id": row.get("spare_part_id", row.get("part_code")),
                "spare_part_name": row.get("spare_part_name", row.get("part_code")),
                "part_code": row.get("part_code", ""),
                "usage_cycles": usage_cycles,
                "max_cycles": max_cycles,
                "usage_ratio": round(usage_ratio, 3),
                "remaining_ratio": round(remaining_ratio, 3),
                "status": status
            })
        
        # 统计关键备件数量
        critical_count = 0
        for s in spare_parts_info:
            if s["status"] == "critical":
                critical_count = critical_count + 1
        
        return {
            "status": "ok",
            "spare_parts": spare_parts_info,
            "total_parts": len(spare_parts_info),
            "critical_count": critical_count,
        }



if __name__ == "__main__":
    from backend.algorithm.demo_runner import run_demo

    run_demo()