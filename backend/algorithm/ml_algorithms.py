# coding:utf-8
import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd, json
from pathlib import Path
from backend.algorithm.data_service import load_data_from_csv, prepare_metric_series


logging.basicConfig(
    level=logging.INFO,  
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger()


@dataclass
class TrendResults:
    alpha: float
    beta: float
    R2: float
    t_days: np.ndarray
    y_items: np.ndarray



# 线性回归
def linearRegression(df: pd.DataFrame) -> Optional[TrendResults]:
    '''最小二乘直线拟合: return beta, alpha, R^2'''
    try:
        required_cols = {"record_time", "quality"}
        if not required_cols.issubset(df.columns):
            raise KeyError(f"linearRegression 输入缺少列: {required_cols - set(df.columns)}")
        # 可靠性检验
        if len(df) < 3:
            return None
        t0 = df["record_time"].min()
        t_days = (df["record_time"] - t0).dt.total_seconds() / 86400.0
        # print(t_days)
        y = df["value"].astype(float).to_numpy()
        # 中心化
        t_center = t_days - t_days.mean()
        y_center = y - y.mean()
        denominator = np.sum(t_center ** 2)
        if denominator == 0:
            return None
        
        beta = float(sum(t_center * y_center) / denominator)  # β
        alpha = float(y.mean() - beta * t_days.mean())  # α
        y_fitting = alpha + beta * t_days
        s_res = float(np.sum((y - y_fitting) ** 2))  # 计算残差平方和
        s_tot = float(np.sum((y - y.mean()) ** 2)) or 1e-6  # 计算总平方和
        r2 = 1.0 - s_res / s_tot    # 决定系数

        log.info(f'R^2: {r2}')
        return TrendResults(
            alpha=alpha, beta=beta, R2=r2, t_days=t_days.to_numpy(), y_items=y
        )
    except ArithmeticError as ae:
        log.info(f"failed to linear regression : {ae}")


# 机器剩余寿命RUL
def estimate_rul(df: Dict, trend: TrendResults, max_days: int=365) -> Any:
    '''基于线性趋势的撞线时间,即何时撞见最大阈值c'''
    d = df["trend_direction"]  # 恶化方向
    z_c = d * df["crit_threshold"]
    # 趋势参数映射到统一坐标系 z
    alpha_z = d * trend.alpha
    beta_z = d * trend.beta
    t_now = np.asarray(trend.t_days)[-1]
    if beta_z <= 0:
        return None, "状态稳定"
    t_end = (z_c - alpha_z) / beta_z   # 撞线时间
    rul = t_end - t_now
    if rul <= 0:
        return 0, "设备寿命已经结束"
    if rul > max_days:
        return max_days, "长期安全"
    return int(rul), "正在衰退"

# 


if __name__ == "__main__":
    data = load_data_from_csv()
    reg_df = prepare_metric_series(data=data, metric_key="spindle_vibration")
    # dmd = prepare_metric_series(data=data, metric_key="crit_threshold")
    trend = linearRegression(df=reg_df, metric_key="device_metric_definitions")
    rul1 = estimate_rul(df=dmd, trend=TrendResults)
    if trend:
        print(f"alpha={trend.alpha:.2f}, beta={trend.beta:.4f}, R2={trend.R2:.5f}")



