# coding:utf-8
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd, json
from pathlib import Path


CRIT_WEIGHT = {"high": 1.2, "medium": 1.0, "low": 0.8}

BASE_DIR = Path(__file__).resolve().parents[2]
CSV_PATH = BASE_DIR / "data" / "csv"


@dataclass
class TrendResults:
    alpha: float
    beta: float
    R2: float
    t_days: np.ndarray
    y_items: np.ndarray


def load_data_from_csv(base=CSV_PATH) -> pd.DataFrame:
    """Load all CSVs and parse JSON-ish columns."""
    return {
        "device_metric_definitions": pd.read_csv(f"{base}/device_metric_definitions.csv").assign(
            feature_snapshot=lambda df: df["feature_snapshot"].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x)
        ),
        "device_models": pd.read_csv(f"{base}/device_models.csv"),
        "devices": pd.read_csv(f"{base}/devices.csv"),
        "energy_consumption": pd.read_csv(f"{base}/energy_consumption.csv"),
        "equipment_and_asset_management": pd.read_csv(f"{base}/equipment_and_asset_management.csv"),
        "inspection_submits": pd.read_csv(f"{base}/inspection_submits.csv", parse_dates=['recorded_at']).assign(
            # 对 CSV 里的 metrics 列,逐个检查转换每个单元格: str -> Dict,List
            metrics=lambda df: df["metrics"].apply(lambda x: json.loads(x) if isinstance(x, str) else x),
            data_origin=lambda df: df.get("data_origin", "IoT"),
            data_quality_score=lambda df: df.get("data_quality_score", 1.0),
            validation_notes=lambda df: df.get("validation_notes", "")
        ),
        "maintenance_costs": pd.read_csv(f"{base}/maintenance_costs.csv", parse_dates=['period_start', 'period_end']),
        "metric_ai_analysis": pd.read_csv(f"{base}/metric_ai_analysis.csv", parse_dates=['calc_time']).assign(
            curve_points=lambda df: df["curve_points"].apply(lambda x: json.loads(x) if isinstance(x, str) else x),
            extra_info=lambda df: df["extra_info"].apply(lambda x: json.loads(x) if isinstance(x, str) else x),
            feature_snapshot=lambda df: df["feature_snapshot"].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x)
        ),
        "oee_stats": pd.read_csv(f"{base}/oee_stats.csv", parse_dates=['period_start', 'period_end']),
        "spare_usage_cycles": pd.read_csv(f"{base}/spare_usage_cycles.csv")
    }


# 数据准备
def prepare_metric_series(
        data, metric_key, device_id=None, window_days=30, safety_Confidence=0.8
) -> pd.DataFrame:
    df_irf = data["inspection_submits"]
    if device_id is not None:
        df_irf = df_irf[df_irf["device_id"] == device_id]
    # copy
    df_irf = df_irf.copy()
    df_irf["value"] = df_irf["metrics"].apply(
        lambda line: float(line.get(metric_key, np.nan)) if isinstance(line, dict) else np.nan
    )
    df_irf["quality"] = df_irf["data_quality_score"].fillna(0.8) if "data_quality_score" in df_irf else 0.8
    # 去掉空值
    df_irf = df_irf.dropna(subset=["value"])
    # convert datetime
    df_irf["record_time"] = pd.to_datetime(df_irf["recorded_at"])
    # window filter
    cutoff = df_irf["record_time"].max() - pd.Timedelta(days=window_days)
    df_irf = df_irf[(df_irf["record_time"] >= cutoff) & (df_irf["quality"] >= safety_Confidence)]
    return df_irf.sort_values("record_time")[["record_time", "quality"]]


# 线性回归
def linearRegression(df: pd.DataFrame) -> Optional[TrendResults]:
    required_cols = {"record_time", "quality"}
    if not required_cols.issubset(df.columns):
        raise KeyError(f"linearRegression 输入缺少列: {required_cols - set(df.columns)}")
    # 可靠性检验
    if len(df) < 3:
        return None
    t0 = df["record_time"].min()
    t_days = (df["record_time"] - t0).dt.total_seconds() / 86400.0
    print(t_days)


if __name__ == "__main__":
    data = load_data_from_csv()
    series_df = prepare_metric_series(data=data, metric_key="spindle_vibration")
    test2 = linearRegression(df=series_df)
    print(test2)
