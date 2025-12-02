# coding:utf-8

from __future__ import annotations

import json
import logging
import sys
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import select, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from backend.db import SessionLocal, engine as default_engine
from backend import models

log = logging.getLogger(__name__)

CRIT_WEIGHT = {"high": 1.2, "medium": 1.0, "low": 0.8}

CSV_PATH = BASE_DIR / "data" / "csv"

# 做 JSON 解析列定义
JSON_FIELDS: Dict[str, List[str]] = {
    "device_metric_definitions": ["feature_snapshot"],
    "inspection_submits": ["metrics"],
    "metric_ai_analysis": ["curve_points", "extra_info", "feature_snapshot"],
    "inspection_metric_values": ["metrics_data"],
}

# 时间列定义，便于读取时自动转为 datetime
DATE_FIELDS: Dict[str, List[str]] = {
    "inspection_submits": ["recorded_at"],
    "metric_ai_analysis": ["calc_time"],
    "maintenance_costs": ["period_start", "period_end"],
    "oee_stats": ["period_start", "period_end"],
}

# 默认需要加载的表名列表
DEFAULT_TABLES: List[str] = [
    "device_metric_definitions",
    "device_models",
    "devices",
    "energy_consumption",
    "equipment_and_asset_management",
    "inspection_submits",
    "inspection_logs",
    "inspection_metric_values",
    "maintenance_costs",
    "metric_ai_analysis",
    "oee_stats",
    "spare_usage_cycles",
]


def validate_db_config(db_engine: Optional[Engine] = None) -> None:
    """数据库配置校验，确保连接字符串合法"""
    engine_to_check = db_engine or default_engine
    url = str(engine_to_check.url)
    if "://" not in url:
        raise ValueError("数据库连接字符串格式不正确，请检查 DB_URL / DATABASE_URL 配置")
    # 可在此处扩展更多连接池、超时等校验


def _safe_json_load(value: Any) -> Any:
    """安全 JSON 解析：仅对字符串尝试 json.loads，异常时返回原值"""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        log.warning("JSON 解析失败，保持原值: %s", value)
        return value


def _apply_json_fields(df: pd.DataFrame, fields: Iterable[str]) -> pd.DataFrame:
    """对指定列执行 JSON 解析，返回新的 DataFrame"""
    for col in fields:
        if col in df.columns:
            df[col] = df[col].apply(_safe_json_load)
    return df


def _discover_csv_tables(base: Path) -> List[str]:
    """自动扫描 CSV 目录，返回所有文件名。"""
    base_path = Path(base)
    if not base_path.exists():
        raise FileNotFoundError(f"CSV 目录不存在: {base_path}")
    tables = sorted(p.stem for p in base_path.glob("*.csv"))
    if not tables:
        log.warning("CSV 目录中未发现任何文件: %s", base_path)
    return tables


def _resolve_csv_tables(tables: Optional[Iterable[str]], base: Path) -> List[str]:
    """
    决定 CSV 读取列表：
        tables 未指定时读取目录下全部 CSV
        tables 指定时保持顺序并去重
    """
    if tables:
        seen = set()
        ordered: List[str] = []
        for name in tables:
            if name not in seen:
                ordered.append(name)
                seen.add(name)
        return ordered
    discovered = _discover_csv_tables(base)
    return discovered or DEFAULT_TABLES


def _get_data_from_csv(
    tables: Optional[Iterable[str]] = None,
    base: Path = CSV_PATH,
    verbose: bool = False,
) -> Dict[str, pd.DataFrame]:
    """从 CSV 目录批量读取数据"""
    tables_to_read = _resolve_csv_tables(tables, base)
    data: Dict[str, pd.DataFrame] = {}
    for table in tables_to_read:
        path = Path(base) / f"{table}.csv"
        if not path.exists():
            log.error("CSV 文件不存在: %s", path)
            raise FileNotFoundError(f"CSV 文件不存在: {path}")
        parse_dates = DATE_FIELDS.get(table, [])
        try:
            df = pd.read_csv(path, parse_dates=parse_dates)
        except Exception as exc:
            log.exception("读取 CSV 失败: %s", path)
            raise ValueError(f"读取 CSV 失败: {path}") from exc
        json_cols = JSON_FIELDS.get(table, [])
        df = _apply_json_fields(df, json_cols)
        data[table] = df
        if verbose:
            print(f"[CSV] 表: {table} | 行数: {len(df)} | 列: {list(df.columns)}")
            print(df.head(3))
    return data


@lru_cache(maxsize=32)
def _cached_db_query(
    engine_url: str,
    table_name: str,
    limit: Optional[int],
    offset: Optional[int],
) -> List[Dict[str, Any]]:
    """简单 LRU 缓存，降低重复查询压力"""
    engine_to_use = default_engine if str(default_engine.url) == engine_url else create_engine(engine_url)
    with engine_to_use.connect() as conn:
        query = f"SELECT * FROM {table_name}"
        if offset is not None:
            query += f" LIMIT {limit or 0} OFFSET {offset}"
        elif limit is not None:
            query += f" LIMIT {limit}"
        return pd.read_sql(query, conn).to_dict(orient="records")


def _get_data_from_db(
    tables: Optional[Iterable[str]] = None,
    session: Optional[Session] = None,
    db_engine: Optional[Engine] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """从数据库读取数据，保持与 CSV 相同的数据结构"""
    validate_db_config(db_engine)
    engine_to_use = db_engine or default_engine
    tables_to_use = list(tables) if tables else DEFAULT_TABLES

    orm_map: Dict[str, Any] = {
        "device_metric_definitions": models.DeviceMetricDefinition,
        "device_models": models.DeviceModel,
        "devices": models.Device,
        "energy_consumption": models.EnergyConsumption,
        "equipment_and_asset_management": models.EquipmentAndAssetManagement,
        "inspection_submits": models.InspectionLog,  # CSV 兼容字段
        "inspection_logs": models.InspectionLog,
        "inspection_metric_values": models.InspectionMetricValue,
        "maintenance_costs": models.MaintenanceCost,
        "metric_ai_analysis": models.MetricAIAnalysis,
        "oee_stats": models.OEEStat,
        "spare_usage_cycles": models.SpareUsage,
    }
    if session is None:
        # 根据传入的 db_engine 创建会话，确保连接池和事务配置一致
        if db_engine is None:
            local_session = SessionLocal()
        else:
            SessionMaker = sessionmaker(bind=db_engine)
            local_session = SessionMaker()
        created_session = True
    else:
        local_session = session
        created_session = False
    data: Dict[str, pd.DataFrame] = {}
    try:
        for table in tables_to_use:
            model_cls = orm_map.get(table)
            if not model_cls:
                log.warning("未找到 ORM 映射，跳过表: %s", table)
                continue
            try:
                if limit is not None or offset is not None:
                    # 使用缓存避免重复分页查询
                    rows_dict = _cached_db_query(str(engine_to_use.url), table, limit, offset)
                    df = pd.DataFrame(rows_dict)
                else:
                    stmt = select(model_cls)
                    rows = local_session.execute(stmt).scalars().all()
                    df = pd.DataFrame([row.__dict__ for row in rows])
                if not df.empty and "_sa_instance_state" in df.columns:
                    df = df.drop(columns=["_sa_instance_state"])
                json_cols = JSON_FIELDS.get(table, [])
                data[table] = _apply_json_fields(df, json_cols)
            except Exception as exc:
                log.exception("数据库读取失败: %s", table)
                raise ValueError(f"数据库读取失败: {table}") from exc
    finally:
        if created_session:
            local_session.close()
    return data


def get_data(
    source_type: str = "csv",
    tables: Optional[Iterable[str]] = None,
    base: Path = CSV_PATH,
    session: Optional[Session] = None,
    db_engine: Optional[Engine] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    统一数据获取入口
        支持 CSV/DB 两种来源（source_type: csv/db）
        自动处理 JSON 字段
        返回标准化结构：{"data": {...}, "metadata": {...}}
    """
    if source_type == "csv" and not tables:
        metadata_tables = _resolve_csv_tables(tables=None, base=base)
    else:
        metadata_tables = list(tables) if tables else DEFAULT_TABLES
    metadata: Dict[str, Any] = {
        "source": source_type,
        "tables": metadata_tables,
        "errors": [],
        "pagination": {"limit": limit, "offset": offset},
    }
    try:
        if source_type == "csv":
            data = _get_data_from_csv(tables=tables, base=base, verbose=verbose)
        elif source_type == "db":
            data = _get_data_from_db(
                tables=tables,
                session=session,
                db_engine=db_engine,
                limit=limit,
                offset=offset,
            )
        else:
            raise ValueError(f"未知数据源类型: {source_type}")
    except Exception as exc:
        metadata["errors"].append(str(exc))
        log.exception("获取数据失败: %s", exc)
        raise
    return {"data": data, "metadata": metadata}


def load_data_from_csv(
    base: Path = CSV_PATH,
    tables: Optional[Iterable[str]] = None,
    verbose: bool = False,
) -> Dict[str, pd.DataFrame]:
    result = get_data(source_type="csv", tables=tables, base=base, verbose=verbose)
    return result["data"]


def load_data_from_db(
    tables: Optional[Iterable[str]] = None,
    session: Optional[Session] = None,
    db_engine: Optional[Engine] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    result = get_data(
        source_type="db",
        tables=tables,
        session=session,
        db_engine=db_engine,
        limit=limit,
        offset=offset,
    )
    data = result["data"]
    return _ensure_inspection_submits(data)


def _ensure_inspection_submits(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Ensure data bundle contains inspection_submits dataframe similar to CSV dataset.
    - If DB only has inspection_logs + inspection_metric_values, compose them.
    - If inspection_submits exists but没有 metrics 列，也会用 logs+metric_values 重建。
    """
    logs_df = None
    if "inspection_submits" in data:
        # 兼容：如果已有 CSV 风格的数据且包含 metrics 列，直接返回
        existing = data["inspection_submits"]
        if "metrics" in existing.columns:
            return data
        logs_df = existing
    if logs_df is None:
        logs_df = data.get("inspection_logs") or data.get("inspection_submits")
    if logs_df is None or logs_df.empty:
        return data

    logs_df = logs_df.copy()
    metrics_df = data.get("inspection_metric_values")
    if metrics_df is not None and not metrics_df.empty:
        merged = logs_df.merge(
            metrics_df,
            left_on="id",
            right_on="log_id",
            how="left",
            suffixes=("", "_metric"),
        )
        merged["metrics"] = merged.get("metrics_data", pd.Series([{}] * len(merged))).apply(
            lambda val: val if isinstance(val, dict) else (val or {})
        )
        merged = merged.drop(columns=[col for col in ("log_id", "metrics_data") if col in merged.columns])
    else:
        merged = logs_df
        merged["metrics"] = [{}] * len(merged)

    # Align column name with CSV expectation
    if "record_time" in merged.columns and "recorded_at" not in merged.columns:
        merged = merged.rename(columns={"record_time": "recorded_at"})

    data["inspection_submits"] = merged
    return data


def _normalize_series(values: pd.Series) -> pd.Series:
    """最小-最大归一化，避免除零"""
    if values.empty:
        return values
    v_min, v_max = values.min(), values.max()
    if v_max == v_min:
        return pd.Series([0.0] * len(values), index=values.index)
    return (values - v_min) / (v_max - v_min)


def prepare_metric_series(
    data: Optional[Dict[str, pd.DataFrame]] = None,
    metric_key: str = "",
    device_id: Optional[int] = None,
    window_days: int = 30,
    safety_confidence: float = 0.8,
    source_type: str = "csv",
    base: Path = CSV_PATH,
    session: Optional[Session] = None,
) -> pd.DataFrame:
    if data is None:
        data = get_data(source_type=source_type, base=base, session=session)["data"]
    df_irf = data.get("inspection_submits", pd.DataFrame()).copy()
    if df_irf.empty:
        return pd.DataFrame(columns=["record_time", "value", "quality", "value_norm"])
    if device_id is not None and "device_id" in df_irf.columns:
        df_irf = df_irf[df_irf["device_id"] == device_id]
    df_irf["record_time"] = pd.to_datetime(df_irf.get("recorded_at"), errors="coerce")
    df_irf["value"] = df_irf["metrics"].apply(
        lambda line: float(line.get(metric_key, np.nan)) if isinstance(line, dict) else np.nan
    )
    df_irf["quality"] = df_irf.get("data_quality_score", 0.8).fillna(0.8)
    df_irf = df_irf.dropna(subset=["record_time", "value"])
    if df_irf.empty:
        return pd.DataFrame(columns=["record_time", "value", "quality", "value_norm"])

    # 窗口过滤
    cutoff = df_irf["record_time"].max() - pd.Timedelta(days=window_days)
    df_irf = df_irf[(df_irf["record_time"] >= cutoff) & (df_irf["quality"] >= safety_confidence)]
    if df_irf.empty:
        return pd.DataFrame(columns=["record_time", "value", "quality", "value_norm"])

    # 按日对齐时间序列并填充缺失
    df_irf = df_irf.sort_values("record_time").set_index("record_time")
    full_range = pd.date_range(start=df_irf.index.min(), end=df_irf.index.max(), freq="D")
    df_irf = df_irf.reindex(full_range)
    df_irf["value"] = df_irf["value"].interpolate().fillna(method="bfill").fillna(method="ffill")
    df_irf["quality"] = df_irf["quality"].fillna(df_irf["quality"].mean())

    # 归一化
    df_irf["value_norm"] = _normalize_series(df_irf["value"])
    df_irf = df_irf.reset_index().rename(columns={"index": "record_time"})
    return df_irf[["record_time", "value", "quality", "value_norm"]]


def prepare_rul_series(
    data: Optional[Dict[str, pd.DataFrame]] = None,
    metric_key: str = "",
    device_id: Optional[int] = None,
    window_days: int = 30,
    source_type: str = "csv",
    base: Path = CSV_PATH,
    session: Optional[Session] = None,
) -> pd.DataFrame:
    """
    获取 RUL 序列，识别特征列并返回标准化格式。
    """
    if data is None:
        data = get_data(source_type=source_type, base=base, session=session)["data"]
    df = data.get("metric_ai_analysis", pd.DataFrame()).copy()
    if df.empty:
        return pd.DataFrame(columns=["record_time", "rul_days", "rul_estimated_end"])
    if device_id is not None and "device_id" in df.columns:
        df = df[df["device_id"] == device_id]
    if "metric_key" in df.columns:
        df = df[df["metric_key"] == metric_key]
    if df.empty or "rul_days" not in df.columns:
        return pd.DataFrame(columns=["record_time", "rul_days", "rul_estimated_end"])
    df["record_time"] = pd.to_datetime(df.get("calc_time"), errors="coerce")
    df = df.dropna(subset=["record_time", "rul_days"])
    if df.empty:
        return pd.DataFrame(columns=["record_time", "rul_days", "rul_estimated_end"])
    cutoff = df["record_time"].max() - pd.Timedelta(days=window_days)
    df = df[df["record_time"] >= cutoff]
    df["rul_estimated_end"] = df["record_time"] + pd.to_timedelta(df["rul_days"], unit="D")
    return df.sort_values("record_time")[["record_time", "rul_days", "rul_estimated_end"]]


def get_metric_def(
    data: Dict[str, pd.DataFrame],
    metric_key: str,
    device_id: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """提取阈值/趋势方向"""
    df = data.get("device_metric_definitions", pd.DataFrame())
    if device_id is not None and "devices" in data:
        device_row = data["devices"][data["devices"]["id"] == device_id]
        if not device_row.empty:
            df = df[df["model_id"] == device_row.iloc[0]["model_id"]]
    df = df[df["metric_key"] == metric_key]
    if df.empty:
        return None
    row = df.iloc[0]
    return {
        "crit_threshold": row.get("crit_threshold"),
        "warn_threshold": row.get("warn_threshold"),
        "trend_direction": row.get("trend_direction"),
        "weight_in_health": row.get("weight_in_health", 1.0),
        "criticality": row.get("criticality", "medium"),
    }


def _flatten_metrics(df_irf: pd.DataFrame, metrics_filter: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    展开 inspection_submits.metrics 字段。
        将嵌套的 metrics 字典拆成 device_id/metric_key/value 结构
        metrics_filter 可仅保留关心的指标，减少无关计算量
    """
    records: List[Dict[str, Any]] = []
    metric_set = set(metrics_filter) if metrics_filter else None
    for _, row in df_irf.iterrows():
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        device_id = row.get("device_id")
        record_time = row.get("recorded_at")
        for k, v in metrics.items():
            if metric_set and k not in metric_set:
                continue
            try:
                value = float(v)
            except (TypeError, ValueError):
                continue
            records.append({"device_id": device_id, "metric_key": k, "value": value, "record_time": record_time})
    return pd.DataFrame(records)


def _clip_outliers(series: pd.Series) -> pd.Series:
    """
    对指标值做分位截断，过滤掉极端值对健康分的影响。
    """
    if series.empty:
        return series
    q_low, q_high = series.quantile(0.01), series.quantile(0.99)
    return series.clip(lower=q_low, upper=q_high)


def aggregate_device_health(
    source_type: str = "csv",
    base: Path = CSV_PATH,
    session: Optional[Session] = None,
    metrics_of_interest: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    设备健康度聚合
        负责：取数 + 指标展开 + 聚合统计 + 分数计算
        返回：以 device_id 为单位的健康度表，并附带各指标的均值/极值等统计列
    """
    payload = get_data(source_type=source_type, base=base, session=session)
    data = payload["data"]
    df_irf = data.get("inspection_submits", pd.DataFrame())
    defs = data.get("device_metric_definitions", pd.DataFrame())
    if df_irf.empty or defs.empty:
        return pd.DataFrame(columns=["device_id", "health_score"])

    metric_keys = list(metrics_of_interest) if metrics_of_interest else defs.get("metric_key", []).tolist()
    metrics_df = _flatten_metrics(df_irf, metrics_filter=metric_keys)
    if metrics_df.empty:
        return pd.DataFrame(columns=["device_id", "health_score"])

    metrics_df["value"] = _clip_outliers(metrics_df["value"])

    agg_df = metrics_df.groupby(["device_id", "metric_key"])["value"].agg(["mean", "max", "std"]).reset_index()

    weight_map = {row["metric_key"]: row.get("weight_in_health", 1.0) for _, row in defs.iterrows()}
    warn_map = {row["metric_key"]: row.get("warn_threshold") for _, row in defs.iterrows()}
    crit_map = {row["metric_key"]: row.get("crit_threshold") for _, row in defs.iterrows()}

    def _metric_score(row: pd.Series) -> float:
        """根据阈值与均值计算单指标健康度 0-100"""
        metric_key = row["metric_key"]
        value_mean = row["mean"]
        warn = warn_map.get(metric_key)
        crit = crit_map.get(metric_key)
        if warn is None or crit is None or crit == warn:
            return 100.0
        ratio = (value_mean - warn) / (crit - warn)
        ratio = float(np.clip(ratio, 0.0, 1.0))
        return float(round(100.0 * (1.0 - ratio), 1))

    agg_df["metric_score"] = agg_df.apply(_metric_score, axis=1)
    agg_df["weight"] = agg_df["metric_key"].apply(lambda k: weight_map.get(k, 1.0))

    device_scores = (
        agg_df.groupby("device_id")
        .apply(
            lambda g: pd.Series(
                {
                    "health_score": round(np.average(g["metric_score"], weights=g["weight"]), 1)
                    if g["weight"].sum() > 0
                    else 0.0
                }
            )
        )
        .reset_index()
    )

    stats_wide = agg_df.pivot_table(
        index="device_id",
        columns="metric_key",
        values=["mean", "max", "std"],
    )
    stats_wide.columns = [f"{stat}_{metric}" for stat, metric in stats_wide.columns]
    stats_wide = stats_wide.reset_index()

    result = device_scores.merge(stats_wide, on="device_id", how="left")
    return result


@contextmanager
def transaction_scope(existing_session: Optional[Session] = None) -> Any:
    """
    事务上下文管理器，支持 with 使用，自动提交/回滚。
    """
    session = existing_session or SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        if existing_session is None:
            session.close()

