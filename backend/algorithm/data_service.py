"""
EMSforAI 数据服务层

本模块负责从数据库加载数据并转换为算法引擎所需的格式。
适配4域8表架构，提供数据查询、过滤、预处理等功能。

主要功能：
- 从数据库加载所有的数据
- 准备过程数据时间序列
- 加载波形二进制数据
- 提取维护标签（用于AI训练）
- 提取知识库上下文（用于LLM RAG）

Author: EMSforAI Team
License: MIT
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from backend.db import SessionLocal, engine as default_engine
from backend import models

log = logging.getLogger(__name__)


def load_data_from_db(
    session: Optional[Session] = None,
    asset_id: Optional[str] = None,
    start_time: Optional[pd.Timestamp] = None,
    end_time: Optional[pd.Timestamp] = None,
) -> Dict[str, pd.DataFrame]:
    """
    从数据库加载所有域的数据
    
    Args:
        session: 数据库会话
        asset_id: 设备ID（可选，过滤特定设备）
        start_time: 开始时间（可选，过滤时间范围）
        end_time: 结束时间（可选，过滤时间范围）
    
    Returns:
        数据字典，包含所有表的数据
    """
    if session is None:
        local_session = SessionLocal()
        created_session = True
    else:
        local_session = session
        created_session = False
    
    data: Dict[str, pd.DataFrame] = {}
    
    try:
        # 一：基础元数据
        # 1. 设备资产表
        stmt = select(models.Asset)
        if asset_id:
            stmt = stmt.where(models.Asset.asset_id == asset_id)
        assets = local_session.execute(stmt).scalars().all()
        data["assets"] = pd.DataFrame([{
            "asset_id": a.asset_id,
            "name": a.name,
            "model_id": a.model_id,
            "location": a.location,
            "commission_date": a.commission_date,
            "status": a.status,
        } for a in assets])
        
        # 2. 测点定义表
        stmt = select(models.MetricDefinition)
        if asset_id:
            stmt = stmt.where(models.MetricDefinition.asset_id == asset_id)
        metrics = local_session.execute(stmt).scalars().all()
        data["metric_definitions"] = pd.DataFrame([{
            "metric_id": m.metric_id,
            "asset_id": m.asset_id,
            "metric_name": m.metric_name,
            "metric_type": m.metric_type.value if m.metric_type else None,
            "unit": m.unit,
            "warn_threshold": m.warn_threshold,
            "crit_threshold": m.crit_threshold,
            "is_condition_dependent": m.is_condition_dependent,
            "sampling_frequency": m.sampling_frequency,
        } for m in metrics])
        
        # 二：动态感知
        # 3. 过程数据
        stmt = select(models.TelemetryProcess)
        if asset_id:
            # 通过metric_id关联到asset_id
            # 重要：如果 asset_id 不存在，metric_definitions 为空，应该返回空结果集
            # 而不是返回所有数据，避免数据泄漏
            metric_ids = data["metric_definitions"]["metric_id"].tolist() if not data["metric_definitions"].empty else []
            if metric_ids:
                # 只有当找到对应的 metric_ids 时才添加过滤条件
                stmt = stmt.where(models.TelemetryProcess.metric_id.in_(metric_ids))
            else:
                # 如果 asset_id 存在但没有任何 metric_definitions，返回空结果集
                # 使用一个不可能匹配的条件来确保返回空结果
                stmt = stmt.where(models.TelemetryProcess.metric_id == "__NONEXISTENT__")
        if start_time:
            stmt = stmt.where(models.TelemetryProcess.timestamp >= start_time)
        if end_time:
            stmt = stmt.where(models.TelemetryProcess.timestamp <= end_time)
        stmt = stmt.order_by(models.TelemetryProcess.timestamp)
        process_data = local_session.execute(stmt).scalars().all()
        data["telemetry_process"] = pd.DataFrame([{
            "id": p.id,
            "timestamp": p.timestamp,
            "metric_id": p.metric_id,
            "value": p.value,
            "quality": p.quality,
            "machine_state": p.machine_state,
        } for p in process_data])
        
        # 4. 波形数据（只加载元数据，不加载二进制数据）
        stmt = select(models.TelemetryWaveform)
        if asset_id:
            stmt = stmt.where(models.TelemetryWaveform.asset_id == asset_id)
        if start_time:
            stmt = stmt.where(models.TelemetryWaveform.timestamp >= start_time)
        if end_time:
            stmt = stmt.where(models.TelemetryWaveform.timestamp <= end_time)
        waveforms = local_session.execute(stmt).scalars().all()
        data["telemetry_waveform"] = pd.DataFrame([{
            "snapshot_id": w.snapshot_id,
            "asset_id": w.asset_id,
            "timestamp": w.timestamp,
            "sampling_rate": w.sampling_rate,
            "duration_ms": w.duration_ms,
            "axis": w.axis,
            "ref_rpm": w.ref_rpm,
            "metric_id": w.metric_id,
            # 注意：data_blob不加载，需要时单独查询
        } for w in waveforms])
        
        # 三：知识与运维
        # 5. 运维记录
        stmt = select(models.MaintenanceRecord)
        if asset_id:
            stmt = stmt.where(models.MaintenanceRecord.asset_id == asset_id)
        if start_time:
            stmt = stmt.where(models.MaintenanceRecord.start_time >= start_time)
        if end_time:
            stmt = stmt.where(models.MaintenanceRecord.end_time <= end_time)
        maintenance = local_session.execute(stmt).scalars().all()
        data["maintenance_records"] = pd.DataFrame([{
            "record_id": m.record_id,
            "asset_id": m.asset_id,
            "start_time": m.start_time,
            "end_time": m.end_time,
            "failure_code": m.failure_code,
            "issue_description": m.issue_description,
            "solution_description": m.solution_description,
            "cost": m.cost,
        } for m in maintenance])
        
        # 6. 知识库
        stmt = select(models.KnowledgeBase)
        if asset_id:
            # 通过model_id关联
            model_ids = data["assets"]["model_id"].unique().tolist() if not data["assets"].empty else []
            if model_ids:
                stmt = stmt.where(models.KnowledgeBase.applicable_model.in_(model_ids))
        knowledge = local_session.execute(stmt).scalars().all()
        data["knowledge_base"] = pd.DataFrame([{
            "doc_id": k.doc_id,
            "applicable_model": k.applicable_model,
            "category": k.category.value if k.category else None,
            "title": k.title,
            "content_chunk": k.content_chunk,
        } for k in knowledge])
        
        # 四：分析结果
        # 7. 健康分析结果
        stmt = select(models.AIHealthAnalysis)
        if asset_id:
            stmt = stmt.where(models.AIHealthAnalysis.asset_id == asset_id)
        if start_time:
            stmt = stmt.where(models.AIHealthAnalysis.calc_time >= start_time)
        if end_time:
            stmt = stmt.where(models.AIHealthAnalysis.calc_time <= end_time)
        stmt = stmt.order_by(models.AIHealthAnalysis.calc_time.desc())
        health_analyses = local_session.execute(stmt).scalars().all()
        data["ai_health_analysis"] = pd.DataFrame([{
            "analysis_id": h.analysis_id,
            "asset_id": h.asset_id,
            "calc_time": h.calc_time,
            "health_score": h.health_score,
            "rul_days": h.rul_days,
            "trend_slope": h.trend_slope,
            "diagnosis_result": h.diagnosis_result,
            "model_version": h.model_version,
            "prediction_confidence": h.prediction_confidence,
        } for h in health_analyses])
        
        # 8. AI报告
        if not data["ai_health_analysis"].empty:
            analysis_ids = data["ai_health_analysis"]["analysis_id"].tolist()
            stmt = select(models.AIReport).where(models.AIReport.analysis_id.in_(analysis_ids))
            reports = local_session.execute(stmt).scalars().all()
            data["ai_reports"] = pd.DataFrame([{
                "report_id": r.report_id,
                "analysis_id": r.analysis_id,
                "generated_content": r.generated_content,
                "user_feedback": r.user_feedback,
                "created_at": r.created_at,
            } for r in reports])
        else:
            data["ai_reports"] = pd.DataFrame()
        
        log.info(f"数据加载完成: {len(data)} 个表")
        for table_name, df in data.items():
            log.debug(f"  {table_name}: {len(df)} 条记录")
        
    except Exception as e:
        log.exception(f"数据加载失败: {e}")
        raise
    finally:
        if created_session:
            local_session.close()
    
    return data


def prepare_process_series(
    data: Dict[str, pd.DataFrame],
    metric_id: str,
    asset_id: Optional[str] = None,
    window_days: int = 30,
    machine_state: Optional[int] = None,
    quality_threshold: int = 1,
) -> pd.DataFrame:
    """
    准备过程数据时间序列
    
    Args:
        data: 数据字典
        metric_id: 测点ID
        asset_id: 设备ID（可选）
        window_days: 时间窗口（天）
        machine_state: 工况状态过滤（None=不过滤，0=停机，1=待机，2=加工）
        quality_threshold: 数据质量阈值（0或1）
    
    Returns:
        时间序列DataFrame，包含timestamp, value, quality, machine_state
    """
    df = data.get("telemetry_process", pd.DataFrame())
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "value", "quality", "machine_state"])
    
    # 过滤测点
    df = df[df["metric_id"] == metric_id].copy()
    
    # 过滤设备（通过metric_id关联）
    if asset_id:
        metrics_df = data.get("metric_definitions", pd.DataFrame())
        if not metrics_df.empty:
            asset_metrics = metrics_df[metrics_df["asset_id"] == asset_id]["metric_id"].tolist()
            df = df[df["metric_id"].isin(asset_metrics)]
    
    # 过滤数据质量
    df = df[df["quality"] >= quality_threshold]
    
    # 过滤工况状态
    if machine_state is not None:
        df = df[df["machine_state"] == machine_state]
    
    # 时间窗口过滤
    if not df.empty:
        max_time = df["timestamp"].max()
        cutoff = max_time - pd.Timedelta(days=window_days)
        df = df[df["timestamp"] >= cutoff]
    
    # 排序
    if not df.empty:
        df = df.sort_values("timestamp")
    
    return df[["timestamp", "value", "quality", "machine_state"]]


def load_waveform_data(
    session: Session,
    snapshot_id: str,
) -> Optional[np.ndarray]:
    """
    加载波形数据的二进制数组
    
    Args:
        session: 数据库会话
        snapshot_id: 快照ID
    
    Returns:
        numpy数组，如果不存在则返回None
    """
    from backend import models
    
    waveform = session.execute(
        select(models.TelemetryWaveform).where(
            models.TelemetryWaveform.snapshot_id == snapshot_id
        )
    ).scalar_one_or_none()
    
    if waveform and waveform.data_blob:
        # 将二进制数据转换为numpy数组
        # 注意：需要知道原始数据类型和形状，这里假设是float32
        try:
            arr = np.frombuffer(waveform.data_blob, dtype=np.float32)
            return arr
        except Exception as e:
            log.warning(f"波形数据解析失败: {e}")
            return None
    
    return None


def get_maintenance_labels(
    data: Dict[str, pd.DataFrame],
    asset_id: str,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> List[Dict[str, Any]]:
    """
    获取维护标签（用于AI训练）
    
    Args:
        data: 数据字典
        asset_id: 设备ID
        start_time: 开始时间
        end_time: 结束时间
    
    Returns:
        标签列表，每个标签包含时间段和故障代码
    """
    df = data.get("maintenance_records", pd.DataFrame())
    if df.empty:
        return []
    
    df = df[df["asset_id"] == asset_id].copy()
    df = df[
        (df["start_time"] >= start_time) & 
        (df["start_time"] <= end_time)
    ]
    
    labels = []
    for _, row in df.iterrows():
        labels.append({
            "start_time": row["start_time"],
            "end_time": row["end_time"] if pd.notna(row["end_time"]) else end_time,
            "failure_code": row["failure_code"],
            "issue_description": row["issue_description"],
        })
    
    return labels


def get_knowledge_context(
    data: Dict[str, pd.DataFrame],
    model_id: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """
    获取知识库上下文（用于LLM RAG）
    
    Args:
        data: 数据字典
        model_id: 型号ID（可选）
        category: 类别（可选）
        limit: 返回数量限制
    
    Returns:
        知识片段列表
    """
    df = data.get("knowledge_base", pd.DataFrame())
    if df.empty:
        return []
    
    if model_id:
        df = df[df["applicable_model"] == model_id]
    if category:
        df = df[df["category"] == category]
    
    df = df.head(limit)
    
    return [
        {
            "doc_id": row["doc_id"],
            "title": row["title"],
            "content": row["content_chunk"],
            "category": row["category"],
        }
        for _, row in df.iterrows()
    ]

