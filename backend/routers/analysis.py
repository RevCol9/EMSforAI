
from __future__ import annotations

from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import models, schemas
from ..db import SessionLocal
from ..algorithm.algorithms_engine import AlgorithmEngine
from ..algorithm.data_service import load_data_from_db


router = APIRouter(prefix="/api/v2", tags=["算法分析"])


def get_db_session():
    """数据库会话依赖"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/device/{device_id}/analysis", response_model=schemas.DeviceAnalysisResponse)
def get_device_analysis(
    device_id: int,
    window_days: int = Query(30, ge=7, le=365, description="分析窗口天数"),
    quality_cutoff: float = Query(0.8, ge=0.0, le=1.0, description="数据质量阈值"),
    db: Session = Depends(get_db_session)
):
    """
    获取设备完整分析
    返回设备健康度、各指标分析、停机风险、产能影响、备件寿命等信息
    """
    # 检查设备是否存在
    dev = db.execute(select(models.Device).where(models.Device.id == device_id)).scalar_one_or_none()
    if not dev:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"设备 {device_id} 不存在"
        )
    
    # 从数据库加载数据
    try:
        data_bundle = load_data_from_db(session=db)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"数据加载失败: {str(e)}"
        )
    
    # 创建算法引擎并分析
    engine = AlgorithmEngine(window_days=window_days, quality_cutoff=quality_cutoff)
    context = engine.load(data_bundle)
    result = engine.analyze_device(context=context, device_id=device_id)
    
    # 转换备件信息格式
    spare_life = result.get("spare_life_info", {})
    spare_parts_list = []
    if isinstance(spare_life, dict) and "spare_parts" in spare_life:
        for part in spare_life.get("spare_parts", []):
            spare_parts_list.append(schemas.SparePartInfo(
                spare_part_id=part.get("spare_part_id"),
                spare_part_name=part.get("spare_part_name"),
                part_code=part.get("part_code"),
                usage_cycles=part.get("usage_cycles", 0),
                max_cycles=part.get("max_cycles", 0),
                usage_ratio=part.get("usage_ratio", 0.0),
                remaining_ratio=part.get("remaining_ratio", 0.0),
                status=part.get("status", "normal")
            ))
    
    return schemas.DeviceAnalysisResponse(
        device_id=result.get("device_id", device_id),
        device_health_score=result.get("device_health_score", 0.0),
        device_alert_level=result.get("device_alert_level", "unknown"),
        metrics=[
            schemas.MetricAnalysisResult(**metric) 
            for metric in result.get("metrics", [])
        ],
        downtime_risk=result.get("downtime_risk", 0.0),
        throughput_impact=result.get("throughput_impact", 0.0),
        spare_life_info=schemas.SpareLifeInfo(
            status=spare_life.get("status", "unknown"),
            spare_parts=spare_parts_list,
            total_parts=spare_life.get("total_parts", 0),
            critical_count=spare_life.get("critical_count", 0)
        )
    )


@router.get("/device/{device_id}/metrics/{metric_key}/analysis", response_model=schemas.MetricAnalysisResult)
def get_metric_analysis(
    device_id: int,
    metric_key: str,
    window_days: int = Query(30, ge=7, le=365, description="分析窗口天数"),
    quality_cutoff: float = Query(0.8, ge=0.0, le=1.0, description="数据质量阈值"),
    db: Session = Depends(get_db_session)
):
    """
    获取单指标分析
    返回指定设备的指定指标的详细分析结果，包括趋势、RUL、健康度等
    """
    # 检查设备是否存在
    dev = db.execute(select(models.Device).where(models.Device.id == device_id)).scalar_one_or_none()
    if not dev:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"设备 {device_id} 不存在"
        )
    
    # 获取指标定义
    def_row = db.execute(
        select(models.DeviceMetricDefinition).where(
            models.DeviceMetricDefinition.model_id == dev.model_id,
            models.DeviceMetricDefinition.metric_key == metric_key
        )
    ).scalar_one_or_none()
    
    if not def_row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"指标 {metric_key} 的定义不存在"
        )
    
    # 构建指标定义字典
    metric_def = {
        "metric_key": def_row.metric_key,
        "warn_threshold": def_row.warn_threshold,
        "crit_threshold": def_row.crit_threshold,
        "trend_direction": def_row.trend_direction,
        "weight_in_health": def_row.weight_in_health or 1.0,
        "criticality": getattr(def_row, "criticality", "medium"),
    }
    
    # 从数据库加载数据
    try:
        data_bundle = load_data_from_db(session=db)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"数据加载失败: {str(e)}"
        )
    
    # 创建算法引擎并分析
    engine = AlgorithmEngine(window_days=window_days, quality_cutoff=quality_cutoff)
    context = engine.load(data_bundle)
    result = engine.analyze_metric(context=context, defn=metric_def, device_id=device_id)
    
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="数据不足，无法进行分析"
        )
    
    return schemas.MetricAnalysisResult(**result)


@router.get("/device/{device_id}/health", response_model=schemas.DeviceOverview)
def get_device_health(
    device_id: int,
    window_days: int = Query(30, ge=7, le=365, description="分析窗口天数"),
    quality_cutoff: float = Query(0.8, ge=0.0, le=1.0, description="数据质量阈值"),
    db: Session = Depends(get_db_session)
):
    """
    获取设备健康度概览
    返回设备健康度分数和各个指标的概览信息（简化版，不包含详细分析）
    """
    # 检查设备是否存在
    dev = db.execute(select(models.Device).where(models.Device.id == device_id)).scalar_one_or_none()
    if not dev:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"设备 {device_id} 不存在"
        )
    
    # 从数据库加载数据
    try:
        data_bundle = load_data_from_db(session=db)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"数据加载失败: {str(e)}"
        )
    
    # 创建算法引擎并分析
    engine = AlgorithmEngine(window_days=window_days, quality_cutoff=quality_cutoff)
    context = engine.load(data_bundle)
    result = engine.analyze_device(context=context, device_id=device_id)
    
    # 构建指标概览列表
    metrics_overview = []
    for metric in result.get("metrics", []):
        metrics_overview.append(schemas.MetricOverviewItem(
            metric_key=metric.get("metric_key", ""),
            last_value=metric.get("current_value", 0.0),
            status=metric.get("alert_level", "unknown"),
            rul_days=metric.get("rul_days")
        ))
    
    return schemas.DeviceOverview(
        device_id=device_id,
        health_score=result.get("device_health_score", 0.0),
        metrics=metrics_overview
    )


@router.get("/devices/analysis")
def get_all_devices_analysis(
    window_days: int = Query(30, ge=7, le=365, description="分析窗口天数"),
    quality_cutoff: float = Query(0.8, ge=0.0, le=1.0, description="数据质量阈值"),
    db: Session = Depends(get_db_session)
):
    """
    获取所有设备的分析结果摘要
    返回所有设备的健康度、告警级别、停机风险、产能影响等摘要信息
    """
    # 获取所有设备
    devices = db.execute(select(models.Device)).scalars().all()
    if not devices:
        return []
    
    # 从数据库加载数据
    try:
        data_bundle = load_data_from_db(session=db)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"数据加载失败: {str(e)}"
        )
    
    # 创建算法引擎
    engine = AlgorithmEngine(window_days=window_days, quality_cutoff=quality_cutoff)
    context = engine.load(data_bundle)
    
    # 分析每个设备
    results = []
    for device in devices:
        try:
            result = engine.analyze_device(context=context, device_id=device.id)
            results.append({
                "device_id": result.get("device_id"),
                "device_health_score": result.get("device_health_score", 0.0),
                "device_alert_level": result.get("device_alert_level", "unknown"),
                "downtime_risk": result.get("downtime_risk", 0.0),
                "throughput_impact": result.get("throughput_impact", 0.0),
                "metrics_count": len(result.get("metrics", []))
            })
        except Exception as e:
            # 如果某个设备分析失败，记录错误但继续处理其他设备
            results.append({
                "device_id": device.id,
                "device_health_score": 0.0,
                "device_alert_level": f"分析失败: {str(e)}",
                "downtime_risk": 0.0,
                "throughput_impact": 0.0,
                "metrics_count": 0
            })
    
    return results

