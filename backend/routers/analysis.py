
"""
EMSforAI 算法分析API路由模块

本模块提供设备健康分析相关的RESTful API接口，包括：
- 设备健康度分析
- 单指标详细分析
- 设备健康度概览
- 批量设备分析摘要

Author: EMSforAI Team
License: MIT
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import models, schemas
from ..db import SessionLocal
from ..algorithm.algorithms_engine import AlgorithmEngine, HealthAnalysisResult
from ..algorithm.data_service import load_data_from_db

BASE_DIR = Path(__file__).resolve().parents[2]


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
    device_id: str,
    window_days: int = Query(30, ge=7, le=365, description="分析窗口天数"),
    quality_threshold: int = Query(1, ge=0, le=1, description="数据质量阈值（0或1）"),
    use_lstm: bool = Query(True, description="是否优先使用LSTM模型（True=LSTM优先，False=仅传统模型）"),
    db: Session = Depends(get_db_session)
):
    """
    获取设备完整分析报告
    
    对指定设备进行全面的健康分析，包括健康度评分、告警级别、停机风险、产能影响等。
    该接口会调用算法引擎进行多维度分析，优先使用LSTM模型进行RUL预测。
    
    Args:
        device_id (str): 设备ID（asset_id），例如："CNC-MAZAK-01"、"COMP-ATLAS-01"
        window_days (int): 分析窗口天数，范围7-365天，默认30天
                          - 窗口越大，分析结果越稳定，但可能包含过多历史数据
                          - 窗口越小，更能反映近期趋势，但可能受噪声影响
        quality_threshold (int): 数据质量阈值，0或1，默认1
                                - 0: 包含所有质量的数据（包括低质量）
                                - 1: 仅使用高质量数据（推荐）
        use_lstm (bool): 是否使用LSTM模型，默认True
                        - True: 使用LSTM模型，如果LSTM模型不存在，返回错误"当前设备暂无预训练LSTM模型"
                        - False: 仅使用传统模型（线性回归、多项式回归、指数回归、分段线性回归）
        db (Session): 数据库会话（自动注入）
    
    Returns:
        DeviceAnalysisResponse: 设备分析结果，包含：
            - device_id: 设备ID
            - device_health_score: 设备健康度分数（0-100）
            - device_alert_level: 告警级别（normal/warning/critical）
            - metrics: 各指标详细分析列表（当前版本为空）
            - downtime_risk: 停机风险（0-1，基于健康度估算）
            - throughput_impact: 产能影响（0-1，基于健康度估算）
            - spare_life_info: 备件寿命信息（当前版本无数据）
    
    Raises:
        HTTPException 404: 设备不存在
        HTTPException 500: 数据加载失败或算法分析异常
    
    Example:
        ```bash
        GET /api/v2/device/CNC-MAZAK-01/analysis?window_days=30&quality_threshold=1
        ```
    """
    # 步骤1: 验证设备是否存在
    # 注意：device_id 直接作为 asset_id 使用（字符串格式）
    asset = db.execute(select(models.Asset).where(models.Asset.asset_id == device_id)).scalar_one_or_none()
    if not asset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"设备 {device_id} 不存在"
        )
    
    # 步骤2: 从数据库加载设备相关数据
    # 包括：测点定义、过程数据、波形数据等
    try:
        data_bundle = load_data_from_db(session=db)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"数据加载失败: {str(e)}"
        )
    
    # 步骤3: 创建算法引擎并执行分析
    # 算法引擎会进行：健康度计算、RUL预测、趋势分析等
    asset_id_str = device_id
    engine = AlgorithmEngine(window_days=window_days, quality_threshold=quality_threshold)
    
    # 如果use_lstm=True，检查是否有LSTM模型，如果没有则返回错误
    require_lstm = use_lstm  # 如果前端选择使用LSTM，则要求必须有模型
    try:
        result = engine.analyze_asset(data_bundle, asset_id_str, use_lstm=use_lstm, require_lstm=require_lstm)
    except ValueError as e:
        # 如果没有LSTM模型且require_lstm=True，返回明确的错误信息
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    # 步骤4: 转换分析结果为API响应格式
    # 新架构返回HealthAnalysisResult对象，需要转换为DeviceAnalysisResponse格式
    from backend.algorithm.algorithms_engine import HealthAnalysisResult
    
    if isinstance(result, HealthAnalysisResult):
        # 提取健康度分数
        health_score = result.health_score
        
        # 根据健康度估算停机风险和产能影响
        # 健康度越低，风险越高（线性映射）
        downtime_risk = max(0.0, min(1.0, (100 - health_score) / 100.0))
        throughput_impact = max(0.0, min(1.0, (100 - health_score) / 100.0))
        
        # 根据健康度确定告警级别
        # 分级标准：>=60正常，30-60警告，<30严重
        if health_score < 30:
            alert_level = "critical"
        elif health_score < 60:
            alert_level = "warning"
        else:
            alert_level = "normal"
        
        return schemas.DeviceAnalysisResponse(
            device_id=device_id,
            device_health_score=health_score,
            device_alert_level=alert_level,
            metrics=[],  # 新架构不提供详细指标列表
            downtime_risk=downtime_risk,
            throughput_impact=throughput_impact,
            spare_life_info=schemas.SpareLifeInfo(
                status="no_data",  # 新架构不提供备件信息
                spare_parts=[],
                total_parts=0,
                critical_count=0
            )
        )
    else:
        # 兼容旧格式（字典）
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
    device_id: str,
    metric_key: str,
    window_days: int = Query(30, ge=7, le=365, description="分析窗口天数"),
    quality_threshold: int = Query(1, ge=0, le=1, description="数据质量阈值（0或1）"),
    use_lstm: bool = Query(True, description="是否使用LSTM模型（True=使用LSTM，False=仅传统模型）"),
    db: Session = Depends(get_db_session)
):
    """
    获取单指标详细分析
    
    对指定设备的指定测点进行深入分析，包括当前值、健康度、RUL预测、趋势分析等。
    该接口适用于需要查看单个测点详细信息的场景。
    
    Args:
        device_id (str): 设备ID（asset_id），例如："CNC-MAZAK-01"、"COMP-ATLAS-01"
        metric_key (str): 测点标识符（metric_id），例如："CNC01_SPINDLE_TEMP"
        window_days (int): 分析窗口天数，范围7-365天，默认30天
        quality_threshold (int): 数据质量阈值，0或1，默认1（仅高质量数据）
        use_lstm (bool): 是否使用LSTM模型，默认True
                        - True: 使用LSTM模型，如果LSTM模型不存在，返回错误
                        - False: 仅使用传统模型（线性回归、多项式回归、指数回归、分段线性回归）
        db (Session): 数据库会话（自动注入）
    
    Returns:
        MetricAnalysisResult: 单指标分析结果，包含：
            - metric_key: 测点标识符
            - device_id: 设备ID
            - current_value: 当前值（当前版本可能为0）
            - health_score: 健康度分数（0-100）
            - alert_level: 告警级别（normal/warning/critical）
            - data_points: 数据点数量（当前版本可能为0）
            - rul_days: 剩余使用寿命（天数），可能为None
            - rul_status: RUL计算状态（"正常"或"无法计算"）
            - trend_beta: 趋势斜率
            - prediction_confidence: 预测置信度（0-1）
    
    Raises:
        HTTPException 404: 设备或测点不存在
        HTTPException 400: 如果use_lstm=True但设备没有预训练LSTM模型，或数据不足无法进行分析
        HTTPException 500: 数据加载失败或算法分析异常
    
    Note:
        当前版本由于架构限制，部分字段（如current_value、data_points）可能返回默认值。
        建议使用 /device/{device_id}/analysis 接口获取更完整的分析结果。
    
    Example:
        ```bash
        GET /api/v2/device/CNC-MAZAK-01/metrics/CNC01_SPINDLE_TEMP/analysis?window_days=30
        ```
    """
    # 步骤1: 验证设备是否存在
    asset = db.execute(select(models.Asset).where(models.Asset.asset_id == device_id)).scalar_one_or_none()
    if not asset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"设备 {device_id} 不存在"
        )
    
    # 步骤2: 验证测点定义是否存在
    # metric_key参数实际对应数据库中的metric_id字段
    asset_id_str = device_id
    def_row = db.execute(
        select(models.MetricDefinition).where(
            models.MetricDefinition.asset_id == asset_id_str,
            models.MetricDefinition.metric_id == metric_key
        )
    ).scalar_one_or_none()
    
    if not def_row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"指标 {metric_key} 的定义不存在"
        )
    
    # 步骤3: 从数据库加载该设备的数据
    try:
        data_bundle = load_data_from_db(session=db, asset_id=asset_id_str)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"数据加载失败: {str(e)}"
        )
    
    # 步骤4: 执行设备整体分析
    # 注意：当前架构的analyze_asset方法返回设备整体分析结果，不提供单指标详细分析
    # 这里使用整体分析结果来构建单指标响应
    engine = AlgorithmEngine(window_days=window_days, quality_threshold=quality_threshold)
    require_lstm = use_lstm  # 如果前端选择使用LSTM，则要求必须有模型
    try:
        result = engine.analyze_asset(data_bundle, asset_id_str, use_lstm=use_lstm, require_lstm=require_lstm)
    except ValueError as e:
        # 如果没有LSTM模型且require_lstm=True，返回明确的错误信息
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    # 步骤5: 转换分析结果为单指标响应格式
    if isinstance(result, HealthAnalysisResult):
        # 根据整体健康度确定告警级别
        if result.health_score >= 60:
            alert_level = "normal"
        elif result.health_score >= 30:
            alert_level = "warning"
        else:
            alert_level = "critical"
        
        return schemas.MetricAnalysisResult(
            metric_key=metric_key,
            device_id=device_id,
            current_value=0.0,  # 当前版本暂不支持单指标当前值提取
            health_score=result.health_score,
            alert_level=alert_level,
            data_points=0,  # 当前版本暂不支持数据点统计
            rul_days=int(result.rul_days) if result.rul_days is not None else None,
            rul_status="无法计算" if result.rul_days is None else "正常",
            trend_alpha=None,  # 当前版本不提供截距
            trend_beta=result.trend_slope,  # 趋势斜率
            trend_r2=None,  # 当前版本不提供R²值
            prediction_confidence=result.prediction_confidence,
            weight_in_health=None,  # 当前版本不提供权重信息
            criticality=None  # 当前版本不提供关键程度
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="数据不足，无法进行分析"
        )


@router.get("/device/{device_id}/health", response_model=schemas.DeviceOverview)
def get_device_health(
    device_id: str,
    window_days: int = Query(30, ge=7, le=365, description="分析窗口天数"),
    quality_threshold: int = Query(1, ge=0, le=1, description="数据质量阈值（0或1）"),
    db: Session = Depends(get_db_session)
):
    """
    获取设备健康度概览（轻量级接口）
    
    快速获取设备健康度分数和指标概览，不包含详细分析。
    适用于需要频繁查询健康状态的场景，性能优于完整分析接口。
    
    Args:
        device_id (str): 设备ID（asset_id），例如："CNC-MAZAK-01"、"COMP-ATLAS-01"
        window_days (int): 分析窗口天数，范围7-365天，默认30天
        quality_threshold (int): 数据质量阈值，0或1，默认1（仅高质量数据）
        db (Session): 数据库会话（自动注入）
    
    Returns:
        DeviceOverview: 设备健康度概览，包含：
            - device_id: 设备ID
            - health_score: 设备健康度分数（0-100）
            - metrics: 各指标概览列表（当前版本为空）
    
    Raises:
        HTTPException 404: 设备不存在
        HTTPException 500: 数据加载失败或算法分析异常
    
    Note:
        该接口返回的是简化版数据，如需详细信息请使用 /device/{device_id}/analysis 接口。
    
    Example:
        ```bash
        GET /api/v2/device/CNC-MAZAK-01/health?window_days=30
        ```
    """
    # 步骤1: 验证设备是否存在
    asset = db.execute(select(models.Asset).where(models.Asset.asset_id == device_id)).scalar_one_or_none()
    if not asset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"设备 {device_id} 不存在"
        )
    
    # 步骤2: 从数据库加载设备数据
    try:
        data_bundle = load_data_from_db(session=db)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"数据加载失败: {str(e)}"
        )
    
    # 步骤3: 执行健康度分析
    asset_id_str = device_id
    engine = AlgorithmEngine(window_days=window_days, quality_threshold=quality_threshold)
    result = engine.analyze_asset(data_bundle, asset_id_str)
    
    # 步骤4: 构建概览响应
    # 当前版本不提供详细指标列表，返回空列表
    metrics_overview = []
    if isinstance(result, HealthAnalysisResult):
        # 预留：未来可以在这里提取各指标的概览信息
        pass
    
    return schemas.DeviceOverview(
        device_id=device_id,
        health_score=result.health_score if isinstance(result, HealthAnalysisResult) else 0.0,
        metrics=metrics_overview
    )


@router.get("/devices/analysis")
def get_all_devices_analysis(
    window_days: int = Query(30, ge=7, le=365, description="分析窗口天数"),
    quality_threshold: int = Query(1, ge=0, le=1, description="数据质量阈值（0或1）"),
    db: Session = Depends(get_db_session)
):
    """
    获取所有设备的分析结果摘要（批量查询接口）
    
    对系统中的所有设备进行批量分析，返回每个设备的健康度、告警级别、风险等摘要信息。
    适用于设备监控大屏、设备列表页面等需要展示多设备状态的场景。
    
    Args:
        window_days (int): 分析窗口天数，范围7-365天，默认30天
        quality_threshold (int): 数据质量阈值，0或1，默认1（仅高质量数据）
        db (Session): 数据库会话（自动注入）
    
    Returns:
        List[Dict]: 设备分析结果列表，每个元素包含：
            - device_id: 设备ID（字符串格式的asset_id）
            - device_health_score: 设备健康度分数（0-100）
            - device_alert_level: 告警级别（normal/warning/critical）
            - downtime_risk: 停机风险（0-1）
            - throughput_impact: 产能影响（0-1）
            - metrics_count: 指标数量（当前版本为0）
    
    Note:
        - 如果某个设备分析失败，会在结果中标记错误信息，不影响其他设备的分析
        - 该接口可能耗时较长，建议前端使用分页或异步加载
        - 返回的设备ID为字符串格式（asset_id），而非整数格式
    
    Example:
        ```bash
        GET /api/v2/devices/analysis?window_days=30&quality_threshold=1
        ```
    """
    # 步骤1: 获取系统中所有设备
    devices = db.execute(select(models.Asset)).scalars().all()
    if not devices:
        return []
    
    # 步骤2: 从数据库加载所有设备的数据
    # 一次性加载所有数据，避免重复查询数据库
    try:
        data_bundle = load_data_from_db(session=db)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"数据加载失败: {str(e)}"
        )
    
    # 步骤3: 创建算法引擎（所有设备共享同一配置）
    engine = AlgorithmEngine(window_days=window_days, quality_threshold=quality_threshold)
    
    # 步骤4: 遍历每个设备进行分析
    results = []
    for device in devices:
        try:
            asset_id_str = device.asset_id
            result = engine.analyze_asset(data_bundle, asset_id_str)
            
            if isinstance(result, HealthAnalysisResult):
                # 提取分析结果并计算衍生指标
                health_score = result.health_score
                
                # 根据健康度计算停机风险和产能影响（线性映射）
                downtime_risk = max(0.0, min(1.0, (100 - health_score) / 100.0))
                throughput_impact = max(0.0, min(1.0, (100 - health_score) / 100.0))
                
                # 根据健康度确定告警级别
                if health_score < 30:
                    alert_level = "critical"
                elif health_score < 60:
                    alert_level = "warning"
                else:
                    alert_level = "normal"
                
                results.append({
                    "device_id": device.asset_id,  # 使用字符串格式的asset_id
                    "device_health_score": health_score,
                    "device_alert_level": alert_level,
                    "downtime_risk": downtime_risk,
                    "throughput_impact": throughput_impact,
                    "metrics_count": 0  # 当前版本不提供指标数量统计
                })
            else:
                # 兼容旧格式返回结果（字典格式）
                results.append({
                    "device_id": result.get("device_id"),
                    "device_health_score": result.get("device_health_score", 0.0),
                    "device_alert_level": result.get("device_alert_level", "unknown"),
                    "downtime_risk": result.get("downtime_risk", 0.0),
                    "throughput_impact": result.get("throughput_impact", 0.0),
                    "metrics_count": len(result.get("metrics", []))
                })
        except Exception as e:
            # 如果某个设备分析失败，记录错误信息但继续处理其他设备
            # 这确保了批量查询的健壮性：一个设备失败不影响其他设备
            results.append({
                "device_id": device.asset_id,
                "device_health_score": 0.0,
                "device_alert_level": f"分析失败: {str(e)}",
                "downtime_risk": 0.0,
                "throughput_impact": 0.0,
                "metrics_count": 0
            })
    
    return results


@router.get("/device/{device_id}/metrics", response_model=List[schemas.MetricListItem])
def get_device_metrics(
    device_id: str,
    db: Session = Depends(get_db_session)
):
    """
    获取设备的所有测点列表
    
    返回指定设备的所有测点定义，包括测点ID、名称、类型、阈值等信息。
    前端可以使用此接口来展示测点选择器，让用户选择要分析的测点。
    
    Args:
        device_id (str): 设备ID（asset_id），例如："CNC-MAZAK-01"、"COMP-ATLAS-01"
        db (Session): 数据库会话（自动注入）
    
    Returns:
        List[MetricListItem]: 测点列表，每个元素包含：
            - metric_id: 测点ID
            - metric_name: 测点名称
            - metric_type: 测点类型（PROCESS/WAVEFORM等）
            - unit: 单位
            - warn_threshold: 警告阈值
            - crit_threshold: 临界阈值
            - has_lstm_model: 是否有LSTM模型
    
    Raises:
        HTTPException 404: 设备不存在
    
    Example:
        ```bash
        GET /api/v2/device/CNC-MAZAK-01/metrics
        ```
    """
    # 步骤1: 验证设备是否存在
    asset_id_str = device_id
    asset = db.execute(select(models.Asset).where(models.Asset.asset_id == asset_id_str)).scalar_one_or_none()
    if not asset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"设备 {device_id} 不存在"
        )
    
    # 步骤2: 查询该设备的所有测点定义
    metric_defs = db.execute(
        select(models.MetricDefinition).where(
            models.MetricDefinition.asset_id == asset_id_str
        )
    ).scalars().all()
    
    # 步骤3: 检查每个测点是否有LSTM模型
    models_dir = BASE_DIR / "models" / "lstm"
    results = []
    for metric_def in metric_defs:
        model_path = models_dir / f"{asset_id_str}_{metric_def.metric_id}_lstm.pt"
        has_lstm = model_path.exists()
        
        results.append(schemas.MetricListItem(
            metric_id=metric_def.metric_id,
            metric_name=metric_def.metric_name,
            metric_type=metric_def.metric_type.value if metric_def.metric_type else None,
            unit=metric_def.unit,
            warn_threshold=metric_def.warn_threshold,
            crit_threshold=metric_def.crit_threshold,
            has_lstm_model=has_lstm
        ))
    
    return results


@router.get("/device/{device_id}/radar", response_model=schemas.RadarChartResponse)
def get_device_radar_chart(
    device_id: str,
    window_days: int = Query(30, ge=7, le=365, description="分析窗口天数"),
    quality_threshold: int = Query(1, ge=0, le=1, description="数据质量阈值（0或1）"),
    db: Session = Depends(get_db_session)
):
    """
    获取设备雷达图数据（多测点综合健康度）
    
    返回设备各维度的健康度评分，用于前端绘制雷达图。
    雷达图展示设备在不同维度（如温度、振动、负载等）的健康状态。
    
    Args:
        device_id (str): 设备ID（asset_id），例如："CNC-MAZAK-01"、"COMP-ATLAS-01"
        window_days (int): 分析窗口天数，范围7-365天，默认30天
        quality_threshold (int): 数据质量阈值，0或1，默认1（仅高质量数据）
        db (Session): 数据库会话（自动注入）
    
    Returns:
        RadarChartResponse: 雷达图数据，包含：
            - device_id: 设备ID
            - device_health_score: 设备整体健康度（0-100）
            - dimensions: 各维度评分列表，每个维度包含：
                - dimension_name: 维度名称（如"温度"、"振动"等）
                - metric_id: 测点ID
                - health_score: 健康度分数（0-100）
                - current_value: 当前值
                - warn_threshold: 警告阈值
                - crit_threshold: 临界阈值
                - trend: 趋势（上升/下降/稳定）
                - alert_level: 告警级别（normal/warning/critical）
    
    Raises:
        HTTPException 404: 设备不存在
        HTTPException 500: 数据加载失败或算法分析异常
    
    Example:
        ```bash
        GET /api/v2/device/CNC-MAZAK-01/radar?window_days=30&quality_threshold=1
        ```
    """
    # 步骤1: 验证设备是否存在
    asset_id_str = device_id
    asset = db.execute(select(models.Asset).where(models.Asset.asset_id == asset_id_str)).scalar_one_or_none()
    if not asset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"设备 {device_id} 不存在"
        )
    
    # 步骤2: 从数据库加载设备数据
    try:
        data_bundle = load_data_from_db(session=db, asset_id=asset_id_str)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"数据加载失败: {str(e)}"
        )
    
    # 步骤3: 执行健康度分析（获取维度评分）
    engine = AlgorithmEngine(window_days=window_days, quality_threshold=quality_threshold)
    result = engine.analyze_asset(data_bundle, asset_id_str)
    
    # 步骤4: 提取雷达图数据
    if isinstance(result, HealthAnalysisResult):
        dimension_scores = result.dimension_scores
        
        # 转换为API响应格式
        dimensions = [
            schemas.RadarDimensionScore(
                dimension_name=dim.dimension_name,
                metric_id=dim.metric_id,
                health_score=dim.health_score,
                current_value=dim.current_value,
                warn_threshold=dim.warn_threshold,
                crit_threshold=dim.crit_threshold,
                trend=dim.trend,
                alert_level=dim.alert_level
            )
            for dim in dimension_scores
        ]
        
        return schemas.RadarChartResponse(
            device_id=device_id,
            device_health_score=result.health_score,
            dimensions=dimensions
        )
    else:
        # 兼容旧格式（如果没有维度评分，返回空列表）
        return schemas.RadarChartResponse(
            device_id=device_id,
            device_health_score=result.get("device_health_score", 0.0) if isinstance(result, dict) else 0.0,
            dimensions=[]
        )

