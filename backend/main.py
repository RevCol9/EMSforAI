"""
EMSforAI 主应用入口

FastAPI应用主文件，负责：
- 应用初始化
- 路由注册
- 数据库表创建
- 全局接口定义（健康检查、巡检数据提交等）

Author: EMSforAI Team
License: MIT
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径，支持直接运行
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from sqlalchemy import select
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict

# 支持相对导入（作为模块）和绝对导入（直接运行）
try:
    from .db import SessionLocal, engine, Base
    from .models import Asset, TelemetryProcess, MetricDefinition
    from .schemas import InspectionSubmit, InspectionSubmitResponse
    from .routers import analysis, devices
except ImportError:
    # 如果相对导入失败，使用绝对导入（直接运行时）
    from backend.db import SessionLocal, engine, Base
    from backend.models import Asset, TelemetryProcess, MetricDefinition
    from backend.schemas import InspectionSubmit, InspectionSubmitResponse
    from backend.routers import analysis, devices


def get_db():
    """
    数据库会话依赖注入函数
    
    用于FastAPI的依赖注入系统，为每个请求提供独立的数据库会话。
    请求结束后自动关闭会话，确保资源正确释放。
    
    Yields:
        Session: SQLAlchemy数据库会话对象
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# 创建FastAPI应用实例
app = FastAPI(
    title="EMSforAI",
    description="设备管理系统 (Equipment Management System) 的 AI 算法引擎",
    version="2.0.0"
)

# 注册路由模块
app.include_router(devices.router)  # 设备管理相关接口：/devices/*
app.include_router(analysis.router)  # 算法分析相关接口：/api/v2/*

# 初始化数据库表结构（如果不存在则创建）
Base.metadata.create_all(bind=engine)


@app.get("/health")
def health():
    """
    健康检查接口
    
    用于检查服务是否正常运行，常用于：
    - 负载均衡器健康检查
    - 监控系统服务状态
    - 容器编排系统（如Kubernetes）的存活探针
    
    Returns:
        Dict: 健康状态，包含：
            - status: 服务状态（"ok"表示正常）
    
    Example:
        ```bash
        GET /health
        ```
    """
    return {"status": "ok"}


@app.post("/api/inspection/submit", response_model=InspectionSubmitResponse)
def submit_inspection(payload: InspectionSubmit, db: Session = Depends(get_db)):
    """
    提交巡检数据
    
    接收来自前端或数据采集系统的巡检数据，保存到数据库并返回简单的异常检测结果。
    该接口主要用于数据采集，详细的算法分析请使用 /api/v2/device/{device_id}/analysis 接口。
    
    Args:
        payload (InspectionSubmit): 巡检数据提交请求体，包含：
            - device_id: 设备ID（整数格式，内部转换为字符串asset_id）
            - user_id: 提交用户ID
            - recorded_at: 记录时间（可选，默认当前时间）
            - metrics: 测点数据字典，格式为 {metric_key: value}
            - data_origin: 数据来源标签（可选）
            - collector_id: 采集设备/传感器编号（可选）
            - data_quality_score: 数据质量评分（0-1，可选）
            - validation_notes: 校验说明（可选）
        db (Session): 数据库会话（自动注入）
    
    Returns:
        InspectionSubmitResponse: 提交结果，包含：
            - status: 提交状态（"success"或"error"）
            - anomalies: 异常检测结果字典，格式为 {metric_key: is_anomaly}
    
    Note:
        - 如果设备不存在，返回status="error"，anomalies为空字典
        - 如果测点定义不存在，该测点的数据会被跳过，anomalies中标记为False
        - 数据质量评分>=0.8时，quality字段为1（高质量），否则为0（低质量）
        - 工况状态默认为2（RUNNING，加工状态）
        - 当前版本的异常检测逻辑较简单，仅返回False（无异常）
    
    Example:
        ```json
        POST /api/inspection/submit
        {
            "device_id": "CNC-MAZAK-01",
            "user_id": 100,
            "recorded_at": "2024-01-01T10:00:00+08:00",
            "metrics": {
                "SPINDLE_TEMP": 45.5,
                "SPINDLE_VIB": 2.3
            },
            "data_quality_score": 0.95
        }
        ```
    """
    # device_id 直接作为 asset_id 使用（字符串格式）
    asset_id_str = payload.device_id
    asset = db.execute(select(Asset).where(Asset.asset_id == asset_id_str)).scalar_one_or_none()
    if not asset:
        return {"status": "error", "anomalies": {}}
    
    tz = ZoneInfo("Asia/Shanghai")
    recorded_at = payload.recorded_at or datetime.now(tz)
    
    # 获取该设备的所有测点定义
    # 用于将metric_key映射到metric_id
    metrics_data = payload.metrics
    anomalies: Dict[str, bool] = {}
    
    metric_defs = db.execute(
        select(MetricDefinition).where(MetricDefinition.asset_id == asset_id_str)
    ).scalars().all()
    
    # 创建metric_key到metric_id的映射
    # 
    # 映射策略说明：
    # 1. 直接使用完整的metric_id作为key（保证唯一性，避免冲突）
    #    例如：CNC01_SPINDLE_TEMP 和 CNC01_BEARING_TEMP 不会冲突
    # 2. 同时支持从metric_id中提取的短名称作为key（向后兼容）
    #    例如：如果payload中的metric_key是"TEMP"，也能匹配到对应的metric_id
    # 
    # 注意：如果多个metric_id提取的短名称相同（如SPINDLE_TEMP和BEARING_TEMP都提取为TEMP），
    # 则短名称映射会被最后一个覆盖，但完整metric_id映射仍然可用
    metric_id_map = {}
    for metric_def in metric_defs:
        metric_id = metric_def.metric_id
        
        # 策略1: 使用完整的metric_id作为key（主要映射，保证唯一性）
        metric_id_map[metric_id] = metric_id
        
        # 策略2: 提取短名称作为key（向后兼容，但可能冲突）
        # 如果metric_id包含下划线，取最后一部分作为短名称
        if '_' in metric_id:
            short_key = metric_id.split('_')[-1]
            # 只有当短名称不存在或与完整metric_id相同时才添加
            # 这样可以避免不同metric_id的短名称互相覆盖
            if short_key not in metric_id_map or metric_id_map[short_key] == short_key:
                metric_id_map[short_key] = metric_id
    
    # 将巡检数据转换为TelemetryProcess记录并保存
    for metric_key, value in metrics_data.items():
        # 查找对应的metric_id
        metric_id = metric_id_map.get(metric_key)
        if not metric_id:
            # 如果找不到对应的测点定义，跳过该数据点
            anomalies[metric_key] = False
            continue
        
        # 确定数据质量和工况状态
        # 数据质量：quality_score>=0.8为高质量(1)，否则为低质量(0)
        quality = 1 if (payload.data_quality_score is None or payload.data_quality_score >= 0.8) else 0
        # 工况状态：默认为2（RUNNING，加工状态）
        machine_state = 2
        
        # 创建TelemetryProcess记录
        process_record = TelemetryProcess(
            timestamp=recorded_at,
            metric_id=metric_id,
            value=float(value),
            quality=quality,
            machine_state=machine_state
        )
        db.add(process_record)
        
        # 基础异常检测逻辑（当前版本简单返回False）
        # 未来可以基于MetricDefinition中的阈值进行异常检测
        anomalies[metric_key] = False
    
    # 提交事务
    db.commit()
    return {"status": "success", "anomalies": anomalies}


# 支持直接运行（开发模式）
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


