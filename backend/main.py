from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from sqlalchemy import select
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict

from .db import SessionLocal, engine, Base
from .models import Device, InspectionLog, InspectionMetricValue
from .schemas import InspectionSubmit, InspectionSubmitResponse
from .routers import analysis, devices


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


app = FastAPI(title="EMSforAI")

# 注册路由模块
app.include_router(devices.router)  # 设备管理相关接口
app.include_router(analysis.router)  # 算法分析相关接口

Base.metadata.create_all(bind=engine)


@app.get("/health")
def health():
    """健康检查接口"""
    return {"status": "ok"}


@app.post("/api/inspection/submit", response_model=InspectionSubmitResponse)
def submit_inspection(payload: InspectionSubmit, db: Session = Depends(get_db)):
    """
    提交巡检数据
    
    保存巡检记录到数据库，返回简单的异常检测结果
    详细的算法分析请使用 /api/v2/device/{device_id}/analysis 接口
    """
    dev = db.execute(select(Device).where(Device.id == payload.device_id)).scalar_one_or_none()
    if not dev:
        return {"status": "error", "anomalies": {}}
    
    tz = ZoneInfo("Asia/Shanghai")
    recorded_at = payload.recorded_at or datetime.now(tz)
    log = InspectionLog(device_id=payload.device_id, user_id=payload.user_id, recorded_at=recorded_at)
    db.add(log)
    db.flush()
    
    metrics_data = payload.metrics
    db.add(InspectionMetricValue(log_id=log.id, metrics_data=metrics_data))
    
    # 异常检测：当前返回空结果，详细分析请使用算法引擎接口
    anomalies: Dict[str, bool] = {}
    for mk, val in metrics_data.items():
        # 基础异常检测逻辑：可根据指标定义进行阈值检查
        anomalies[mk] = False
    
    db.commit()
    return {"status": "success", "anomalies": anomalies}


