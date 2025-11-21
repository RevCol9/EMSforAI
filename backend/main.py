from typing import List, Dict, Optional
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from sqlalchemy import select, desc
from datetime import datetime

from .db import SessionLocal, engine, Base
from .models import (
    DeviceType,
    Device,
    DeviceMetricDefinition,
    InspectionLog,
    InspectionMetricValue,
    MetricAIAnalysis,
)
from .schemas import (
    InspectionSubmit,
    InspectionSubmitResponse,
    MetricAIResponse,
    CurvePoint,
    DeviceOverview,
    MetricOverviewItem,
)
from .ai_service import EquipmentAnalyzer


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


app = FastAPI(title="EMS for AI Demo")


Base.metadata.create_all(bind=engine)


def init_sample_data(db: Session):
    dt = db.execute(select(DeviceType).where(DeviceType.name == "Pump")).scalar_one_or_none()
    if not dt:
        dt = DeviceType(name="Pump")
        db.add(dt)
        db.flush()
    dev = db.execute(select(Device).where(Device.name == "Pump-001")).scalar_one_or_none()
    if not dev:
        dev = Device(device_type_id=dt.id, name="Pump-001")
        db.add(dev)
        db.flush()
    defs = [
        ("vibration", "Vibration", "mm/s", 6.0, 8.0, 0.0, 50.0, 1, 1.0),
        ("temperature", "Temperature", "°C", 70.0, 85.0, 0.0, 150.0, 1, 1.0),
    ]
    for mk, mn, unit, warn, crit, vmin, vmax, dirc, w in defs:
        exists = db.execute(
            select(DeviceMetricDefinition).where(
                DeviceMetricDefinition.device_type_id == dt.id,
                DeviceMetricDefinition.metric_key == mk,
            )
        ).scalar_one_or_none()
        if not exists:
            db.add(
                DeviceMetricDefinition(
                    device_type_id=dt.id,
                    metric_key=mk,
                    metric_name=mn,
                    unit=unit,
                    data_type="float",
                    warn_threshold=warn,
                    crit_threshold=crit,
                    valid_min=vmin,
                    valid_max=vmax,
                    trend_direction=dirc,
                    weight_in_health=w,
                    is_ai_analyzed=True,
                )
            )
    db.commit()


@app.on_event("startup")
def startup_event():
    with SessionLocal() as db:
        init_sample_data(db)


@app.post("/api/inspection/submit", response_model=InspectionSubmitResponse)
def submit_inspection(payload: InspectionSubmit, db: Session = Depends(get_db)):
    dev = db.execute(select(Device).where(Device.id == payload.device_id)).scalar_one_or_none()
    if not dev:
        return {"status": "error", "anomalies": {}}
    recorded_at = payload.recorded_at or datetime.utcnow()
    log = InspectionLog(
        device_id=payload.device_id,
        user_id=payload.user_id,
        recorded_at=recorded_at,
    )
    db.add(log)
    db.flush()
    metrics_data = payload.metrics
    db.add(InspectionMetricValue(log_id=log.id, metrics_data=metrics_data))

    anomalies: Dict[str, bool] = {}
    for mk, val in metrics_data.items():
        ddef = db.execute(
            select(DeviceMetricDefinition).where(
                DeviceMetricDefinition.device_type_id == dev.device_type_id,
                DeviceMetricDefinition.metric_key == mk,
            )
        ).scalar_one_or_none()
        history_rows = (
            db.execute(
                select(InspectionLog, InspectionMetricValue)
                .join(InspectionMetricValue, InspectionMetricValue.log_id == InspectionLog.id)
                .where(InspectionLog.device_id == payload.device_id)
                .order_by(InspectionLog.recorded_at)
            ).all()
        )
        hist: List[Dict[str, float]] = []
        for lr, mv in history_rows:
            if mv.metrics_data and mk in mv.metrics_data:
                hist.append({"date": lr.recorded_at.date().isoformat(), "value": float(mv.metrics_data[mk])})
        analyzer = EquipmentAnalyzer(hist)
        minv = ddef.valid_min if ddef else None
        maxv = ddef.valid_max if ddef else None
        anomalies[mk] = analyzer.detect_input_anomaly(float(val), minv, maxv)
        trend_dir = ddef.trend_direction if ddef else 1
        threshold = ddef.warn_threshold if ddef and ddef.warn_threshold is not None else float(val)
        rul = analyzer.predict_rul(threshold, trend_dir)
        curve = analyzer.generate_smooth_curve(points=60)
        db.add(
            MetricAIAnalysis(
                device_id=payload.device_id,
                metric_key=mk,
                model_version="v1",
                rul_days=rul.get("rul_days"),
                trend_r2=None,
                last_value=float(val),
                curve_points=curve,
                extra_info={"status": rul.get("status")},
            )
        )
    db.commit()
    return {"status": "success", "anomalies": anomalies}


@app.get("/api/device/{device_id}/metrics/{metric_key}/history")
def get_metric_history(device_id: int, metric_key: str, db: Session = Depends(get_db)):
    rows = (
        db.execute(
            select(InspectionLog, InspectionMetricValue)
            .join(InspectionMetricValue, InspectionMetricValue.log_id == InspectionLog.id)
            .where(InspectionLog.device_id == device_id)
            .order_by(InspectionLog.recorded_at)
        ).all()
    )
    out = []
    for lr, mv in rows:
        if mv.metrics_data and metric_key in mv.metrics_data:
            out.append({"date": lr.recorded_at.date().isoformat(), "value": float(mv.metrics_data[metric_key])})
    analyzer = EquipmentAnalyzer(out)
    return {"history": out, "curve": analyzer.generate_smooth_curve(points=60)}


@app.get("/api/device/{device_id}/metrics/{metric_key}/ai_analysis", response_model=MetricAIResponse)
def get_metric_ai(device_id: int, metric_key: str, db: Session = Depends(get_db)):
    row = db.execute(
        select(MetricAIAnalysis)
        .where(MetricAIAnalysis.device_id == device_id, MetricAIAnalysis.metric_key == metric_key)
        .order_by(desc(MetricAIAnalysis.calc_time))
    ).scalar_one_or_none()
    if row:
        curve = [CurvePoint(date=p["date"], value=p["value"]) for p in (row.curve_points or [])]
        return {
            "device_id": device_id,
            "metric_key": metric_key,
            "last_value": float(row.last_value or 0.0),
            "status": (row.extra_info or {}).get("status", "unknown"),
            "rul_days": row.rul_days,
            "curve": curve,
        }
    hist = get_metric_history(device_id, metric_key, db)
    analyzer = EquipmentAnalyzer(hist["history"]) if isinstance(hist, dict) else EquipmentAnalyzer([])
    curve = analyzer.generate_smooth_curve(points=60)
    return {
        "device_id": device_id,
        "metric_key": metric_key,
        "last_value": float(hist["history"][-1]["value"]) if hist["history"] else 0.0,
        "status": "unknown",
        "rul_days": None,
        "curve": [CurvePoint(date=p["date"], value=p["value"]) for p in curve],
    }


@app.get("/api/device/{device_id}/ai_overview", response_model=DeviceOverview)
def get_device_overview(device_id: int, db: Session = Depends(get_db)):
    dev = db.execute(select(Device).where(Device.id == device_id)).scalar_one_or_none()
    if not dev:
        return {"device_id": device_id, "health_score": 0.0, "metrics": []}
    defs = db.execute(select(DeviceMetricDefinition).where(DeviceMetricDefinition.device_type_id == dev.device_type_id)).scalars().all()
    latest_rows = db.execute(
        select(MetricAIAnalysis)
        .where(MetricAIAnalysis.device_id == device_id)
        .order_by(desc(MetricAIAnalysis.calc_time))
    ).scalars().all()
    latest_by_key: Dict[str, MetricAIAnalysis] = {}
    for r in latest_rows:
        if r.metric_key not in latest_by_key:
            latest_by_key[r.metric_key] = r
    items: List[MetricOverviewItem] = []
    score = 100.0
    for d in defs:
        row = latest_by_key.get(d.metric_key)
        if not row:
            continue
        val = float(row.last_value or 0.0)
        status = "normal"
        rul_days = row.rul_days
        if d.warn_threshold is not None and val >= d.warn_threshold:
            status = "warn"
        if d.crit_threshold is not None and val >= d.crit_threshold:
            status = "danger"
        delta = 0.0
        if d.warn_threshold is not None:
            delta = max(0.0, val - d.warn_threshold) * (d.weight_in_health or 1.0)
        score -= delta
        items.append(MetricOverviewItem(metric_key=d.metric_key, last_value=val, status=status, rul_days=rul_days))
    score = max(0.0, min(100.0, score))
    return DeviceOverview(device_id=device_id, health_score=round(score, 2), metrics=items)

