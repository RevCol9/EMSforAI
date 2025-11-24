from typing import Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel


class DeviceModelCreate(BaseModel):
    name: str


class DeviceCreate(BaseModel):
    name: str
    model_id: int


class InspectionSubmit(BaseModel):
    device_id: int
    user_id: int
    recorded_at: Optional[datetime] = None
    metrics: Dict[str, float]


class InspectionSubmitResponse(BaseModel):
    status: str
    anomalies: Dict[str, bool]


class CurvePoint(BaseModel):
    date: str
    value: float


class MetricAIResponse(BaseModel):
    device_id: int
    metric_key: str
    last_value: float
    status: str
    rul_days: Optional[int]
    curve: List[CurvePoint]


class MetricOverviewItem(BaseModel):
    metric_key: str
    last_value: float
    status: str
    rul_days: Optional[int]


class DeviceOverview(BaseModel):
    device_id: int
    health_score: float
    metrics: List[MetricOverviewItem]
