from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class DeviceModelBase(BaseModel):
    name: str = Field(..., description="设备型号名称")


class DeviceModelCreate(DeviceModelBase):
    pass


class DeviceModelRead(DeviceModelBase):
    id: int

    class Config:
        orm_mode = True


class DeviceBase(BaseModel):
    name: str = Field(..., description="设备名称/资产编号")
    model_id: int = Field(..., description="关联设备型号ID")
    serial_number: Optional[str] = Field(None, description="序列号")
    location: Optional[str] = Field(None, description="位置/产线/区域")
    status: Optional[str] = Field("active", description="状态")


class DeviceCreate(DeviceBase):
    pass


class DeviceRead(DeviceBase):
    id: int
    model: Optional[DeviceModelRead]

    class Config:
        orm_mode = True


class InspectionSubmit(BaseModel):
    device_id: int
    user_id: int
    recorded_at: Optional[datetime] = None
    metrics: Dict[str, float]
    data_origin: Optional[str] = Field(None, description="数据来源标签")
    collector_id: Optional[str] = Field(None, description="采集设备/传感器编号")
    data_quality_score: Optional[float] = Field(None, description="数据质量评分")
    validation_notes: Optional[str] = Field(None, description="校验说明")


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
    prediction_confidence: Optional[float] = Field(None, description="预测置信度0-1")
    health_score: Optional[float] = Field(None, description="健康度")


class MetricOverviewItem(BaseModel):
    metric_key: str
    last_value: float
    status: str
    rul_days: Optional[int]


class DeviceOverview(BaseModel):
    device_id: int
    health_score: float
    metrics: List[MetricOverviewItem]
