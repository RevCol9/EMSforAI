from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    JSON,
    Text,
    PrimaryKeyConstraint,
    Index,
)
from sqlalchemy.orm import relationship
from datetime import datetime
from .db import Base


class DeviceType(Base):
    __tablename__ = "device_types"
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)


class Device(Base):
    __tablename__ = "devices"
    id = Column(Integer, primary_key=True)
    device_type_id = Column(Integer, ForeignKey("device_types.id"), nullable=False)
    name = Column(String(100), unique=True, nullable=False)
    device_type = relationship("DeviceType")


class DeviceMetricDefinition(Base):
    __tablename__ = "device_metric_definitions"
    id = Column(Integer, primary_key=True)
    device_type_id = Column(Integer, ForeignKey("device_types.id"), nullable=False)
    metric_key = Column(String(50), nullable=False)
    metric_name = Column(String(50), nullable=False)
    unit = Column(String(20))
    data_type = Column(String(20))
    warn_threshold = Column(Float)
    crit_threshold = Column(Float)
    valid_min = Column(Float)
    valid_max = Column(Float)
    trend_direction = Column(Integer, default=1)
    weight_in_health = Column(Float, default=1.0)
    is_ai_analyzed = Column(Boolean, default=True)
    device_type = relationship("DeviceType")
    __table_args__ = (
        Index("idx_metric_def_device_type_key", "device_type_id", "metric_key", unique=True),
    )


class InspectionLog(Base):
    __tablename__ = "inspection_logs"
    id = Column(Integer, primary_key=True)
    device_id = Column(Integer, ForeignKey("devices.id"), nullable=False)
    user_id = Column(Integer, nullable=False)
    recorded_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    device = relationship("Device")
    __table_args__ = (
        Index("idx_logs_device", "device_id"),
        Index("idx_logs_recorded_at", "recorded_at"),
    )


class InspectionMetricValue(Base):
    __tablename__ = "inspection_metric_values"
    log_id = Column(Integer, ForeignKey("inspection_logs.id"), primary_key=True)
    metrics_data = Column(JSON)
    log = relationship("InspectionLog")


class MetricAIAnalysis(Base):
    __tablename__ = "metric_ai_analysis"
    device_id = Column(Integer, ForeignKey("devices.id"), nullable=False)
    metric_key = Column(String(50), nullable=False)
    calc_time = Column(DateTime, default=datetime.utcnow, nullable=False)
    model_version = Column(String(20), nullable=False)
    rul_days = Column(Integer)
    trend_r2 = Column(Float)
    last_value = Column(Float)
    curve_points = Column(JSON)
    extra_info = Column(JSON)
    device = relationship("Device")
    __table_args__ = (
        PrimaryKeyConstraint("device_id", "metric_key", "calc_time", name="pk_metric_ai"),
        Index("idx_metric_ai_latest", "device_id", "metric_key"),
    )

