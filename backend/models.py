from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    JSON,
    Index,
    PrimaryKeyConstraint,
    BigInteger,
    Text,
    Date,
)
from sqlalchemy.orm import relationship
from datetime import datetime
from .db import Base


class DeviceModel(Base):
    __tablename__ = "device_models"
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    manufacturer = Column(String(100))
    description = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class Device(Base):
    __tablename__ = "devices"
    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey("device_models.id"), nullable=False)
    name = Column(String(100), unique=True, nullable=False)
    serial_number = Column(String(100), unique=True, nullable=False)
    location = Column(String(100))
    status = Column(String(50), default="active", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_service_date = Column(DateTime)
    model = relationship("DeviceModel")
    __table_args__ = (
        Index("idx_devices_model", "model_id"),
        Index("idx_devices_serial", "serial_number", unique=True),
    )


class DeviceMetricDefinition(Base):
    __tablename__ = "device_metric_definitions"
    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey("device_models.id"), nullable=False)
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
    model = relationship("DeviceModel")
    __table_args__ = (Index("idx_metric_def_model_key", "model_id", "metric_key", unique=True),)


class InspectionLog(Base):
    __tablename__ = "inspection_logs"
    id = Column(Integer, primary_key=True)
    device_id = Column(Integer, ForeignKey("devices.id"), nullable=False)
    user_id = Column(Integer, nullable=False)
    recorded_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    device = relationship("Device")
    __table_args__ = (Index("idx_logs_device", "device_id"), Index("idx_logs_recorded_at", "recorded_at"))


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


class Location(Base):
    __tablename__ = "locations"
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    parent_id = Column(Integer, ForeignKey("locations.id"))
    meta_data = Column("metadata", JSON)
    __table_args__ = (Index("idx_locations_parent", "parent_id"),)


class Vendor(Base):
    __tablename__ = "vendors"
    id = Column(Integer, primary_key=True)
    name = Column(String(150), unique=True, nullable=False)
    contact = Column(String(150))
    phone = Column(String(50))
    email = Column(String(150))


class DeviceWarranty(Base):
    __tablename__ = "device_warranty"
    id = Column(Integer, primary_key=True)
    device_id = Column(Integer, ForeignKey("devices.id"), nullable=False)
    vendor_id = Column(Integer, ForeignKey("vendors.id"))
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    terms = Column(Text)
    __table_args__ = (Index("idx_warranty_device", "device_id", unique=True),)


class DeviceStatusHistory(Base):
    __tablename__ = "device_status_history"
    id = Column(Integer, primary_key=True)
    device_id = Column(Integer, ForeignKey("devices.id"), nullable=False)
    status = Column(String(50), nullable=False)
    changed_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    reason = Column(String(255))
    __table_args__ = (Index("idx_status_device_changed", "device_id", "changed_at"),)


class WorkOrder(Base):
    __tablename__ = "work_orders"
    id = Column(Integer, primary_key=True)
    device_id = Column(Integer, ForeignKey("devices.id"), nullable=False)
    type = Column(String(50), nullable=False)
    priority = Column(String(20), nullable=False)
    status = Column(String(20), nullable=False)
    scheduled_at = Column(DateTime)
    completed_at = Column(DateTime)
    assignee = Column(String(100))
    notes = Column(Text)
    __table_args__ = (
        Index("idx_work_orders_device_status", "device_id", "status"),
        Index("idx_work_orders_scheduled", "scheduled_at"),
    )


class MaintenanceTask(Base):
    __tablename__ = "maintenance_tasks"
    id = Column(Integer, primary_key=True)
    device_id = Column(Integer, ForeignKey("devices.id"), nullable=False)
    work_order_id = Column(Integer, ForeignKey("work_orders.id"))
    task_type = Column(String(50), nullable=False)
    due_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime)
    result = Column(String(50))
    remarks = Column(Text)
    __table_args__ = (Index("idx_tasks_device_due", "device_id", "due_at"),)


class Telemetry(Base):
    __tablename__ = "telemetry"
    id = Column(BigInteger, primary_key=True)
    device_id = Column(Integer, ForeignKey("devices.id"), nullable=False)
    metric = Column(String(50), nullable=False)
    value = Column(Float, nullable=False)
    unit = Column(String(20))
    recorded_at = Column(DateTime, nullable=False)
    ingested_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    __table_args__ = (Index("idx_telemetry_device_metric", "device_id", "metric", "recorded_at"),)


class Anomaly(Base):
    __tablename__ = "anomalies"
    id = Column(BigInteger, primary_key=True)
    device_id = Column(Integer, ForeignKey("devices.id"), nullable=False)
    source = Column(String(50), nullable=False)
    metric = Column(String(50))
    score = Column(Float, nullable=False)
    threshold = Column(Float)
    detected_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    context = Column(JSON)
    __table_args__ = (Index("idx_anomalies_device_detected", "device_id", "detected_at"),)


class HealthScore(Base):
    __tablename__ = "health_scores"
    id = Column(BigInteger, primary_key=True)
    device_id = Column(Integer, ForeignKey("devices.id"), nullable=False)
    score = Column(Float, nullable=False)
    model_version = Column(String(50))
    computed_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    meta_data = Column("metadata", JSON)
    __table_args__ = (Index("idx_health_device_computed", "device_id", "computed_at"),)


class MaintenanceRecommendation(Base):
    __tablename__ = "maintenance_recommendations"
    id = Column(BigInteger, primary_key=True)
    device_id = Column(Integer, ForeignKey("devices.id"), nullable=False)
    recommendation = Column(Text, nullable=False)
    priority = Column(String(20), nullable=False)
    source = Column(String(50), nullable=False)
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime)
    __table_args__ = (
        Index("idx_reco_device_priority", "device_id", "priority"),
        Index("idx_reco_created", "created_at"),
    )


class AuditEvent(Base):
    __tablename__ = "audit_events"
    id = Column(BigInteger, primary_key=True)
    actor = Column(String(100), nullable=False)
    action = Column(String(100), nullable=False)
    entity = Column(String(100), nullable=False)
    entity_id = Column(String(100))
    payload = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    __table_args__ = (Index("idx_audit_entity", "entity", "entity_id"), Index("idx_audit_created", "created_at"),)
