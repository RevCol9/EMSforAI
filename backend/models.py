from datetime import datetime
from zoneinfo import ZoneInfo
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
    CheckConstraint,
)
from sqlalchemy.orm import relationship

from .db import Base

CST = ZoneInfo("Asia/Shanghai")


def now_cst():
    return datetime.now(CST)


class DeviceModel(Base):
    __tablename__ = "device_models"
    id = Column(Integer, primary_key=True, comment="主键")
    name = Column(String(100), unique=True, nullable=False, comment="设备型号名称")


class Device(Base):
    __tablename__ = "devices"
    id = Column(Integer, primary_key=True, comment="主键")
    model_id = Column(Integer, ForeignKey("device_models.id"), nullable=False, comment="关联设备型号")
    name = Column(String(100), unique=True, nullable=False, comment="设备名称/资产编号")
    serial_number = Column(String(100), unique=True, comment="序列号")
    location = Column(String(100), comment="位置/产线/区域")
    status = Column(String(50), default="active", comment="状态：在用/停机/维修")
    model = relationship("DeviceModel")
    __table_args__ = (Index("idx_devices_model", "model_id"),)


class DeviceMetricDefinition(Base):
    __tablename__ = "device_metric_definitions"
    id = Column(Integer, primary_key=True, comment="主键")
    model_id = Column(Integer, ForeignKey("device_models.id"), nullable=False, comment="关联设备型号")
    metric_key = Column(String(50), nullable=False, comment="指标键（型号内唯一）")
    metric_name = Column(String(50), nullable=False, comment="指标显示名称")
    unit = Column(String(20), comment="原始单位")
    base_unit = Column(String(20), comment="内部基准单位")
    display_unit = Column(String(20), comment="展示/输入单位")
    unit_scale = Column(Float, comment="单位换算乘数")
    unit_offset = Column(Float, comment="单位换算偏移")
    decimal_precision = Column(Integer, comment="展示/存储小数位")
    data_type = Column(String(20), comment="数据类型：float/int/bool")
    warn_threshold = Column(Float, comment="预警阈值")
    crit_threshold = Column(Float, comment="严重阈值")
    valid_min = Column(Float, comment="最小有效值")
    valid_max = Column(Float, comment="最大有效值")
    trend_direction = Column(Integer, default=1, comment="趋势方向 1/-1")
    weight_in_health = Column(Float, default=1.0, comment="健康度权重")
    is_ai_analyzed = Column(Boolean, default=True, comment="是否参与AI分析")
    sampling_frequency = Column(Float, comment="采样频率 Hz")
    collection_source = Column(String(100), comment="采集来源：IoT/SCADA/人工等")
    collector_id = Column(String(100), comment="采集设备/传感器编号")
    standard_code = Column(String(100), comment="标准编号 ISO/GB 等")
    alarm_delay_seconds = Column(Integer, comment="报警延迟/抑制秒数")
    criticality = Column(String(50), comment="重要性等级")
    data_origin = Column(String(50), comment="数据来源标签：原始/清洗/衍生")
    validation_required = Column(Boolean, default=False, comment="是否需要校验")
    is_validated = Column(Boolean, comment="校验结果")
    validation_notes = Column(String(255), comment="校验说明")
    feature_snapshot = Column(JSON, comment="模型特征/配置快照")
    prediction_confidence = Column(Float, comment="预测置信度 0-1")
    min_sampling_interval = Column(Float, comment="最小采样间隔（秒）")
    max_sampling_interval = Column(Float, comment="最大采样间隔（秒）")
    model = relationship("DeviceModel")
    __table_args__ = (
        Index("idx_metric_def_model_key", "model_id", "metric_key", unique=True),
        CheckConstraint("sampling_frequency IS NULL OR sampling_frequency > 0", name="ck_metric_def_sampling_positive"),
        CheckConstraint("unit_scale IS NULL OR unit_scale > 0", name="ck_metric_def_unit_scale_positive"),
        CheckConstraint("alarm_delay_seconds IS NULL OR alarm_delay_seconds > 0", name="ck_metric_def_alarm_delay_positive"),
        CheckConstraint("min_sampling_interval IS NULL OR min_sampling_interval > 0", name="ck_metric_def_min_sampling_positive"),
        CheckConstraint("max_sampling_interval IS NULL OR max_sampling_interval > 0", name="ck_metric_def_max_sampling_positive"),
        CheckConstraint(
            "max_sampling_interval IS NULL OR min_sampling_interval IS NULL OR max_sampling_interval >= min_sampling_interval",
            name="ck_metric_def_sampling_bounds",
        ),
        CheckConstraint(
            "prediction_confidence IS NULL OR (prediction_confidence >= 0 AND prediction_confidence <= 1)",
            name="ck_metric_def_confidence_range",
        ),
    )


class InspectionLog(Base):
    __tablename__ = "inspection_logs"
    id = Column(Integer, primary_key=True, comment="主键")
    device_id = Column(Integer, ForeignKey("devices.id"), nullable=False, comment="关联设备")
    user_id = Column(Integer, nullable=False, comment="巡检用户ID")
    recorded_at = Column(DateTime(timezone=True), nullable=False, comment="巡检发生时间")
    created_at = Column(DateTime(timezone=True), default=now_cst, comment="记录创建时间")
    updated_at = Column(DateTime(timezone=True), default=now_cst, onupdate=now_cst, comment="记录更新时间")
    data_origin = Column(String(50), comment="数据来源标签")
    collector_id = Column(String(100), comment="采集设备/传感器编号")
    data_quality_score = Column(Float, comment="数据质量评分")
    is_validated = Column(Boolean, comment="校验结果")
    validation_notes = Column(String(255), comment="校验说明")
    created_by = Column(String(100), comment="创建人")
    updated_by = Column(String(100), comment="更新人")
    deleted_at = Column(DateTime(timezone=True), comment="软删除时间")
    device = relationship("Device")
    __table_args__ = (
        Index("idx_logs_device", "device_id"),
        Index("idx_logs_recorded_at", "recorded_at"),
    )


class InspectionMetricValue(Base):
    __tablename__ = "inspection_metric_values"
    log_id = Column(Integer, ForeignKey("inspection_logs.id"), primary_key=True, comment="关联巡检记录")
    metrics_data = Column(JSON, comment="本次巡检的指标数据JSON")
    log = relationship("InspectionLog")


class MetricAIAnalysis(Base):
    __tablename__ = "metric_ai_analysis"
    device_id = Column(Integer, ForeignKey("devices.id"), nullable=False, comment="关联设备")
    metric_key = Column(String(50), nullable=False, comment="指标键")
    calc_time = Column(DateTime(timezone=True), default=now_cst, nullable=False, comment="分析时间")
    model_version = Column(String(20), nullable=False, comment="模型版本")
    rul_days = Column(Integer, comment="剩余寿命天数")
    trend_r2 = Column(Float, comment="趋势拟合R2")
    last_value = Column(Float, comment="最新指标值")
    curve_points = Column(JSON, comment="平滑曲线点")
    extra_info = Column(JSON, comment="附加信息/调试")
    health_score = Column(Float, comment="指标健康度")
    prediction_confidence = Column(Float, comment="预测置信度 0-1")
    feature_snapshot = Column(JSON, comment="输入特征快照")
    alert_level = Column(String(50), comment="告警等级")
    alert_status = Column(String(50), comment="告警状态")
    acknowledged_at = Column(DateTime(timezone=True), comment="告警确认时间")
    ack_by = Column(String(100), comment="告警确认人")
    downtime_risk = Column(Float, comment="停机风险评分")
    throughput_impact = Column(Float, comment="产能影响评估")
    device = relationship("Device")
    __table_args__ = (
        PrimaryKeyConstraint("device_id", "metric_key", "calc_time", name="pk_metric_ai"),
        Index("idx_metric_ai_latest", "device_id", "metric_key"),
        CheckConstraint(
            "prediction_confidence IS NULL OR (prediction_confidence >= 0 AND prediction_confidence <= 1)",
            name="ck_metric_ai_confidence_range",
        ),
    )


class EquipmentAndAssetManagement(Base):
    __tablename__ = "equipment_and_asset_management"

    id = Column(Integer, primary_key=True, comment="主键")
    device_id = Column(Integer, ForeignKey("devices.id"), nullable=False, comment="关联设备")
    serial_number = Column(String(100), unique=True, nullable=False, comment="序列号/资产编码")
    asset_code = Column(String(100), comment="备用资产编码")
    location = Column(String(100), comment="位置/区域")
    status = Column(String(100), default="active", nullable=False, comment="状态")
    vendor = Column(String(100), comment="供应商")
    brand = Column(String(100), comment="品牌")
    model_revision = Column(String(100), comment="型号版本")
    installed_at = Column(DateTime, comment="安装日期")
    commissioned_at = Column(DateTime, comment="投运日期")
    warranty_end = Column(DateTime, comment="保修到期日")
    maintenance_contract_id = Column(String(100), comment="维保合同ID")
    management_team = Column(String(100), comment="责任班组")
    maintainer_id = Column(Integer, comment="维护人ID")
    created_at = Column(DateTime(timezone=True), default=now_cst, nullable=False, comment="创建时间")
    updated_at = Column(DateTime(timezone=True), default=now_cst, onupdate=now_cst, nullable=False, comment="更新时间")

    device = relationship("Device")

    __table_args__ = (
        Index("idx_eam_device", "device_id"),
        Index("idx_eam_location", "location"),
    )


class SpareUsage(Base):
    __tablename__ = "spare_usage_cycles"

    id = Column(Integer, primary_key=True, comment="主键")
    device_id = Column(Integer, ForeignKey("devices.id"), nullable=False, comment="关联设备")
    part_code = Column(String(100), nullable=False, comment="备件编码")
    usage_cycles = Column(Integer, default=0, comment="使用循环次数")
    updated_at = Column(DateTime(timezone=True), default=now_cst, onupdate=now_cst, comment="更新时间")
    created_at = Column(DateTime(timezone=True), default=now_cst, comment="创建时间")

    device = relationship("Device")

    __table_args__ = (
        Index("idx_spare_device", "device_id"),
        CheckConstraint("usage_cycles >= 0", name="ck_spare_usage_nonnegative"),
    )


class EnergyConsumption(Base):
    __tablename__ = "energy_consumption"

    id = Column(Integer, primary_key=True, comment="主键")
    device_id = Column(Integer, ForeignKey("devices.id"), nullable=False, comment="关联设备")
    recorded_at = Column(DateTime(timezone=True), nullable=False, comment="计量时间")
    energy_kwh = Column(Float, comment="电耗 kWh")
    gas_nm3 = Column(Float, comment="气耗 Nm3")
    water_ton = Column(Float, comment="水耗 吨")
    source = Column(String(50), comment="数据来源/系统")
    created_at = Column(DateTime(timezone=True), default=now_cst, comment="创建时间")

    device = relationship("Device")

    __table_args__ = (
        Index("idx_energy_device", "device_id"),
        Index("idx_energy_recorded_at", "recorded_at"),
        CheckConstraint("energy_kwh IS NULL OR energy_kwh >= 0", name="ck_energy_kwh_nonnegative"),
        CheckConstraint("gas_nm3 IS NULL OR gas_nm3 >= 0", name="ck_energy_gas_nonnegative"),
        CheckConstraint("water_ton IS NULL OR water_ton >= 0", name="ck_energy_water_nonnegative"),
    )


class OEEStat(Base):
    __tablename__ = "oee_stats"

    id = Column(Integer, primary_key=True, comment="主键")
    device_id = Column(Integer, ForeignKey("devices.id"), nullable=False, comment="关联设备")
    period_start = Column(DateTime(timezone=True), nullable=False, comment="统计开始时间")
    period_end = Column(DateTime(timezone=True), nullable=False, comment="统计结束时间")
    availability = Column(Float, comment="稼动率 0-1")
    performance = Column(Float, comment="性能系数 0-1")
    quality_rate = Column(Float, comment="良品率 0-1")
    created_at = Column(DateTime(timezone=True), default=now_cst, comment="创建时间")

    device = relationship("Device")

    __table_args__ = (
        Index("idx_oee_device", "device_id"),
        Index("idx_oee_period_start", "period_start"),
        CheckConstraint(
            "availability IS NULL OR (availability >= 0 AND availability <= 1)",
            name="ck_oee_availability_range",
        ),
        CheckConstraint(
            "performance IS NULL OR (performance >= 0 AND performance <= 1)",
            name="ck_oee_performance_range",
        ),
        CheckConstraint(
            "quality_rate IS NULL OR (quality_rate >= 0 AND quality_rate <= 1)",
            name="ck_oee_quality_range",
        ),
    )


class MaintenanceCost(Base):
    __tablename__ = "maintenance_costs"

    id = Column(Integer, primary_key=True, comment="主键")
    device_id = Column(Integer, ForeignKey("devices.id"), nullable=False, comment="关联设备")
    period_start = Column(DateTime(timezone=True), comment="费用期间起")
    period_end = Column(DateTime(timezone=True), comment="费用期间止")
    cost = Column(Float, comment="维护费用")
    budget_center = Column(String(100), comment="预算/成本中心")
    note = Column(String(255), comment="备注")
    created_at = Column(DateTime(timezone=True), default=now_cst, comment="创建时间")

    device = relationship("Device")

    __table_args__ = (
        Index("idx_cost_device", "device_id"),
        CheckConstraint("cost IS NULL OR cost >= 0", name="ck_cost_nonnegative"),
    )
