"""
EMSforAI 数据模型定义

本模块定义了系统的核心数据模型，采用4域8表架构设计：
- 域一：基础元数据域（设备资产、测点定义）
- 域二：动态感知域（过程数据、波形数据）
- 域三：知识与运维域（运维记录、知识库）
- 域四：分析结果域（健康分析、AI报告）

所有模型均基于SQLAlchemy ORM，支持关系型数据库存储。
波形数据建议在生产环境中使用TSDB（如InfluxDB/TDengine）存储。

Author: EMSforAI Team
License: MIT
"""
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
    Text,
    Enum,
    JSON,
    Index,
    PrimaryKeyConstraint,
    CheckConstraint,
    LargeBinary,
)
from sqlalchemy.orm import relationship
import enum

from .db import Base

# 时区配置：使用中国标准时间
CST = ZoneInfo("Asia/Shanghai")


def now_cst() -> datetime:
    """
    获取当前中国标准时间
    
    Returns:
        datetime: 带时区的当前时间
    """
    return datetime.now(CST)


# ============================================================================
# 域一：基础元数据域 (Metadata Layer)
# ============================================================================
# 本域包含系统的静态配置信息，定义了设备、测点及其阈值等基础元数据。
# 这些数据通常不会频繁变更，是系统运行的基础骨架。

class Asset(Base):
    """
    设备资产表
    
    存储设备的基本信息，包括设备标识、名称、型号、位置等。
    每个设备对应一个唯一的asset_id，作为系统的主键标识。
    
    Attributes:
        asset_id: 设备唯一标识（主键），建议使用有意义的编码，如"CNC-MAZAK-01"
        name: 设备名称，用于显示和识别
        model_id: 设备型号ID，关联到设备型号表（当前版本简化，直接存储型号字符串）
        location: 设备物理位置，如"一车间-C区"
        commission_date: 设备投产日期，用于计算设备服役时长
        status: 设备状态，可选值：active（在用）、maintenance（维修中）、down（停机）
        created_at: 记录创建时间
        updated_at: 记录最后更新时间
    """
    __tablename__ = "assets"
    
    asset_id = Column(String(100), primary_key=True, comment="设备唯一标识，建议使用有意义的编码")
    name = Column(String(200), nullable=False, comment="设备名称，用于显示和识别")
    model_id = Column(String(100), nullable=False, comment="设备型号ID，用于关联设备型号信息")
    location = Column(String(200), comment="设备物理位置，如车间、产线、区域等")
    commission_date = Column(DateTime, comment="设备投产日期，用于计算设备服役时长")
    status = Column(String(50), default="active", comment="设备状态：active(在用)/maintenance(维修中)/down(停机)")
    created_at = Column(DateTime(timezone=True), default=now_cst, comment="记录创建时间")
    updated_at = Column(DateTime(timezone=True), default=now_cst, onupdate=now_cst, comment="记录最后更新时间")
    
    __table_args__ = (
        Index("idx_assets_model", "model_id"),
        Index("idx_assets_location", "location"),
    )


class MetricType(enum.Enum):
    """
    测点类型枚举
    
    用于区分不同类型的测点数据，不同类型的测点需要采用不同的分析方法：
    - PROCESS: 过程量，如温度、压力、转速、负载等标量数据
    - VIBRATION: 振动量，如振动加速度、速度等，通常需要FFT分析
    """
    PROCESS = "PROCESS"      # 过程量：温度、压力、转速、负载等标量数据
    VIBRATION = "VIBRATION"  # 振动量：振动加速度、速度等，通常需要频域分析


class MetricDefinition(Base):
    """
    测点与阈值定义表
    
    定义每个设备上的测点及其阈值配置。测点是设备健康监测的基本单元，
    每个测点都有对应的预警阈值和临界阈值，用于健康度计算和告警判断。
    
    Attributes:
        metric_id: 测点唯一标识（主键），建议格式：设备ID_测点名称，如"CNC01_SPINDLE_TEMP"
        asset_id: 所属设备ID，外键关联assets表
        metric_name: 测点显示名称，用于前端展示
        metric_type: 测点类型，PROCESS或VIBRATION
        unit: 测点单位，如"℃"、"mm/s"、"MPa"等
        warn_threshold: 预警阈值，超过此值触发警告
        crit_threshold: 临界阈值，超过此值触发严重告警
        is_condition_dependent: 是否依赖工况，True表示阈值在不同工况下可能不同
        sampling_frequency: 采样频率（Hz），用于数据采集配置
        collector_id: 采集设备/传感器编号，用于追溯数据来源
    """
    __tablename__ = "metric_definitions"
    
    metric_id = Column(String(100), primary_key=True, comment="测点唯一标识，建议格式：设备ID_测点名称")
    asset_id = Column(String(100), ForeignKey("assets.asset_id", ondelete="CASCADE"), nullable=False, comment="所属设备ID")
    metric_name = Column(String(200), nullable=False, comment="测点显示名称，用于前端展示")
    metric_type = Column(Enum(MetricType), nullable=False, comment="测点类型：PROCESS(过程量)或VIBRATION(振动量)")
    unit = Column(String(50), comment="测点单位，如℃、mm/s、MPa等")
    warn_threshold = Column(Float, comment="预警阈值，超过此值触发警告告警")
    crit_threshold = Column(Float, comment="临界阈值，超过此值触发严重告警，设备需要立即维护")
    is_condition_dependent = Column(Boolean, default=False, comment="是否依赖工况，True表示阈值在不同工况下可能不同")
    sampling_frequency = Column(Float, comment="采样频率（Hz），用于数据采集配置")
    collector_id = Column(String(100), comment="采集设备/传感器编号，用于追溯数据来源")
    created_at = Column(DateTime(timezone=True), default=now_cst, comment="记录创建时间")
    
    asset = relationship("Asset", backref="metrics")
    
    __table_args__ = (
        Index("idx_metric_asset", "asset_id"),
        Index("idx_metric_type", "metric_type"),
    )


# ============================================================================
# 域二：动态感知域 (Sensing Layer)
# ============================================================================
# 本域包含设备的实时感知数据，是AI算法的输入数据源。
# 过程数据是高频标量数据，波形数据是低频但高维的数组数据。

class MachineState(enum.IntEnum):
    """
    工况状态枚举
    
    用于标识设备在数据采集时的运行状态，不同状态下的数据具有不同的意义：
    - STOPPED(0): 停机状态，设备未运行
    - STANDBY(1): 待机状态，设备运行但未加工
    - RUNNING(2): 加工状态，设备正在执行加工任务
    
    注意：算法分析时通常只使用RUNNING状态的数据，以确保分析结果的一致性。
    """
    STOPPED = 0   # 停机状态，设备未运行
    STANDBY = 1   # 待机状态，设备运行但未加工
    RUNNING = 2   # 加工状态，设备正在执行加工任务


class TelemetryProcess(Base):
    """
    过程数据表 - 高频标量数据
    
    存储设备的实时过程量数据，如温度、压力、转速、负载等。
    这些数据通常是高频采集（1Hz或更高），建议在生产环境中使用TSDB存储。
    
    注意：本表使用关系型数据库存储，适合小规模数据。生产环境建议使用：
    - InfluxDB
    - TDengine
    - TimescaleDB
    
    Attributes:
        id: 自增主键
        timestamp: 数据采集时间（带时区）
        metric_id: 测点ID，外键关联metric_definitions表
        value: 测点的物理数值
        quality: 数据质量标记，0=传感器故障/数据异常，1=正常数据
        machine_state: 采集时的工况状态，0=停机，1=待机，2=加工
    """
    __tablename__ = "telemetry_process"
    
    id = Column(Integer, primary_key=True, autoincrement=True, comment="自增主键")
    timestamp = Column(DateTime(timezone=True), nullable=False, comment="数据采集时间（带时区）")
    metric_id = Column(String(100), ForeignKey("metric_definitions.metric_id", ondelete="CASCADE"), nullable=False, comment="测点ID")
    value = Column(Float, nullable=False, comment="测点的物理数值")
    quality = Column(Integer, default=1, comment="数据质量标记：0=传感器故障/数据异常，1=正常数据")
    machine_state = Column(Integer, default=2, comment="采集时的工况状态：0=停机，1=待机，2=加工")
    
    metric = relationship("MetricDefinition", backref="process_data")
    
    __table_args__ = (
        Index("idx_telemetry_time", "timestamp"),
        Index("idx_telemetry_metric", "metric_id"),
        Index("idx_telemetry_state", "machine_state"),
        Index("idx_telemetry_metric_time", "metric_id", "timestamp"),
    )


class TelemetryWaveform(Base):
    """
    波形数据表 - 高频波形数组
    
    存储设备的振动波形数据，用于深度故障诊断。波形数据通常是低频采集（10-30分钟/次），
    但每次采集的数据量很大（数千到数万个采样点）。
    
    注意：
    - data_blob字段存储二进制波形数据，实际生产环境建议使用对象存储（S3/MinIO）
    - 或使用TSDB的波形数据存储功能
    - 本表主要用于存储波形元数据和引用路径
    
    Attributes:
        snapshot_id: 波形快照唯一标识（主键）
        asset_id: 设备ID，外键关联assets表
        timestamp: 波形采集时间
        sampling_rate: 采样率（Hz），如12800Hz，用于FFT分析
        duration_ms: 采集时长（毫秒），如1000ms
        axis: 采集轴向，X/Y/Z，用于多轴分析
        data_blob: 原始波形数组的二进制数据（LargeBinary）
        ref_rpm: 参考转速（RPM），采集波形瞬间的主轴转速，用于计算倍频
        metric_id: 关联的测点ID（可选），通常为振动测点
    """
    __tablename__ = "telemetry_waveform"
    
    snapshot_id = Column(String(100), primary_key=True, comment="波形快照唯一标识")
    asset_id = Column(String(100), ForeignKey("assets.asset_id", ondelete="CASCADE"), nullable=False, comment="设备ID")
    timestamp = Column(DateTime(timezone=True), nullable=False, comment="波形采集时间")
    sampling_rate = Column(Integer, nullable=False, comment="采样率（Hz），如12800Hz，用于FFT频域分析")
    duration_ms = Column(Integer, nullable=False, comment="采集时长（毫秒），如1000ms")
    axis = Column(String(10), nullable=False, comment="采集轴向：X/Y/Z，用于多轴振动分析")
    data_blob = Column(LargeBinary, comment="原始波形数组的二进制数据，建议使用对象存储或TSDB")
    ref_rpm = Column(Float, comment="参考转速（RPM），采集波形瞬间的主轴转速，用于计算1X、2X倍频")
    metric_id = Column(String(100), ForeignKey("metric_definitions.metric_id", ondelete="SET NULL"), comment="关联的测点ID（通常为振动测点）")
    
    asset = relationship("Asset", backref="waveforms")
    metric = relationship("MetricDefinition", backref="waveform_data")

    __table_args__ = (
        Index("idx_waveform_asset_time", "asset_id", "timestamp"),
        Index("idx_waveform_metric", "metric_id"),
    )


# ============================================================================
# 域三：知识与运维域 (Knowledge Layer)
# ============================================================================
# 本域包含运维知识和历史记录，用于AI训练和LLM语义分析。

class MaintenanceRecord(Base):
    """
    运维工单表
    
    存储设备的维护和故障记录，这些记录有两个主要用途：
    1. AI训练标签：failure_code作为分类标签，用于训练故障诊断模型
    2. LLM语料：issue_description和solution_description作为上下文，用于生成报告
    
    Attributes:
        record_id: 工单唯一标识（主键）
        asset_id: 设备ID，外键关联assets表
        start_time: 故障开始时间，用于标记"这段时间的数据是坏的"
        end_time: 故障解决时间，用于计算故障持续时间
        failure_code: 标准故障代码，如"ERR_BRG_01"（轴承故障），作为AI分类标签
        issue_description: 故障现象描述，用于LLM理解故障情况
        solution_description: 维修过程记录，用于LLM学习维修知识
        cost: 维修成本（元），用于成本分析
    """
    __tablename__ = "maintenance_records"

    record_id = Column(String(100), primary_key=True, comment="工单唯一标识")
    asset_id = Column(String(100), ForeignKey("assets.asset_id", ondelete="CASCADE"), nullable=False, comment="设备ID")
    start_time = Column(DateTime(timezone=True), nullable=False, comment="故障开始时间，用于标记异常数据区间")
    end_time = Column(DateTime(timezone=True), comment="故障解决时间，用于计算故障持续时间")
    failure_code = Column(String(100), comment="标准故障代码，如ERR_BRG_01，作为AI分类标签")
    issue_description = Column(Text, comment="故障现象描述，用于LLM理解故障情况")
    solution_description = Column(Text, comment="维修过程记录，用于LLM学习维修知识和生成报告")
    cost = Column(Float, comment="维修成本（元），用于成本分析和预算管理")
    created_at = Column(DateTime(timezone=True), default=now_cst, comment="记录创建时间")

    asset = relationship("Asset", backref="maintenance_records")

    __table_args__ = (
        Index("idx_maintenance_asset", "asset_id"),
        Index("idx_maintenance_time", "start_time"),
        Index("idx_maintenance_code", "failure_code"),
    )


class KnowledgeCategory(enum.Enum):
    """
    知识库类型枚举
    
    用于分类不同类型的知识文档，不同类型的知识在LLM RAG检索时具有不同的权重：
    - MANUAL: 设备手册，权威性最高
    - CASE: 维修案例，实用性最高
    - STANDARD: 行业标准，规范性最高
    """
    MANUAL = "手册"    # 设备手册，包含设备技术参数和维护标准
    CASE = "案例"      # 维修案例，包含实际故障处理经验
    STANDARD = "标准"  # 行业标准，包含ISO、GB等标准规范


class KnowledgeBase(Base):
    """
    知识库表 - LLM RAG语料库
    
    存储设备维护相关的知识文档，用于LLM的RAG（检索增强生成）功能。
    知识库中的内容会被切分成chunk（文本切片），每个chunk可以单独检索。
    
    注意：
    - content_chunk字段存储文本切片，建议每个chunk 200-500字
    - embedding字段可选，用于向量数据库加速检索
    - 如果使用向量数据库，embedding字段可以存储向量数据的JSON格式
    
    Attributes:
        doc_id: 文档唯一标识（主键）
        applicable_model: 适用设备型号，用于过滤知识范围
        category: 知识类型，手册/案例/标准
        title: 文档标题，用于显示和检索
        content_chunk: 文本切片内容，用于LLM RAG检索
        embedding: 向量数据（JSON格式），可选，用于向量数据库检索加速
    """
    __tablename__ = "knowledge_base"
    
    doc_id = Column(String(100), primary_key=True, comment="文档唯一标识")
    applicable_model = Column(String(100), comment="适用设备型号，用于过滤知识范围")
    category = Column(Enum(KnowledgeCategory), nullable=False, comment="知识类型：手册/案例/标准")
    title = Column(String(500), comment="文档标题，用于显示和检索")
    content_chunk = Column(Text, nullable=False, comment="文本切片内容，用于LLM RAG检索，建议200-500字/切片")
    embedding = Column(JSON, comment="向量数据（JSON格式），可选，用于向量数据库检索加速")
    created_at = Column(DateTime(timezone=True), default=now_cst, comment="记录创建时间")

    __table_args__ = (
        Index("idx_knowledge_model", "applicable_model"),
        Index("idx_knowledge_category", "category"),
    )


# ============================================================================
# 域四：分析结果域 (Analysis Output Layer)
# ============================================================================
# 本域包含AI算法的分析结果，是系统输出的核心数据。

class AIHealthAnalysis(Base):
    """
    健康度与寿命预测结果表
    
    存储AI算法对设备健康状态的分析结果，包括健康分、剩余寿命、趋势分析、故障诊断等。
    这些数据是系统输出的核心，用于前端展示和决策支持。
    
    Attributes:
        analysis_id: 分析记录唯一标识（主键）
        asset_id: 设备ID，外键关联assets表
        calc_time: 分析计算时间
        health_score: 设备健康分（0-100），100表示完全健康，0表示严重故障
        rul_days: 剩余使用寿命（天），基于趋势预测达到临界阈值的时间
        trend_slope: 趋势斜率，正值表示上升趋势（恶化），负值表示下降趋势（改善）
        diagnosis_result: 故障诊断结果（JSON格式），键为故障代码，值为概率，如{"bearing_wear": 0.85, "unbalance": 0.10}
        model_version: 使用的模型版本，用于模型版本管理
        prediction_confidence: 预测置信度（0-1），综合考虑数据量和趋势拟合度
    """
    __tablename__ = "ai_health_analysis"
    
    analysis_id = Column(String(100), primary_key=True, comment="分析记录唯一标识")
    asset_id = Column(String(100), ForeignKey("assets.asset_id", ondelete="CASCADE"), nullable=False, comment="设备ID")
    calc_time = Column(DateTime(timezone=True), nullable=False, comment="分析计算时间")
    health_score = Column(Float, comment="设备健康分（0-100），100表示完全健康，0表示严重故障")
    rul_days = Column(Float, comment="剩余使用寿命（天），基于趋势预测达到临界阈值的时间，None表示无法计算")
    trend_slope = Column(Float, comment="趋势斜率，正值表示上升趋势（恶化），负值表示下降趋势（改善）")
    diagnosis_result = Column(JSON, comment="故障诊断结果（JSON），键为故障代码，值为概率，如{'bearing_wear': 0.85}")
    model_version = Column(String(50), comment="使用的模型版本，用于模型版本管理和追溯")
    prediction_confidence = Column(Float, comment="预测置信度（0-1），综合考虑数据量和趋势拟合度")

    asset = relationship("Asset", backref="health_analyses")

    __table_args__ = (
        Index("idx_health_asset_time", "asset_id", "calc_time"),
        CheckConstraint(
            "health_score IS NULL OR (health_score >= 0 AND health_score <= 100)",
            name="ck_health_score_range"
        ),
        CheckConstraint(
            "prediction_confidence IS NULL OR (prediction_confidence >= 0 AND prediction_confidence <= 1)",
            name="ck_confidence_range"
        ),
    )


class AIReport(Base):
    """
    AI生成的报告表
    
    存储LLM生成的设备健康分析报告，报告基于ai_health_analysis的数值分析结果，
    结合knowledge_base的知识和maintenance_records的历史记录生成。
    
    Attributes:
        report_id: 报告唯一标识（主键）
        analysis_id: 关联的数值分析ID，外键关联ai_health_analysis表
        generated_content: LLM生成的报告内容（Markdown格式），包含自然语言描述和建议
        user_feedback: 用户反馈评分（1-5星），用于RLHF（人类反馈强化学习）优化模型
    """
    __tablename__ = "ai_reports"
    
    report_id = Column(String(100), primary_key=True, comment="报告唯一标识")
    analysis_id = Column(String(100), ForeignKey("ai_health_analysis.analysis_id", ondelete="CASCADE"), comment="关联的数值分析ID")
    generated_content = Column(Text, nullable=False, comment="LLM生成的报告内容（Markdown格式），包含自然语言描述和建议")
    user_feedback = Column(Integer, comment="用户反馈评分（1-5星），用于RLHF优化模型")
    created_at = Column(DateTime(timezone=True), default=now_cst, comment="报告创建时间")

    analysis = relationship("AIHealthAnalysis", backref="reports")

    __table_args__ = (
        Index("idx_report_analysis", "analysis_id"),
        CheckConstraint(
            "user_feedback IS NULL OR (user_feedback >= 1 AND user_feedback <= 5)",
            name="ck_feedback_range"
        ),
    )
