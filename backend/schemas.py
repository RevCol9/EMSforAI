"""
EMSforAI API数据验证模型

本模块定义了所有API接口的请求和响应数据模型，使用Pydantic进行数据验证。
这些模型确保API接口接收和返回的数据格式正确，并提供自动的文档生成。

主要模型分类：
- 设备管理模型：设备创建、查询等
- 巡检数据模型：巡检数据提交、响应等
- 算法分析模型：健康分析结果、指标分析结果等

Author: EMSforAI Team
License: MIT
"""
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ========== 设备型号相关模型 ==========

class DeviceModelBase(BaseModel):
    """设备型号基础模型"""
    name: str = Field(..., description="设备型号名称")


class DeviceModelCreate(DeviceModelBase):
    """创建设备型号请求模型"""
    pass


class DeviceModelRead(DeviceModelBase):
    """设备型号查询响应模型"""
    id: int

    class Config:
        orm_mode = True


# ========== 设备资产相关模型 ==========

class DeviceBase(BaseModel):
    """设备基础模型"""
    name: str = Field(..., description="设备名称/资产编号")
    model_id: int = Field(..., description="关联设备型号ID")
    serial_number: Optional[str] = Field(None, description="序列号")
    location: Optional[str] = Field(None, description="位置/产线/区域")
    status: Optional[str] = Field("active", description="设备状态：active(在用)/maintenance(维修中)/down(停机)")


class DeviceCreate(DeviceBase):
    """创建设备请求模型"""
    pass


class DeviceRead(DeviceBase):
    """设备查询响应模型"""
    id: int
    model: Optional[DeviceModelRead]

    class Config:
        orm_mode = True


# ========== 巡检数据相关模型 ==========

class InspectionSubmit(BaseModel):
    """巡检数据提交请求模型
    
    用于接收来自前端或数据采集系统的巡检数据。
    """
    device_id: str = Field(..., description="设备ID（asset_id），例如：'CNC-MAZAK-01'")
    user_id: int
    recorded_at: Optional[datetime] = None
    metrics: Dict[str, float]
    data_origin: Optional[str] = Field(None, description="数据来源标签")
    collector_id: Optional[str] = Field(None, description="采集设备/传感器编号")
    data_quality_score: Optional[float] = Field(None, description="数据质量评分")
    validation_notes: Optional[str] = Field(None, description="校验说明")


class InspectionSubmitResponse(BaseModel):
    """巡检数据提交响应模型"""
    status: str = Field(..., description="提交状态：success(成功)/error(失败)")
    anomalies: Dict[str, bool] = Field(..., description="异常检测结果，格式为{metric_key: is_anomaly}")


# ========== 算法分析相关模型 ==========

class CurvePoint(BaseModel):
    """时间序列曲线点模型（用于图表展示）"""
    date: str = Field(..., description="日期字符串（ISO格式）")
    value: float = Field(..., description="数值")


class MetricAIResponse(BaseModel):
    """单指标AI分析响应模型（旧版API兼容）"""
    device_id: str
    metric_key: str
    last_value: float
    status: str
    rul_days: Optional[int]
    curve: List[CurvePoint]
    prediction_confidence: Optional[float] = Field(None, description="预测置信度0-1")
    health_score: Optional[float] = Field(None, description="健康度")


class MetricOverviewItem(BaseModel):
    """指标概览项模型（用于设备概览接口）"""
    metric_key: str = Field(..., description="测点标识符")
    last_value: float = Field(..., description="最新值")
    status: str = Field(..., description="状态：normal/warning/critical")
    rul_days: Optional[int] = Field(None, description="剩余使用寿命（天数）")


class DeviceOverview(BaseModel):
    """设备健康度概览模型"""
    device_id: str = Field(..., description="设备ID（asset_id）")
    health_score: float = Field(..., description="设备健康度分数（0-100）")
    metrics: List[MetricOverviewItem] = Field(default_factory=list, description="各指标概览列表")


# ========== 算法引擎 API 响应模型 ==========

class MetricAnalysisResult(BaseModel):
    """单指标详细分析结果模型
    
    包含单个测点的完整分析信息，包括当前值、健康度、RUL预测、趋势分析等。
    """
    metric_key: str
    device_id: Optional[str]
    current_value: float
    health_score: float
    alert_level: str = Field(..., description="告警级别: normal, warning, critical")
    data_points: int
    rul_days: Optional[int] = None
    rul_status: Optional[str] = None
    trend_alpha: Optional[float] = None
    trend_beta: Optional[float] = None
    trend_r2: Optional[float] = None
    prediction_confidence: Optional[float] = Field(None, description="预测置信度 0-1")
    weight_in_health: Optional[float] = Field(None, description="在设备健康度计算中的权重")
    criticality: Optional[str] = Field(None, description="指标关键程度: high, medium, low")


class SparePartInfo(BaseModel):
    """备件信息模型
    
    用于描述设备备件的使用情况和寿命状态。
    """
    spare_part_id: Optional[str] = None
    spare_part_name: Optional[str] = None
    part_code: Optional[str] = None
    usage_cycles: int
    max_cycles: int
    usage_ratio: float
    remaining_ratio: float
    status: str = Field(..., description="状态: normal, warning, critical")


class SpareLifeInfo(BaseModel):
    """备件寿命信息模型
    
    汇总设备所有备件的寿命信息，包括总数、关键数量等。
    """
    status: str
    spare_parts: List[SparePartInfo]
    total_parts: int
    critical_count: int


class DeviceAnalysisResponse(BaseModel):
    """设备完整分析结果模型
    
    包含设备健康分析的所有信息，包括健康度、告警级别、各指标分析、
    停机风险、产能影响、备件寿命等。
    """
    device_id: str
    device_health_score: float
    device_alert_level: str
    metrics: List[MetricAnalysisResult]
    downtime_risk: float = Field(..., description="停机风险 0-1")
    throughput_impact: float = Field(..., description="产能影响 0-1")
    spare_life_info: SpareLifeInfo


# ========== 雷达图相关模型 ==========

class RadarDimensionScore(BaseModel):
    """雷达图维度评分模型
    
    用于前端绘制雷达图，展示设备各维度的健康度。
    """
    dimension_name: str = Field(..., description="维度名称，如'温度'、'振动'等")
    metric_id: str = Field(..., description="测点ID")
    health_score: float = Field(..., description="健康度分数 0-100")
    current_value: float = Field(..., description="当前值")
    warn_threshold: float = Field(..., description="警告阈值")
    crit_threshold: float = Field(..., description="临界阈值")
    trend: str = Field(..., description="趋势：上升/下降/稳定")
    alert_level: str = Field(..., description="告警级别：normal/warning/critical")


class RadarChartResponse(BaseModel):
    """雷达图数据响应模型
    
    包含设备多维度健康度评分，用于前端绘制雷达图。
    """
    device_id: str
    device_health_score: float = Field(..., description="设备整体健康度 0-100")
    dimensions: List[RadarDimensionScore] = Field(..., description="各维度评分列表")


# ========== 测点列表相关模型 ==========

class MetricListItem(BaseModel):
    """测点列表项模型"""
    metric_id: str = Field(..., description="测点ID")
    metric_name: Optional[str] = Field(None, description="测点名称")
    metric_type: Optional[str] = Field(None, description="测点类型：PROCESS/WAVEFORM等")
    unit: Optional[str] = Field(None, description="单位")
    warn_threshold: Optional[float] = Field(None, description="警告阈值")
    crit_threshold: Optional[float] = Field(None, description="临界阈值")
    has_lstm_model: bool = Field(False, description="是否有LSTM模型")
