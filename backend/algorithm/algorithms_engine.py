"""
EMSforAI 算法引擎 V2 - LSTM 版

1. 多维度健康评分，直接服务前端雷达图
2. LSTM 驱动的剩余寿命（RUL）预测
3. 标准化的时间序列与统计输出，方便折线图/统计图复用

Author: EMSforAI Team
License: MIT
"""
import logging  
import sys
from pathlib import Path  
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import numpy as np  
import pandas as pd  
from dataclasses import dataclass, field

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

SCIPY_AVAILABLE = False
try:
    from scipy import signal
    from scipy.fft import fft, fftfreq
    from scipy.stats import linregress
    from scipy.optimize import curve_fit
    SCIPY_AVAILABLE = True
except ImportError:
    pass

SKLEARN_AVAILABLE = False
try:
    # 仅用于传统回归/评分，不影响核心 LSTM 功能
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    pass

# LSTM模型
LSTM_AVAILABLE = False
try:
    from backend.algorithm.lstm_model import LSTMRULPredictor, MultiVariateLSTMPredictor
    LSTM_AVAILABLE = True
except ImportError:
    pass

log = logging.getLogger(__name__)


@dataclass
class MetricDimensionScore:
    """单个维度的健康评分（用于雷达图）"""
    dimension_name: str  # 维度名称，如"温度"、"振动"等
    metric_id: str  # 测点ID
    health_score: float  # 0-100
    current_value: float  # 当前值
    warn_threshold: float  # 警告阈值
    crit_threshold: float  # 临界阈值
    trend: str  # 趋势：上升/下降/稳定
    alert_level: str  # 告警级别：normal/warning/critical


@dataclass
class TimeSeriesPoint:
    """时间序列数据点（用于曲线图）"""
    timestamp: str  # ISO格式时间字符串
    value: float  # 数值
    smoothed_value: Optional[float] = None  # 平滑后的值（可选）


@dataclass
class TrendAnalysis:
    """趋势分析结果"""
    slope: float  # 趋势斜率
    intercept: float  # 截距
    r2: float  # R²值
    p_value: Optional[float] = None  # p值（如果可用）
    trend_type: str = "lstm"  # 趋势类型
    model_name: str = "LSTM"  # 使用的模型名称


@dataclass
class HealthAnalysisResult:
    """健康分析结果（增强版）"""
    health_score: float  # 0-100，总体健康分
    rul_days: Optional[float]  # 剩余寿命天数
    trend_slope: float  # 平均趋势斜率
    diagnosis_result: Dict[str, float]  # 诊断概率
    prediction_confidence: float  # 预测置信度 0-1
    model_version: str = "2.0"
    
    # 新增：多维度评分（用于雷达图）
    dimension_scores: List[MetricDimensionScore] = field(default_factory=list)
    
    # 新增：时间序列数据（用于曲线图）
    time_series_data: Dict[str, List[TimeSeriesPoint]] = field(default_factory=dict)
    
    # 新增：统计信息（用于统计图）
    statistics: Dict[str, Any] = field(default_factory=dict)


class AlgorithmEngine:
    """
    算法引擎主类（V2 - LSTM 增强版）
    
    负责执行设备健康分析的所有算法，针对实际生产数据进行优化：
    - 多维度健康评分（支持雷达图）
    - 智能 RUL 预测（LSTM 神经网络 + 传统回归模型）
    - 时间序列分析（趋势、周期、异常提取）
    - 前端友好的数据结构（适配可视化需求）
    """
    
    def _get_or_load_multivariate_lstm_model(self, asset_id: str) -> Optional[MultiVariateLSTMPredictor]:
        """
        获取或加载多变量LSTM模型（一个设备一个模型）
        
        Args:
            asset_id: 设备ID
        
        Returns:
            MultiVariateLSTMPredictor 实例，如果模型不存在或加载失败则返回 None
        """
        if not LSTM_AVAILABLE:
            return None
        
        # 检查缓存
        if asset_id in self._multivariate_lstm_cache:
            return self._multivariate_lstm_cache[asset_id]
        
        # 模型不在缓存中，尝试加载
        models_dir = BASE_DIR / "models" / "lstm"
        model_path = models_dir / f"{asset_id}_multivariate_lstm.pt"
        
        if not model_path.exists():
            log.debug(f"多变量LSTM模型不存在: {model_path}")
            return None
        
        try:
            predictor = MultiVariateLSTMPredictor()
            predictor.load_model(str(model_path))
            # 加载成功后缓存
            self._multivariate_lstm_cache[asset_id] = predictor
            log.debug(f"多变量LSTM模型已加载并缓存: {asset_id}")
            return predictor
        except Exception as e:
            log.warning(f"加载多变量LSTM模型失败 {asset_id}: {e}")
            return None
    
    def _try_multivariate_lstm_prediction(
        self,
        data: Dict[str, pd.DataFrame],
        asset_id: str,
        metric_ids: List[str],
        crit_thresholds: Dict[str, float],
    ) -> Optional[Tuple[Optional[float], TrendAnalysis]]:
        """
        尝试使用多变量LSTM模型进行RUL预测
        
        Args:
            data: 数据字典
            asset_id: 设备ID
            metric_ids: 测点ID列表（模型训练时使用的测点）
            crit_thresholds: 每个测点的临界阈值字典
        
        Returns:
            (rul_days, trend_analysis) 或 None（如果多变量LSTM不可用）
        """
        if not LSTM_AVAILABLE:
            return None
        
        try:
            # 使用缓存机制获取或加载模型
            predictor = self._get_or_load_multivariate_lstm_model(asset_id)
            if predictor is None:
                return None
            
            # 准备每个测点的时间序列数据
            from backend.algorithm.data_service import prepare_process_series
            
            metric_data = {}
            for metric_id in metric_ids:
                if metric_id not in predictor.feature_names:
                    log.warning(f"测点 {metric_id} 不在模型的特征列表中，跳过")
                    continue
                
                df = prepare_process_series(
                    data,
                    metric_id,
                    asset_id=asset_id,
                    window_days=self.window_days,
                    machine_state=self.machine_state_filter,
                    quality_threshold=self.quality_threshold,
                )
                
                if df.empty or len(df) < predictor.sequence_length:
                    log.debug(f"测点 {metric_id} 数据不足，跳过")
                    return None
                
                df = df.sort_values("timestamp")
                metric_data[metric_id] = df["value"].values
            
            if not metric_data:
                log.debug("没有足够的测点数据用于多变量LSTM预测")
                return None
            
            # 标准化数据
            data_dict_scaled = predictor.transform_multivariate_values(metric_data)
            
            # 构建推理张量
            X_input = predictor.build_multivariate_inference_tensor(data_dict_scaled)
            
            # 预测RUL
            rul_predicted_scaled = predictor.predict(X_input)[0]
            
            # 反缩放RUL（从[0,1]恢复到天数）
            from backend.algorithm.training_utils import MAX_RUL_DAYS
            rul_predicted = rul_predicted_scaled * MAX_RUL_DAYS
            
            # 计算趋势（使用关键测点的数据）
            key_metric = metric_ids[0]  # 使用第一个测点计算趋势
            if key_metric in metric_data:
                values = metric_data[key_metric]
                if len(values) >= 10 and SCIPY_AVAILABLE:
                    t_days = np.arange(len(values))
                    slope, intercept, r_value, _, _ = linregress(t_days[-30:], values[-30:])
                    r2 = r_value ** 2
                else:
                    slope = 0.0
                    intercept = values[-1] if len(values) > 0 else 0.0
                    r2 = 0.0
            else:
                slope = 0.0
                intercept = 0.0
                r2 = 0.0
            
            # 创建趋势分析对象
            trend_analysis = TrendAnalysis(
                slope=float(slope),
                intercept=float(intercept),
                r2=float(r2),
                trend_type="multivariate_lstm",
                model_name="MultiVariateLSTM",
            )
            
            # 确保RUL非负
            rul_days = max(0.0, float(rul_predicted))
            
            return rul_days, trend_analysis
            
        except Exception as e:
            log.warning(f"多变量LSTM预测失败: {e}")
            import traceback
            log.debug(traceback.format_exc())
            return None
    
    def _try_lstm_prediction(
        self,
        df: pd.DataFrame,
        asset_id: str,
        metric_id: str,
        crit_threshold: float,
    ) -> Optional[Tuple[Optional[float], TrendAnalysis]]:
        """
        尝试使用LSTM模型进行RUL预测（使用模型缓存）
            
        Returns:
            (rul_days, trend_analysis) 或 None（如果LSTM不可用）
        """
        if not LSTM_AVAILABLE:
            return None
        
        try:
            # 使用缓存机制获取或加载模型
            predictor = self._get_or_load_lstm_model(asset_id, metric_id)
            if predictor is None:
                return None
            
            df = df.sort_values("timestamp")
            values = df["value"].values
            
            if len(values) < predictor.sequence_length:
                log.debug(
                    "数据点不足，模型需要至少%s个点，当前只有%s个",
                    predictor.sequence_length,
                    len(values),
                )
                return None
            
            values_scaled = predictor.transform_values(values.reshape(-1, 1))
            X_input = predictor.build_inference_tensor(values_scaled)
            
            rul_predicted = predictor.predict(X_input)[0]
            
            # 计算趋势（使用最近的数据点）
            # 注意：需要检查 SCIPY_AVAILABLE，因为 linregress 来自 scipy.stats
            if len(values) >= 10 and SCIPY_AVAILABLE:
                t_days = np.arange(len(values))
                slope, intercept, r_value, _, _ = linregress(t_days[-30:], values[-30:])
                r2 = r_value ** 2
            else:
                # 数据点不足或 scipy 不可用时，使用简单的默认趋势
                if len(values) < 10:
                    log.debug("数据点不足（<10），使用默认趋势")
                if not SCIPY_AVAILABLE:
                    log.debug("scipy 不可用，使用默认趋势")
                slope = 0.0
                intercept = values[-1] if len(values) > 0 else 0.0
                r2 = 0.0
            
            # 创建趋势分析对象
            trend_analysis = TrendAnalysis(
                slope=float(slope),
                intercept=float(intercept),
                r2=float(r2),
                trend_type="lstm",
                model_name="LSTM",
            )
            
            # 确保RUL非负
            rul_days = max(0.0, float(rul_predicted))
            
            return rul_days, trend_analysis
            
        except Exception as e:
            log.warning(f"LSTM预测失败: {e}")
            import traceback
            log.debug(traceback.format_exc())
            return None
    
    def __init__(
        self,
        window_days: int = 30,
        quality_threshold: int = 1,
        machine_state_filter: Optional[int] = 2,  # 默认只分析加工状态
        preload_lstm_models: bool = False,  # 是否预加载所有LSTM模型
    ):
        """
        初始化算法引擎
        
        Args:
            window_days: 分析窗口天数
            quality_threshold: 数据质量阈值（0或1）
            machine_state_filter: 工况状态过滤（None=不过滤，2=只分析加工状态）
            preload_lstm_models: 是否预加载所有LSTM模型（True=启动时加载，False=按需加载并缓存）
        """
        self.window_days = window_days
        self.quality_threshold = quality_threshold
        self.machine_state_filter = machine_state_filter
        
        # LSTM模型缓存：
        # - 单变量模型：key为 (asset_id, metric_id)，value为 LSTMRULPredictor 实例
        # - 多变量模型：key为 asset_id，value为 MultiVariateLSTMPredictor 实例
        self._lstm_model_cache: Dict[Tuple[str, str], LSTMRULPredictor] = {}
        self._multivariate_lstm_cache: Dict[str, MultiVariateLSTMPredictor] = {}
        
        # 如果启用预加载，扫描并加载所有可用的LSTM模型
        if preload_lstm_models and LSTM_AVAILABLE:
            self._preload_all_lstm_models()
    
    def _preload_all_lstm_models(self):
        """
        预加载所有可用的LSTM模型到内存缓存
        
        扫描 models/lstm/ 目录，加载所有 .pt 模型文件。
        适用于需要频繁使用多个模型的场景，可以提升响应速度。
        """
        if not LSTM_AVAILABLE:
            return
        
        models_dir = BASE_DIR / "models" / "lstm"
        if not models_dir.exists():
            log.debug("LSTM模型目录不存在，跳过预加载")
            return
        
        # 扫描所有 .pt 文件
        model_files = list(models_dir.glob("*_*_lstm.pt"))
        log.info(f"发现 {len(model_files)} 个LSTM模型文件，开始预加载...")
        
        loaded_count = 0
        for model_path in model_files:
            try:
                # 从文件名解析 asset_id 和 metric_id
                # 格式：{asset_id}_{metric_id}_lstm.pt
                filename = model_path.stem  # 去掉 .pt 后缀
                if not filename.endswith("_lstm"):
                    continue
                
                parts = filename.replace("_lstm", "").rsplit("_", 1)
                if len(parts) != 2:
                    log.warning(f"无法解析模型文件名: {model_path.name}")
                    continue
                
                asset_id, metric_id = parts
                cache_key = (asset_id, metric_id)
                
                # 如果已经缓存，跳过
                if cache_key in self._lstm_model_cache:
                    continue
                
                # 加载模型
                predictor = LSTMRULPredictor()
                predictor.load_model(str(model_path))
                self._lstm_model_cache[cache_key] = predictor
                loaded_count += 1
                log.debug(f"预加载LSTM模型: {asset_id}/{metric_id}")
                
            except Exception as e:
                log.warning(f"预加载LSTM模型失败 {model_path.name}: {e}")
        
        log.info(f"LSTM模型预加载完成: {loaded_count}/{len(model_files)} 个模型已加载")
    
    def _get_or_load_lstm_model(self, asset_id: str, metric_id: str) -> Optional[LSTMRULPredictor]:
        """
        获取或加载LSTM模型（带缓存）
        
        如果模型已在缓存中，直接返回；否则加载并缓存。
        
        Args:
            asset_id: 设备ID
            metric_id: 测点ID
        
        Returns:
            LSTMRULPredictor 实例，如果模型不存在或加载失败则返回 None
        """
        if not LSTM_AVAILABLE:
            return None
        
        cache_key = (asset_id, metric_id)
        
        # 检查缓存
        if cache_key in self._lstm_model_cache:
            return self._lstm_model_cache[cache_key]
        
        # 模型不在缓存中，尝试加载
        models_dir = BASE_DIR / "models" / "lstm"
        model_path = models_dir / f"{asset_id}_{metric_id}_lstm.pt"
        
        if not model_path.exists():
            log.debug(f"LSTM模型不存在: {model_path}")
            return None
        
        try:
            predictor = LSTMRULPredictor()
            predictor.load_model(str(model_path))
            # 加载成功后缓存
            self._lstm_model_cache[cache_key] = predictor
            log.debug(f"LSTM模型已加载并缓存: {asset_id}/{metric_id}")
            return predictor
        except Exception as e:
            log.warning(f"加载LSTM模型失败 {asset_id}/{metric_id}: {e}")
            return None
    
    def analyze_health_score(
        self,
        data: Dict[str, pd.DataFrame],
        asset_id: str,
        metric_ids: Optional[List[str]] = None,
    ) -> Tuple[float, List[MetricDimensionScore]]:
        """
        计算健康分（多维度评分）
        
        算法流程：
        1. 提取所有PROCESS类型测点的最新值
        2. 为每个测点计算健康分（基于阈值）
        3. 按维度分组（温度、振动、负载等）
        4. 计算总体健康分（加权平均）
        
        Returns:
            (总体健康分, 维度评分列表)
        """
        metric_defs = data.get("metric_definitions", pd.DataFrame())
        process_df = data.get("telemetry_process", pd.DataFrame())
        
        if metric_defs.empty or process_df.empty:
            log.warning(f"设备 {asset_id} 缺少测点定义或过程数据")
            return 50.0, []
        
        # 过滤设备相关的测点
        asset_metrics = metric_defs[
            (metric_defs["asset_id"] == asset_id) &
            (metric_defs["metric_type"] == "PROCESS")
        ]
        
        if metric_ids:
            asset_metrics = asset_metrics[asset_metrics["metric_id"].isin(metric_ids)]
        
        if asset_metrics.empty:
            log.warning(f"设备 {asset_id} 没有PROCESS类型的测点")
            return 50.0, []
        
        dimension_scores = []
        dimension_health_scores = {}  # 按维度分组
        
        for _, metric in asset_metrics.iterrows():
            metric_id = metric["metric_id"]
            metric_name = metric.get("metric_name", metric_id)
            warn_threshold = metric.get("warn_threshold")
            crit_threshold = metric.get("crit_threshold")
            
            if warn_threshold is None or crit_threshold is None:
                continue
            
            # 获取最新值（过滤质量和工况）
            df = process_df[
                (process_df["metric_id"] == metric_id) &
                (process_df["quality"] >= self.quality_threshold)
            ]
            
            if self.machine_state_filter is not None and "machine_state" in process_df.columns:
                df = df[df["machine_state"] == self.machine_state_filter]
            
            if df.empty:
                continue
            
            # 获取时间窗口内的数据
            max_time = df["timestamp"].max()
            cutoff = max_time - pd.Timedelta(days=self.window_days)
            df_window = df[df["timestamp"] >= cutoff]
            
            if df_window.empty:
                continue
            
            # 计算当前值（使用最近N个值的平均值，减少噪声）
            recent_values = df_window["value"].tail(min(10, len(df_window)))
            current_value = float(recent_values.mean())

            # 计算健康分（基于阈值）
            if current_value >= crit_threshold:
                health_score = 0.0
                alert_level = "critical"
            elif current_value >= warn_threshold:
                # 警告区间：线性插值
                health_score = 100.0 * (1.0 - (current_value - warn_threshold) / (crit_threshold - warn_threshold))
                alert_level = "warning"
            else:
                health_score = 100.0
                alert_level = "normal"
            
            # 计算趋势（简单线性回归）
            if len(df_window) >= 3:
                df_window = df_window.sort_values("timestamp")
                t_days = (df_window["timestamp"] - df_window["timestamp"].min()).dt.total_seconds() / 86400.0
                y = df_window["value"].values
                
                try:
                    slope, intercept, r_value, p_value, std_err = linregress(t_days.values, y)
                    if abs(slope) < 1e-6:
                        trend = "稳定"
                    elif slope > 0:
                        trend = "上升"
                    else:
                        trend = "下降"
                except:
                    trend = "未知"
            else:
                trend = "未知"
            
            # 确定维度名称（从测点名称提取）
            dimension_name = self._extract_dimension_name(metric_name, metric_id)
            
            # 创建维度评分
            dim_score = MetricDimensionScore(
                dimension_name=dimension_name,
                metric_id=metric_id,
                health_score=round(health_score, 1),
                current_value=round(current_value, 2),
                warn_threshold=float(warn_threshold),
                crit_threshold=float(crit_threshold),
                trend=trend,
                alert_level=alert_level,
            )
            
            dimension_scores.append(dim_score)
            
            # 按维度分组（同一维度取最低分）
            if dimension_name not in dimension_health_scores:
                dimension_health_scores[dimension_name] = []
            dimension_health_scores[dimension_name].append(health_score)
        
        # 计算总体健康分（按维度加权，每个维度取最低分）
        if dimension_health_scores:
            # 每个维度取最低分（最差维度决定整体）
            dimension_min_scores = [min(scores) for scores in dimension_health_scores.values()]
            overall_health_score = np.mean(dimension_min_scores)
        else:
            overall_health_score = 50.0
        
        return round(overall_health_score, 1), dimension_scores
    
    def _extract_dimension_name(self, metric_name: str, metric_id: str) -> str:
        """从测点名称提取维度名称"""
        # 中文名称提取
        if "温度" in metric_name:
            return "温度"
        elif "振动" in metric_name:
            return "振动"
        elif "负载" in metric_name or "LOAD" in metric_id:
            return "负载"
        elif "压力" in metric_name or "PRESSURE" in metric_id:
            return "压力"
        elif "流量" in metric_name or "FLOW" in metric_id:
            return "流量"
        elif "润滑" in metric_name or "LUBE" in metric_id:
            return "润滑"
        elif "进给" in metric_name or "FEED" in metric_id:
            return "进给速度"
        elif "冷却" in metric_name or "COOLANT" in metric_id:
            return "冷却系统"
        else:
            return "其他"
    
    def analyze_rul(
        self,
        data: Dict[str, pd.DataFrame],
        asset_id: str,
        metric_id: str,
        use_lstm: bool = True,
        require_lstm: bool = False,
    ) -> Tuple[Optional[float], TrendAnalysis, Optional[str]]:
        """
        预测剩余寿命（RUL）- LSTM增强版
        
        算法流程：
        1. 如果use_lstm=True，优先尝试LSTM模型（如果已训练）
        2. 如果LSTM不可用或use_lstm=False，使用传统模型：
           - 时间序列预处理（去噪、平滑）
           - 时间序列分解（趋势+周期+残差）
           - 多模型拟合（线性、多项式、指数、分段线性）
           - 选择最佳模型（基于R²）
        3. 预测达到阈值的时间
        
        Args:
            use_lstm: 是否使用LSTM模型（True=优先使用LSTM，False=仅使用传统模型）
            require_lstm: 是否要求必须使用LSTM（True=如果没有LSTM模型则返回错误，False=自动回退到传统模型）
            
        Returns:
            (rul_days, trend_analysis, error_message) 元组
            - rul_days: 剩余寿命（天），如果无法预测则为None
            - trend_analysis: 趋势分析结果
            - error_message: 错误信息，如果require_lstm=True且没有LSTM模型则返回错误信息
        """
        from backend.algorithm.data_service import prepare_process_series
        
        # 准备时间序列
        df = prepare_process_series(
            data,
            metric_id,
            asset_id=asset_id,
            window_days=self.window_days,
            machine_state=self.machine_state_filter,
            quality_threshold=self.quality_threshold,
        )
        
        if df.empty or len(df) < 5:  # 至少需要5个点
            return None, TrendAnalysis(slope=0.0, intercept=0.0, r2=0.0), None
        
        # 获取阈值
        metric_defs = data.get("metric_definitions", pd.DataFrame())
        metric_def = metric_defs[metric_defs["metric_id"] == metric_id]
        
        if metric_def.empty:
            return None, TrendAnalysis(slope=0.0, intercept=0.0, r2=0.0), None
        
        crit_threshold = metric_def.iloc[0]["crit_threshold"]
        if crit_threshold is None:
            return None, TrendAnalysis(slope=0.0, intercept=0.0, r2=0.0), None
        
        # 尝试使用LSTM模型（优先多变量，然后单变量）
        if use_lstm and LSTM_AVAILABLE:
            # 首先尝试多变量LSTM模型（一个设备一个模型）
            models_dir = BASE_DIR / "models" / "lstm"
            multivariate_model_path = models_dir / f"{asset_id}_multivariate_lstm.pt"
            
            if multivariate_model_path.exists():
                # 获取设备的所有测点
                asset_metrics = metric_defs[
                    (metric_defs["asset_id"] == asset_id) &
                    (metric_defs["metric_type"] == "PROCESS")
                ]
                
                if not asset_metrics.empty:
                    # 获取所有测点的metric_id和crit_threshold
                    all_metric_ids = asset_metrics["metric_id"].tolist()
                    all_crit_thresholds = {
                        row["metric_id"]: row["crit_threshold"]
                        for _, row in asset_metrics.iterrows()
                        if row["crit_threshold"] is not None
                    }
                    
                    # 尝试使用多变量LSTM预测
                    multivariate_result = self._try_multivariate_lstm_prediction(
                        data, asset_id, all_metric_ids, all_crit_thresholds
                    )
                    
                    if multivariate_result is not None:
                        rul_days, trend_analysis = multivariate_result
                        if rul_days is not None:
                            log.info(f"使用多变量LSTM模型预测RUL: {rul_days:.1f}天 (R²={trend_analysis.r2:.3f})")
                            return rul_days, trend_analysis, None
            
            # 多变量模型不存在或预测失败，尝试单变量LSTM模型
            single_model_path = models_dir / f"{asset_id}_{metric_id}_lstm.pt"
            
            if not single_model_path.exists():
                if require_lstm:
                    error_msg = f"当前设备({asset_id})的测点({metric_id})暂无预训练LSTM模型（单变量或多变量）"
                    log.warning(error_msg)
                    return None, TrendAnalysis(slope=0.0, intercept=0.0, r2=0.0), error_msg
                else:
                    log.debug(f"LSTM模型不存在，回退到传统模型: {single_model_path}")
            else:
                # 单变量模型存在，尝试使用
                lstm_result = self._try_lstm_prediction(df, asset_id, metric_id, crit_threshold)
                if lstm_result is not None:
                    rul_days, trend_analysis = lstm_result
                    if rul_days is not None:
                        log.info(f"使用单变量LSTM模型预测RUL: {rul_days:.1f}天 (R²={trend_analysis.r2:.3f})")
                        return rul_days, trend_analysis, None
        
        # LSTM不可用或预测失败，使用传统模型
        if use_lstm:
            log.debug(f"LSTM不可用或预测失败，使用传统模型预测RUL")
        else:
            log.debug(f"使用传统模型预测RUL（use_lstm=False）")
        
        # 时间序列预处理
        df = df.sort_values("timestamp")
        t0 = df["timestamp"].min()
        t_days = (df["timestamp"] - t0).dt.total_seconds() / 86400.0
        y_raw = df["value"].values
        
        # 1. 数据平滑（使用Savitzky-Golay滤波器，保留趋势）
        if len(y_raw) >= 7 and SCIPY_AVAILABLE:
            try:
                window_length = min(7, len(y_raw) // 2 * 2 - 1)  # 必须是奇数
                if window_length >= 3:
                    y_smooth = signal.savgol_filter(y_raw, window_length, 2)
                else:
                    y_smooth = y_raw
            except:
                y_smooth = y_raw
        else:
            # 简单移动平均
            window_size = max(3, min(len(y_raw) // 10, 5))
            if window_size > 1:
                y_smooth = pd.Series(y_raw).rolling(window=window_size, center=True).mean()
                y_smooth = y_smooth.bfill().ffill()
                y_smooth = y_smooth.values
            else:
                y_smooth = y_raw
        
        y = y_smooth
        
        # 2. 时间序列分解（提取趋势和周期）
        trend_component, seasonal_component = self._decompose_time_series(t_days.values, y)
        
        # 3. 多模型拟合
        best_model = None
        best_r2 = -np.inf
        best_trend_analysis = None
        
        models_to_try = []
        
        # 3.1 线性回归
        if SKLEARN_AVAILABLE:
            try:
                X = t_days.values.reshape(-1, 1)
                model_linear = LinearRegression()
                model_linear.fit(X, y)
                y_pred = model_linear.predict(X)
                r2_linear = r2_score(y, y_pred)
                
                models_to_try.append({
                    "model": model_linear,
                    "r2": r2_linear,
                    "name": "LinearRegression",
                    "type": "linear",
                    "predict_func": lambda t: model_linear.predict(t.reshape(-1, 1)),
                    "slope": float(model_linear.coef_[0]),
                    "intercept": float(model_linear.intercept_),
                })
            except Exception as e:
                log.debug(f"线性回归失败: {e}")
        
        # 3.2 多项式回归（2-4次）
        if SKLEARN_AVAILABLE and len(y) >= 6:
            for degree in [2, 3, 4]:
                if len(y) < degree + 2:
                    continue
                try:
                    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
                    X_poly = poly_features.fit_transform(X)
                    model_poly = LinearRegression()
                    model_poly.fit(X_poly, y)
                    y_pred = model_poly.predict(X_poly)
                    r2_poly = r2_score(y, y_pred)
                    
                    # 计算平均斜率
                    t_pred = np.linspace(t_days.min(), t_days.max(), 100)
                    y_pred_curve = model_poly.predict(poly_features.transform(t_pred.reshape(-1, 1)))
                    slope = (y_pred_curve[-1] - y_pred_curve[0]) / (t_pred[-1] - t_pred[0])
                    
                    models_to_try.append({
                        "model": model_poly,
                        "poly_features": poly_features,
                        "r2": r2_poly,
                        "name": f"Polynomial({degree})",
                        "type": "polynomial",
                        "predict_func": lambda t, m=model_poly, pf=poly_features: m.predict(pf.transform(t.reshape(-1, 1))),
                        "slope": float(slope),
                        "intercept": float(y_pred_curve[0]),
                    })
                except Exception as e:
                    log.debug(f"多项式回归(degree={degree})失败: {e}")
        
        # 3.3 指数回归（y = a * exp(b * t) + c）
        if SCIPY_AVAILABLE and len(y) >= 5:
            try:
                y_min = float(y.min())
                y_offset = y - y_min + 1e-6
                y_log = np.log(y_offset)
                
                # 使用scipy的curve_fit
                def exp_func(t, a, b):
                    return a * np.exp(b * t) + y_min - 1e-6
                
                popt, _ = curve_fit(exp_func, t_days.values, y, p0=[y_offset[0], 0.01], maxfev=1000)
                y_pred = exp_func(t_days.values, *popt)
                r2_exp = r2_score(y, y_pred)
                
                # 计算平均斜率
                t_pred = np.linspace(t_days.min(), t_days.max(), 100)
                y_pred_curve = exp_func(t_pred, *popt)
                slope = (y_pred_curve[-1] - y_pred_curve[0]) / (t_pred[-1] - t_pred[0])
                
                models_to_try.append({
                    "model": popt,
                    "y_min": y_min,
                    "r2": r2_exp,
                    "name": "Exponential",
                    "type": "exponential",
                    "predict_func": lambda t, p=popt, ym=y_min: p[0] * np.exp(p[1] * t) + ym - 1e-6,
                    "slope": float(slope),
                    "intercept": float(y_pred_curve[0]),
                })
            except Exception as e:
                log.debug(f"指数回归失败: {e}")
        
        # 3.4 分段线性回归（如果数据有明显转折点）
        if len(y) >= 10:
            try:
                # 使用简单的分段线性：找到最佳分割点
                best_split_r2 = -np.inf
                best_split_idx = len(y) // 2
                
                for split_idx in range(len(y) // 3, len(y) * 2 // 3):
                    t1 = t_days.values[:split_idx]
                    t2 = t_days.values[split_idx:]
                    y1 = y[:split_idx]
                    y2 = y[split_idx:]
                    
                    if len(t1) >= 3 and len(t2) >= 3:
                        slope1, intercept1, _, _, _ = linregress(t1, y1)
                        slope2, intercept2, _, _, _ = linregress(t2, y2)
                        
                        y_pred1 = slope1 * t1 + intercept1
                        y_pred2 = slope2 * t2 + intercept2
                        y_pred_all = np.concatenate([y_pred1, y_pred2])
                        
                        r2_split = r2_score(y, y_pred_all)
                        if r2_split > best_split_r2:
                            best_split_r2 = r2_split
                            best_split_idx = split_idx
                
                if best_split_r2 > 0.3:  # 只有当R²足够好时才使用分段模型
                    t1 = t_days.values[:best_split_idx]
                    t2 = t_days.values[best_split_idx:]
                    y1 = y[:best_split_idx]
                    y2 = y[best_split_idx:]
                    
                    slope1, intercept1, _, _, _ = linregress(t1, y1)
                    slope2, intercept2, _, _, _ = linregress(t2, y2)
                    
                    # 使用第二段的斜率（更接近当前趋势）
                    slope = slope2
                    
                    models_to_try.append({
                        "model": {"slope1": slope1, "intercept1": intercept1, "slope2": slope2, "intercept2": intercept2, "split_idx": best_split_idx},
                        "r2": best_split_r2,
                        "name": "PiecewiseLinear",
                        "type": "piecewise",
                        "predict_func": None,  # 分段函数需要特殊处理
                        "slope": float(slope),
                        "intercept": float(intercept2),
                    })
            except Exception as e:
                log.debug(f"分段线性回归失败: {e}")
        
        # 4. 选择最佳模型（R²最高，且R² > 0.1）
        if models_to_try:
            # 过滤掉R²太低的模型
            valid_models = [m for m in models_to_try if m["r2"] > 0.1]
            if valid_models:
                best_model = max(valid_models, key=lambda x: x["r2"])
                best_r2 = best_model["r2"]
                best_trend_analysis = TrendAnalysis(
                    slope=best_model["slope"],
                    intercept=best_model["intercept"],
                    r2=best_r2,
                    trend_type=best_model["type"],
                    model_name=best_model["name"],
                )
            else:
                # 如果没有R² > 0.1的模型，使用R²最高的
                best_model = max(models_to_try, key=lambda x: x["r2"])
                best_r2 = best_model["r2"]
                best_trend_analysis = TrendAnalysis(
                    slope=best_model["slope"],
                    intercept=best_model["intercept"],
                    r2=best_r2,
                    trend_type=best_model["type"],
                    model_name=best_model["name"],
                )
        else:
            # 回退到简单线性回归
            slope, intercept, r_value, p_value, std_err = linregress(t_days.values, y)
            best_trend_analysis = TrendAnalysis(
                slope=float(slope),
                intercept=float(intercept),
                r2=float(r_value ** 2),
                p_value=float(p_value) if p_value else None,
            )
            best_model = {"slope": slope, "intercept": intercept, "predict_func": lambda t: slope * t + intercept}
            best_r2 = float(r_value ** 2)
        
        # 5. 预测RUL
        if best_model is None or abs(best_trend_analysis.slope) < 1e-6:
            return None, best_trend_analysis, None
        
        current_value = float(y[-1])
        t_current = float(t_days.iloc[-1])
        
        # 判断趋势方向
        is_rising = best_trend_analysis.slope > 0
        
        # 预测达到阈值的时间
        try:
            if best_model["type"] == "piecewise":
                # 分段线性：使用第二段预测
                slope2 = best_model["model"]["slope2"]
                intercept2 = best_model["model"]["intercept2"]
                if abs(slope2) < 1e-6:
                    return None, best_trend_analysis, None
                
                if is_rising:
                    if current_value >= crit_threshold:
                        return 0.0, best_trend_analysis, None
                    rul_days = float((crit_threshold - intercept2) / slope2 - t_current)
                else:
                    if current_value <= crit_threshold:
                        return 0.0, best_trend_analysis, None
                    rul_days = float((crit_threshold - intercept2) / slope2 - t_current)
            elif best_model["type"] == "exponential":
                # 指数模型：二分搜索
                rul_days = self._predict_rul_exponential(
                    best_model["predict_func"], t_current, current_value, crit_threshold, is_rising
                )
            elif best_model["type"] == "polynomial":
                # 多项式模型：二分搜索
                rul_days = self._predict_rul_polynomial(
                    best_model["predict_func"], t_current, current_value, crit_threshold, is_rising
                )
            else:
                # 线性模型
                slope = best_trend_analysis.slope
                if is_rising:
                    if current_value >= crit_threshold:
                        return 0.0, best_trend_analysis, None
                    rul_days = float((crit_threshold - current_value) / slope)
                else:
                    if current_value <= crit_threshold:
                        return 0.0, best_trend_analysis, None
                    rul_days = float((current_value - crit_threshold) / abs(slope))
            
            if rul_days is None or rul_days < 0:
                return None, best_trend_analysis, None
            
            rul_days = max(0.0, min(365.0, rul_days))
            
            # 根据R²调整置信度提示
            if best_r2 < 0.3:
                log.warning(f"指标 {metric_id} R²={best_r2:.3f}较低（模型={best_model['name']}），RUL预测可能不准确")
            elif best_r2 < 0.6:
                log.info(f"指标 {metric_id} R²={best_r2:.3f}中等（模型={best_model['name']}），RUL预测可信度一般")
            else:
                log.info(f"指标 {metric_id} R²={best_r2:.3f}良好（模型={best_model['name']}），RUL预测可信")
            
            return round(rul_days, 1), best_trend_analysis, None
            
        except Exception as e:
            log.warning(f"RUL预测失败: {e}")
            import traceback
            log.debug(traceback.format_exc())
            return None, best_trend_analysis, None
    
    def _decompose_time_series(self, t: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        时间序列分解：提取趋势和周期成分
            
        Returns:
            (trend_component, seasonal_component)
        """
        # 简单方法：使用移动平均提取趋势，残差作为周期
        window_size = max(3, min(len(y) // 7, 7))
        if window_size > 1 and len(y) >= window_size * 2:
            trend = pd.Series(y).rolling(window=window_size, center=True).mean()
            trend = trend.bfill().ffill()
            trend = trend.values
            seasonal = y - trend
        else:
            trend = y.copy()
            seasonal = np.zeros_like(y)
        
        return trend, seasonal
    
    def _predict_rul_exponential(
        self,
        predict_func,
        t_current: float,
        current_value: float,
        crit_threshold: float,
        is_rising: bool,
    ) -> Optional[float]:
        """使用指数模型预测RUL（二分搜索）"""
        try:
            t_min = float(t_current)
            t_max = float(t_current + 365)
            
            for _ in range(50):
                t_mid = (t_min + t_max) / 2.0
                y_pred = float(predict_func(np.array([t_mid]))[0])
                
                if is_rising:
                    if y_pred >= crit_threshold:
                        t_max = t_mid
                    else:
                        t_min = t_mid
                else:
                    if y_pred <= crit_threshold:
                        t_max = t_mid
                    else:
                        t_min = t_mid
                
                if abs(t_max - t_min) < 0.01:
                    break
            
            rul_days = (t_min + t_max) / 2.0 - t_current
            return float(max(0.0, rul_days))
        except:
            return None
    
    def _predict_rul_polynomial(
        self,
        predict_func,
        t_current: float,
        current_value: float,
        crit_threshold: float,
        is_rising: bool,
    ) -> Optional[float]:
        """使用多项式模型预测RUL（二分搜索）"""
        try:
            t_min = float(t_current)
            t_max = float(t_current + 365)
            
            for _ in range(50):
                t_mid = (t_min + t_max) / 2.0
                y_pred = float(predict_func(np.array([t_mid]))[0])
                
                if is_rising:
                    if y_pred >= crit_threshold:
                        t_max = t_mid
                    else:
                        t_min = t_mid
                else:
                    if y_pred <= crit_threshold:
                        t_max = t_mid
                    else:
                        t_min = t_mid
                
                if abs(t_max - t_min) < 0.01:
                    break
            
            rul_days = (t_min + t_max) / 2.0 - t_current
            return float(max(0.0, rul_days))
        except:
            return None
    
    def get_time_series_data(
        self, 
        data: Dict[str, pd.DataFrame],
        asset_id: str,
        metric_id: str,
        max_points: int = 500,
    ) -> List[TimeSeriesPoint]:
        """
        获取时间序列数据（用于曲线图）
        
        Args:
            data: 数据字典
            asset_id: 设备ID
            metric_id: 测点ID
            max_points: 最大数据点数（用于降采样）
        
        Returns:
            时间序列数据点列表
        """
        from backend.algorithm.data_service import prepare_process_series
        
        df = prepare_process_series(
            data,
            metric_id,
            asset_id=asset_id,
            window_days=self.window_days,
            machine_state=self.machine_state_filter,
            quality_threshold=self.quality_threshold,
        )
        
        if df.empty:
            return []
        
        df = df.sort_values("timestamp")
        
        # 降采样（如果数据点太多，按步长抽样）
        if len(df) > max_points:
            step = max(1, len(df) // max_points)
            df = df.iloc[::step]
        
        # 计算平滑值（与 RUL 预测中保持一致的平滑逻辑）
        y_raw = df["value"].values
        if len(y_raw) >= 7 and SCIPY_AVAILABLE:
            try:
                window_length = min(7, len(y_raw) // 2 * 2 - 1)  # 必须是奇数
                if window_length >= 3:
                    y_smooth = signal.savgol_filter(y_raw, window_length, 2)
                else:
                    y_smooth = y_raw
            except Exception:
                y_smooth = y_raw
        else:
            # 简单移动平均
            window_size = max(3, min(len(y_raw) // 10, 5))
            if window_size > 1:
                y_smooth = pd.Series(y_raw).rolling(window=window_size, center=True).mean()
                y_smooth = y_smooth.bfill().ffill()
                y_smooth = y_smooth.values
            else:
                y_smooth = y_raw
        
        # 构建时间序列点
        points = []
        for i, (idx, row) in enumerate(df.iterrows()):
            timestamp_str = row["timestamp"].isoformat() if hasattr(row["timestamp"], "isoformat") else str(row["timestamp"])
            smoothed_val = round(float(y_smooth[i]), 2) if i < len(y_smooth) else None
            points.append(TimeSeriesPoint(
                timestamp=timestamp_str,
                value=round(float(row["value"]), 2),
                smoothed_value=smoothed_val,
            ))
        
        return points
    
    def diagnose_fault(
        self,
        waveform_data: np.ndarray,
        sampling_rate: int,
        ref_rpm: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        故障诊断（基于波形FFT分析）
        
        算法流程：
        1. 对波形进行FFT变换
        2. 提取频域特征（1X, 2X倍频，轴承故障频率等）
        3. 输入分类器得到故障概率
        
        Args:
            waveform_data: 波形数据数组
            sampling_rate: 采样率（Hz）
            ref_rpm: 参考转速（RPM）
        
        Returns:
            故障概率字典
        """
        if not SCIPY_AVAILABLE or waveform_data is None or len(waveform_data) == 0:
            return {"normal": 0.5}
        
        try:
            # FFT变换
            fft_result = fft(waveform_data)
            freqs = fftfreq(len(waveform_data), 1.0 / sampling_rate)
            
            # 提取特征（简化处理）
            # 实际应用中需要训练分类器
            return {
                "bearing_wear": 0.3,
                "normal": 0.7,
            }
        except Exception as e:
            log.warning(f"故障诊断失败: {e}")
            return {"unknown": 0.5}
    
    def analyze_asset(
        self,
        data: Dict[str, pd.DataFrame],
        asset_id: str,
        use_lstm: bool = True,
        require_lstm: bool = False,
    ) -> HealthAnalysisResult:
        """
        综合分析设备（增强版）
        
        返回包含多维度评分、时间序列数据、统计信息的完整结果
        
        Args:
            data: 数据字典
            asset_id: 设备ID
        
        Returns:
            健康分析结果（包含多维度评分和时间序列数据）
        """
        # 1. 计算健康分（多维度）
        health_score, dimension_scores = self.analyze_health_score(data, asset_id)
        
        # 2. 计算RUL（使用主要测点）
        metric_defs = data.get("metric_definitions", pd.DataFrame())
        rul_days = None
        trend_slopes = []
        trend_analyses = []
        
        if not metric_defs.empty:
            asset_metrics = metric_defs[
                (metric_defs["asset_id"] == asset_id) &
                (metric_defs["metric_type"] == "PROCESS")
            ]
            
            if not asset_metrics.empty:
                # 分析所有测点（不再限制为5个）
                # 这样可以对所有指标进行RUL预测，提供更全面的分析
                for _, metric in asset_metrics.iterrows():
                    metric_id = metric["metric_id"]
                    rul, trend_analysis, error_msg = self.analyze_rul(
                        data, asset_id, metric_id, 
                        use_lstm=use_lstm, 
                        require_lstm=require_lstm
                    )
                    
                    # 如果require_lstm=True且没有LSTM模型，抛出异常
                    if require_lstm and error_msg:
                        raise ValueError(error_msg)
                    
                    if rul is not None:
                        # 取所有测点中最短的剩余寿命，作为设备的整体 RUL
                        if rul_days is None:
                            rul_days = rul
                        else:
                            rul_days = min(rul_days, rul)
                    
                    if trend_analysis and abs(trend_analysis.slope) > 1e-6:
                        trend_slopes.append(trend_analysis.slope)
                        trend_analyses.append(trend_analysis)
        
        trend_slope = float(np.mean(trend_slopes)) if trend_slopes else 0.0
        
        # 3. 故障诊断
        waveform_df = data.get("telemetry_waveform", pd.DataFrame())
        diagnosis_result = {"normal": 0.5}
        
        if not waveform_df.empty and "asset_id" in waveform_df.columns:
            waveform_df = waveform_df[waveform_df["asset_id"] == asset_id]
            if not waveform_df.empty:
                # 简化处理：实际需要加载二进制数据
                diagnosis_result = {
                    "bearing_wear": 0.3,
                    "normal": 0.7,
                }
        
        # 4. 计算置信度：综合数据量与趋势拟合优度（R²）
        process_df = data.get("telemetry_process", pd.DataFrame())
        if process_df.empty or metric_defs.empty:
            prediction_confidence = 0.0
        else:
            asset_metrics = metric_defs[metric_defs["asset_id"] == asset_id]
            if asset_metrics.empty or "metric_id" not in process_df.columns:
                prediction_confidence = 0.0
            else:
                asset_process = process_df[
                    (process_df["metric_id"].isin(asset_metrics["metric_id"])) &
                    (process_df["quality"] >= self.quality_threshold)
                ]
                data_points = len(asset_process)
                # 基础置信度：数据点越多，置信度越高（200 个数据点视为 1.0）
                base_confidence = min(1.0, data_points / 200.0)
                
                # 如果有趋势分析，根据平均 R² 调整置信度
                if trend_analyses:
                    avg_r2 = np.mean([ta.r2 for ta in trend_analyses])
                    r2_factor = min(1.0, avg_r2 * 1.2)  # R² 越高，置信度越高
                    prediction_confidence = base_confidence * 0.7 + r2_factor * 0.3
                else:
                    prediction_confidence = base_confidence
        
        # 5. 获取时间序列数据（用于曲线图）
        time_series_data = {}
        if not metric_defs.empty:
            asset_metrics = metric_defs[
                (metric_defs["asset_id"] == asset_id) &
                (metric_defs["metric_type"] == "PROCESS")
            ]
            
            # 为每个维度选择一个代表性测点
            dimension_metrics = {}
            for _, metric in asset_metrics.iterrows():
                metric_id = metric["metric_id"]
                metric_name = metric.get("metric_name", metric_id)
                dimension_name = self._extract_dimension_name(metric_name, metric_id)
                
                if dimension_name not in dimension_metrics:
                    dimension_metrics[dimension_name] = metric_id
            
            for dimension_name, metric_id in dimension_metrics.items():
                points = self.get_time_series_data(data, asset_id, metric_id, max_points=200)
                if points:
                    time_series_data[dimension_name] = points
        
        # 6. 计算统计信息（用于统计图）
        statistics = {}
        if not process_df.empty:
            asset_metrics = metric_defs[metric_defs["asset_id"] == asset_id]
            if not asset_metrics.empty:
                asset_process = process_df[
                    (process_df["metric_id"].isin(asset_metrics["metric_id"])) &
                    (process_df["quality"] >= self.quality_threshold)
                ]
                
                if not asset_process.empty:
                    statistics = {
                        "total_data_points": len(asset_process),
                        "date_range": {
                            "start": asset_process["timestamp"].min().isoformat() if hasattr(asset_process["timestamp"].min(), "isoformat") else str(asset_process["timestamp"].min()),
                            "end": asset_process["timestamp"].max().isoformat() if hasattr(asset_process["timestamp"].max(), "isoformat") else str(asset_process["timestamp"].max()),
                        },
                        "data_quality_rate": len(asset_process[asset_process["quality"] == 1]) / len(asset_process) if len(asset_process) > 0 else 0.0,
                        "machine_state_distribution": asset_process["machine_state"].value_counts().to_dict() if "machine_state" in asset_process.columns else {},
                    }

        return HealthAnalysisResult(
            health_score=health_score,
            rul_days=rul_days,
            trend_slope=trend_slope,
            diagnosis_result=diagnosis_result,
            prediction_confidence=round(prediction_confidence, 3),
            model_version="2.0",
            dimension_scores=dimension_scores,
            time_series_data=time_series_data,
            statistics=statistics,
        )