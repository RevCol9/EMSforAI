# AI 计算引擎设计草案（制造业场景）

本设计围绕现有数据表（设备、巡检、遥测、异常、健康度、维保与审计）规划一套可迭代的 AI 计算引擎，目标是：
1. 支撑在线/离线批处理，产出健康度、RUL、异常检测、维修建议等结果；
2. 兼容新增扩展表（`telemetry`、`anomalies`、`health_scores`、`maintenance_recommendations` 等）；
3. 便于逐步替换/升级模型（规则→统计→ML），并可追踪版本与输入输出。 

## 核心组件
- **数据接入层**：从 `inspection_logs`/`inspection_metric_values`（低频巡检）与 `telemetry`（高频传感）读取原始数据，统一时间线并做清洗/对齐。
- **特征层**：计算滚动统计、频域特征（如振动 RMS/峰值、温度移动平均、趋势斜率）、质量标签（缺失率/波动度），特征写入特征缓存或临时表。
- **任务编排**：
  - 周期调度：按设备/指标创建任务（cron 或队列），分别用于趋势建模、异常检测、健康度汇总、建议生成。
  - 事件驱动：新遥测/巡检写入时触发轻量实时检测（阈值+3σ）并生成 `anomalies` 记录。
- **模型服务**：
  - 趋势与 RUL：线性/指数回归、Cox 退化、Prophet、LSTM 等，输出拟合度、RUL 天数、关键阈值命中时间。
  - 异常检测：基于阈值、统计控制（EWMA/Shewhart）、孤立森林/LOF、自编码器（时序/频域）。
  - 健康评分：按指标权重、告警状态、数据质量综合评分，写入 `health_scores`。
  - 维保建议：基于规则库 + 模型输出（风险等级/剩余寿命/异常模式）生成 `maintenance_recommendations`，附置信度与过期时间。
- **结果落库与审计**：所有计算结果写入 `metric_ai_analysis`、`anomalies`、`health_scores`、`maintenance_recommendations`，并在 `audit_events` 记录任务、模型版本、输入摘要。 

## 数据流示意
1. **采集阶段**：
   - 巡检：写入 `inspection_logs` + `inspection_metric_values`（JSON）。
   - 遥测：写入 `telemetry`，索引 `idx_telemetry_device_metric` 便于按设备/指标时间窗扫描。
2. **预处理**：
   - 缺失/异常点剔除、重采样到统一时间步；
   - 单位换算与有效区间裁剪（利用 `device_metric_definitions.valid_min/valid_max`）。
3. **检测/建模**：
   - 实时阈值检测 → `anomalies`（source="rule"）。
   - 滚动回归/预测 → `metric_ai_analysis`（存 `curve_points`、`model_version`）。
   - 数据质量与状态聚合 → `health_scores`。
4. **决策输出**：
   - 建议/工单：生成 `maintenance_recommendations`，必要时更新 `work_orders`/`maintenance_tasks`。
   - 审计：写 `audit_events` 记录计算任务、模型、输入时间窗。 

## 任务与接口建议
- **定时任务入口**：新增 `backend/cron.py`（或 Celery/Arq 任务队列），统一封装：
  - `run_trend_and_rul(device_id, metric_key, window_days=30)`：读取最近窗口数据，拟合并落地 `metric_ai_analysis`；
  - `run_anomaly_scan(device_id, metric_key, window_hours=24)`：3σ + 阈值检测，写入 `anomalies`；
  - `recompute_health(device_id)`：汇总最新 `metric_ai_analysis` + 数据质量，写 `health_scores`；
  - `generate_recommendations(device_id)`：基于健康度/异常/RUL 生成建议。
- **API 预留**：
  - `GET /devices/{id}/health`：返回最新 `health_scores`、关键异常和建议。
  - `GET /devices/{id}/metrics/{metric_key}/ai_analysis?latest=1`：读取最新 AI 分析结果。
  - `POST /ai/tasks`：触发特定设备/指标的重新计算（便于回溯/调参）。
- **模型版本管理**：结果表包含 `model_version`，建议使用 `{模块}-{算法}-{数据窗}-{超参}` 的可读格式，便于回滚与 A/B。 

## 与现有代码的衔接
- `backend/ai_service.py` 保留轻量趋势/异常逻辑，可在队列任务中直接复用或替换为更高阶模型。
- `backend/models.py` 已有索引满足按设备/指标/时间查询，适合窗口扫描；如需高频数据，可进一步分表/冷热分层。
- 结果回写时应更新 `MetricAIAnalysis.calc_time` 与 `AuditEvent`，保证可追踪性。 

## 后续迭代清单
- 引入任务队列（Celery/Arq/RQ）与结果表的幂等写入策略（`ON CONFLICT` upsert）。
- 增加数据质量标注与回灌：在 `telemetry`/`inspection_metric_values` 存储质量评分/过滤标签。 
- 引入模型注册/配置表（如 `ai_models`, `ai_jobs`）以支持多模型并存与动态调度。
- 对接可视化：提供曲线（`curve_points`）、异常时间轴、健康评分趋势的统一查询接口。
