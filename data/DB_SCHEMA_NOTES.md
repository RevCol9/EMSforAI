# 设备与巡检数据库字段说明（V2 架构）与制造业扩展建议

本说明文档与当前代码中的 ORM 模型（`backend/models.py`）保持一致，围绕“四域八表”架构，详细描述各表字段含义与使用场景。

- 一：基础元数据域（设备资产、测点定义）
- 二：动态感知域（过程数据、波形数据）
- 三：知识与运维域（运维记录、知识库）
- 四：分析结果域（健康分析、AI 报告）

---

## 一：基础元数据（Metadata Layer）

### 表：`assets`（设备资产表）

对应模型：`Asset`

- **asset_id** `String(100)`：设备唯一标识，**主键**。  
  建议使用业务可读编码，如 `CNC-MAZAK-01`，在全系统中统一使用该字段作为设备主键。
- **name** `String(200)`：设备名称/展示名，用于前端显示和查询。
- **model_id** `String(100)`：设备型号 ID，当前版本简化为字符串，可与 PLM/ERP 型号编码对接。
- **location** `String(200)`：设备物理位置，如“车间-产线-工位”，用于区域化统计与告警路由。
- **commission_date** `DateTime`：设备投运日期，可用于寿命模型、折旧和资产年限分析。
- **status** `String(50)`：设备状态：`active`（在用）/`maintenance`（维修中）/`down`（停机），支持 OEE 计算和运维看板。
- **created_at** `DateTime(timezone=True)`：记录创建时间（CST）。
- **updated_at** `DateTime(timezone=True)`：记录最后更新时间（CST）。

索引：
- `idx_assets_model (model_id)`：同型号设备查询。
- `idx_assets_location (location)`：按位置查询设备。

> 兼容性说明：V1 中的 `devices` / `device_models` 已被 `assets` 所取代，**不再使用整型 `device_id` 主键**，统一使用字符串 `asset_id`。

---

### 表：`metric_definitions`（测点与阈值定义表）

对应模型：`MetricDefinition`

- **metric_id** `String(100)`：测点唯一标识，**主键**。  
  推荐格式：`资产ID_测点名`，如 `CNC-MAZAK-01_SPINDLE_TEMP`。
- **asset_id** `String(100)` 外键 → `assets.asset_id`：所属设备。
- **metric_name** `String(200)`：测点展示名称（中文），如“主轴温度”“主轴振动”。
- **metric_type** `Enum(MetricType)`：测点类型：  
  - `PROCESS`：过程量（温度/压力/转速/负载等标量数据）；  
  - `VIBRATION`：振动量（需要波形/频域分析）。
- **unit** `String(50)`：物理单位，如 `℃`、`mm/s`、`MPa`。
- **warn_threshold** `Float`：预警阈值，超过此值进入“警告”区间。
- **crit_threshold** `Float`：临界阈值，超过此值进入“严重告警”，通常需要立即维护。
- **is_condition_dependent** `Boolean`：是否依赖工况，不同工况下阈值可能不同（当前版本主要用于标记）。
- **sampling_frequency** `Float`：采样频率（Hz），用于采集配置。
- **collector_id** `String(100)`：采集设备/传感器编号，便于溯源与设备管理。
- **created_at** `DateTime(timezone=True)`：记录创建时间。

关系与索引：
- `asset = relationship("Asset", backref="metrics")`：一台设备下可有多个测点。  
- `idx_metric_asset (asset_id)`：按设备查找测点。  
- `idx_metric_type (metric_type)`：按测点类型过滤。

---

## 二：动态感知域（Sensing Layer）

### 表：`telemetry_process`（过程数据表）

对应模型：`TelemetryProcess`  
存放高频标量过程数据，是健康度、趋势与 RUL 分析的核心输入。

- **id** `Integer`：自增主键。
- **timestamp** `DateTime(timezone=True)`：数据采集时间（带时区）。
- **metric_id** `String(100)` 外键 → `metric_definitions.metric_id`：测点 ID。
- **value** `Float`：采集到的物理量数值。
- **quality** `Integer`：数据质量标记：`0=异常/故障`，`1=正常`。算法会优先使用 `quality=1` 的数据。  
- **machine_state** `Integer`：工况状态：  
  - `0`：停机（STOPPED）；  
  - `1`：待机（STANDBY）；  
  - `2`：加工（RUNNING，默认重点分析对象）。

关系与索引：
- `metric = relationship("MetricDefinition", backref="process_data")`。  
- 索引：
  - `idx_telemetry_time (timestamp)`：按时间范围检索。  
  - `idx_telemetry_metric (metric_id)`：按测点检索。  
  - `idx_telemetry_state (machine_state)`：按工况过滤。  
  - `idx_telemetry_metric_time (metric_id, timestamp)`：测点+时间复合索引，服务时间序列查询。

---

### 表：`telemetry_waveform`（波形数据表）

对应模型：`TelemetryWaveform`  
存放振动等波形数据的元数据与二进制内容，适合结合对象存储或 TSDB 使用。

- **snapshot_id** `String(100)`：波形快照唯一标识，**主键**。  
- **asset_id** `String(100)` 外键 → `assets.asset_id`：关联设备。  
- **timestamp** `DateTime(timezone=True)`：采集时间。  
- **sampling_rate** `Integer`：采样率（Hz），如 12800 Hz。  
- **duration_ms** `Integer`：采集时长（毫秒）。  
- **axis** `String(10)`：采集轴向：`X/Y/Z`。  
- **data_blob** `LargeBinary`：波形数组的二进制数据（例如 `float32` 编码后的字节流）。  
- **ref_rpm** `Float`：采集时的主轴转速（RPM），用于倍频分析。  
- **metric_id** `String(100)` 外键 → `metric_definitions.metric_id`（可空）：关联振动测点。

索引：
- `idx_waveform_asset_time (asset_id, timestamp)`：按设备+时间查询波形历史。  
- `idx_waveform_metric (metric_id)`：按测点查询波形。

---

## 三：知识与运维域（Knowledge Layer）

### 表：`maintenance_records`（运维工单表）

对应模型：`MaintenanceRecord`  
记录设备故障与维修历史，可作为 AI 训练标签与 LLM 语料。

- **record_id** `String(100)`：工单唯一标识，**主键**。  
- **asset_id** `String(100)` 外键 → `assets.asset_id`：关联设备。  
- **start_time** `DateTime(timezone=True)`：故障开始时间，用于标记异常数据区间。  
- **end_time** `DateTime(timezone=True)`：故障结束时间（可空），用于计算故障持续时间。  
- **failure_code** `String(100)`：标准故障代码，如 `ERR_BRG_01`。  
- **issue_description** `Text`：故障现象描述。  
- **solution_description** `Text`：维修过程与解决方案记录。  
- **cost** `Float`：维修成本（元）。  
- **created_at** `DateTime(timezone=True)`：记录创建时间。

索引：
- `idx_maintenance_asset (asset_id)`：按设备查历史工单。  
- `idx_maintenance_time (start_time)`：按时间范围查询工单。  
- `idx_maintenance_code (failure_code)`：按故障代码做统计分析。

---

### 枚举：`KnowledgeCategory` 与表：`knowledge_base`（知识库表）

对应模型：`KnowledgeBase`

#### 枚举 `KnowledgeCategory`

- `MANUAL` → `"手册"`：设备手册、技术文档。
- `CASE` → `"案例"`：维修案例、实践经验。  
- `STANDARD` → `"标准"`：行业标准或企业标准。

> CSV 导入时，代码支持中文值（手册/案例/标准）与英文键名（MANUAL/CASE/STANDARD）的双向映射，详见 `import_csv_data.py`。

#### 表 `knowledge_base`

- **doc_id** `String(100)`：文档唯一标识，**主键**。  
- **applicable_model** `String(100)`：适用设备型号，用于过滤检索范围。  
- **category** `Enum(KnowledgeCategory)`：知识类型。  
- **title** `String(500)`：文档标题。  
- **content_chunk** `Text`：文本切片内容，用于 LLM RAG 检索（推荐 200–500 字）。  
- **embedding** `JSON`：可选，存储向量表示，用于向量数据库检索。  
- **created_at** `DateTime(timezone=True)`：记录创建时间。

索引：
- `idx_knowledge_model (applicable_model)`：按型号过滤知识文档。  
- `idx_knowledge_category (category)`：按知识类型过滤。

---

## 四：分析结果域（Analysis Output Layer）

### 表：`ai_health_analysis`（健康分析结果表）

对应模型：`AIHealthAnalysis`  
持久化保存算法引擎的数值输出，便于历史追踪与报表展示。

- **analysis_id** `String(100)`：分析记录唯一标识，**主键**。  
- **asset_id** `String(100)` 外键 → `assets.asset_id`：对应设备。  
- **calc_time** `DateTime(timezone=True)`：分析计算时间。  
- **health_score** `Float`：设备健康分（0–100）。  
- **rul_days** `Float`：剩余寿命（天），`NULL` 表示无法计算。  
- **trend_slope** `Float`：趋势斜率，正值常表示恶化，负值表示改善。  
- **diagnosis_result** `JSON`：故障诊断结果，如 `{"bearing_wear": 0.85}`。  
- **model_version** `String(50)`：模型版本号。  
- **prediction_confidence** `Float`：预测置信度（0–1），综合数据量与趋势拟合优度计算。

索引与约束：
- `idx_health_asset_time (asset_id, calc_time)`：按设备+时间查询历史。  
- `ck_health_score_range`：健康分范围约束：`0 <= health_score <= 100`。  
- `ck_confidence_range`：置信度范围约束：`0 <= prediction_confidence <= 1`。

---

### 表：`ai_reports`（AI 报告表）

对应模型：`AIReport`  
存储基于 `ai_health_analysis` + `knowledge_base` + `maintenance_records` 生成的 LLM 报告。

- **report_id** `String(100)`：报告唯一标识，**主键**。  
- **analysis_id** `String(100)` 外键 → `ai_health_analysis.analysis_id`：关联的数值分析记录。  
- **generated_content** `Text`：LLM 生成的 Markdown 报告内容。  
- **user_feedback** `Integer`：用户反馈评分（1–5 星）。  
- **created_at** `DateTime(timezone=True)`：报告生成时间。

索引与约束：
- `idx_report_analysis (analysis_id)`：按分析记录查询报告。  
- `ck_feedback_range`：保证 `user_feedback` 在 1–5 之间。

### 索引与约束建议
- 为高频查询（按设备、时间范围、测点）补充组合索引，例如：  
  `(asset_id, timestamp)`、`(asset_id, metric_id, timestamp desc)` 等。
- 对关键业务字段（序列号、工单ID）设置唯一约束或外键，保证数据一致性。

## 兼容性小贴士
- 新增字段时保持空值兼容（`nullable=True` 或设默认值），避免线上迁移风险。
- 对 JSON 字段定义存储结构约定，避免前后端语义不一致。
- 为时序数据考虑分区或归档策略（如按月分区或历史归档），减轻主库压力。


