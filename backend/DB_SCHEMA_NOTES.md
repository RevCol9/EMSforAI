# 设备与巡检数据库字段说明与制造业扩展建议

## 已有表设计解读

### `device_models`
- **id**：设备型号主键。
- **name**：设备型号名称，要求唯一，用于区分不同规格/系列。

### `devices`
- **id**：设备主键。
- **model_id**：关联设备型号，用于继承该型号的指标定义。
- **name**：设备名称或资产编号，要求唯一，便于资产管理。
- **索引 `idx_devices_model`**：加速按型号查询设备。

### `device_metric_definitions`
- **id**：指标定义主键。
- **model_id**：指标定义所属的设备型号。
- **metric_key**：指标键（英文唯一标识）。
- **metric_name**：指标展示名称。
- **unit**：单位（如 °C、rpm）。
- **data_type**：数据类型描述（如 float、int、bool）。
- **warn_threshold / crit_threshold**：预警/告警阈值。
- **valid_min / valid_max**：有效取值范围，用于校验异常数据。
- **trend_direction**：趋势方向，1 表示数值上升为好或坏（业务可自定义约定）。
- **weight_in_health**：健康度计算权重。
- **is_ai_analyzed**：是否参与 AI 分析。
- **唯一索引 `idx_metric_def_model_key`**：保证同一型号下 `metric_key` 唯一。

### `inspection_logs`
- **id**：巡检记录主键。
- **device_id**：被巡检的设备。
- **user_id**：执行巡检的用户。
- **recorded_at**：巡检发生时间（现场采集时间）。
- **created_at**：记录入库时间（系统时间）。
- **索引**：`idx_logs_device` 支持按设备过滤，`idx_logs_recorded_at` 支持按时间排序/筛选。

### `inspection_metric_values`
- **log_id**：巡检日志外键，亦为主键（一条日志对应一份指标数据）。
- **metrics_data**：JSON 存放该次巡检的全部指标及数值，结构与 `device_metric_definitions` 对应。

### `metric_ai_analysis`
- **device_id / metric_key**：针对某设备某指标的 AI 分析结果。
- **calc_time**：分析计算时间（纳入主键，用于版本化记录）。
- **model_version**：AI 模型版本。
- **rul_days**：剩余寿命（Remaining Useful Life）估计。
- **trend_r2**：趋势拟合优度。
- **last_value**：最近一次数值。
- **curve_points / extra_info**：用于前端或模型调试的曲线点、附加信息。
- **组合主键 `pk_metric_ai`**：保证同一设备/指标/时间唯一；索引 `idx_metric_ai_latest` 便于获取最新记录。

## 面向制造业的扩展字段建议
以下建议可按需选择，并配合索引/约束完善：

### 设备与资产管理
- **序列号 / 资产编码**（`serial_number`）：与厂家或ERP资产对应，唯一索引。
- **生产线 / 车间 / 区域**（`line_id`, `workshop`, `location_code`）：便于分区查询与告警路由，可加组合索引。
- **设备状态**（`status`：运行/停机/维修/备用）：支持OEE统计。
- **供应商 / 品牌 / 型号版本**（`vendor`, `brand`, `model_revision`）：追溯可靠性。
- **安装日期 / 投运日期**（`installed_at`, `commissioned_at`）：用于寿命模型和折旧计算。
- **保修到期日 / 维保合同ID**（`warranty_end`, `maintenance_contract_id`）。
- **责任班组 / 维护人**（`owner_team`, `maintainer_id`）。

### 指标定义与采集
- **采样频率 / 采集来源**（`sample_rate_hz`, `source`）：区分在线监测 vs 手工巡检。
- **合规标准编号**（`standard_code`）：对应GB/ISO或企业标准。
- **单位换算信息**（`unit_scale`, `unit_offset`）：便于多源数据统一。
- **报警滞后/抑制时间**（`alarm_delay_seconds`）：防止抖动告警。
- **质量等级或权重分组**（`criticality`）：标记关键指标。

### 巡检与工单协同
- **巡检类型**（`inspection_type`：点巡检/预防性/临时性）。
- **工单ID / 计划ID**（`work_order_id`, `plan_id`）：联动CMMS/工单系统。
- **环境条件**（`ambient_temp`, `humidity`）：辅助判断异常原因。
- **巡检备注 / 附件路径**（`remark`, `attachments` JSON）：上传照片、声音。
- **地理坐标 / 定位**（`geo_lat`, `geo_lng`, `area_grid`）：大厂园区定位。

### 数据质量与审计
- **数据来源标签**（`data_origin`：IoT网关/人工录入/API）。
- **采集设备ID**（`collector_id`：传感器或采集器编号）。
- **数据质量评分**（`data_quality_score`）：便于过滤噪声。
- **校验状态**（`is_validated`, `validation_notes`）：记录异常值处理。
- **审计字段**（`created_by`, `updated_by`, `updated_at`, `deleted_at` 软删）。

### AI 分析与健康管理
- **健康评分**（`health_score`）：按指标/设备聚合的健康度。
- **预测置信度**（`prediction_confidence`）：RUL或异常检测可信度。
- **模型输入特征摘要**（`feature_snapshot` JSON）：便于复现实验。
- **告警级别与处理状态**（`alert_level`, `alert_status`, `acknowledged_at`, `ack_by`）。
- **停机风险 / 产能影响评估**（`downtime_risk`, `throughput_impact`）。

### 运维与成本
- **耗材/备件寿命计数**（`spare_usage_cycles`）：对应刀具、轴承等。
- **能源计量**（`energy_kwh`, `gas_nm3`, `water_ton`）：支撑能耗分析。
- **OEE 相关字段**（`availability`, `performance`, `quality_rate`）：对接生产效率。
- **维护成本 / 预算归集**（`maintenance_cost`, `budget_center`）。

### 索引与约束建议
- 为高频查询（按设备、时间范围、指标键）补充组合索引，例如 `(device_id, recorded_at)`、`(device_id, metric_key, calc_time desc)`。
- 对关键业务字段（序列号、工单ID）设置唯一约束或外键，保证数据一致性。

## 兼容性小贴士
- 新增字段时保持空值兼容（`nullable=True` 或设默认值），避免线上迁移风险。
- 对 JSON 字段定义存储结构约定，避免前后端语义不一致。
- 为时序数据考虑分区或归档策略，减轻历史数据压力。
