"""
生成示例数据脚本
用于测试和演示

本脚本生成4域8表架构的示例数据，包括：
- 设备资产数据
- 测点定义数据（多维度，支持雷达图）
- 过程数据（1年，高频采集，满足曲线图和统计图需求）
- 波形数据元数据
- 运维记录
- 知识库数据

数据特点：
- 高频采集：每天10-20次（模拟实际IoT采集）
- 多维度测点：每个设备5-8个测点（支持雷达图）
- 长期趋势：1年数据，包含趋势变化
- 足够数据量：每个设备约3000-5000条数据

Author: EMSforAI Team
License: MIT
"""
import csv
import sys
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import random
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

CST = ZoneInfo("Asia/Shanghai")
CSV_DIR = BASE_DIR / "data" / "csv_v2"


def generate_telemetry_process():
    """
    生成过程数据示例（优化版：确保RUL标签分布均衡）
    
    数据特点：
    - 时间跨度：1年（2024-01-01 至 2024-12-31）
    - 采集频率：每天10-20次（工作日更频繁）
    - 每个设备5-8个测点（多维度，支持雷达图）
    - 包含多个退化周期，确保RUL标签分布均衡
    - 总数据量：每个设备约3000-5000条
    
    改进：
    - 为每个测点生成多个退化周期（60-120天）
    - 每个周期从健康状态逐渐退化到故障
    - 确保RUL标签分布：0-30天、30-60天、60-90天、90-120天都有一定比例
    """
    print("生成过程数据（优化版：RUL标签分布均衡）...")
    
    # 设备ID和测点配置（多维度，支持雷达图）
    # 格式：(metric_id, 均值, 标准差, 警告阈值, 临界阈值)
    metrics_config = {
        "CNC-MAZAK-01": [
            # 温度维度
            ("CNC01_SPINDLE_TEMP", 60.0, 5.0, 65.0, 80.0),  # 主轴温度
            ("CNC01_BEARING_TEMP", 55.0, 4.0, 60.0, 75.0),  # 轴承温度
            # 振动维度
            ("CNC01_SPINDLE_VIB_X", 3.5, 0.8, 4.5, 7.0),  # X轴振动
            ("CNC01_SPINDLE_VIB_Y", 3.2, 0.7, 4.5, 7.0),  # Y轴振动
            ("CNC01_SPINDLE_VIB_Z", 3.8, 0.9, 4.5, 7.0),  # Z轴振动
            # 负载维度
            ("CNC01_LOAD", 50.0, 10.0, 75.0, 90.0),  # 主轴负载
            ("CNC01_FEED_RATE", 1200.0, 150.0, 1500.0, 1800.0),  # 进给速度
            # 润滑维度
            ("CNC01_LUBE_PRESSURE", 2.5, 0.3, 2.0, 1.5),  # 润滑压力（下降趋势为坏）
        ],
        "CNC-MAZAK-02": [
            ("CNC02_SPINDLE_TEMP", 62.0, 5.0, 65.0, 80.0),
            ("CNC02_BEARING_TEMP", 57.0, 4.5, 60.0, 75.0),
            ("CNC02_SPINDLE_VIB_X", 3.8, 0.9, 4.5, 7.0),
            ("CNC02_SPINDLE_VIB_Y", 3.5, 0.8, 4.5, 7.0),
            ("CNC02_SPINDLE_VIB_Z", 4.0, 1.0, 4.5, 7.0),
            ("CNC02_LOAD", 52.0, 11.0, 75.0, 90.0),
            ("CNC02_FEED_RATE", 1250.0, 160.0, 1500.0, 1800.0),
            ("CNC02_LUBE_PRESSURE", 2.4, 0.3, 2.0, 1.5),
        ],
        "COMP-ATLAS-01": [
            # 压力维度
            ("COMP01_DISCHARGE_PRESSURE", 0.75, 0.05, 0.78, 0.95),  # 排气压力
            ("COMP01_SUCTION_PRESSURE", 0.15, 0.02, 0.12, 0.10),  # 吸气压力
            # 温度维度
            ("COMP01_DISCHARGE_TEMP", 95.0, 5.0, 95.0, 110.0),  # 排气温度
            ("COMP01_OIL_TEMP", 70.0, 4.0, 75.0, 85.0),  # 油温
            # 流量维度
            ("COMP01_FLOW_RATE", 12.5, 1.2, 10.0, 8.0),  # 流量（下降趋势为坏）
            # 振动维度
            ("COMP01_VIBRATION", 2.8, 0.6, 4.0, 6.0),  # 整体振动
        ],
        "COMP-ATLAS-02": [
            ("COMP02_DISCHARGE_PRESSURE", 0.73, 0.05, 0.78, 0.95),
            ("COMP02_SUCTION_PRESSURE", 0.16, 0.02, 0.12, 0.10),
            ("COMP02_DISCHARGE_TEMP", 97.0, 5.5, 95.0, 110.0),
            ("COMP02_OIL_TEMP", 72.0, 4.5, 75.0, 85.0),
            ("COMP02_FLOW_RATE", 12.0, 1.3, 10.0, 8.0),
            ("COMP02_VIBRATION", 3.0, 0.7, 4.0, 6.0),
        ],
        "MILL-HAAS-01": [
            ("MILL01_SPINDLE_TEMP", 58.0, 4.5, 65.0, 80.0),
            ("MILL01_SPINDLE_VIB", 3.0, 0.6, 4.5, 7.0),
            ("MILL01_LOAD", 45.0, 9.0, 75.0, 90.0),
            ("MILL01_COOLANT_TEMP", 25.0, 2.0, 30.0, 35.0),
            ("MILL01_COOLANT_FLOW", 8.5, 0.8, 7.0, 5.0),
        ],
    }
    
    # 生成1年的数据（2024年全年）
    start_date = datetime(2024, 1, 1, 6, 0, 0, tzinfo=CST)
    end_date = datetime(2024, 12, 31, 22, 0, 0, tzinfo=CST)
    total_days = (end_date - start_date).days + 1
    
    records = []
    
    # 为每个测点生成多个退化周期，确保RUL标签分布均衡
    for asset_id, metrics in metrics_config.items():
        for metric_id, mean_value, std_value, warn_threshold, crit_threshold in metrics:
            # 判断趋势方向
            is_rising = "TEMP" in metric_id or "VIB" in metric_id or "LOAD" in metric_id or "FEED" in metric_id
            
            # 生成4-5个退化周期，每个周期90-150天（更长，确保有更多健康状态样本）
            num_cycles = random.randint(4, 5)
            cycle_lengths = []
            cycle_starts = []
            
            # 计算周期长度和起始时间，确保覆盖全年
            remaining_days = total_days
            current_start = 0
            
            for cycle_idx in range(num_cycles):
                if cycle_idx == num_cycles - 1:
                    # 最后一个周期使用剩余所有天数
                    cycle_length = remaining_days
                else:
                    # 每个周期90-150天（更长，确保退化更慢）
                    cycle_length = random.randint(90, 150)
                    remaining_days -= cycle_length
                    if remaining_days < 90:
                        cycle_length += remaining_days
                        remaining_days = 0
                
                cycle_lengths.append(cycle_length)
                cycle_starts.append(current_start)
                current_start += cycle_length
                
                if remaining_days <= 0:
                    break
            
            # 为每个周期生成数据
            for cycle_idx, (cycle_start, cycle_length) in enumerate(zip(cycle_starts, cycle_lengths)):
                # 计算周期内的起始值和目标值
                if is_rising:
                    # 上升趋势：从健康值逐渐上升到临界值
                    cycle_start_value = mean_value * (0.85 + random.uniform(-0.05, 0.05))  # 健康状态，略低于均值
                    cycle_end_value = crit_threshold * (0.95 + random.uniform(0, 0.05))  # 接近或达到临界值
                else:
                    # 下降趋势：从健康值逐渐下降到临界值
                    cycle_start_value = mean_value * (1.15 + random.uniform(-0.05, 0.05))  # 健康状态，略高于均值
                    cycle_end_value = crit_threshold * (1.05 + random.uniform(-0.05, 0))  # 接近或达到临界值
                
                # 生成周期内的数据点
                for day_in_cycle in range(cycle_length):
                    current_date = start_date + timedelta(days=cycle_start + day_in_cycle)
                    if current_date > end_date:
                        break
                    
        # 工作日更频繁采集
        is_weekday = current_date.weekday() < 5
                    is_holiday = False
        
                # 高频采集：工作日每天15-20次，周末每天5-10次
                if is_weekday and not is_holiday:
                    num_collections = random.randint(15, 20)
                else:
                    num_collections = random.randint(5, 10)
                
                for i in range(num_collections):
                        # 计算周期内的进度（0-1）
                        progress = day_in_cycle / max(1, cycle_length - 1)
                        
                        # 使用更慢的退化曲线，确保有更多长期RUL样本
                        # 使用分段线性退化：前60%时间慢速退化，后40%时间加速退化
                        # 这样可以确保有更多健康状态样本（RUL 60-120天）
                        if progress < 0.6:
                            # 前60%时间：慢速退化（线性）
                            degradation_factor = progress / 0.6 * 0.3  # 只退化30%
                        else:
                            # 后40%时间：加速退化（非线性）
                            remaining_progress = (progress - 0.6) / 0.4  # 0-1
                            degradation_factor = 0.3 + remaining_progress ** 1.5 * 0.7  # 从30%到100%
                        
                        if is_rising:
                            # 上升趋势
                            base_value = cycle_start_value + (cycle_end_value - cycle_start_value) * degradation_factor
                    else:
                            # 下降趋势
                            base_value = cycle_start_value - (cycle_start_value - cycle_end_value) * degradation_factor
                    
                        # 添加噪声和波动
                    # 季节性波动（温度相关指标）
                        days_offset = (current_date - start_date).days
                    if "TEMP" in metric_id:
                            seasonal = 2.0 * np.sin(2 * np.pi * days_offset / 365)  # 年度周期
                    else:
                        seasonal = 0.0
                    
                        # 周期性波动（周周期）
                        cycle_factor = 0.02 * abs(np.sin(2 * np.pi * days_offset / 7))
                    
                        # 日内波动
                    hour = random.randint(6, 22)
                    if 8 <= hour <= 18:
                            daily_factor = 0.01
                    else:
                            daily_factor = -0.005
                    
                    # 组合所有因素
                        value = base_value + seasonal + cycle_factor * base_value + daily_factor * base_value + random.gauss(0, std_value)
                    
                    # 确保值在合理范围内
                    if "TEMP" in metric_id:
                        value = max(20, min(120, value))
                    elif "LOAD" in metric_id or "FEED" in metric_id:
                        value = max(0, min(100 if "LOAD" in metric_id else 2000, value))
                    elif "VIB" in metric_id:
                        value = max(0, min(15, value))
                    elif "PRESSURE" in metric_id:
                        value = max(0.1, min(1.5, value))
                    elif "FLOW" in metric_id:
                        value = max(5, min(20, value))
                    elif "LUBE" in metric_id:
                        value = max(1.0, min(3.5, value))
                    
                        # 生成采集时间
                    hour = random.randint(6, 22)
                    minute = random.randint(0, 59)
                    second = random.randint(0, 59)
                    collection_time = current_date.replace(hour=hour, minute=minute, second=second)
                    
                    # 数据质量（95%高质量，5%低质量）
                    quality = 1 if random.random() > 0.05 else 0
                    
                    # 工况状态（工作日主要是加工状态）
                    if is_weekday and 8 <= hour <= 18:
                        machine_state = random.choices([0, 1, 2], weights=[0.02, 0.08, 0.90])[0]
                    elif is_weekday:
                        machine_state = random.choices([0, 1, 2], weights=[0.10, 0.30, 0.60])[0]
                    else:
                        machine_state = random.choices([0, 1, 2], weights=[0.40, 0.50, 0.10])[0]
                    
                    records.append({
                        "timestamp": collection_time.strftime("%Y-%m-%dT%H:%M:%S+08:00"),
                        "metric_id": metric_id,
                        "value": round(value, 2),
                        "quality": quality,
                        "machine_state": machine_state,
                    })
    
    # 按时间排序
    records.sort(key=lambda x: x["timestamp"])
    
    # 写入CSV
    csv_path = CSV_DIR / "telemetry_process.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "metric_id", "value", "quality", "machine_state"])
        writer.writeheader()
        writer.writerows(records)
    
    print(f"  ✓ 生成 {len(records):,} 条过程数据记录")
    print(f"  ✓ 时间范围: {start_date.date()} 到 {end_date.date()}")
    print(f"  ✓ 平均每天: {len(records) // 365:.0f} 条")
    print(f"  ✓ 保存到: {csv_path}")
    
    # 统计每个设备的数据量
    device_counts = {}
    for record in records:
        asset_id = record["metric_id"].split("_")[0]
        if asset_id not in device_counts:
            device_counts[asset_id] = 0
        device_counts[asset_id] += 1
    
    print(f"  ✓ 各设备数据量:")
    for asset_id, count in sorted(device_counts.items()):
        print(f"    - {asset_id}: {count:,} 条")


def generate_maintenance_records():
    """生成运维记录示例（用于AI训练标签）"""
    print("生成运维记录...")
    
    # 生成更多运维记录，覆盖全年
    records = []
    record_counter = 1
    
    # 为每个设备生成3-5条运维记录
    assets = ["CNC-MAZAK-01", "CNC-MAZAK-02", "COMP-ATLAS-01", "COMP-ATLAS-02", "MILL-HAAS-01"]
    
    failure_types = {
        "CNC": [
            ("ERR_BRG_01", "轴承故障", "主轴在3000转时有啸叫，且电流波动大。振动值达到6.8mm/s，超过警告阈值。", "拆解发现保持架断裂，更换 NSK-7014 轴承。检查润滑系统，补充润滑脂。", 3200.0),
            ("ERR_TEMP_01", "温度故障", "主轴温度持续升高，超过80度。冷却系统效率下降，冷却液循环不畅。", "清洗冷却系统，更换冷却液。检查冷却泵，清理过滤器。", 1500.0),
            ("ERR_VIB_01", "振动故障", "主轴振动异常，X轴方向振动值达到5.2mm/s，存在不平衡问题。", "进行动平衡校正，调整主轴配重。检查刀具安装，确保同心度。", 2800.0),
            ("ERR_LUBE_01", "润滑故障", "润滑压力下降至1.8MPa，低于正常值。润滑系统堵塞。", "清洗润滑管路，更换润滑泵过滤器。补充润滑脂。", 1200.0),
        ],
        "COMP": [
            ("ERR_PRESSURE_01", "压力故障", "排气压力波动大，压力值在0.85-0.92MPa之间波动，超出正常范围。", "更换空气过滤器，清洗压力阀。检查管路密封性，紧固连接。", 1200.0),
            ("ERR_TEMP_02", "温度故障", "排气温度持续偏高，达到105℃，接近临界值。冷却器效率下降。", "清洗冷却器，更换冷却介质。检查风扇运行状态，清理散热片。", 2100.0),
            ("ERR_FLOW_01", "流量故障", "流量下降至9.5m³/min，低于正常值。过滤器堵塞。", "更换空气过滤器，清洗管路。检查流量传感器。", 800.0),
        ],
        "MILL": [
            ("ERR_COOLANT_01", "冷却故障", "冷却液温度升高至32℃，流量下降。冷却系统效率下降。", "更换冷却液，清洗冷却系统。检查冷却泵。", 900.0),
            ("ERR_VIB_02", "振动故障", "主轴振动值达到5.5mm/s，存在不平衡。", "进行动平衡校正，检查刀具安装。", 1500.0),
        ],
    }
    
    # 生成时间点（全年分布）
    base_date = datetime(2024, 1, 15, 8, 0, 0, tzinfo=CST)
    
    for asset_id in assets:
        asset_type = asset_id.split("-")[0]
        failure_list = failure_types.get(asset_type, failure_types["CNC"])
        
        # 每个设备3-5条记录
        num_records = random.randint(3, 5)
        for i in range(num_records):
            # 随机分布在全年
            days_offset = random.randint(0, 300)
            start_time = base_date + timedelta(days=days_offset)
            
            # 随机选择故障类型
            failure_code, failure_name, issue_desc, solution_desc, cost = random.choice(failure_list)
            
            # 故障持续时间（4-8小时）
            duration_hours = random.randint(4, 8)
            end_time = start_time + timedelta(hours=duration_hours)
            
            record_id = f"MAINT-2024-{record_counter:03d}"
            records.append({
                "record_id": record_id,
                "asset_id": asset_id,
                "start_time": start_time.strftime("%Y-%m-%dT%H:%M:%S+08:00"),
                "end_time": end_time.strftime("%Y-%m-%dT%H:%M:%S+08:00"),
                "failure_code": failure_code,
                "issue_description": issue_desc,
                "solution_description": solution_desc,
                "cost": str(cost),
            })
            record_counter += 1
    
    csv_path = CSV_DIR / "maintenance_records.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["record_id", "asset_id", "start_time", "end_time", 
                                                "failure_code", "issue_description", "solution_description", "cost"])
        writer.writeheader()
        writer.writerows(records)
    
    print(f"  ✓ 生成 {len(records)} 条运维记录")
    print(f"  ✓ 保存到: {csv_path}")


def generate_knowledge_base():
    """生成知识库示例（用于LLM RAG）"""
    print("生成知识库...")
    
    records = [
        {
            "doc_id": "KB-001",
            "applicable_model": "MAZAK-H800",
            "category": "手册",
            "title": "主轴维护手册-轴承部分",
            "content_chunk": "主轴轴承更换标准：当振动值超过7.0mm/s或温度超过80℃时，需要检查轴承状态。正常维护周期为5000小时。轴承型号：NSK-7014。更换时需注意保持架方向，确保润滑充分。",
        },
        {
            "doc_id": "KB-002",
            "applicable_model": "MAZAK-H800",
            "category": "手册",
            "title": "主轴维护手册-温度控制",
            "content_chunk": "主轴温度正常范围：50-70℃。超过70℃需要检查冷却系统。冷却液应每6个月更换一次。冷却泵流量不应低于额定值的80%。",
        },
        {
            "doc_id": "KB-003",
            "applicable_model": "MAZAK-H800",
            "category": "案例",
            "title": "轴承故障处理案例-保持架断裂",
            "content_chunk": "案例：某设备出现3000转啸叫，检查发现保持架断裂。处理：立即停机，更换NSK-7014轴承，检查润滑系统。预防措施：定期检查轴承状态，确保润滑充分。",
        },
        {
            "doc_id": "KB-004",
            "applicable_model": "MAZAK-H800",
            "category": "案例",
            "title": "振动异常处理案例-不平衡",
            "content_chunk": "案例：主轴X轴方向振动值达到5.2mm/s，存在不平衡问题。处理：进行动平衡校正，调整主轴配重。检查刀具安装，确保同心度。预防措施：定期进行动平衡检查。",
        },
        {
            "doc_id": "KB-005",
            "applicable_model": "ATLAS-GA75",
            "category": "标准",
            "title": "空压机运行标准-压力",
            "content_chunk": "空压机排气压力正常范围：0.6-0.9MPa。压力波动不应超过±0.05MPa。压力过高会导致能耗增加，压力过低影响用气设备。",
        },
        {
            "doc_id": "KB-006",
            "applicable_model": "ATLAS-GA75",
            "category": "标准",
            "title": "空压机运行标准-温度",
            "content_chunk": "空压机排气温度不应超过110℃。正常范围：85-100℃。温度过高会导致润滑油失效，加速磨损。冷却器应定期清洗。",
        },
        {
            "doc_id": "KB-007",
            "applicable_model": "ATLAS-GA75",
            "category": "案例",
            "title": "压力波动处理案例",
            "content_chunk": "案例：排气压力在0.85-0.92MPa之间波动，超出正常范围。处理：更换空气过滤器，清洗压力阀。检查管路密封性，紧固连接。预防措施：定期更换过滤器，检查管路。",
        },
        {
            "doc_id": "KB-008",
            "applicable_model": "MAZAK-H800",
            "category": "标准",
            "title": "振动监测标准",
            "content_chunk": "主轴振动监测标准：正常值<4.5mm/s，警告值4.5-7.0mm/s，临界值>7.0mm/s。振动监测应在加工状态下进行，不同转速下阈值不同。",
        },
        {
            "doc_id": "KB-009",
            "applicable_model": "MAZAK-H800",
            "category": "手册",
            "title": "润滑系统维护",
            "content_chunk": "润滑系统压力正常范围：2.0-3.0MPa。压力低于2.0MPa需要检查润滑泵和过滤器。润滑脂应每3000小时更换一次。",
        },
        {
            "doc_id": "KB-010",
            "applicable_model": "HAAS-VF2",
            "category": "手册",
            "title": "冷却系统维护",
            "content_chunk": "冷却液温度正常范围：20-28℃。流量不应低于7.0L/min。冷却液应每6个月更换一次。定期清洗冷却系统。",
        },
    ]
    
    csv_path = CSV_DIR / "knowledge_base.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id", "applicable_model", "category", "title", "content_chunk"])
        writer.writeheader()
        writer.writerows(records)
    
    print(f"  ✓ 生成 {len(records)} 条知识库记录")
    print(f"  ✓ 保存到: {csv_path}")


def generate_telemetry_waveform():
    """生成波形数据元数据（实际波形数据是二进制，这里只生成元数据）"""
    print("生成波形数据元数据...")
    
    records = []
    
    # 为每个设备生成一些波形快照
    # 定义每个设备对应的振动测点ID（必须与metric_definitions.csv中的定义一致）
    asset_vib_metrics = {
        "CNC-MAZAK-01": {
            "X": "CNC01_SPINDLE_VIB_X",
            "Y": "CNC01_SPINDLE_VIB_Y",
            "Z": "CNC01_SPINDLE_VIB_Z",
        },
        "CNC-MAZAK-02": {
            "X": "CNC02_SPINDLE_VIB_X",
            "Y": "CNC02_SPINDLE_VIB_Y",
            "Z": "CNC02_SPINDLE_VIB_Z",
        },
        "MILL-HAAS-01": {
            "X": "MILL01_SPINDLE_VIB",  # MILL设备只有一个振动测点，不分轴
            "Y": "MILL01_SPINDLE_VIB",
            "Z": "MILL01_SPINDLE_VIB",
        },
    }
    
    axes = ["X", "Y", "Z"]
    
    # 生成1年的数据，每5-7天一次采集（更频繁）
    start_date = datetime(2024, 1, 1, 10, 0, 0, tzinfo=CST)
    end_date = datetime(2024, 12, 31, 16, 0, 0, tzinfo=CST)
    
    snapshot_counter = 1
    current_date = start_date
    
    while current_date <= end_date:
        # 每5-7天采集一次（更频繁）
        if random.random() < 0.2:  # 约20%的概率采集
            for asset_id, vib_metrics in asset_vib_metrics.items():
                for axis in axes:
                    snapshot_id = f"WAVEFORM-{snapshot_counter:06d}"
                    
                    # 生成参考转速（加工状态下的典型转速）
                    ref_rpm = random.choice([1500, 2000, 2500, 3000])
                    
                    # 获取对应的metric_id（如果设备有该轴的测点定义）
                    metric_id = vib_metrics.get(axis)
                    
                    records.append({
                        "snapshot_id": snapshot_id,
                        "asset_id": asset_id,
                        "timestamp": current_date.strftime("%Y-%m-%dT%H:%M:%S+08:00"),
                        "sampling_rate": "12800",
                        "duration_ms": "1000",
                        "axis": axis,
                        "ref_rpm": str(ref_rpm),
                        "metric_id": metric_id,
                    })
                    
                    snapshot_counter += 1
        
        current_date += timedelta(days=1)
    
    csv_path = CSV_DIR / "telemetry_waveform.csv"
    if records:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["snapshot_id", "asset_id", "timestamp", "sampling_rate", 
                                                    "duration_ms", "axis", "ref_rpm", "metric_id"])
            writer.writeheader()
            writer.writerows(records)
        
        print(f"  ✓ 生成 {len(records)} 条波形数据元数据")
        print(f"  ✓ 保存到: {csv_path}")
        print(f"  ⚠️  注意: 实际波形数据（data_blob）需要单独存储，这里只生成元数据")
    else:
        print(f"  ⚠️  未生成波形数据（采集频率较低）")


def main():
    """主函数"""
    print("=" * 60)
    print("生成示例数据（增强版 - 支持雷达图、曲线图、统计图）")
    print("=" * 60)
    print()
    
    # 确保目录存在
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    
    # 生成数据
    generate_telemetry_process()
    print()
    generate_telemetry_waveform()
    print()
    generate_maintenance_records()
    print()
    generate_knowledge_base()
    print()
    
    print("=" * 60)
    print("示例数据生成完成！")
    print("=" * 60)
    print("\n数据特点:")
    print("  ✓ 高频采集：每天10-20次（工作日更频繁）")
    print("  ✓ 多维度测点：每个设备5-8个测点（支持雷达图）")
    print("  ✓ 多个退化周期：每个测点3-4个周期，确保RUL标签分布均衡")
    print("  ✓ 非线性退化：使用幂函数模拟实际退化过程")
    print("  ✓ 足够数据量：每个设备约3000-5000条数据")
    print("\nRUL标签分布优化:")
    print("  ✓ 每个周期60-120天，从健康状态逐渐退化到故障")
    print("  ✓ 确保0-30天、30-60天、60-90天、90-120天都有一定比例")
    print("\n下一步:")
    print("1. 运行导入脚本: python backend/import_csv_data.py")
    print("2. 训练模型验证: python backend/algorithm/train_lstm.py")
    print("3. 查看RUL标签分布统计（在训练日志中）")


if __name__ == "__main__":
    main()
