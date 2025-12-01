"""
CSV数据导入脚本
将data/csv目录下的CSV文件导入到数据库
"""
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session
from sqlalchemy import select, inspect, text

from .db import SessionLocal, engine, Base
from .models import (
    DeviceModel,
    Device,
    DeviceMetricDefinition,
    InspectionLog,
    InspectionMetricValue,
    MetricAIAnalysis,
    EquipmentAndAssetManagement,
    SpareUsage,
    EnergyConsumption,
    OEEStat,
    MaintenanceCost,
)

CST = ZoneInfo("Asia/Shanghai")
CSV_DIR = Path(__file__).parent.parent / "data" / "csv"


def parse_datetime(value: str) -> Optional[datetime]:
    """解析日期时间字符串"""
    if not value or value.strip() == "":
        return None
    try:
        # 尝试多种格式
        for fmt in [
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ]:
            try:
                dt = datetime.strptime(value.strip(), fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=CST)
                return dt
            except ValueError:
                continue
        return None
    except Exception:
        return None


def parse_bool(value: str) -> Optional[bool]:
    """解析布尔值"""
    if not value or value.strip() == "":
        return None
    v = value.strip().lower()
    if v in ("true", "1", "yes", "y", "t"):
        return True
    if v in ("false", "0", "no", "n", "f"):
        return False
    return None


def parse_json(value: str) -> Optional[Dict]:
    """解析JSON字符串"""
    if not value or value.strip() == "":
        return None
    try:
        # CSV中的JSON可能使用双引号转义，先尝试直接解析
        return json.loads(value)
    except json.JSONDecodeError:
        # 如果失败，尝试替换双引号转义
        try:
            # CSV中 "" 表示一个双引号
            cleaned = value.replace('""', '"')
            return json.loads(cleaned)
        except Exception:
            return None
    except Exception:
        return None


def parse_float(value: str) -> Optional[float]:
    """解析浮点数"""
    if not value or value.strip() == "":
        return None
    try:
        return float(value.strip())
    except Exception:
        return None


def parse_int(value: str) -> Optional[int]:
    """解析整数"""
    if not value or value.strip() == "":
        return None
    try:
        return int(value.strip())
    except Exception:
        return None


def import_device_models(db: Session, csv_path: Path) -> Dict[int, int]:
    """
    导入设备型号
    返回: 旧ID -> 新ID的映射
    """
    print(f"导入设备型号: {csv_path.name}")
    id_mapping = {}
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            old_id = parse_int(row.get("id"))
            name = row.get("name", "").strip()
            
            if not name:
                continue
            
            # 检查是否已存在
            existing = db.execute(
                select(DeviceModel).where(DeviceModel.name == name)
            ).scalar_one_or_none()
            
            if existing:
                id_mapping[old_id] = existing.id
                print(f"  设备型号已存在: {name} (ID: {existing.id})")
            else:
                model = DeviceModel(name=name)
                db.add(model)
                db.flush()
                id_mapping[old_id] = model.id
                print(f"  导入设备型号: {name} (ID: {model.id})")
    
    db.commit()
    return id_mapping


def import_devices(db: Session, csv_path: Path, model_id_mapping: Dict[int, int]):
    """
    导入设备
    """
    print(f"导入设备: {csv_path.name}")
    id_mapping = {}
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            old_id = parse_int(row.get("id"))
            old_model_id = parse_int(row.get("model_id"))
            name = row.get("name", "").strip()
            serial_number = row.get("serial_number", "").strip()
            location = row.get("location", "").strip()
            status = row.get("status", "active").strip()
            
            if not name or old_model_id not in model_id_mapping:
                print(f"  跳过设备: {name} (缺少名称或无效的model_id)")
                continue
            
            new_model_id = model_id_mapping[old_model_id]
            
            # 检查是否已存在
            existing = db.execute(
                select(Device).where(Device.name == name)
            ).scalar_one_or_none()
            
            if existing:
                id_mapping[old_id] = existing.id
                print(f"  设备已存在: {name} (ID: {existing.id})")
            else:
                device = Device(
                    model_id=new_model_id,
                    name=name,
                    serial_number=serial_number if serial_number else None,
                    location=location if location else None,
                    status=status,
                )
                db.add(device)
                db.flush()
                id_mapping[old_id] = device.id
                print(f"  导入设备: {name} (ID: {device.id})")
    
    db.commit()
    return id_mapping


def import_device_metric_definitions(
    db: Session, csv_path: Path, model_id_mapping: Dict[int, int]
):
    """导入设备指标定义"""
    print(f"导入设备指标定义: {csv_path.name}")
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            old_model_id = parse_int(row.get("model_id"))
            metric_key = row.get("metric_key", "").strip()
            
            if not metric_key or old_model_id not in model_id_mapping:
                continue
            
            new_model_id = model_id_mapping[old_model_id]
            
            # 检查是否已存在
            existing = db.execute(
                select(DeviceMetricDefinition).where(
                    DeviceMetricDefinition.model_id == new_model_id,
                    DeviceMetricDefinition.metric_key == metric_key,
                )
            ).scalar_one_or_none()
            
            if existing:
                print(f"  指标定义已存在: {metric_key}")
                continue
            
            metric = DeviceMetricDefinition(
                model_id=new_model_id,
                metric_key=metric_key,
                metric_name=row.get("metric_name", "").strip(),
                unit=row.get("unit", "").strip() or None,
                base_unit=row.get("base_unit", "").strip() or None,
                display_unit=row.get("display_unit", "").strip() or None,
                unit_scale=parse_float(row.get("unit_scale")),
                unit_offset=parse_float(row.get("unit_offset")),
                decimal_precision=parse_int(row.get("decimal_precision")),
                data_type=row.get("data_type", "").strip() or None,
                warn_threshold=parse_float(row.get("warn_threshold")),
                crit_threshold=parse_float(row.get("crit_threshold")),
                valid_min=parse_float(row.get("valid_min")),
                valid_max=parse_float(row.get("valid_max")),
                trend_direction=parse_int(row.get("trend_direction")) or 1,
                weight_in_health=parse_float(row.get("weight_in_health")) or 1.0,
                is_ai_analyzed=parse_bool(row.get("is_ai_analyzed")) if row.get("is_ai_analyzed") else True,
                sampling_frequency=parse_float(row.get("sampling_frequency")),
                collection_source=row.get("collection_source", "").strip() or None,
                collector_id=row.get("collector_id", "").strip() or None,
                standard_code=row.get("standard_code", "").strip() or None,
                alarm_delay_seconds=parse_int(row.get("alarm_delay_seconds")),
                criticality=row.get("criticality", "").strip() or None,
                data_origin=row.get("data_origin", "").strip() or None,
                validation_required=parse_bool(row.get("validation_required")) if row.get("validation_required") else False,
                feature_snapshot=parse_json(row.get("feature_snapshot")),
                min_sampling_interval=parse_float(row.get("min_sampling_interval")),
                max_sampling_interval=parse_float(row.get("max_sampling_interval")),
            )
            db.add(metric)
            print(f"  导入指标定义: {metric_key}")
    
    db.commit()


def import_inspection_submits(
    db: Session, csv_path: Path, device_id_mapping: Dict[int, int]
):
    """导入巡检记录"""
    print(f"导入巡检记录: {csv_path.name}")
    count = 0
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            old_device_id = parse_int(row.get("device_id"))
            if old_device_id not in device_id_mapping:
                continue
            
            device_id = device_id_mapping[old_device_id]
            user_id = parse_int(row.get("user_id")) or 0
            recorded_at = parse_datetime(row.get("recorded_at"))
            
            if not recorded_at:
                continue
            
            metrics_data = parse_json(row.get("metrics", "{}"))
            if not metrics_data:
                metrics_data = {}
            
            log = InspectionLog(
                device_id=device_id,
                user_id=user_id,
                recorded_at=recorded_at,
                data_origin=row.get("data_origin", "").strip() or None,
                collector_id=row.get("collector_id", "").strip() or None,
                data_quality_score=parse_float(row.get("data_quality_score")),
                validation_notes=row.get("validation_notes", "").strip() or None,
            )
            db.add(log)
            db.flush()
            
            metric_value = InspectionMetricValue(
                log_id=log.id,
                metrics_data=metrics_data,
            )
            db.add(metric_value)
            count += 1
    
    db.commit()
    print(f"  导入 {count} 条巡检记录")


def import_metric_ai_analysis(
    db: Session, csv_path: Path, device_id_mapping: Dict[int, int]
):
    """导入AI分析结果"""
    print(f"导入AI分析结果: {csv_path.name}")
    count = 0
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            old_device_id = parse_int(row.get("device_id"))
            if old_device_id not in device_id_mapping:
                continue
            
            device_id = device_id_mapping[old_device_id]
            metric_key = row.get("metric_key", "").strip()
            calc_time = parse_datetime(row.get("calc_time"))
            
            if not metric_key or not calc_time:
                continue
            
            existing = db.get(MetricAIAnalysis, (device_id, metric_key, calc_time))
            if existing:
                print(
                    f"  AI分析已存在: device={device_id}, metric={metric_key}, calc_time={calc_time.isoformat()}，跳过"
                )
                continue

            analysis = MetricAIAnalysis(
                device_id=device_id,
                metric_key=metric_key,
                calc_time=calc_time,
                model_version=row.get("model_version", "v1").strip(),
                rul_days=parse_int(row.get("rul_days")),
                trend_r2=parse_float(row.get("trend_r2")),
                last_value=parse_float(row.get("last_value")),
                curve_points=parse_json(row.get("curve_points")),
                extra_info=parse_json(row.get("extra_info")),
                health_score=parse_float(row.get("health_score")),
                prediction_confidence=parse_float(row.get("prediction_confidence")),
                feature_snapshot=parse_json(row.get("feature_snapshot")),
                alert_level=row.get("alert_level", "").strip() or None,
                alert_status=row.get("alert_status", "").strip() or None,
                acknowledged_at=parse_datetime(row.get("acknowledged_at")),
                ack_by=row.get("ack_by", "").strip() or None,
                downtime_risk=parse_float(row.get("downtime_risk")),
                throughput_impact=parse_float(row.get("throughput_impact")),
            )
            db.add(analysis)
            count += 1
    
    db.commit()
    print(f"  导入 {count} 条AI分析记录")


def import_equipment_and_asset_management(
    db: Session, csv_path: Path, device_id_mapping: Dict[int, int]
):
    """导入资产扩展信息"""
    print(f"导入资产扩展信息: {csv_path.name}")
    count = 0
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            old_device_id = parse_int(row.get("device_id"))
            if old_device_id not in device_id_mapping:
                continue
            
            device_id = device_id_mapping[old_device_id]
            serial_number = row.get("serial_number", "").strip()
            
            if not serial_number:
                continue
            
            # 检查是否已存在
            existing = db.execute(
                select(EquipmentAndAssetManagement).where(
                    EquipmentAndAssetManagement.serial_number == serial_number
                )
            ).scalar_one_or_none()
            
            if existing:
                continue
            
            eam = EquipmentAndAssetManagement(
                device_id=device_id,
                serial_number=serial_number,
                asset_code=row.get("asset_code", "").strip() or None,
                location=row.get("location", "").strip() or None,
                status=row.get("status", "active").strip(),
                vendor=row.get("vendor", "").strip() or None,
                brand=row.get("brand", "").strip() or None,
                model_revision=row.get("model_revision", "").strip() or None,
                installed_at=parse_datetime(row.get("installed_at")),
                commissioned_at=parse_datetime(row.get("commissioned_at")),
                warranty_end=parse_datetime(row.get("warranty_end")),
                maintenance_contract_id=row.get("maintenance_contract_id", "").strip() or None,
                management_team=row.get("management_team", "").strip() or None,
                maintainer_id=parse_int(row.get("maintainer_id")),
            )
            db.add(eam)
            count += 1
    
    db.commit()
    print(f"  导入 {count} 条资产扩展信息")


def import_spare_usage_cycles(
    db: Session, csv_path: Path, device_id_mapping: Dict[int, int]
):
    """导入备件寿命计数"""
    print(f"导入备件寿命计数: {csv_path.name}")
    count = 0
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            old_device_id = parse_int(row.get("device_id"))
            if old_device_id not in device_id_mapping:
                continue
            
            device_id = device_id_mapping[old_device_id]
            part_code = row.get("part_code", "").strip()
            
            if not part_code:
                continue
            
            spare = SpareUsage(
                device_id=device_id,
                part_code=part_code,
                usage_cycles=parse_int(row.get("usage_cycles")) or 0,
            )
            db.add(spare)
            count += 1
    
    db.commit()
    print(f"  导入 {count} 条备件寿命记录")


def import_energy_consumption(
    db: Session, csv_path: Path, device_id_mapping: Dict[int, int]
):
    """导入能耗计量"""
    print(f"导入能耗计量: {csv_path.name}")
    count = 0
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            old_device_id = parse_int(row.get("device_id"))
            if old_device_id not in device_id_mapping:
                continue
            
            device_id = device_id_mapping[old_device_id]
            recorded_at = parse_datetime(row.get("recorded_at"))
            
            if not recorded_at:
                continue
            
            energy = EnergyConsumption(
                device_id=device_id,
                recorded_at=recorded_at,
                energy_kwh=parse_float(row.get("energy_kwh")),
                gas_nm3=parse_float(row.get("gas_nm3")),
                water_ton=parse_float(row.get("water_ton")),
                source=row.get("source", "").strip() or None,
            )
            db.add(energy)
            count += 1
    
    db.commit()
    print(f"  导入 {count} 条能耗记录")


def import_oee_stats(db: Session, csv_path: Path, device_id_mapping: Dict[int, int]):
    """导入OEE统计"""
    print(f"导入OEE统计: {csv_path.name}")
    count = 0
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            old_device_id = parse_int(row.get("device_id"))
            if old_device_id not in device_id_mapping:
                continue
            
            device_id = device_id_mapping[old_device_id]
            period_start = parse_datetime(row.get("period_start"))
            period_end = parse_datetime(row.get("period_end"))
            
            if not period_start or not period_end:
                continue
            
            oee = OEEStat(
                device_id=device_id,
                period_start=period_start,
                period_end=period_end,
                availability=parse_float(row.get("availability")),
                performance=parse_float(row.get("performance")),
                quality_rate=parse_float(row.get("quality_rate")),
            )
            db.add(oee)
            count += 1
    
    db.commit()
    print(f"  导入 {count} 条OEE统计记录")


def import_maintenance_costs(
    db: Session, csv_path: Path, device_id_mapping: Dict[int, int]
):
    """导入维护成本"""
    print(f"导入维护成本: {csv_path.name}")
    count = 0
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            old_device_id = parse_int(row.get("device_id"))
            if old_device_id not in device_id_mapping:
                continue
            
            device_id = device_id_mapping[old_device_id]
            
            cost = MaintenanceCost(
                device_id=device_id,
                period_start=parse_datetime(row.get("period_start")),
                period_end=parse_datetime(row.get("period_end")),
                cost=parse_float(row.get("cost")),
                budget_center=row.get("budget_center", "").strip() or None,
                note=row.get("note", "").strip() or None,
            )
            db.add(cost)
            count += 1
    
    db.commit()
    print(f"  导入 {count} 条维护成本记录")


def main():
    """主函数：导入所有CSV数据"""
    print("=" * 60)
    print("开始导入CSV数据到数据库")
    print("=" * 60)
    
    # 显示数据库连接信息（隐藏密码）
    from urllib.parse import urlparse
    db_url = os.getenv("DB_URL") or os.getenv("DATABASE_URL") or "sqlite:///emsforai.db"
    parsed = urlparse(db_url)
    if parsed.password:
        # 隐藏密码
        safe_url = db_url.replace(parsed.password, "***")
    else:
        safe_url = db_url
    print(f"数据库连接: {safe_url}")
    print()
    
    # 测试数据库连接
    print("测试数据库连接...")
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✓ 数据库连接成功")
    except Exception as e:
        print(f"✗ 数据库连接失败: {e}")
        print("\n请检查:")
        print("1. 数据库服务是否运行")
        print("2. 环境变量 DB_URL 或 DATABASE_URL 是否正确配置")
        print("3. 数据库用户权限是否正确")
        if "postgresql" in db_url.lower():
            print("4. 如果使用Docker，请运行: docker-compose up -d db")
        return
    
    # 确保数据库表已创建
    print("\n创建数据库表...")
    try:
        Base.metadata.create_all(bind=engine)
        print("✓ 数据库表创建完成")
        
        # 验证表是否创建成功
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        print(f"✓ 已创建 {len(tables)} 个表: {', '.join(tables)}")
    except Exception as e:
        print(f"✗ 创建数据库表失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print()
    db = SessionLocal()
    try:
        # 按依赖顺序导入
        # 1. 设备型号（无依赖）
        model_id_mapping = {}
        csv_path = CSV_DIR / "device_models.csv"
        if csv_path.exists():
            model_id_mapping = import_device_models(db, csv_path)
        else:
            print(f"警告: 未找到文件 {csv_path}")
        
        # 2. 设备（依赖设备型号）
        device_id_mapping = {}
        csv_path = CSV_DIR / "devices.csv"
        if csv_path.exists():
            device_id_mapping = import_devices(db, csv_path, model_id_mapping)
        else:
            print(f"警告: 未找到文件 {csv_path}")
        
        # 3. 设备指标定义（依赖设备型号）
        csv_path = CSV_DIR / "device_metric_definitions.csv"
        if csv_path.exists():
            import_device_metric_definitions(db, csv_path, model_id_mapping)
        else:
            print(f"警告: 未找到文件 {csv_path}")
        
        # 4. 其他数据（依赖设备）
        csv_path = CSV_DIR / "inspection_submits.csv"
        if csv_path.exists():
            import_inspection_submits(db, csv_path, device_id_mapping)
        
        csv_path = CSV_DIR / "metric_ai_analysis.csv"
        if csv_path.exists():
            import_metric_ai_analysis(db, csv_path, device_id_mapping)
        
        csv_path = CSV_DIR / "equipment_and_asset_management.csv"
        if csv_path.exists():
            import_equipment_and_asset_management(db, csv_path, device_id_mapping)
        
        csv_path = CSV_DIR / "spare_usage_cycles.csv"
        if csv_path.exists():
            import_spare_usage_cycles(db, csv_path, device_id_mapping)
        
        csv_path = CSV_DIR / "energy_consumption.csv"
        if csv_path.exists():
            import_energy_consumption(db, csv_path, device_id_mapping)
        
        csv_path = CSV_DIR / "oee_stats.csv"
        if csv_path.exists():
            import_oee_stats(db, csv_path, device_id_mapping)
        
        csv_path = CSV_DIR / "maintenance_costs.csv"
        if csv_path.exists():
            import_maintenance_costs(db, csv_path, device_id_mapping)
        
        print("=" * 60)
        print("CSV数据导入完成！")
        print("=" * 60)
        
    except Exception as e:
        db.rollback()
        print(f"导入过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()

