"""
EMSforAI CSV数据导入脚本

本脚本用于将CSV格式的数据导入到数据库中，支持4域8表架构的所有表。
导入顺序按照表之间的依赖关系：先导入基础元数据，再导入动态数据。

支持的表：
- 一：assets, metric_definitions
- 二：telemetry_process, telemetry_waveform
- 三：maintenance_records, knowledge_base
- 四：ai_health_analysis, ai_reports（通常由算法生成，不在此导入）

Author: EMSforAI Team
License: MIT
"""
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session
from sqlalchemy import select, inspect, text

# 处理直接运行时的导入问题
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))
    from backend.db import SessionLocal, engine, Base
    from backend import models
else:
    from .db import SessionLocal, engine, Base
    from . import models

CST = ZoneInfo("Asia/Shanghai")
CSV_DIR = Path(__file__).parent.parent / "data" / "csv_v2"


def parse_datetime(value: str) -> Optional[datetime]:
    """解析日期时间字符串"""
    if not value or value.strip() == "":
        return None
    try:
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


def import_assets(db: Session, csv_path: Path):
    """导入设备资产"""
    print(f"导入设备资产: {csv_path.name}")
    count = 0
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            asset_id = row.get("asset_id", "").strip()
            if not asset_id:
                continue
            
            existing = db.get(models.Asset, asset_id)
            if existing:
                print(f"  设备已存在: {asset_id}")
                continue
            
            asset = models.Asset(
                asset_id=asset_id,
                name=row.get("name", "").strip(),
                model_id=row.get("model_id", "").strip(),
                location=row.get("location", "").strip() or None,
                commission_date=parse_datetime(row.get("commission_date")),
                status=row.get("status", "active").strip(),
            )
            db.add(asset)
            count += 1
    
    db.commit()
    print(f"  导入 {count} 条设备记录")


def import_metric_definitions(db: Session, csv_path: Path):
    """导入测点定义"""
    print(f"导入测点定义: {csv_path.name}")
    count = 0
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric_id = row.get("metric_id", "").strip()
            if not metric_id:
                continue
            
            existing = db.get(models.MetricDefinition, metric_id)
            if existing:
                print(f"  测点已存在: {metric_id}")
                continue
            
            metric_type_str = row.get("metric_type", "PROCESS").strip().upper()
            try:
                metric_type = models.MetricType[metric_type_str]
            except KeyError:
                metric_type = models.MetricType.PROCESS
            
            metric = models.MetricDefinition(
                metric_id=metric_id,
                asset_id=row.get("asset_id", "").strip(),
                metric_name=row.get("metric_name", "").strip(),
                metric_type=metric_type,
                unit=row.get("unit", "").strip() or None,
                warn_threshold=parse_float(row.get("warn_threshold")),
                crit_threshold=parse_float(row.get("crit_threshold")),
                is_condition_dependent=parse_bool(row.get("is_condition_dependent")) or False,
                sampling_frequency=parse_float(row.get("sampling_frequency")),
                collector_id=row.get("collector_id", "").strip() or None,
            )
            db.add(metric)
            count += 1
    
    db.commit()
    print(f"  导入 {count} 条测点定义")


def import_telemetry_process(db: Session, csv_path: Path):
    """导入过程数据"""
    print(f"导入过程数据: {csv_path.name}")
    count = 0
    batch_size = 1000
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        batch = []
        
        for row in reader:
            timestamp = parse_datetime(row.get("timestamp"))
            if not timestamp:
                continue
            
            telemetry = models.TelemetryProcess(
                timestamp=timestamp,
                metric_id=row.get("metric_id", "").strip(),
                value=parse_float(row.get("value")) or 0.0,
                quality=parse_int(row.get("quality")) or 1,
                machine_state=parse_int(row.get("machine_state")) or 2,
            )
            batch.append(telemetry)
            
            if len(batch) >= batch_size:
                db.bulk_save_objects(batch)
                db.commit()
                count += len(batch)
                batch = []
        
        if batch:
            db.bulk_save_objects(batch)
            db.commit()
            count += len(batch)
    
    print(f"  导入 {count} 条过程数据记录")


def import_telemetry_waveform(db: Session, csv_path: Path):
    """导入波形数据元数据（注意：实际波形数据是二进制，需要单独处理）"""
    print(f"导入波形数据元数据: {csv_path.name}")
    count = 0
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            snapshot_id = row.get("snapshot_id", "").strip()
            if not snapshot_id:
                continue
            
            existing = db.get(models.TelemetryWaveform, snapshot_id)
            if existing:
                print(f"  快照已存在: {snapshot_id}")
                continue
            
            waveform = models.TelemetryWaveform(
                snapshot_id=snapshot_id,
                asset_id=row.get("asset_id", "").strip(),
                timestamp=parse_datetime(row.get("timestamp")),
                sampling_rate=parse_int(row.get("sampling_rate")) or 12800,
                duration_ms=parse_int(row.get("duration_ms")) or 1000,
                axis=row.get("axis", "X").strip(),
                ref_rpm=parse_float(row.get("ref_rpm")),
                metric_id=row.get("metric_id", "").strip() or None,
                # 注意：data_blob需要单独处理，这里不导入
            )
            db.add(waveform)
            count += 1
    
    db.commit()
    print(f"  导入 {count} 条波形数据元数据")
    print(f"  ⚠️  注意: 实际波形数据（data_blob）需要单独导入或通过API上传")


def import_maintenance_records(db: Session, csv_path: Path):
    """导入运维记录"""
    print(f"导入运维记录: {csv_path.name}")
    count = 0
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            record_id = row.get("record_id", "").strip()
            if not record_id:
                continue
            
            existing = db.get(models.MaintenanceRecord, record_id)
            if existing:
                print(f"  记录已存在: {record_id}")
                continue
            
            record = models.MaintenanceRecord(
                record_id=record_id,
                asset_id=row.get("asset_id", "").strip(),
                start_time=parse_datetime(row.get("start_time")),
                end_time=parse_datetime(row.get("end_time")),
                failure_code=row.get("failure_code", "").strip() or None,
                issue_description=row.get("issue_description", "").strip() or None,
                solution_description=row.get("solution_description", "").strip() or None,
                cost=parse_float(row.get("cost")),
            )
            db.add(record)
            count += 1
    
    db.commit()
    print(f"  导入 {count} 条运维记录")


def import_knowledge_base(db: Session, csv_path: Path):
    """导入知识库"""
    print(f"导入知识库: {csv_path.name}")
    count = 0
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_id = row.get("doc_id", "").strip()
            if not doc_id:
                continue
            
            existing = db.get(models.KnowledgeBase, doc_id)
            if existing:
                print(f"  文档已存在: {doc_id}")
                continue
            
            category_str = row.get("category", "手册").strip()
            # 映射中文字符到枚举键
            category_map = {
                "手册": models.KnowledgeCategory.MANUAL,
                "案例": models.KnowledgeCategory.CASE,
                "标准": models.KnowledgeCategory.STANDARD,
                # 也支持英文键名（向后兼容）
                "MANUAL": models.KnowledgeCategory.MANUAL,
                "CASE": models.KnowledgeCategory.CASE,
                "STANDARD": models.KnowledgeCategory.STANDARD,
            }
            # 先尝试直接映射（支持中文）
            category = category_map.get(category_str)
            if category is None:
                # 如果直接映射失败，尝试使用枚举键名（英文大写）
                try:
                    category = models.KnowledgeCategory[category_str.upper()]
                except KeyError:
                    # 默认使用 MANUAL
                    category = models.KnowledgeCategory.MANUAL
            
            doc = models.KnowledgeBase(
                doc_id=doc_id,
                applicable_model=row.get("applicable_model", "").strip() or None,
                category=category,
                title=row.get("title", "").strip() or None,
                content_chunk=row.get("content_chunk", "").strip(),
            )
            db.add(doc)
            count += 1
    
    db.commit()
    print(f"  导入 {count} 条知识库记录")


def main():
    """主函数：导入所有CSV数据"""
    print("=" * 60)
    print("开始导入CSV数据到数据库")
    print("=" * 60)
    
    # 确保CSV目录存在
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    
    # 测试数据库连接
    print("测试数据库连接...")
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✓ 数据库连接成功")
    except Exception as e:
        print(f"✗ 数据库连接失败: {e}")
        return
    
    # 创建数据库表
    print("\n创建数据库表...")
    try:
        Base.metadata.create_all(bind=engine)
        print("✓ 数据库表创建完成")
    except Exception as e:
        print(f"✗ 创建数据库表失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print()
    db = SessionLocal()
    try:
        # 按依赖顺序导入
        csv_path = CSV_DIR / "assets.csv"
        if csv_path.exists():
            import_assets(db, csv_path)
        else:
            print(f"警告: 未找到文件 {csv_path}")
        
        csv_path = CSV_DIR / "metric_definitions.csv"
        if csv_path.exists():
            import_metric_definitions(db, csv_path)
        else:
            print(f"警告: 未找到文件 {csv_path}")
        
        csv_path = CSV_DIR / "telemetry_process.csv"
        if csv_path.exists():
            import_telemetry_process(db, csv_path)
        else:
            print(f"警告: 未找到文件 {csv_path}")
        
        csv_path = CSV_DIR / "telemetry_waveform.csv"
        if csv_path.exists():
            import_telemetry_waveform(db, csv_path)
        else:
            print(f"警告: 未找到文件 {csv_path}")
        
        csv_path = CSV_DIR / "maintenance_records.csv"
        if csv_path.exists():
            import_maintenance_records(db, csv_path)
        else:
            print(f"警告: 未找到文件 {csv_path}")
        
        csv_path = CSV_DIR / "knowledge_base.csv"
        if csv_path.exists():
            import_knowledge_base(db, csv_path)
        else:
            print(f"警告: 未找到文件 {csv_path}")
        
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

