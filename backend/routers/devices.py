"""
EMSforAI 设备管理API路由模块

本模块提供设备资产管理相关的RESTful API接口，包括：
- 创建设备资产
- 查询设备列表

Author: EMSforAI Team
License: MIT
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import models, schemas
from ..db import SessionLocal

router = APIRouter(prefix="/devices", tags=["设备管理"])


def get_db_session():
    """
    数据库会话依赖注入函数
    
    用于FastAPI的依赖注入系统，为每个请求提供独立的数据库会话。
    请求结束后自动关闭会话，确保资源正确释放。
    
    Yields:
        Session: SQLAlchemy数据库会话对象
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/", status_code=status.HTTP_201_CREATED)
def create_device(payload: schemas.DeviceCreate, db: Session = Depends(get_db_session)):
    """
    创建设备资产
    
    在系统中注册一个新的设备资产，包括设备名称、型号、位置等信息。
    设备创建后，可以为其配置测点定义并开始采集数据。
    
    Args:
        payload (DeviceCreate): 设备创建请求体，包含：
            - name: 设备名称/资产编号（必填，将作为asset_id使用）
            - model_id: 关联的设备型号ID（必填，整数格式，内部转换为字符串）
            - location: 设备位置/产线/区域（可选）
            - status: 设备状态（可选，默认"active"）
        db (Session): 数据库会话（自动注入）
    
    Returns:
        Dict: 创建成功的设备信息，包含：
            - asset_id: 设备资产ID（字符串格式，与name相同）
            - name: 设备名称
            - model_id: 设备型号ID（字符串格式）
    
    Raises:
        HTTPException 400: 设备已存在（asset_id重复）
        HTTPException 500: 数据库操作失败
    
    Note:
        - asset_id使用设备名称（name字段），确保名称唯一性
        - model_id在数据库中存储为字符串格式，但API接受整数格式以保持兼容性
        - 设备创建后状态默认为"active"（活跃）
    
    Example:
        ```json
        POST /devices/
        {
            "name": "CNC-MAZAK-01",
            "model_id": 1,
            "location": "产线A",
            "status": "active"
        }
        ```
    """
    # 步骤1: 检查设备是否已存在
    # 使用设备名称作为asset_id，确保唯一性
    asset_id = payload.name
    exists_stmt = select(models.Asset).where(models.Asset.asset_id == asset_id)
    if db.execute(exists_stmt).scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"设备 {asset_id} 已存在"
        )

    # 步骤2: 创建设备资产记录
    # 注意：model_id在数据库中为字符串类型，需要转换
    asset = models.Asset(
        asset_id=asset_id,
        name=payload.name,
        model_id=str(payload.model_id),  # 将整数转换为字符串
        location=payload.location,
        status=payload.status or "active"  # 默认状态为"active"
    )
    
    # 步骤3: 保存到数据库
    db.add(asset)
    db.commit()
    db.refresh(asset)
    
    return {
        "asset_id": asset.asset_id,
        "name": asset.name,
        "model_id": asset.model_id
    }


@router.get("/")
def list_devices(db: Session = Depends(get_db_session)):
    """
    列出所有设备资产
    
    查询系统中所有已注册的设备资产，返回设备的基本信息列表。
    适用于设备管理页面、设备选择器等场景。
    
    Args:
        db (Session): 数据库会话（自动注入）
    
    Returns:
        List[Dict]: 设备列表，每个元素包含：
            - asset_id: 设备资产ID（字符串格式）
            - name: 设备名称
            - model_id: 设备型号ID（字符串格式）
            - location: 设备位置/产线/区域
            - status: 设备状态（如"active"、"inactive"等）
    
    Note:
        - 返回所有设备，无论状态如何
        - 如需过滤特定状态的设备，建议在前端或通过其他接口实现
        - 返回的model_id为字符串格式，与数据库存储格式一致
    
    Example:
        ```bash
        GET /devices/
        ```
    """
    # 查询所有设备资产
    stmt = select(models.Asset)
    rows = db.execute(stmt).scalars().all()
    
    # 转换为字典列表格式
    return [{
        "asset_id": r.asset_id,
        "name": r.name,
        "model_id": r.model_id,
        "location": r.location,
        "status": r.status
    } for r in rows]
