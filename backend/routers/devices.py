from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import models, schemas
from ..db import SessionLocal

router = APIRouter(prefix="/devices", tags=["devices"])


def get_db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/models", status_code=status.HTTP_201_CREATED)
def create_device_model(payload: schemas.DeviceModelCreate, db: Session = Depends(get_db_session)):
    exists_stmt = select(models.DeviceModel).where(models.DeviceModel.name == payload.name)
    if db.execute(exists_stmt).scalar_one_or_none():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Model already exists")

    model = models.DeviceModel(name=payload.name)
    db.add(model)
    db.commit()
    db.refresh(model)
    return {"id": model.id, "name": model.name}


@router.get("/models")
def list_device_models(db: Session = Depends(get_db_session)):
    stmt = select(models.DeviceModel)
    rows = db.execute(stmt).scalars().all()
    return [{"id": r.id, "name": r.name} for r in rows]


@router.post("/", status_code=status.HTTP_201_CREATED)
def create_device(payload: schemas.DeviceCreate, db: Session = Depends(get_db_session)):
    model = db.get(models.DeviceModel, payload.model_id)
    if not model:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Device model not found")

    device = models.Device(name=payload.name, model_id=payload.model_id)
    db.add(device)
    db.commit()
    db.refresh(device)
    return {"id": device.id, "name": device.name, "model_id": device.model_id}


@router.get("/")
def list_devices(db: Session = Depends(get_db_session)):
    stmt = select(models.Device)
    rows = db.execute(stmt).scalars().all()
    return [{"id": r.id, "name": r.name, "model_id": r.model_id} for r in rows]
