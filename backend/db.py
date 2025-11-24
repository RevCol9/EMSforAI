import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

load_dotenv()
# Prefer DB_URL; fall back to DATABASE_URL for compatibility
db_url = os.getenv("DB_URL") or os.getenv("DATABASE_URL") or "sqlite:///emsforai.db"
engine = create_engine(db_url, connect_args={"check_same_thread": False} if db_url.startswith("sqlite") else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
