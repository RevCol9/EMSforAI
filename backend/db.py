"""
EMSforAI 数据库配置模块

本模块负责数据库连接和会话管理，支持SQLite和PostgreSQL数据库。
使用SQLAlchemy ORM进行数据库操作，提供统一的数据库接口。

配置说明：
- 默认使用SQLite数据库（emsforai.db）
- 可通过环境变量DB_URL或DATABASE_URL配置数据库连接
- 支持PostgreSQL、MySQL等关系型数据库

Author: EMSforAI Team
License: MIT
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

# 加载环境变量（从.env文件）
load_dotenv()

# 数据库连接URL配置
# 优先级：DB_URL > DATABASE_URL > 默认SQLite
# 示例：
#   SQLite: sqlite:///emsforai.db
#   PostgreSQL: postgresql://user:password@localhost/emsforai
#   MySQL: mysql+pymysql://user:password@localhost/emsforai
db_url = os.getenv("DB_URL") or os.getenv("DATABASE_URL") or "sqlite:///emsforai.db"

# 创建数据库引擎
# SQLite需要设置check_same_thread=False以支持多线程
# 其他数据库不需要此参数
engine = create_engine(
    db_url,
    connect_args={"check_same_thread": False} if db_url.startswith("sqlite") else {}
)

# 创建会话工厂
# autocommit=False: 需要手动提交事务
# autoflush=False: 需要手动刷新会话
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 声明式基类，所有ORM模型都继承自此类
Base = declarative_base()
