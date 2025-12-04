"""
EMSforAI CSV数据导入入口脚本

便捷的CSV数据导入入口，调用backend模块的导入功能。
支持从data/csv_v2目录导入所有CSV文件到数据库。

使用方法：
    python import_csv.py

Author: EMSforAI Team
License: MIT
"""
# coding: utf-8

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backend.import_csv_data import main

if __name__ == "__main__":
    main()

