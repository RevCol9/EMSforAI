# EMSforAI

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**设备管理系统 (Equipment Management System) 的 AI 算法引擎**

基于 FastAPI 构建，提供设备健康度分析、剩余使用寿命预测、停机风险评估等功能。

[功能特性](#功能特性) • [快速开始](#快速开始) • [API 文档](#api-接口) • [贡献指南](CONTRIBUTING.md) • [许可证](LICENSE)

</div>

---

## 功能特性

- 🎯 **设备健康度分析**：基于多指标加权评估设备综合健康状态，支持雷达图可视化
- 📈 **RUL 预测**：使用 LSTM 神经网络和传统回归模型预测设备剩余使用寿命
- ⚠️ **智能告警系统**：支持三级告警（正常/警告/严重），及时发现问题
- 📊 **风险评估**：计算停机风险和产能影响，辅助决策制定
- 🔧 **备件管理**：跟踪备件使用周期，预测更换时间
- 🚀 **RESTful API**：完整的 FastAPI 接口，自动生成交互式 API 文档
- 🧠 **深度学习支持**：基于 PyTorch 的 LSTM 模型，支持多变量时间序列预测

## 技术栈

- **后端框架**：FastAPI 0.110+
- **数据库**：SQLAlchemy ORM，支持 SQLite/PostgreSQL
- **数据处理**：Pandas, NumPy, SciPy
- **机器学习**：scikit-learn, PyTorch
- **深度学习**：PyTorch (LSTM 神经网络)
- **数据可视化**：Matplotlib (训练曲线、预测散点图)
- **环境管理**：Conda

## 项目结构

```
EMSforAI/
├── backend/                    # 后端代码
│   ├── algorithm/             # 算法引擎核心模块
│   │   ├── algorithms_engine.py  # 主算法引擎（健康度、RUL预测）
│   │   ├── lstm_model.py      # LSTM神经网络模型定义
│   │   ├── train.py           # LSTM模型训练脚本
│   │   ├── training_utils.py  # 训练数据准备工具
│   │   ├── visualization.py   # 训练可视化工具
│   │   ├── data_service.py    # 数据加载和预处理
│   │   ├── constants.py       # 算法常量定义
│   │   ├── base.py            # 算法引擎抽象基类
│   │   └── demo_runner.py     # 演示脚本
│   ├── routers/               # API 路由模块
│   │   ├── devices.py         # 设备管理接口
│   │   └── analysis.py        # 算法分析接口
│   ├── models.py              # 数据库模型定义（SQLAlchemy）
│   ├── schemas.py             # Pydantic 数据验证模型
│   ├── db.py                  # 数据库连接配置
│   └── main.py                # FastAPI 应用入口
├── data/                      # 数据文件目录
│   ├── csv/                   # CSV 格式示例数据
│   └── *.json                 # JSON 格式示例数据
├── models/                    # 训练好的模型
│   └── lstm/                  # LSTM模型文件（.pt格式）
│       └── plots/             # 训练曲线和预测散点图
├── docs/                      # 项目文档
├── environment.yml            # Conda 环境配置
├── docker-compose.yml         # Docker 编排配置
├── CONTRIBUTING.md            # 贡献指南
└── README.md                  # 项目说明文档
```

## 快速开始

### 1. 环境准备

创建并激活 Conda 环境：

```bash
conda env create -f environment.yml
conda activate emsforai
```

### 2. 数据库配置

项目默认使用 SQLite，数据库文件会在首次运行时自动创建。

如需使用 PostgreSQL，请设置环境变量：

```bash
export DATABASE_URL="postgresql://user:password@localhost/emsforai"
```

### 3. 导入示例数据（可选）

```bash
python import_csv.py
# 或
python -m backend.import_csv_data
```

### 4. 启动服务

```bash
# 开发模式（自动重载）
fastapi dev backend/main.py

# 或使用 uvicorn
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

服务启动后，访问 `http://localhost:8000/docs` 查看 API 文档。

### 5. 运行算法演示

```bash
python -m backend.algorithm.demo_runner
```

## API 接口

### 设备管理

- `POST /devices/models` - 创建设备型号
- `GET /devices/models` - 列出所有设备型号
- `POST /devices` - 创建设备
- `GET /devices` - 列出所有设备

### 算法分析

- `GET /api/v2/device/{device_id}/health` - 获取设备健康度概览
- `GET /api/v2/device/{device_id}/analysis` - 获取设备完整分析报告
- `GET /api/v2/device/{device_id}/metrics/{metric_key}/analysis` - 获取单指标分析
- `GET /api/v2/devices/analysis` - 获取所有设备分析摘要

### 数据提交

- `POST /api/inspection/submit` - 提交巡检数据

## 算法说明

### 健康度计算

设备健康度基于多个指标的加权平均计算：

1. **单指标健康度**：基于当前值与警告/临界阈值的线性插值，范围 0-100
2. **综合健康度**：加权平均 = Σ(指标健康度 × 基础权重 × 关键程度系数) / Σ(权重)

### 剩余使用寿命 (RUL) 预测

系统支持两种 RUL 预测方法：

#### 1. LSTM 神经网络（推荐）

- 使用多层 LSTM 网络学习时间序列的长期依赖关系
- 支持多变量输入，能够捕捉复杂的非线性趋势
- 自动特征提取，无需手动特征工程
- 训练好的模型保存在 `models/lstm/` 目录

#### 2. 传统回归模型（备用）

- 线性回归、多项式回归（2-4次）、指数回归
- 自动选择最佳模型（基于 R² 分数）
- 使用 Savitzky-Golay 滤波器平滑数据
- 计算指标达到临界阈值所需时间：`RUL = (阈值 - α) / β`
- 预测置信度 = min(1.0, 样本数/30) × R²

**训练 LSTM 模型**：

```bash
python -m backend.algorithm.train \
    --asset_id COMP-ATLAS-01 \
    --metric_id COMP01_OIL_TEMP \
    --sequence_length 30 \
    --lstm_units 128 64 \
    --epochs 32 \
    --batch_size 64
```

### 告警级别

- **normal**：指标值在正常范围内
- **warning**：指标值超过警告阈值
- **critical**：指标值超过临界阈值

## 开发指南

### 添加新的算法引擎

1. 继承 `backend.algorithm.base.Engine` 抽象基类
2. 实现 `load()`, `analyze_metric()`, `analyze_device()` 方法
3. 在 API 路由中实例化并使用

### 扩展数据模型

1. 在 `backend/models.py` 中定义 SQLAlchemy 模型
2. 在 `backend/schemas.py` 中定义 Pydantic 验证模型
3. 更新 `backend/algorithm/data_service.py` 的数据加载逻辑

## 贡献指南

我们欢迎所有形式的贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详细的贡献指南。

### 快速贡献流程

1. ⭐ Fork 本仓库
2. 🌿 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 💻 编写代码并添加中文注释
4. ✅ 确保代码符合项目规范
5. 📝 提交更改 (`git commit -m 'feat: Add some AmazingFeature'`)
6. 🚀 推送到分支 (`git push origin feature/AmazingFeature`)
7. 🔄 开启 Pull Request

### 代码规范

- 所有注释使用中文
- 使用 `black` 格式化代码
- 遵循 PEP 8 代码风格
- 添加类型提示
- 为新功能添加文档说明

更多详情请参考 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可证

本项目采用 [MIT 许可证](LICENSE)。

## 致谢

感谢所有为这个项目做出贡献的开发者！

## 联系方式

- 📧 提交 Issue: [GitHub Issues](https://github.com/your-org/EMSforAI/issues)
- 📖 查看文档: [项目文档](docs/)
- 💬 讨论交流: [GitHub Discussions](https://github.com/your-org/EMSforAI/discussions)

---

<div align="center">

**如果这个项目对您有帮助，请给我们一个 ⭐ Star！**

Made with ❤️ by EMSforAI Team

</div>
