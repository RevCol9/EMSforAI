# 贡献指南

感谢您对 EMSforAI 项目的关注！我们欢迎所有形式的贡献，包括但不限于：

- 🐛 报告 Bug
- 💡 提出新功能建议
- 📝 改进文档
- 🔧 提交代码修复或新功能
- 🎨 改进代码风格和可读性

## 开发环境设置

### 1. Fork 和克隆仓库

```bash
# Fork 本仓库到您的 GitHub 账户
# 然后克隆您的 Fork
git clone https://github.com/your-username/EMSforAI.git
cd EMSforAI
```

### 2. 创建开发环境

```bash
# 使用 Conda 创建环境
conda env create -f environment.yml
conda activate emsforai

# 或使用 pip（如果已安装 Python 3.11+）
pip install -r requirements.txt  # 如果有 requirements.txt
```

### 3. 安装开发依赖

```bash
# 安装代码格式化工具
pip install black isort flake8 mypy

# 安装测试工具（如果项目有测试）
pip install pytest pytest-cov
```

### 4. 配置数据库

项目默认使用 SQLite，数据库文件会在首次运行时自动创建。

如需使用 PostgreSQL：

```bash
export DB_URL="postgresql://user:password@localhost/emsforai"
```

## 开发流程

### 1. 创建分支

```bash
# 从主分支创建新分支
git checkout -b feature/your-feature-name
# 或
git checkout -b fix/your-bug-fix
```

分支命名规范：
- `feature/` - 新功能
- `fix/` - Bug 修复
- `docs/` - 文档改进
- `refactor/` - 代码重构
- `test/` - 测试相关

### 2. 编写代码

#### 代码风格

- **语言**: Python 3.11+
- **注释**: 所有注释使用中文，遵循 Google 风格
- **命名**: 使用有意义的变量和函数名，优先使用英文
- **格式化**: 使用 `black` 进行代码格式化

```bash
# 格式化代码
black backend/
isort backend/
```

#### 注释规范

所有函数、类、模块都应该有详细的中文文档字符串：

```python
def calculate_health_score(metrics: Dict[str, float]) -> float:
    """
    计算设备健康度分数
    
    基于多个指标的加权平均计算设备综合健康度。
    
    Args:
        metrics: 指标字典，格式为 {metric_id: value}
    
    Returns:
        float: 健康度分数（0-100）
    
    Raises:
        ValueError: 当指标字典为空时抛出
    """
    # 实现代码
    pass
```

#### 类型提示

尽量为所有函数添加类型提示：

```python
from typing import Dict, List, Optional

def process_data(
    data: pd.DataFrame,
    threshold: float = 0.5,
    columns: Optional[List[str]] = None
) -> Dict[str, float]:
    """处理数据并返回结果"""
    pass
```

### 3. 测试代码

在提交代码前，请确保：

- ✅ 代码可以正常运行
- ✅ 没有语法错误和明显的逻辑错误
- ✅ 新功能有相应的文档说明
- ✅ 代码符合项目的风格规范

### 4. 提交更改

```bash
# 添加更改的文件
git add .

# 提交更改（使用清晰的提交信息）
git commit -m "feat: 添加新的健康度计算算法"

# 推送到您的 Fork
git push origin feature/your-feature-name
```

#### 提交信息规范

使用 [Conventional Commits](https://www.conventionalcommits.org/) 规范：

- `feat:` - 新功能
- `fix:` - Bug 修复
- `docs:` - 文档更改
- `style:` - 代码格式（不影响代码运行）
- `refactor:` - 代码重构
- `test:` - 测试相关
- `chore:` - 构建过程或辅助工具的变动

示例：
```
feat: 添加LSTM模型支持RUL预测
fix: 修复健康度计算中的除零错误
docs: 更新API文档说明
refactor: 重构数据加载逻辑
```

### 5. 创建 Pull Request

1. 在 GitHub 上打开您的 Fork
2. 点击 "New Pull Request"
3. 填写 PR 描述，包括：
   - 更改的目的和背景
   - 实现的主要功能或修复的问题
   - 测试情况
   - 相关的 Issue 编号（如果有）

## 代码审查标准

提交的代码应该：

1. **功能完整**: 实现的功能符合需求
2. **代码质量**: 代码清晰、易读、易维护
3. **注释完善**: 关键逻辑有中文注释说明
4. **风格一致**: 符合项目的代码风格
5. **向后兼容**: 不破坏现有功能（除非是重大更新）

## 报告问题

### Bug 报告

在 [GitHub Issues](https://github.com/your-org/EMSforAI/issues) 中创建 Issue，包括：

- **问题描述**: 清晰描述问题
- **复现步骤**: 如何复现该问题
- **预期行为**: 应该发生什么
- **实际行为**: 实际发生了什么
- **环境信息**: Python 版本、操作系统、依赖版本等
- **错误日志**: 如果有错误信息，请附上

### 功能建议

在 Issues 中创建 Feature Request，包括：

- **功能描述**: 详细描述新功能
- **使用场景**: 为什么需要这个功能
- **可能的实现**: 如果有想法，可以描述实现思路

## 项目结构说明

```
EMSforAI/
├── backend/              # 后端代码
│   ├── algorithm/        # 算法引擎
│   ├── routers/         # API 路由
│   ├── models.py        # 数据库模型
│   ├── schemas.py       # API 数据模型
│   └── main.py          # 应用入口
├── data/                # 示例数据
├── docs/                # 文档
├── models/              # 训练好的模型
└── README.md            # 项目说明
```

## 获取帮助

如果您在贡献过程中遇到问题：

1. 查看 [文档](docs/)
2. 搜索已有的 [Issues](https://github.com/your-org/EMSforAI/issues)
3. 创建新的 Issue 描述您的问题

## 行为准则

我们致力于为每个人提供开放和欢迎的环境。请遵守以下行为准则：

- 使用友好和包容的语言
- 尊重不同的观点和经验
- 优雅地接受建设性批评
- 关注对社区最有利的事情
- 对其他社区成员表示同理心

感谢您的贡献！🎉

