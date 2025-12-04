"""
LSTM训练可视化模块

包含训练曲线、预测散点图等可视化功能。

Author: EMSforAI Team
License: MIT
"""
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

# 尝试导入matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    # 设置中文字体（使用更兼容的字体列表）
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False  # 使用ASCII减号而不是Unicode减号
    # 禁用字体警告
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

log = logging.getLogger(__name__)


def plot_training_curves(
    history: Dict[str, List[float]],
    metrics: Dict[str, Dict[str, float]],
    save_path: Path,
    asset_id: str,
    metric_id: str,
    y_true_train: Optional[np.ndarray] = None,
    y_pred_train: Optional[np.ndarray] = None,
    y_true_val: Optional[np.ndarray] = None,
    y_pred_val: Optional[np.ndarray] = None,
):
    """
    绘制训练曲线图
    
    Args:
        history: 训练历史（包含train_loss和val_loss）
        metrics: 评估指标（包含训练集和验证集的MSE、MAE、R2）
        save_path: 保存路径
        asset_id: 设备ID
        metric_id: 测点ID
        y_true_train: 训练集真实值（可选）
        y_pred_train: 训练集预测值（可选）
        y_true_val: 验证集真实值（可选）
        y_pred_val: 验证集预测值（可选）
    """
    if not MATPLOTLIB_AVAILABLE:
        log.warning("matplotlib不可用，跳过绘图")
        return
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 计算平滑损失（移动平均，窗口大小为3）
    def smooth_curve(data, window_size=3):
        """使用移动平均平滑曲线"""
        if len(data) < window_size:
            return data
        smoothed = []
        for i in range(len(data)):
            start = max(0, i - window_size // 2)
            end = min(len(data), i + window_size // 2 + 1)
            smoothed.append(np.mean(data[start:end]))
        return np.array(smoothed)
    
    train_loss_smooth = smooth_curve(history["train_loss"], window_size=5)
    val_loss_smooth = smooth_curve(history["val_loss"], window_size=5)
    
    epochs_range = range(1, len(history["train_loss"]) + 1)
    
    # 1. 损失曲线（原始 + 平滑，对数刻度）
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs_range, history["train_loss"], 'b-', label='训练损失（原始）', linewidth=1, alpha=0.3)
    ax1.plot(epochs_range, history["val_loss"], 'r-', label='验证损失（原始）', linewidth=1, alpha=0.3)
    ax1.plot(epochs_range, train_loss_smooth, 'b-', label='训练损失（平滑）', linewidth=2.5, alpha=0.9)
    ax1.plot(epochs_range, val_loss_smooth, 'r-', label='验证损失（平滑）', linewidth=2.5, alpha=0.9)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (MSE)', fontsize=12)
    ax1.set_title('训练损失曲线（平滑）', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # 使用对数刻度，更好地显示变化
    
    # 2. 损失曲线（原始 + 平滑，线性刻度）
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs_range, history["train_loss"], 'b-', label='训练损失（原始）', linewidth=1, alpha=0.3)
    ax2.plot(epochs_range, history["val_loss"], 'r-', label='验证损失（原始）', linewidth=1, alpha=0.3)
    ax2.plot(epochs_range, train_loss_smooth, 'b-', label='训练损失（平滑）', linewidth=2.5, alpha=0.9)
    ax2.plot(epochs_range, val_loss_smooth, 'r-', label='验证损失（平滑）', linewidth=2.5, alpha=0.9)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss (MSE)', fontsize=12)
    ax2.set_title('训练损失曲线（线性）', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. 指标对比（柱状图）
    ax3 = fig.add_subplot(gs[0, 2])
    categories = ['MSE', 'MAE (天)', 'R2']
    train_values = [
        metrics["train"]["mse"],
        metrics["train"]["mae"],
        metrics["train"]["r2"]
    ]
    val_values = [
        metrics["val"]["mse"],
        metrics["val"]["mae"],
        metrics["val"]["r2"]
    ]
    x = np.arange(len(categories))
    width = 0.35
    bars1 = ax3.bar(x - width/2, train_values, width, label='训练集', alpha=0.8, color='#3498db')
    bars2 = ax3.bar(x + width/2, val_values, width, label='验证集', alpha=0.8, color='#e74c3c')
    ax3.set_ylabel('数值', fontsize=12)
    ax3.set_title('模型性能指标对比', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories, fontsize=10)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}' if abs(height) < 1 else f'{height:.1f}',
                    ha='center', va='bottom', fontsize=9)
    
    # 4. R2指标
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.bar(['训练集', '验证集'], 
            [metrics["train"]["r2"], metrics["val"]["r2"]],
            color=['#3498db', '#e74c3c'], alpha=0.8)
    ax4.set_ylabel('R2 分数', fontsize=12)
    ax4.set_title('R2 决定系数', fontsize=14, fontweight='bold')
    ax4.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax4.grid(True, alpha=0.3, axis='y')
    # 添加数值标签
    for i, (label, value) in enumerate(zip(['训练集', '验证集'], 
                                            [metrics["train"]["r2"], metrics["val"]["r2"]])):
        ax4.text(i, value, f'{value:.4f}', ha='center', 
                va='bottom' if value >= 0 else 'top', fontsize=11, fontweight='bold')
    
    # 5. MAE指标
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.bar(['训练集', '验证集'],
            [metrics["train"]["mae"], metrics["val"]["mae"]],
            color=['#3498db', '#e74c3c'], alpha=0.8)
    ax5.set_ylabel('MAE (天)', fontsize=12)
    ax5.set_title('平均绝对误差', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    # 添加数值标签
    for i, (label, value) in enumerate(zip(['训练集', '验证集'],
                                            [metrics["train"]["mae"], metrics["val"]["mae"]])):
        ax5.text(i, value, f'{value:.2f}天', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 6. 预测vs真实值散点图（验证集，带多项式拟合）
    ax6 = fig.add_subplot(gs[1, 2])
    if y_true_val is not None and y_pred_val is not None:
        ax6.scatter(y_true_val, y_pred_val, alpha=0.5, s=30, edgecolors='black', linewidth=0.3, 
                   label='LSTM预测点', zorder=3)
        min_val = min(y_true_val.min(), y_pred_val.min())
        max_val = max(y_true_val.max(), y_pred_val.max())
        ax6.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2.5, 
                label='完美预测参考线 (y=x)', alpha=0.8, zorder=2)
        
        ax6.set_xlabel('真实RUL (天)', fontsize=10, fontweight='bold')
        ax6.set_ylabel('LSTM预测RUL (天)', fontsize=10, fontweight='bold')
        ax6.set_title(
            f'验证集预测效果\nR2 = {metrics["val"]["r2"]:.3f} | 散点是LSTM预测结果',
            fontsize=11, fontweight='bold'
        )
        ax6.legend(fontsize=7, loc='lower right', framealpha=0.9)
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, '预测vs真实值散点图\n（数据未提供）', 
                ha='center', va='center', fontsize=12, transform=ax6.transAxes)
        ax6.set_title('预测效果', fontsize=14, fontweight='bold')
        ax6.axis('off')
    
    # 添加总标题
    fig.suptitle(
        f'LSTM模型训练报告\n设备: {asset_id} | 测点: {metric_id}',
        fontsize=16, fontweight='bold', y=0.98
    )
    
    # 保存图片
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log.info(f"训练曲线图已保存到: {save_path}")


def plot_prediction_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path,
    asset_id: str,
    metric_id: str,
    dataset_name: str = "验证集",
):
    """
    绘制预测值vs真实值散点图（带多项式拟合曲线）
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        save_path: 保存路径
        asset_id: 设备ID
        metric_id: 测点ID
        dataset_name: 数据集名称
    """
    if not MATPLOTLIB_AVAILABLE:
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 散点图（这是LSTM的实际预测结果）
    ax.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors='black', linewidth=0.5, 
               label='LSTM预测点（每个点代表一个样本）', zorder=3)
    
    # 添加完美预测参考线（y=x）- 这是理想状态的参考线，不是预测结果
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2.5, 
            label='完美预测参考线 (y=x)\n（理想状态，所有点应落在此线上）', alpha=0.8, zorder=2)
    
    # 计算R2
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    
    ax.set_xlabel('真实RUL (天)', fontsize=12, fontweight='bold')
    ax.set_ylabel('LSTM预测RUL (天)', fontsize=12, fontweight='bold')
    ax.set_title(
        f'{dataset_name} - LSTM预测效果评估\n'
        f'R2 = {r2:.4f} | 说明：散点是LSTM的实际预测结果，y=x是完美预测参考线',
        fontsize=13, fontweight='bold'
    )
    ax.legend(fontsize=9, loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # 添加统计信息文本框
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    textstr = f'MAE: {mae:.2f} 天\nRMSE: {rmse:.2f} 天\nR2: {r2:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log.info(f"预测散点图已保存到: {save_path}")

