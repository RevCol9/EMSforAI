"""
训练LSTM神经网络模型进行设备剩余使用寿命（RUL）预测。

主要功能：
- 从数据库加载训练数据
- 准备时间序列训练数据（序列化、归一化等）
- 构建和训练LSTM模型
- 评估模型性能（MSE、MAE、R²等）
- 保存训练好的模型和可视化图表

使用方法：
    python -m backend.algorithm.train --asset_id COMP-ATLAS-01 --metric_id COMP01_OIL_TEMP

Author: EMSforAI Team
License: MIT
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, List
import pandas as pd
from datetime import datetime
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from backend.algorithm.data_service import load_data_from_db
from backend.algorithm.training_utils import (
    prepare_training_data,
    MAX_RUL_DAYS,
    print_section_header,
    print_info_box,
)
from backend.algorithm.lstm_model import LSTMRULPredictor, MultiVariateLSTMPredictor
from backend.algorithm.training_utils_multivariate import prepare_multivariate_training_data
from backend.algorithm.visualization import (
    plot_training_curves,
    plot_prediction_scatter,
    MATPLOTLIB_AVAILABLE,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def train_single_metric(
    asset_id: str,
    metric_id: str,
    sequence_length: int = 30,
    lstm_units: list = [64, 32],
    epochs: int = 32,
    batch_size: int = 64,
    model_save_path: Optional[str] = None,
    save_plots: bool = True,
):
    """
    训练单个测点的LSTM模型
    
    Args:
        asset_id: 设备ID
        metric_id: 测点ID
        sequence_length: 序列长度
        lstm_units: LSTM层单元数
        epochs: 训练轮数
        batch_size: 批次大小
        model_save_path: 模型保存路径
        save_plots: 是否保存训练曲线图
    """
    start_time = datetime.now()
    
    print_section_header("🚀 LSTM模型训练", "═", 70)
    print_info_box("📋 训练配置", {
        "设备ID": asset_id,
        "测点ID": metric_id,
        "序列长度": sequence_length,
        "LSTM单元": f"{lstm_units}",
        "训练轮数": epochs,
        "批次大小": batch_size,
    })
    
    # 加载数据
    print_section_header("📊 步骤 1/5: 加载数据", "─", 70)
    try:
        data = load_data_from_db(asset_id=asset_id)
        print("✓ 数据加载完成")
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 准备训练数据
    print_section_header("🔧 步骤 2/5: 准备训练数据", "─", 70)
    try:
        X_train, y_train, X_val, y_val, scaler = prepare_training_data(
            data, asset_id, metric_id, sequence_length=sequence_length
        )
        print("✓ 训练数据准备完成")
    except Exception as e:
        print(f"✗ 数据准备失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 构建模型
    print_section_header("🏗️  步骤 3/5: 构建LSTM模型", "─", 70)
    predictor = LSTMRULPredictor(
        sequence_length=sequence_length,
        n_features=1,
        lstm_units=lstm_units,
    )
    predictor.scaler = scaler
    
    model = predictor.build_model()
    if model is None:
        print("✗ 模型构建失败")
        return
    
    total_params = sum(p.numel() for p in model.parameters())
    print_info_box("📐 模型结构", {
        "参数数量": f"{total_params:,}",
        "LSTM层": f"{len(lstm_units)}层",
        "LSTM单元": f"{lstm_units}",
        "Dropout率": "0.2",
    })
    
    # 训练模型
    print_section_header("🎯 步骤 4/5: 训练模型", "─", 70)
    
    try:
        history = predictor.train(
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
        )
        print("\n✓ 模型训练完成")
        
        # 显示训练历史摘要
        if history:
            final_train_loss = history.get("train_loss", [])[-1] if history.get("train_loss") else None
            final_val_loss = history.get("val_loss", [])[-1] if history.get("val_loss") else None
            best_epoch = np.argmin(history["val_loss"]) + 1 if history.get("val_loss") else len(history["train_loss"])
            
            print_info_box("📈 训练摘要", {
                "最佳Epoch": best_epoch,
                "最终训练损失": f"{final_train_loss:.6f}" if final_train_loss else "N/A",
                "最终验证损失": f"{final_val_loss:.6f}" if final_val_loss else "N/A",
                "训练轮数": len(history["train_loss"]),
            })
        
        # 评估模型
        print_section_header("📊 步骤 5/5: 评估模型", "─", 70)
        # 预测得到的是归一化后的RUL，需要还原到"天"
        y_pred_train_scaled = predictor.predict(X_train)
        y_pred_val_scaled = predictor.predict(X_val)

        y_train_days = y_train * MAX_RUL_DAYS
        y_val_days = y_val * MAX_RUL_DAYS
        y_pred_train_days = y_pred_train_scaled * MAX_RUL_DAYS
        y_pred_val_days = y_pred_val_scaled * MAX_RUL_DAYS
        
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        train_mse = mean_squared_error(y_train_days, y_pred_train_days)
        train_mae = mean_absolute_error(y_train_days, y_pred_train_days)
        train_r2 = r2_score(y_train_days, y_pred_train_days)
        train_rmse = np.sqrt(train_mse)
        
        val_mse = mean_squared_error(y_val_days, y_pred_val_days)
        val_mae = mean_absolute_error(y_val_days, y_pred_val_days)
        val_r2 = r2_score(y_val_days, y_pred_val_days)
        val_rmse = np.sqrt(val_mse)
        
        metrics = {
            "train": {
                "mse": train_mse,
                "mae": train_mae,
                "r2": train_r2,
                "rmse": train_rmse,
            },
            "val": {
                "mse": val_mse,
                "mae": val_mae,
                "r2": val_r2,
                "rmse": val_rmse,
            }
        }
        
        print_info_box("🎯 训练集指标", {
            "MSE": f"{train_mse:.4f}",
            "RMSE": f"{train_rmse:.2f} 天",
            "MAE": f"{train_mae:.2f} 天",
            "R2": f"{train_r2:.4f}",
        })
        
        print_info_box("🎯 验证集指标", {
            "MSE": f"{val_mse:.4f}",
            "RMSE": f"{val_rmse:.2f} 天",
            "MAE": f"{val_mae:.2f} 天",
            "R2": f"{val_r2:.4f}",
        })
        
        # 性能评价
        print_section_header("💡 性能评价", "─", 70)
        if val_r2 > 0.8:
            performance = "优秀 ⭐⭐⭐⭐⭐"
            color_indicator = "🟢"
        elif val_r2 > 0.6:
            performance = "良好 ⭐⭐⭐⭐"
            color_indicator = "🟡"
        elif val_r2 > 0.4:
            performance = "一般 ⭐⭐⭐"
            color_indicator = "🟠"
        elif val_r2 > 0:
            performance = "较差 ⭐⭐"
            color_indicator = "🔴"
        else:
            performance = "很差 ⭐"
            color_indicator = "🔴"
        
        print(f"{color_indicator} 模型性能: {performance}")
        print(f"   验证集R2 = {val_r2:.4f}")
        print(f"   平均误差 = {val_mae:.2f} 天")
        print()
        
        # 保存模型和图表
        print_section_header("💾 保存结果", "─", 70)
        
        # 保存模型
        if model_save_path:
            model_path = Path(model_save_path)
        else:
            models_dir = BASE_DIR / "models" / "lstm"
            models_dir.mkdir(parents=True, exist_ok=True)
            model_path = models_dir / f"{asset_id}_{metric_id}_lstm.pt"
        
        predictor.save_model(str(model_path))
        print(f"✓ 模型已保存: {model_path}")
        
        # 保存训练曲线图
        if save_plots and MATPLOTLIB_AVAILABLE:
            plots_dir = BASE_DIR / "models" / "lstm" / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            # 训练曲线图（包含预测散点图）
            plot_path = plots_dir / f"{asset_id}_{metric_id}_training_curves.png"
            plot_training_curves(
                history, metrics, plot_path, asset_id, metric_id,
                y_true_train=y_train_days, y_pred_train=y_pred_train_days,
                y_true_val=y_val_days, y_pred_val=y_pred_val_days
            )
            
            # 预测散点图（训练集）
            scatter_train_path = plots_dir / f"{asset_id}_{metric_id}_scatter_train.png"
            plot_prediction_scatter(
                y_train_days, y_pred_train_days,
                scatter_train_path, asset_id, metric_id, "训练集"
            )
            
            # 预测散点图（验证集）
            scatter_val_path = plots_dir / f"{asset_id}_{metric_id}_scatter_val.png"
            plot_prediction_scatter(
                y_val_days, y_pred_val_days,
                scatter_val_path, asset_id, metric_id, "验证集"
            )
        elif save_plots and not MATPLOTLIB_AVAILABLE:
            print("⚠️  matplotlib不可用，跳过绘图")
        
        # 训练总结
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print_section_header("✅ 训练完成", "═", 70)
        print_info_box("⏱️  训练信息", {
            "开始时间": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "结束时间": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "训练时长": f"{duration:.1f} 秒",
            "模型文件": str(model_path),
        })
        
        if save_plots and MATPLOTLIB_AVAILABLE:
            print(f"\n📊 训练曲线图: {plots_dir / f'{asset_id}_{metric_id}_training_curves.png'}")
            print(f"📊 预测散点图: {plots_dir / f'{asset_id}_{metric_id}_scatter_*.png'}")
        
        print()
        
    except Exception as e:
        print(f"\n✗ 训练失败: {e}")
        import traceback
        traceback.print_exc()


def train_multivariate(
    asset_id: str,
    sequence_length: int = 30,
    lstm_units: list = [128, 64],
    epochs: int = 50,
    batch_size: int = 64,
    model_save_path: Optional[str] = None,
    save_plots: bool = True,
):
    """
    训练多变量LSTM模型（一个设备一个模型）
    
    自动从数据库获取设备的所有PROCESS类型测点进行训练。
    多变量模型可以同时学习多个测点之间的关系，通常比单测点模型有更好的预测性能。
    
    训练流程：
    1. 从数据库加载设备数据
    2. 自动获取设备的所有PROCESS类型测点（有临界阈值的）
    3. 准备多变量训练数据（对齐时间戳、标准化等）
    4. 构建多变量LSTM模型
    5. 训练模型（支持早停）
    6. 评估模型性能
    7. 保存模型和scaler
    
    Args:
        asset_id: 设备ID，例如"COMP-ATLAS-01"
        sequence_length: 序列长度，即时间窗口大小（默认30）
        lstm_units: LSTM层单元数列表，例如[128, 64]表示两层LSTM（默认[128, 64]）
        epochs: 训练轮数（默认50）
        batch_size: 批次大小（默认64）
        model_save_path: 模型保存路径，如果为None则使用默认路径（models/lstm/{asset_id}_multivariate_lstm.pt）
        save_plots: 是否保存训练曲线图（默认True）
    
    Note:
        - 模型会自动保存所有测点的scaler，确保推理时数据预处理一致
        - 训练过程中会显示详细的进度信息和性能指标
        - 如果验证集R² > 0.8，模型性能评价为"优秀"
    """
    start_time = datetime.now()
    
    # 加载数据
    print_section_header("📊 步骤 1/6: 加载数据", "─", 70)
    try:
        data = load_data_from_db(asset_id=asset_id)
        print("✓ 数据加载完成")
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 自动获取设备的所有PROCESS类型测点
    print_section_header("🔍 步骤 2/6: 获取测点列表", "─", 70)
    try:
        metric_defs = data.get("metric_definitions", pd.DataFrame())
        if metric_defs.empty:
            print("✗ 未找到测点定义数据")
            return
        
        # 过滤出该设备的PROCESS类型测点
        asset_metrics = metric_defs[
            (metric_defs["asset_id"] == asset_id) &
            (metric_defs["metric_type"] == "PROCESS")
        ]
        
        if asset_metrics.empty:
            print(f"✗ 设备 {asset_id} 没有PROCESS类型的测点")
            return
        
        # 只选择有临界阈值的测点（用于RUL计算）
        asset_metrics = asset_metrics[asset_metrics["crit_threshold"].notna()]
        
        if asset_metrics.empty:
            print(f"✗ 设备 {asset_id} 没有配置临界阈值的测点")
            return
        
        # 获取测点ID列表
        metric_ids = asset_metrics["metric_id"].tolist()
        
        print(f"✓ 找到 {len(metric_ids)} 个PROCESS类型测点:")
        for i, metric_id in enumerate(metric_ids, 1):
            metric_name = asset_metrics[asset_metrics["metric_id"] == metric_id].iloc[0]["metric_name"]
            print(f"  {i}. {metric_id} ({metric_name})")
        
    except Exception as e:
        print(f"✗ 获取测点列表失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print_section_header("🚀 多变量LSTM模型训练", "═", 70)
    print_info_box("📋 训练配置", {
        "设备ID": asset_id,
        "测点数量": len(metric_ids),
        "测点列表": ", ".join(metric_ids[:5]) + (f" ... (共{len(metric_ids)}个)" if len(metric_ids) > 5 else ""),
        "序列长度": sequence_length,
        "LSTM单元": f"{lstm_units}",
        "训练轮数": epochs,
        "批次大小": batch_size,
    })
    
    # 准备训练数据
    print_section_header("🔧 步骤 3/6: 准备多变量训练数据", "─", 70)
    try:
        X_train, y_train, X_val, y_val, scalers, feature_names = prepare_multivariate_training_data(
            data, asset_id, metric_ids, sequence_length=sequence_length
        )
        print("✓ 多变量训练数据准备完成")
    except Exception as e:
        print(f"✗ 训练数据准备失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 构建模型
    print_section_header("🏗️  步骤 4/6: 构建模型", "─", 70)
    try:
        predictor = MultiVariateLSTMPredictor(
            sequence_length=sequence_length,
            n_features=len(feature_names),
            lstm_units=lstm_units,
        )
        predictor.build_model()
        predictor.scalers = scalers  # 保存所有测点的scaler字典
        predictor.feature_names = feature_names  # 保存特征名称列表
        print("✓ 模型构建完成")
    except Exception as e:
        print(f"✗ 模型构建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 训练模型
    print_section_header("🎓 步骤 5/6: 训练模型", "─", 70)
    try:
        history = predictor.train(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            patience=20,
        )
        print("✓ 模型训练完成")
    except Exception as e:
        print(f"✗ 模型训练失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 评估模型
    print_section_header("📈 步骤 6/6: 评估模型", "─", 70)
    try:
        y_pred_train = predictor.predict(X_train)
        y_pred_val = predictor.predict(X_val)
        
        # 反缩放RUL（从[0,1]恢复到天数）
        y_train_days = y_train * MAX_RUL_DAYS
        y_val_days = y_val * MAX_RUL_DAYS
        y_pred_train_days = y_pred_train * MAX_RUL_DAYS
        y_pred_val_days = y_pred_val * MAX_RUL_DAYS
        
        # 计算指标
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        train_mse = mean_squared_error(y_train_days, y_pred_train_days)
        train_rmse = np.sqrt(train_mse)
        train_mae = mean_absolute_error(y_train_days, y_pred_train_days)
        train_r2 = r2_score(y_train_days, y_pred_train_days)
        
        val_mse = mean_squared_error(y_val_days, y_pred_val_days)
        val_rmse = np.sqrt(val_mse)
        val_mae = mean_absolute_error(y_val_days, y_pred_val_days)
        val_r2 = r2_score(y_val_days, y_pred_val_days)
        
        metrics = {
            "train_mse": train_mse,
            "train_rmse": train_rmse,
            "train_mae": train_mae,
            "train_r2": train_r2,
            "val_mse": val_mse,
            "val_rmse": val_rmse,
            "val_mae": val_mae,
            "val_r2": val_r2,
        }
        
        print_info_box("🎯 训练集指标", {
            "MSE": f"{train_mse:.4f}",
            "RMSE": f"{train_rmse:.2f} 天",
            "MAE": f"{train_mae:.2f} 天",
            "R2": f"{train_r2:.4f}",
        })
        
        print_info_box("🎯 验证集指标", {
            "MSE": f"{val_mse:.4f}",
            "RMSE": f"{val_rmse:.2f} 天",
            "MAE": f"{val_mae:.2f} 天",
            "R2": f"{val_r2:.4f}",
        })
        
        # 性能评价
        print_section_header("💡 性能评价", "─", 70)
        if val_r2 > 0.8:
            performance = "优秀 ⭐⭐⭐⭐⭐"
            color_indicator = "🟢"
        elif val_r2 > 0.6:
            performance = "良好 ⭐⭐⭐⭐"
            color_indicator = "🟡"
        elif val_r2 > 0.4:
            performance = "一般 ⭐⭐⭐"
            color_indicator = "🟠"
        elif val_r2 > 0:
            performance = "较差 ⭐⭐"
            color_indicator = "🔴"
        else:
            performance = "很差 ⭐"
            color_indicator = "🔴"
        
        print(f"{color_indicator} 模型性能: {performance}")
        print(f"   验证集R2 = {val_r2:.4f}")
        print(f"   平均误差 = {val_mae:.2f} 天")
        print()
        
    except Exception as e:
        print(f"✗ 模型评估失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 保存模型
    print_section_header("💾 保存结果", "─", 70)
    
    if model_save_path:
        model_path = Path(model_save_path)
    else:
        models_dir = BASE_DIR / "models" / "lstm"
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / f"{asset_id}_multivariate_lstm.pt"
    
    # 保存模型（包含所有scaler）
    predictor.save_model(str(model_path))
    print(f"✓ 模型已保存: {model_path}")
    
    # 训练总结
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print_section_header("✅ 训练完成", "═", 70)
    print_info_box("⏱️  训练信息", {
        "开始时间": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "结束时间": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "训练时长": f"{duration:.1f} 秒",
        "模型文件": str(model_path),
        "测点数量": len(feature_names),
    })
    print()


def main():
    """
    主函数：解析命令行参数并启动训练
    
    支持两种训练模式：
    - single: 单测点模式，训练单个测点的LSTM模型
    - multivariate: 多变量模式（默认），自动获取设备的所有PROCESS类型测点进行训练
    
    使用示例：
        # 多变量模式（推荐）
        python -m backend.algorithm.train --mode multivariate --asset_id COMP-ATLAS-01
        
        # 单测点模式
        python -m backend.algorithm.train --mode single --asset_id COMP-ATLAS-01 --metric_id COMP01_OIL_TEMP
    """
    parser = argparse.ArgumentParser(
        description="训练LSTM模型进行RUL预测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 多变量模式（推荐，自动获取所有测点）
  python -m backend.algorithm.train --mode multivariate --asset_id COMP-ATLAS-01
  
  # 单测点模式
  python -m backend.algorithm.train --mode single --asset_id COMP-ATLAS-01 --metric_id COMP01_OIL_TEMP
  
  # 自定义训练参数
  python -m backend.algorithm.train --mode multivariate --asset_id COMP-ATLAS-01 --epochs 50 --lstm_units 128 64 32
        """
    )
    
    # 训练模式选择
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["single", "multivariate"], 
        default="multivariate",
        help="训练模式：single=单测点，multivariate=多变量（一个设备一个模型，自动获取所有测点）"
    )
    
    # 训练参数配置
    parser.add_argument("--asset_id", type=str, default="COMP-ATLAS-01", help="设备ID")
    parser.add_argument("--metric_id", type=str, default="COMP01_OIL_TEMP", help="测点ID（仅单测点模式需要）")
    parser.add_argument("--sequence_length", type=int, default=30, help="序列长度（时间窗口大小）")
    parser.add_argument("--lstm_units", type=int, nargs="+", default=[128, 64], help="LSTM层单元数，例如：--lstm_units 128 64")
    parser.add_argument("--epochs", type=int, default=32, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--model_path", type=str, default=None, help="模型保存路径（可选，默认自动生成）")
    parser.add_argument("--no_plots", action="store_true", help="不保存训练曲线图")
    
    args = parser.parse_args()
    
    if args.mode == "multivariate":
        # 多变量模式：自动从数据库获取设备的所有PROCESS类型测点
        train_multivariate(
            asset_id=args.asset_id,
            sequence_length=args.sequence_length,
            lstm_units=args.lstm_units,
            epochs=args.epochs,
            batch_size=args.batch_size,
            model_save_path=args.model_path,
            save_plots=not args.no_plots,
        )
    else:
        # 单测点模式
        train_single_metric(
            asset_id=args.asset_id,
            metric_id=args.metric_id,
            sequence_length=args.sequence_length,
            lstm_units=args.lstm_units,
            epochs=args.epochs,
            batch_size=args.batch_size,
            model_save_path=args.model_path,
            save_plots=not args.no_plots,
        )


if __name__ == "__main__":
    main()

