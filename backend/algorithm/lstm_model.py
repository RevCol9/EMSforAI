"""
LSTM神经网络模型

本模块定义了用于RUL预测的LSTM神经网络架构，包括：
1. LSTM模型定义（多层LSTM + 全连接回归头）
2. PyTorch训练循环（支持早停和验证集）
3. 模型预测、保存与加载
4. Scaler序列化，确保推理与训练一致

Author: EMSforAI Team
License: MIT
"""
import logging
import importlib
import pickle
import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

log = logging.getLogger(__name__)

try:
    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")
    optim = importlib.import_module("torch.optim")
    torch_utils_data = importlib.import_module("torch.utils.data")
    DataLoader = getattr(torch_utils_data, "DataLoader")
    TensorDataset = getattr(torch_utils_data, "TensorDataset")
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ImportError("未检测到 PyTorch，请先安装：pip install torch") from exc

try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover - tqdm可选
    tqdm = None


class RULLSTMNet(nn.Module):
    """多层LSTM + 全连接回归头"""

    def __init__(self, n_features: int, lstm_units: List[int], dropout_rate: float):
        super().__init__()
        if not lstm_units:
            raise ValueError("lstm_units 至少需要一个元素")

        self.lstm_layers = nn.ModuleList()
        input_size = n_features
        for hidden in lstm_units:
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden,
                    batch_first=True,
                )
            )
            input_size = hidden

        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(lstm_units[-1])
        self.regressor = nn.Sequential(
            nn.Linear(lstm_units[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = x
        for idx, lstm in enumerate(self.lstm_layers):
            out, _ = lstm(out)
            if idx < len(self.lstm_layers) - 1:
                out = self.dropout(out)

        out = out[:, -1, :]
        out = self.layer_norm(out)
        out = self.dropout(out)
        return self.regressor(out)


class LSTMRULPredictor:
    """基于PyTorch的LSTM RUL预测器"""

    def __init__(
        self,
        sequence_length: int = 30,
        n_features: int = 1,
        lstm_units: Optional[List[int]] = None,
        dropout_rate: float = 0.1,
        learning_rate: float = 0.001,
        device: Optional[str] = None,
    ):
        if lstm_units is None:
            lstm_units = [64, 32]

        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model: Optional[RULLSTMNet] = None
        self.scaler: Optional[MinMaxScaler] = None
        self.is_trained = False
        self.config = {
            "sequence_length": sequence_length,
            "n_features": n_features,
            "lstm_units": lstm_units,
            "dropout_rate": dropout_rate,
            "learning_rate": learning_rate,
        }

    def build_model(self) -> RULLSTMNet:
        self.model = RULLSTMNet(
            n_features=self.n_features,
            lstm_units=self.lstm_units,
            dropout_rate=self.dropout_rate,
        ).to(self.device)
        return self.model

    def prepare_sequences(
        self,
        data: np.ndarray,
        rul_targets: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if len(data) < self.sequence_length + 1:
            raise ValueError(f"数据长度({len(data)})不足，需要至少{self.sequence_length + 1}个样本")

        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i : i + self.sequence_length])
            if rul_targets is not None:
                y.append(rul_targets[i + self.sequence_length])

        X_arr = np.array(X)
        y_arr = np.array(y) if rul_targets is not None else None
        return X_arr, y_arr

    def transform_values(self, values: np.ndarray) -> np.ndarray:
        values = values.reshape(-1, self.n_features)
        if self.scaler is None:
            scaler = MinMaxScaler()
            scaler.fit(values)
            self.scaler = scaler
        return self.scaler.transform(values)

    def build_inference_tensor(self, values_scaled: np.ndarray) -> np.ndarray:
        if len(values_scaled) < self.sequence_length:
            raise ValueError(
                f"推理需要至少{self.sequence_length}个数据点，当前只有{len(values_scaled)}个"
            )
        window = values_scaled[-self.sequence_length :]
        return window.reshape(1, self.sequence_length, self.n_features)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        verbose: int = 1,
        patience: int = 15,
    ) -> Dict[str, List[float]]:
        if self.model is None:
            self.build_model()

        if self.model is None:
            raise RuntimeError("模型构建失败")

        if X_val is None or y_val is None:
            split_idx = int(len(X_train) * (1 - validation_split))
            X_val = X_train[split_idx:]
            y_val = y_train[split_idx:]
            X_train = X_train[:split_idx]
            y_train = y_train[:split_idx]

        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1),
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1),
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        
        # 学习率调度器：当验证损失不再下降时，降低学习率
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )

        history = {"train_loss": [], "val_loss": []}
        best_val = float("inf")
        best_state: Optional[Dict[str, Any]] = None
        wait = 0

        epoch_iter = range(epochs)
        if verbose and tqdm is not None:
            epoch_iter = tqdm(range(epochs), desc="LSTM训练", leave=False)

        for epoch in epoch_iter:
            self.model.train()
            total_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                preds = self.model(batch_X)
                loss = criterion(preds, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()
                total_loss += loss.item() * batch_X.size(0)

            avg_train_loss = total_loss / len(train_loader.dataset)

            self.model.eval()
            val_loss_total = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    preds = self.model(batch_X)
                    loss = criterion(preds, batch_y)
                    val_loss_total += loss.item() * batch_X.size(0)

            avg_val_loss = val_loss_total / len(val_loader.dataset)

            # 更新学习率
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)

            if verbose and tqdm is None:
                log.info(
                    "Epoch %d/%d - train_loss: %.6f, val_loss: %.6f",
                    epoch + 1,
                    epochs,
                    avg_train_loss,
                    avg_val_loss,
                )
            elif verbose and tqdm is not None:
                epoch_iter.set_postfix(
                    {
                        "train_loss": f"{avg_train_loss:.4f}",
                        "val_loss": f"{avg_val_loss:.4f}",
                    }
                )

            if avg_val_loss < best_val:
                best_val = avg_val_loss
                best_state = {
                    "model": self.model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    log.info("验证集损失未提升，提前停止训练")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state["model"])

        self.is_trained = True
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained or self.model is None:
            raise RuntimeError("模型未训练，无法进行预测")

        self.model.eval()
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float32).to(self.device)
            preds = self.model(inputs).cpu().numpy().reshape(-1)
        return preds

    def save_model(self, filepath: str):
        if self.model is None:
            raise RuntimeError("模型不存在，无法保存")

        payload = {
            "state_dict": self.model.state_dict(),
            "config": self.config,
            "scaler": pickle.dumps(self.scaler) if self.scaler is not None else None,
        }
        torch.save(payload, filepath)
        log.info("模型已保存到: %s", filepath)

    def load_model(self, filepath: str):
        # 注意：由于模型文件包含scaler等pickle对象，需要使用weights_only=False
        # 这些模型文件是系统内部生成的，安全性可控
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        config = checkpoint.get("config", {})

        self.sequence_length = config.get("sequence_length", self.sequence_length)
        self.n_features = config.get("n_features", self.n_features)
        self.lstm_units = config.get("lstm_units", self.lstm_units)
        self.dropout_rate = config.get("dropout_rate", self.dropout_rate)
        self.learning_rate = config.get("learning_rate", self.learning_rate)
        self.config = config or self.config

        self.build_model()
        if self.model is None:
            raise RuntimeError("模型构建失败，无法加载权重")

        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)

        scaler_blob = checkpoint.get("scaler")
        if scaler_blob is not None:
            self.scaler = pickle.loads(scaler_blob)

        self.is_trained = True
        log.info("模型已从 %s 加载", filepath)


class MultiVariateLSTMPredictor(LSTMRULPredictor):
    """多变量版本，复用基类逻辑
    
    支持一个设备的所有测点同时训练和预测，利用测点之间的关联性。
    """

    def __init__(
        self,
        sequence_length: int = 30,
        n_features: int = 5,
        lstm_units: Optional[List[int]] = None,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        device: Optional[str] = None,
    ):
        if lstm_units is None:
            lstm_units = [128, 64]
        super().__init__(
            sequence_length=sequence_length,
            n_features=n_features,
            lstm_units=lstm_units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            device=device,
        )
        # 多变量模型需要保存每个测点的scaler
        self.scalers: Optional[Dict[str, MinMaxScaler]] = None
        self.feature_names: Optional[List[str]] = None

    def prepare_multivariate_sequences(
        self,
        data_dict: Dict[str, np.ndarray],
        rul_targets: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        准备多变量序列数据
        
        Args:
            data_dict: 字典，key为测点ID，value为标准化后的值数组
            rul_targets: RUL目标值数组（可选）
        
        Returns:
            (X, y) 元组，X为形状 (n_samples, sequence_length, n_features) 的数组
        """
        min_length = min(len(v) for v in data_dict.values())
        if min_length < self.sequence_length + 1:
            raise ValueError("数据长度不足，无法构建多变量序列")

        n_samples = min_length - self.sequence_length
        n_features = len(data_dict)
        feature_names = list(data_dict.keys())
        self.feature_names = feature_names  # 保存特征名称

        X = np.zeros((n_samples, self.sequence_length, n_features))
        for i in range(n_samples):
            for j, name in enumerate(feature_names):
                X[i, :, j] = data_dict[name][i : i + self.sequence_length]

        if rul_targets is not None:
            y = rul_targets[self.sequence_length : self.sequence_length + n_samples]
        else:
            y = None

        return X, y
    
    def transform_multivariate_values(
        self, 
        data_dict: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        对多变量数据进行标准化
        
        Args:
            data_dict: 字典，key为测点ID，value为原始值数组
        
        Returns:
            标准化后的数据字典
        """
        if self.scalers is None:
            raise RuntimeError("模型未训练，无法进行标准化")
        
        transformed = {}
        for metric_id, values in data_dict.items():
            if metric_id not in self.scalers:
                raise ValueError(f"测点 {metric_id} 的scaler不存在")
            scaler = self.scalers[metric_id]
            values_2d = values.reshape(-1, 1)
            transformed[metric_id] = scaler.transform(values_2d).flatten()
        
        return transformed
    
    def build_multivariate_inference_tensor(
        self, 
        data_dict: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        构建多变量推理张量
        
        Args:
            data_dict: 标准化后的数据字典，key为测点ID，value为值数组
        
        Returns:
            形状为 (1, sequence_length, n_features) 的推理张量
        """
        if self.feature_names is None:
            raise RuntimeError("模型未训练，无法构建推理张量")
        
        # 确保所有测点数据长度一致
        min_length = min(len(v) for v in data_dict.values())
        if min_length < self.sequence_length:
            raise ValueError(
                f"推理需要至少{self.sequence_length}个数据点，当前只有{min_length}个"
            )
        
        # 构建多变量序列（使用最近sequence_length个点）
        window = np.zeros((1, self.sequence_length, len(self.feature_names)))
        for j, metric_id in enumerate(self.feature_names):
            if metric_id not in data_dict:
                raise ValueError(f"缺少测点 {metric_id} 的数据")
            window[0, :, j] = data_dict[metric_id][-self.sequence_length:]
        
        return window
    
    def save_model(self, filepath: str):
        """保存多变量模型（包含所有scaler）"""
        if self.model is None:
            raise RuntimeError("模型不存在，无法保存")

        # 序列化所有scaler
        scalers_blob = None
        if self.scalers is not None:
            scalers_blob = pickle.dumps(self.scalers)

        payload = {
            "state_dict": self.model.state_dict(),
            "config": self.config,
            "scaler": pickle.dumps(self.scaler) if self.scaler is not None else None,  # 保留兼容性
            "scalers": scalers_blob,  # 多变量scaler字典
            "feature_names": self.feature_names,  # 特征名称列表
        }
        torch.save(payload, filepath)
        log.info("多变量模型已保存到: %s", filepath)

    def load_model(self, filepath: str):
        """加载多变量模型（包含所有scaler）"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        config = checkpoint.get("config", {})

        self.sequence_length = config.get("sequence_length", self.sequence_length)
        self.n_features = config.get("n_features", self.n_features)
        self.lstm_units = config.get("lstm_units", self.lstm_units)
        self.dropout_rate = config.get("dropout_rate", self.dropout_rate)
        self.learning_rate = config.get("learning_rate", self.learning_rate)
        self.config = config or self.config

        self.build_model()
        if self.model is None:
            raise RuntimeError("模型构建失败，无法加载权重")

        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)

        # 加载多变量scaler（优先）
        scalers_blob = checkpoint.get("scalers")
        if scalers_blob is not None:
            self.scalers = pickle.loads(scalers_blob)
            log.info("已加载多变量scaler（%d个测点）", len(self.scalers))
        else:
            # 兼容单变量模型（向后兼容）
            scaler_blob = checkpoint.get("scaler")
            if scaler_blob is not None:
                self.scaler = pickle.loads(scaler_blob)
                log.info("已加载单变量scaler（兼容模式）")

        # 加载特征名称
        self.feature_names = checkpoint.get("feature_names")

        self.is_trained = True
        log.info("多变量模型已从 %s 加载", filepath)

