from typing import List, Dict, Optional
import numpy as np
from datetime import datetime, timedelta

try:
    from scipy.interpolate import CubicSpline
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


class EquipmentAnalyzer:
    def __init__(self, history_data: List[Dict[str, float]]):
        self.data = sorted(history_data, key=lambda x: x["date"]) if history_data else []
        self.dates = [datetime.strptime(d["date"], "%Y-%m-%d") for d in self.data]
        self.values = np.array([float(d["value"]) for d in self.data]) if self.data else np.empty(0)
        self.base_date = self.dates[0] if self.dates else None
        self.x = (
            np.array([(d - self.base_date).total_seconds() / 86400.0 for d in self.dates]).reshape(-1, 1)
            if self.base_date is not None
            else np.empty((0, 1))
        )

    def _lin_reg(self):
        if len(self.values) < 2:
            return None
        x = self.x.flatten()
        slope, intercept = np.polyfit(x, self.values, 1)
        y_pred = slope * x + intercept
        ss_res = float(np.sum((self.values - y_pred) ** 2))
        ss_tot = float(np.sum((self.values - np.mean(self.values)) ** 2)) or 1e-6
        r2 = 1.0 - ss_res / ss_tot
        return slope, intercept, r2

    def predict_rul(self, 
                    limit_threshold: float, 
                    trend_direction: int = 1, 
                    max_days: int = 365) -> Dict[str, Optional[int]]:
        if len(self.values) < 3 or self.base_date is None:
            return {"status": "insufficient", "rul_days": None}
        res = self._lin_reg()
        if res is None:
            return {"status": "insufficient", "rul_days": None}
        slope, intercept, r2 = res
        if r2 < 0.6:
            return {"status": "uncertain", "rul_days": None}
        if trend_direction == 1 and slope <= 0:
            return {"status": "stable", "rul_days": None}
        if trend_direction == -1 and slope >= 0:
            return {"status": "stable", "rul_days": None}
        target_x = (limit_threshold - intercept) / slope
        target_date = self.base_date + timedelta(days=float(target_x))
        days_left = (target_date - datetime.now()).days
        if days_left > max_days:
            return {"status": ">%d" % max_days, "rul_days": max_days}
        return {"status": "degrading", "rul_days": max(0, days_left)}

    def generate_smooth_curve(self, points: int = 100) -> List[Dict[str, float]]:
        if len(self.values) < 3 or self.base_date is None:
            return self.data
        x = self.x.flatten()
        if HAS_SCIPY and len(self.values) >= 3:
            cs = CubicSpline(x, self.values)
            dense_x = np.linspace(x[0], x[-1], points)
            dense_y = cs(dense_x)
        else:
            dense_x = np.linspace(x[0], x[-1], points)
            dense_y = np.interp(dense_x, x, self.values)
        result = []
        for xi, yi in zip(dense_x, dense_y):
            d = self.base_date + timedelta(days=float(xi))
            result.append({"date": d.date().isoformat(), "value": round(float(yi), 2)})
        return result

    def detect_input_anomaly(self, new_value: float, valid_min: Optional[float] = None, valid_max: Optional[float] = None) -> bool:
        if valid_min is not None and new_value < valid_min:
            return True
        if valid_max is not None and new_value > valid_max:
            return True
        if len(self.values) < 5:
            return False
        mean = float(np.mean(self.values))
        std = float(np.std(self.values)) or 1e-6
        return abs(new_value - mean) > 3.0 * std
