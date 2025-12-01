"""Simple CLI helper to run AlgorithmEngine against database data."""
from __future__ import annotations

from typing import Any
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from backend.algorithm.algorithms_engine import AlgorithmEngine
from backend.algorithm.data_service import load_data_from_db


def run_demo(device_id: int = 1, window_days: int = 50, quality_cutoff: float = 0.8) -> Any:
    """Load database data and print device health summary."""
    data_bundle = load_data_from_db()
    engine = AlgorithmEngine(window_days=window_days, quality_cutoff=quality_cutoff)
    result = engine.analyze_device(context=data_bundle, device_id=device_id)
    print(result)
    return result


if __name__ == "__main__":
    run_demo()

