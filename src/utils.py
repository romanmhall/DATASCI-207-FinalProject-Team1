import os
import random
from pathlib import Path
import numpy as np
import pandas as pd


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent


def set_seeds(seed: int = 1234) -> None:
    """Set global seeds for reproducibility across numpy, python, and tensorflow."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def load_processed_data(file_path: Path = None) -> pd.DataFrame:
    """Load the consolidated processed features parquet dataset."""
    if file_path is None:
        file_path = get_project_root() / 'data' / 'processed' / 'processed_features.parquet'
    if not file_path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {file_path}")
    return pd.read_parquet(file_path)
