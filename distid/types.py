from dataclasses import dataclass
from typing import Dict, Optional, Callable, Any
import numpy as np

@dataclass
class FitResult:
    name: str
    params: Dict[str, float]
    rmse: float
    mass_err: float
    score: float
    extras: Dict[str, float]
    k: np.ndarray
    logpmf_theory: np.ndarray
    valid: bool

Recognizer = Callable[[Any, dict], Optional[FitResult]]
