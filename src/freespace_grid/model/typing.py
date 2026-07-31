"""Array type aliases shared by every layer."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["BoolArray", "FloatArray", "IntArray"]

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int64]
type BoolArray = NDArray[np.bool_]
