from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SceneSample:
    name: str
    image_paths: list[str]
    gt_extrinsics: np.ndarray
    category: Optional[str] = None
