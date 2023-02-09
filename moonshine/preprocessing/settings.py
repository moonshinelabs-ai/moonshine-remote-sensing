import dataclasses

import numpy as np


@dataclasses.dataclass
class DatasetSettings(object):
    mean: np.ndarray
    std: np.ndarray


@dataclasses.dataclass
class ModelSettings(object):
    colorspace: str
