from abc import ABC, abstractmethod

import numpy as np

from ..encodings import Encoding
from ..architectures import Architecture


class BaseExecutor(ABC):

    @abstractmethod
    def execute(
        self,
        encoding: Encoding,
        architecture: Architecture,
        x: np.ndarray,
        weights,
        bias,
    ):
        ...
