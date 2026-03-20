from abc import ABC, abstractmethod

import numpy as np


class Trainer(ABC):

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        ...
