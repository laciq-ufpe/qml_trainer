from abc import ABC, abstractmethod

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class DimensionalityProjector(ABC):
    @abstractmethod
    def fit(self, X, n_qubits: int) -> None:
        ...

    @abstractmethod
    def transform(self, X) -> np.ndarray:
        ...


class PCAProjector(DimensionalityProjector):
    def __init__(self):
        self._scaler = None
        self._pca = None

    def fit(self, X, n_qubits: int) -> None:
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        self._pca = PCA(n_components=n_qubits)
        self._pca.fit(X_scaled)

    def transform(self, X) -> np.ndarray:
        X_scaled = self._scaler.transform(X)
        return self._pca.transform(X_scaled)
