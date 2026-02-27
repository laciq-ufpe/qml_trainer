import pennylane as qml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class AngleEncoding:
    def __init__(self):
        self._scaler = None
        self._pca = None

    def fit(self, X, n_qubits):
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        self._pca = PCA(n_components=n_qubits)
        self._pca.fit(X_scaled)

    def transform(self, X):
        X_scaled = self._scaler.transform(X)
        return self._pca.transform(X_scaled)

    def apply(self, x, wires):
        qml.AngleEmbedding(x, wires=wires, rotation='Z')
