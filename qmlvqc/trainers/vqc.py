import numpy as np
import pennylane as qml
import pennylane.numpy as pnp

from ..architectures import Architecture
from ..encodings import AngleEncoding, Encoding
from ..executors import PennylaneExecutor, BaseExecutor
from ..losses import Loss
from .base_trainer import Trainer


class VQC(Trainer):

    def __init__(
        self,
        n_qubits: int,
        architecture: Architecture,
        loss: Loss,
        encoding: Encoding = None,
        executor: BaseExecutor = None,
    ):
        self.n_qubits = n_qubits
        self.architecture = architecture
        self.loss = loss
        self.encoding = encoding if encoding is not None else AngleEncoding()
        self.executor = executor if executor is not None else PennylaneExecutor(n_qubits)

        self._weights = None
        self._bias = None
        self.history_: list[float] = []

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 50,
        lr: float = 0.01,
        batch_size: int = 16,
    ) -> None:
        self.encoding.fit(X_train, self.n_qubits)
        X_enc = self.encoding.transform(X_train)

        weight_shape = self.architecture.weight_shape(self.n_qubits)

        self._weights = pnp.random.uniform(
            0,
            2 * pnp.pi,
            size=weight_shape,
            requires_grad=True,
        )
        self._bias = pnp.array(0.0, requires_grad=True)

        optimizer = qml.AdamOptimizer(stepsize=lr)
        n_samples = len(X_enc)
        self.history_ = []

        for epoch in range(epochs):
            perm = np.random.permutation(n_samples)
            X_shuffled = X_enc[perm]
            y_shuffled = y_train[perm]

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, batch_size):
                X_batch = X_shuffled[start:start + batch_size]
                y_batch = y_shuffled[start:start + batch_size]

                def cost(weights, bias):
                    preds = pnp.array([
                        self.executor.execute(self.encoding, self.architecture, x, weights, bias)
                        for x in X_batch
                    ])
                    return self.loss(preds, y_batch)

                self._weights, self._bias = optimizer.step(cost, self._weights, self._bias)

                epoch_loss += float(cost(self._weights, self._bias))
                n_batches += 1

            mean_loss = epoch_loss / n_batches
            self.history_.append(mean_loss)
            print(f"Epoch {epoch + 1:3d}/{epochs} | loss: {mean_loss:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_enc = self.encoding.transform(X)
        raw = np.array([
            float(self.executor.execute(self.encoding, self.architecture, x, self._weights, self._bias))
            for x in X_enc
        ])
        return self.loss.to_label(raw)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        return float(np.mean(preds == y.astype(int)))
