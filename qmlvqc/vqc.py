import numpy as np
import pennylane as qml
import pennylane.numpy as pnp

from .encodings import AngleEncoding


class VQC:
    def __init__(self, n_qubits, architecture, loss, encoding=None):
        self.n_qubits = n_qubits
        self.architecture = architecture
        self.loss = loss
        self.encoding = encoding if encoding is not None else AngleEncoding()
        self._weights = None
        self._bias = None
        self._circuit = None

    def _build_circuit(self, dev):
        encoding = self.encoding
        architecture = self.architecture
        n_qubits = self.n_qubits

        @qml.qnode(dev)
        def circuit(x, weights):
            encoding.apply(x, wires=range(n_qubits))
            architecture.apply(weights, wires=range(n_qubits))
            return qml.expval(qml.PauliZ(0))

        return circuit

    def fit(self, X_train, y_train, epochs=50, lr=0.01, batch_size=16):
        self.encoding.fit(X_train, self.n_qubits)
        X_enc = self.encoding.transform(X_train)

        dev = qml.device("default.qubit", wires=self.n_qubits)
        circuit = self._build_circuit(dev)
        self._circuit = circuit

        weight_shape = self.architecture.weight_shape(self.n_qubits)
        self._weights = pnp.random.uniform(0, 2 * pnp.pi, size=weight_shape, requires_grad=True)
        self._bias = pnp.array(0.0, requires_grad=True)

        optimizer = qml.AdamOptimizer(stepsize=lr)

        n_samples = len(X_enc)
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
                    preds = pnp.array([circuit(x, weights) + bias for x in X_batch])
                    return self.loss(preds, y_batch)

                self._weights, self._bias = optimizer.step(cost, self._weights, self._bias)
                batch_loss = float(cost(self._weights, self._bias))
                epoch_loss += batch_loss
                n_batches += 1

            print(f"Epoch {epoch + 1:3d}/{epochs} | loss: {epoch_loss / n_batches:.4f}")

    def predict(self, X):
        X_enc = self.encoding.transform(X)
        raw = np.array([float(self._circuit(x, self._weights) + self._bias) for x in X_enc])
        return self.loss.to_label(raw)

    def score(self, X, y):
        preds = self.predict(X)
        return float(np.mean(preds == y.astype(int)))
