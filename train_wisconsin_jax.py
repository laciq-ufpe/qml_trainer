import os
os.environ["JAX_TRACEBACK_FILTERING"] = "off"

import logging
logging.basicConfig(level=logging.INFO)

from qmlvqc import VQCJax, StronglyEntangled, BCEJax, PhaseEncodingJax
from qmlvqc.data import load_wisconsin

X_train, X_test, y_train, y_test = load_wisconsin()
model = VQCJax(
    n_qubits=10,
    architecture=StronglyEntangled(n_layers=10),
    loss=BCEJax(),
    encoding=PhaseEncodingJax(),
    seed=42,
)
model.fit(X_train, y_train, epochs=10, lr=1e-4, batch_size=255)

print("Train acc:", model.score(X_train, y_train))
print("Test acc: ", model.score(X_test, y_test))
