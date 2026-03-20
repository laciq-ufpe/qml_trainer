import numpy as np
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo


def load_wisconsin(
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataset = fetch_ucirepo(id=17)
    X = dataset.data.features.values
    targets = dataset.data.targets
    y = (targets['Diagnosis'] == 'M').astype(float).values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test
