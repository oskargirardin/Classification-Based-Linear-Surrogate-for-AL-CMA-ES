import numpy as np

class ConnectingHyperplane(object):
    def __init__(self) -> None:
        self.coef_ = None
        self.intercept_ = None


    @property
    def parameters(self):
        return self.coef_, self.intercept_

    def fit(self, X: np.ndarray):
        self.dim = X.shape[0]
        assert self.dim == X.shape[1], "X must be symmetric"
        return self._fit(X)

    def _fit(self, X):
        try:
            X_inv = np.linalg.inv(X)
        except np.linalg.LinAlgError:
            print(X)
            raise np.linalg.LinAlgError
        beta = X_inv @ (- np.ones((self.dim, 1)))
        beta0 = 1
        self.coef_ = beta.reshape(-1)
        self.intercept_ = beta0
        return self