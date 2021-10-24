from sklearn import svm
import numpy as np

from source.utils import Utils


class SVMCHandler(Utils):

    def __init__(self, X, Y, **kwargs):
        super().__init__()
        hyperparams = {
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'C': np.logspace(-2, 10, 5),
            'gamma': np.logspace(-9, 3, 5),
        }
        self.n_splits = kwargs['n_splits'] if 'n_splits' in kwargs else 5
        if 'n_splits' in kwargs:
            del kwargs['n_splits']
        self.model = svm.SVC(**kwargs)
        self.X = self.do_scaling(self._ensure_dimensionalit(X))
        self.Y = Y
        assert(len(Y.shape) == 1)
        self.grid = self._gen_gridSearch(
            self.model, hyperparams, self.n_splits)
        self.grid_flag = False

    def fit(self, with_score=True, with_grid=True):
        if with_grid:
            self.grid.fit(self.X, self.Y)
            print(f"[INFO] The best parameters are {self.grid.best_params_}")
            print(f"[INFO] The best score is {self.grid.best_score_:.4f}")
            self.model = self.model.__class__(**self.grid.best_params_)
            self.model.fit(self.X, self.Y)
            self.grid_flag = True
        else:
            self.model.fit(self.X, self.Y)
        if with_score:
            pred = self.predict(self.X)
            print(f"Train acc  is : {self._acc(pred, self.Y)}")

    def predict(self, X):
        X = self._ensure_dimensionalit(X)
        return self.model.predict(X)