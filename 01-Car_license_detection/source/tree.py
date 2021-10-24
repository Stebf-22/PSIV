from sklearn import tree
import numpy as np

from source.utils import Utils


class DTCHandler(Utils):

    def __init__(self, X, Y, **kwargs):
        super(Utils, self).__init__()
        hyperparams = {
            'criterion': ['gini', 'entropy'],
            'max_features': ['auto', 'sqrt', 'log2'],
            'splitter': ['best', 'random'],
        }
        self.n_splits = kwargs['n_splits'] if 'n_splits' in kwargs else 5
        if 'n_splits' in kwargs:
            del kwargs['n_splits']
        self.model = tree.DecisionTreeClassifier(**kwargs)
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