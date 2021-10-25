import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import neighbors
from sklearn import tree
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
import pickle

from source.utils import Utils


class ModelHandler(Utils):

    models = {'SVM': svm.SVC, 'KNN': neighbors.KNeighborsClassifier,
              'DT': tree.DecisionTreeClassifier, 'XGB': XGBClassifier}
    hyperparams = {'SVM': {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'C': np.logspace(-2, 10, 5),
    }, 'KNN': {
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'n_neighbors': np.arange(3, 10, 2),
        'p': np.arange(1, 3),
    }, 'DT': {
        'criterion': ['gini', 'entropy'],
        'max_features': ['auto', 'sqrt', 'log2'],
        'splitter': ['best', 'random'],
    }, 'XGB': {
        'min_child_weight': [1, 5],
        'gamma': [0.5, 1, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4]
    }}

    def __init__(self, X, Y, model: str, **kwargs):

        super().__init__()

        self.n_splits = kwargs['n_splits'] if 'n_splits' in kwargs else 5
        if 'n_splits' in kwargs:
            del kwargs['n_splits']
        self.hyperparam = self.hyperparams[model]
        self.hyperparam.update(kwargs)
        self.model = self.models[model](
            **{x: kwargs[x][0] if type(kwargs[x]) == list else kwargs[x] for x in kwargs})
        self.X = self.do_scaling(self._ensure_dimensionalit(X))
        self.Y = Y
        assert(len(Y.shape) == 1)
        self.grid = self._gen_gridSearch(
            self.model, self.hyperparam, self.n_splits)
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
            print(f"[INFO] Train acc  is : {self._acc(pred, self.Y):.4f}")

        def predict(self, X):
        X = self._ensure_dimensionalit(X)
        return self.model.predict(X)

    def available_models(self):
        return self.models.keys()
    
    def save(self):
        pickle.dump(self.model, open(self.model.__class__,'wb'))
    
    def load(self,path):
        self.model = pickle.load(open(path,'rb'))