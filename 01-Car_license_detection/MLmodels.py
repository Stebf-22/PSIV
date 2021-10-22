import numpy as np
import os
import pandas as pd
import scipy

import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn import neighbors
from sklearn import tree
from sklearn import svm

from xgboost import XGBClassifier

## General utils

class Utils(object):
    
    def func_overX(self, X):
        Y = []
        for element in X:
            Y.append( sum(element.flatten())  > element.flatten().shape[0] //2)
        return np.asarray(Y)

    def testing(self):

        ##### SVM Testing #####
        X = np.random.rand(200 , 60,30)
        Y = func_overX(X)

        ### GridSearch ###
        model = SVMCHandler(X,Y)
        model.fit(with_score=True, with_grid=True)


    def _ensure_dimensionalit(self,arr):
            return arr if len(arr[0].shape) == 1 else [x.flatten() for x in arr]

    def _acc(self, y_pred, y_target):

        if type(y_pred) == np.array and type(y_target) == np.array :
            assert(y_pred.shape == y_target.shape)
            mask = y_pred == y_target

        else: 
            assert(len(y_pred) == len(y_target))
            mask = [x==y for x,y in zip(y_pred, y_target)]
        return sum(mask)/len(mask)


    def do_scaling(self, X):
        Scaler = StandardScaler()
        return Scaler.fit_transform(X)

    def _gen_gridSearch(self, model,hyperparams, n_splits = 5):
        cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state= 42)
        grid = GridSearchCV(model , param_grid= hyperparams, cv=cv, n_jobs=6, verbose=3)
        return grid 

    def df_Grid(self):
        if self.grid_flag:
            c = self.grid.__dict__['cv_results_']['params']           
            a = ['params'] + [f'split{n}_test_score' for n in range(self.n_splits)]
            dic = {h : i for h,i in zip(a, (c, *[self.grid.__dict__['cv_results_'][f'split{n}_test_score'] for n in range(self.n_splits)]))}    
            return pd.DataFrame(dic)
        else:
            print('Grid has not been calculated')
            raise NotImplementedError
            
    def ci(self, alpha):
        def f(x):
            return scipy.stats.t.interval(alpha = alpha, df = self.n_splits - 1, loc = x['mean'], scale = x['sem'])
        return f
    
    def top_params(self, alpha = 0.95, n = None): #retorna els parametres amb millor ci acc
        df = self.df_Grid()
        df['mean'] = df.filter(regex='split').mean(axis = 1) #agafem columnes nombrades 'split*' calculem mitja
        df['sem'] = df.filter(regex='split').apply(scipy.stats.sem, axis = 1) + 1e-8 #standard error of mean            
        df['ci'] = df.apply(self.ci(alpha), axis = 1)
        df = df.sort_values('ci', ascending=False)
        if n:
            return df[:n]
        return df


## SVM

class SVMCHandler(Utils):
    
    def __init__(self, X, Y, **kwargs):
        super().__init__()
        hyperparams = {
            'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
            'C': np.logspace(-2, 10 , 5),
            'gamma': np.logspace(-9,3,5),
        }
        self.n_splits  = kwargs['n_splits'] if 'n_splits' in kwargs else 5
        if 'n_splits' in kwargs: del kwargs['n_splits']
        self.model = svm.SVC(**kwargs)
        self.X = self.do_scaling(self._ensure_dimensionalit(X))
        self.Y = Y
        assert(len(Y.shape) ==1 )
        self.grid = self._gen_gridSearch(self.model, hyperparams, self.n_splits)
        self.grid_flag= False
    
    def fit(self, with_score = True, with_grid=True):
        if with_grid:
            self.grid.fit(self.X, self.Y)
            print(f"The best parameters are {self.grid.best_params_} and the best score is {self.grid.best_score_}")
            self.model = self.model.__class__(**self.grid.best_params_)
            self.model.fit(self.X,self.Y)
            self.grid_flag = True
        else : 
            self.model.fit(self.X, self.Y )
        if with_score:
            pred = self.predict(self.X)
            print(f"Train acc  is : {self._acc(pred, self.Y)}")

    def predict(self, X):
        X = self._ensure_dimensionalit(X)
        return self.model.predict(X)
    

## KNN

class KNNhandler(Utils):

    def __init__(self, X, Y, **kwargs):
        super().__init__()
        hyperparams = {
            'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
            'n_neighbors': np.arange(3, 10, 2),
            'p': np.arange(1,3),
        }
        self.n_splits  = kwargs['n_splits'] if 'n_splits' in kwargs else 5
        if 'n_splits' in kwargs: del kwargs['n_splits']
        self.model = neighbors.KNeighborsClassifier(**kwargs)
        self.X = self.do_scaling(self._ensure_dimensionalit(X))
        self.Y = Y
        assert(len(Y.shape) ==1 )
        self.grid = self._gen_gridSearch(self.model, hyperparams, self.n_splits)
        self.grid_flag= False
      
    
    def fit(self, with_score = True, with_grid=True):
        if with_grid:
            self.grid.fit(self.X, self.Y)
            print(f"The best parameters are {self.grid.best_params_} and the best score is {self.grid.best_score_}")
            self.model = self.model.__class__(**self.grid.best_params_)
            self.model.fit(self.X,self.Y)
            self.grid_flag = True
        else : 
            self.model.fit(self.X, self.Y )
        if with_score:
            pred = self.predict(self.X)
            print(f"Train acc  is : {self._acc(pred, self.Y)}")

    def predict(self, X):
        X = self._ensure_dimensionalit(X)
        return self.model.predict(X)
    


## Decision tree

class DTCHandler(Utils):

    def __init__(self, X, Y, **kwargs):
        super().__init__()
        hyperparams = {
            'criterion':['gini', 'entropy'],
            'max_features': ['auto', 'sqrt', 'log2'],
            'splitter': ['best', 'random'],
        }
        self.n_splits  = kwargs['n_splits'] if 'n_splits' in kwargs else 5
        if 'n_splits' in kwargs: del kwargs['n_splits']
        self.model = tree.DecisionTreeClassifier(**kwargs)
        self.X = self.do_scaling(self._ensure_dimensionalit(X))
        self.Y = Y
        assert(len(Y.shape) ==1 )
        self.grid = self._gen_gridSearch(self.model, hyperparams, self.n_splits)
        self.grid_flag= False
    
    def fit(self, with_score = True, with_grid=True):
        if with_grid:
            self.grid.fit(self.X, self.Y)
            print(f"The best parameters are {self.grid.best_params_} and the best score is {self.grid.best_score_}")
            self.model = self.model.__class__(**self.grid.best_params_)
            self.model.fit(self.X,self.Y)
            self.grid_flag = True

        else : 
            self.model.fit(self.X, self.Y )
        if with_score:
            pred = self.predict(self.X)
            print(f"Train acc  is : {self._acc(pred, self.Y)}")

    def predict(self, X):
        X = self._ensure_dimensionalit(X)
        return self.model.predict(X)


##XGBoost

class XGBHandler(Utils):

    def __init__(self, X, Y, **kwargs):
        super().__init__()
        hyperparams = {
            'min_child_weight': [1, 5],
            'gamma': [0.5, 1, 2, 5],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'max_depth': [3, 4]
        }
        self.n_splits  = kwargs['n_splits'] if 'n_splits' in kwargs else 5
        if 'n_splits' in kwargs: del kwargs['n_splits']
        self.model = XGBClassifier(**kwargs)
        self.X = self.do_scaling(self._ensure_dimensionalit(X))
        self.Y = Y
        assert(len(Y.shape) ==1 )
        self.grid = self._gen_gridSearch(self.model, hyperparams, self.n_splits)
        self.grid_flag= False
    
    def fit(self, with_score = True, with_grid=True):
        if with_grid:
            self.grid.fit(self.X, self.Y)
            print(f"The best parameters are {self.grid.best_params_} and the best score is {self.grid.best_score_}")
            self.model = self.model.__class__(**self.grid.best_params_)
            self.model.fit(self.X,self.Y)
            self.grid_flag = True

        else : 
            self.model.fit(self.X, self.Y )
        if with_score:
            pred = self.predict(self.X)
            print(f"Train acc  is : {self._acc(pred, self.Y)}")

    def predict(self, X):
        X = self._ensure_dimensionalit(X)
        return self.model.predict(X)
