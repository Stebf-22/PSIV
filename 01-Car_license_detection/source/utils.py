import numpy as np
import pandas as pd
import scipy
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


class Utils(object):

    def func_overX(self, X):
        Y = []
        for element in X:
            Y.append(sum(element.flatten()) > element.flatten().shape[0] // 2)
        return np.asarray(Y)

    def testing(self):

        ### SVM Testing ###
        X = np.random.rand(200, 60, 30)
        Y = func_overX(X)

        ### GridSearch ###
        model = SVMCHandler(X, Y)
        model.fit(with_score=True, with_grid=True)

    def _ensure_dimensionalit(self, arr):
        return arr if len(arr[0].shape) == 1 else [x.flatten() for x in arr]

    def _acc(self, y_pred, y_target):

        if type(y_pred) == np.array and type(y_target) == np.array:
            assert(y_pred.shape == y_target.shape)
            mask = y_pred == y_target

        else:
            assert(len(y_pred) == len(y_target))
            mask = [x == y for x, y in zip(y_pred, y_target)]
        return sum(mask)/len(mask)

    def do_scaling(self, X):
        Scaler = StandardScaler()
        return Scaler.fit_transform(X)

    def _gen_gridSearch(self, model, hyperparams, n_splits=5):
        cv = StratifiedShuffleSplit(
            n_splits=n_splits, test_size=0.2, random_state=42)
        grid = GridSearchCV(model, param_grid=hyperparams,
                            cv=cv, n_jobs=6, verbose=3)
        return grid

    def df_Grid(self):
        if self.grid_flag:
            c = self.grid.__dict__['cv_results_']['params']
            a = ['params'] + \
                [f'split{n}_test_score' for n in range(self.n_splits)]
            data = pd.DataFrame({h: i for h, i in zip(
                a, (c, *[self.grid.__dict__['cv_results_'][f'split{n}_test_score'] for n in range(self.n_splits)]))})
            
            data[list(data['params'][0].keys())] = pd.DataFrame(data['params'].tolist())
            
            return data
        else:
            print('Grid has not been calculated')
                       
    def ci(self, alpha):
        def f(x):
            return scipy.stats.t.interval(alpha = alpha, df = self.n_splits - 1, loc = x['mean'], scale = x['sem'])
        return f

    def top_params(self, alpha = 0.95, n = None): #retorna els parametres amb millor ci acc
        df = self.df_Grid()
        df['mean'] = df.filter(regex='test').mean(axis = 1) #agafem columnes nombrades 'split*' calculem mitja
        df['sem'] = df.filter(regex='test').apply(scipy.stats.sem, axis = 1) + 1e-8 #standard error of mean
        df['ci'] = df.apply(self.ci(alpha), axis = 1)
        df['sort'] = [0.5 * x[1] + abs(x[0] - x[1]) * 0.5 for x in df['ci']] 
        df = df.sort_values('sort', ascending=False)
        
        if n:
            return df[:n]
        return df[:]
