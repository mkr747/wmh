import numpy as np
from attribute import Attribute


class DataProvider:
    def __init__(self):
        self.data_names = [
            'AMZN 1Y.csv', 'AMZN 3M.csv', 'AMZ 6M.csv', 'apple 1Y', 'apple 3M.csv', 'apple 6M.csv', 'FB 1Y.csv', 'FB 3M.csv', 'FB 6M.csv']
        self.cvs = np.arange(1, 10)
        self.kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
        self.degrees = np.arange(1, 10, 1)
        self.gammas = np.arange(0.0, 1.0, 0.01)
        self.coef0s = np.arange(0, 20, 0.1)
        self.tols = np.arange(0, 5, 0.01)
        self.cs = np.arange(0.0, 10)
        self.epsilon = np.arange(0.0, 5, 0.1)
        self.shrinking = [True, False]
        self.verbose = [False]
        attributes = {}
        attributes['data_name'] = Attribute(
            name='data_name', is_enabled=True, values=self.data_names)
        attributes['cv'] = Attribute(
            name='cv', is_enabled=True, values=self.cvs)
        attributes['kernels'] = Attribute(
            name='kernels', is_enabled=True, values=self.kernels)
        attributes['degree'] = Attribute(
            name='degree', is_enabled=True, values=self.degrees)
        attributes['gamma'] = Attribute(
            name='gamma', is_enabled=True, values=self.gammas)
        attributes['coef0'] = Attribute(
            name='coef0', is_enabled=True, values=self.coef0s)
        attributes['tol'] = Attribute(
            name='tol', is_enabled=True, values=self.tols)
        attributes['C'] = Attribute(
            name='C', is_enabled=True, values=self.cs)
        attributes['epsilon'] = Attribute(
            name='epsilon', is_enabled=True, values=self.epsilon)
        attributes['shrinking'] = Attribute(
            name='shrinking', is_enabled=True, values=self.shrinking)
        attributes['verbose'] = Attribute(
            name='verbose', is_enabled=True, values=self.verbose)
        self.attributes = attributes.copy()
        print(self.attributes['data_name'].values)

    def get_linear_params(self):
        params = {}
        params['data_name'] = Attribute(
            name='data_name', is_enabled=True, values=[])
        params['cv'] = Attribute(
            name='cv', is_enabled=True, values=self.cvs)
        params['kernels'] = Attribute(
            'kernels', is_enabled=True, values=['linear'])
        params['coef0'] = Attribute(
            'coef0', is_enabled=True, values=self.coef0s)
        params['tol'] = Attribute(
            'tol', is_enabled=True, values=self.tols)
        params['C'] = Attribute('C', is_enabled=True, values=self.cs)

        return params

    def get_poly(self):
        pass

    def get_rbf(self):
        pass

    def get_sigmoid(self):
        pass
