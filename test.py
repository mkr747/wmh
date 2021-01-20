from svr_calculator import SVRCalculator
import numpy as np
from datetime import datetime


class Test:
    def __init__(self):
        self.degrees = np.arange(1, 10, 0.1)
        self.gammas = np.arange(0.0, 1.0, 0.01)
        self.coef0s = np.arange(0, 20, 0.1)
        self.tols = np.arange(0, 2, 0.001)
        self.Cs = np.arange(1, 100)
        self.epsilons = [0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
        print(self.epsilons)
        self.shrinking = [True, False]
        self.calculator = SVRCalculator()

    def linear_test(self, data_name):
        for tol in self.tols:
            for C in self.Cs:
                for epsilon in self.epsilons:
                    for shrink in self.shrinking:
                        print(
                            f'tol: {tol}, C: {C}, epsilon: {epsilon}, shrink: {shrink}')
                        self.calculator.linear(
                            data_name=data_name, tol=tol, C=C, epsilon=epsilon, shrinking=shrink, visualization=False)

    def poly_test(self, data_name):
        for degree in self.degrees:
            for gamma in self.gammas:
                for coef0 in self.coef0s:
                    for tol in self.tols:
                        for C in self.Cs:
                            for epsilon in self.epsilons:
                                for shrink in self.shrinking:
                                    self.calculator.poly(data_name=data_name, degree=degree, gamma=gamma, coef0=coef0,
                                                         tol=tol, C=C, epsilon=epsilon, shrinking=shrink, visualization=False)

    def rbf_test(self, data_name):
        for gamma in self.gammas:
            for tol in self.tols:
                for C in self.Cs:
                    for epsilon in self.epsilons:
                        for shrink in self.shrinking:
                            self.calculator.rbf(data_name=data_name, gamma=gamma, tol=tol,
                                                C=C, epsilon=epsilon, shrinking=shrink, visualization=False)

    def sigmoid_test(self, data_name):
        for gamma in self.gammas:
            for coef0 in self.coef0s:
                for tol in self.tols:
                    for C in self.Cs:
                        for epsilon in self.epsilons:
                            for shrink in self.shrinking:
                                self.calculator.sigmoid(data_name=data_name, gamma=gamma, coef0=coef0,
                                                        tol=tol, C=C, epsilon=epsilon, shrinking=shrink, visualization=False)

    def precomputed_test(self, data_name):
        for tol in self.tols:
            for C in self.Cs:
                for epsilon in self.epsilons:
                    for shrink in self.shrinking:
                        self.calculator.precomputed(
                            data_name=data_name, tol=tol, C=C, epsilon=epsilon, shrinking=shrink, visualization=False)

    def linear_search_grid_test(self, data_name):
        params = {
            'kernel': ['linear'],
            'tol': self.tols,
            'C': self.Cs,
            'epsilon': self.epsilons,
            'shrinking': self.shrinking
        }

        today = datetime.today()
        print(today)
        self.calculator.grid_search(params, data_name)
        end = datetime.today()
        print(end)

    def rbf_search_grid_test(self, data_name):
        params = {
            'kernel': ['rbf'],
            'gamma': self.gammas,
            'tol': self.tols,
            'C': self.Cs,
            'epsilon': self.epsilons,
            'shrinking': self.shrinking
        }

        today = datetime.today()
        print(today)
        self.calculator.grid_search(params, data_name)
        end = datetime.today()
        print(end)

    def poly_search_grid_test(self, data_name):
        params = {
            'kernel': ['poly'],
            'degree': self.degrees,
            'gamma': self.gammas,
            'coef0': self.coef0s,
            'tol': self.tols,
            'C': self.Cs,
            'epsilon': self.epsilons,
            'shrinking': self.shrinking
        }

        today = datetime.today()
        print(today)
        self.calculator.grid_search(params, data_name)
        end = datetime.today()
        print(end)

    def sigmoid_search_grid_test(self, data_name):
        params = {
            'kernel': ['poly'],
            'gamma': self.gammas,
            'coef0': self.coef0s,
            'tol': self.tols,
            'C': self.Cs,
            'epsilon': self.epsilons,
            'shrinking': self.shrinking
        }

        today = datetime.today()
        print(today)
        self.calculator.grid_search(params, data_name)
        end = datetime.today()
        print(end)

    def precomputed_search_grid_test(self, data_name):
        params = {
            'kernel': ['poly'],
            'tol': self.tols,
            'C': self.Cs,
            'epsilon': self.epsilons,
            'shrinking': self.shrinking
        }

        today = datetime.today()
        print(today)
        self.calculator.grid_search(params, data_name)
        end = datetime.today()
        print(end)
