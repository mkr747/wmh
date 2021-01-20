import pandas as pd
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


class SVRCalculator:
    def __init__(self):
        pass

    def linear(self, data_name, tol, C, epsilon, shrinking, visualization=False):
        regr = SVR(kernel='linear', C=C, tol=tol,
                   epsilon=epsilon, shrinking=shrinking)
        SVRCalculator.calculate_svr(regr, data_name, visualization)

    def poly(self, data_name, degree, gamma, coef0, tol, C, epsilon, shrinking, visualization=False):
        regr = SVR(kernel='poly', degree=degree, gamma=gamma, coef0=coef0,
                   tol=tol, C=C, epsilon=epsilon, shrinking=shrinking)
        SVRCalculator.calculate_svr(regr, data_name, visualization)

    def rbf(self, data_name, gamma, tol, C, epsilon, shrinking, visualization=False):
        regr = SVR(kernel='rbf', gamma=gamma, tol=tol, C=C,
                   epsilon=epsilon, shrinking=shrinking)
        SVRCalculator.calculate_svr(regr, data_name, visualization)

    def sigmoid(self, data_name, gamma, coef0, tol, C, epsilon, shrinking, visualization=False):
        regr = SVR(kernel='sigmoid', gamma=gamma, coef0=coef0,
                   tol=tol, C=C, epsilon=epsilon, shrinking=shrinking)
        SVRCalculator.calculate_svr(regr, data_name, visualization)

    def precomputed(self, data_name, tol, C, epsilon, shrinking, visualization=False):
        regr = SVR(kernel='precomputed', tol=tol, C=C,
                   epsilon=epsilon, shrinking=shrinking)
        SVRCalculator.calculate_svr(regr, data_name, visualization)

    def grid_search(self, params, data_name):
        regr = SVR(kernel='rbf', C=1.0, epsilon=0.2)
        print(regr.get_params().keys())
        min_date, max_date, y, X = SVRCalculator.get_data(data_name)
        X, y, sc_y, next_day, next_week, next_month, sc_X = SVRCalculator.data_normalization(
            max_date, min_date, y, X)
        grid_search = GridSearchCV(estimator=regr, param_grid=params,
                                   cv=5, n_jobs=-1, verbose=0, refit=True)

        grid_search.fit(X, y[0])
        next_day_prediction = sc_y.inverse_transform(
            grid_search.predict(next_day))
        next_week_prediction = sc_y.inverse_transform(
            grid_search.predict(next_week))
        next_month_prediction = sc_y.inverse_transform(
            grid_search.predict(next_month))

        print("Next day prediction: ", next_day_prediction)
        print("Next week prediction: ", next_week_prediction)
        print("Next month prediction: ", next_month_prediction)

        SVRCalculator.count_errors(y, X, sc_y, regr)

    @staticmethod
    def calculate_svr(regr, data_name, visualization=False):
        min_date, max_date, y, X = SVRCalculator.get_data(data_name)
        X, y, sc_y, next_day, next_week, next_month, sc_X = SVRCalculator.data_normalization(
            max_date, min_date, y, X)

        # regression
        regr.fit(X, y)

        # inverse transform is needed for readable form
        next_day_prediction = sc_y.inverse_transform(regr.predict(next_day))
        next_week_prediction = sc_y.inverse_transform(regr.predict(next_week))
        next_month_prediction = sc_y.inverse_transform(
            regr.predict(next_month))

        print("Next day prediction: ", next_day_prediction)
        print("Next week prediction: ", next_week_prediction)
        print("Next month prediction: ", next_month_prediction)

        # error
        SVRCalculator.count_errors(y, X, sc_y, regr)
        # visualization
        if(visualization):
            SVRCalculator.visual_presentation(
                data_name, sc_X, X, sc_y, y, regr)

    @staticmethod
    def count_errors(y, X, sc_y, regr):
        mse = mean_squared_error(sc_y.inverse_transform(
            y), sc_y.inverse_transform(regr.predict(X)))
        print("Mse: ", mse)
        scores = cross_val_score(regr, sc_y.inverse_transform(
            y), sc_y.inverse_transform(regr.predict(X)))
        print(scores)

    @staticmethod
    def data_normalization(max_date, min_date, y, X):
        # to be predicted
        days_between = max_date - min_date
        next_day = [days_between.days + 1]
        next_week = [days_between.days + 7]
        next_month = [days_between.days + 30]

        #values normalization
        sc_y = StandardScaler()
        y = pd.DataFrame(sc_y.fit_transform(y.values.reshape(-1, 1)))

        sc_X = StandardScaler()
        to_be_transformed = X.append(next_day).append(
            next_week).append(next_month)
        transformed_arguments = sc_X.fit_transform(to_be_transformed)
        X = pd.DataFrame(transformed_arguments[:-3])
        next_day = [transformed_arguments[-3]]
        next_week = [transformed_arguments[-2]]
        next_month = [transformed_arguments[-1]]

        return X, y, sc_y, next_day, next_week, next_month, sc_X

    @staticmethod
    def get_data(data_name):
        data_dir = './data/'
        WINDOW_SIZE = 5

        dataset = pd.read_csv(data_dir + data_name)

        # Dates cannot be processed by svr, need to convert into numbers
        dates = pd.to_datetime(dataset.loc[:, 'Date'])
        min_date = dates.min()
        max_date = dates.max()
        dates = pd.DataFrame([(date - min_date).days for date in dates])

        # window operation
        y = dataset.loc[:, 'Open']
        y = y.rolling(WINDOW_SIZE).mean()[WINDOW_SIZE - 1:]
        X = dates[WINDOW_SIZE - 1:]

        return min_date, max_date, y, X
