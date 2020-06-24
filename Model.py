import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, BayesianRidge, SGDRegressor
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor


class ModelCreator:
    """
    Models Are; svr, knn, tree, logistic, linear, ridge, lasso, bayesian, sgd
    Set test_split_available to split test and train
    """
    def __init__(self, x_train, y_train, model_name, test_split_available=False):
        if test_split_available:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x_train, y_train, test_size=0.1, shuffle=False)
            # self.x_train = x_train[:int(0.9*len(x_train))]
            # self.y_train = y_train[:int(0.9*len(y_train))]
            # self.x_test = x_train[int(0.9*len(x_train)):]
            # self.y_test = y_train[int(0.9*len(x_train)):]
            print(self.x_train, self.x_test)
        else:
            self.x_test = x_train
            self.y_test = y_train
            self.x_train = x_train
            self.y_train = y_train
        self.y_predict_test = {}
        self.y_predict_train = {}
        self.models = {'svr': SVR(), 'knn': KNeighborsRegressor(), 'tree': DecisionTreeRegressor(),
                       'logistic': LogisticRegression(), 'linear': LinearRegression(), 'ridge': Ridge(),
                       'lasso': Lasso(), 'bayesian': BayesianRidge(), 'sgd': SGDRegressor()}
        self.model = self.models[model_name]
        self.model_name = model_name

    def fit(self, show_train_error=False, show_output=False):
        regr = self.model
        regr.fit(self.x_train, self.y_train)
        self.y_predict_test = regr.predict(self.x_test)
        self.y_predict_train = regr.predict(self.x_train)
        if show_output:
            df = pd.DataFrame({'Actual': self.y_test, 'Predicted': self.y_predict_test})
            print(df)

        print("########### Test Error for Model name: ", self.model_name, " ###########")
        print('Mean Absolute Error:', mean_absolute_error(self.y_test, self.y_predict_test))
        print('Mean Squared Error:', mean_squared_error(self.y_test, self.y_predict_test))
        print('Root Mean Squared Error:', np.sqrt(mean_squared_error(self.y_test, self.y_predict_test)))

        if show_train_error:
            print("########### Train Error for ###########")
            print('Mean Absolute Error:', mean_absolute_error(self.y_train, self.y_predict_train))
            print('Mean Squared Error:', mean_squared_error(self.y_train, self.y_predict_train))
            print('Root Mean Squared Error:', np.sqrt(mean_squared_error(self.y_train, self.y_predict_train)))
        print()

    def plot_input(self, custom_figure, custom_column):
        custom_figure()
        plt.title('Daily New Corona Virus Cases')
        plt.xlabel('Day')
        plt.ylabel('New Cases')
        plt.plot(self.y_train)
        plt.show()
        plt.title(custom_column)
        plt.xlabel('Day')
        plt.ylabel(custom_column)
        plt.plot(self.x_train[custom_column])
        plt.show()

    def plot_output(self, custom_figure, test_target):
        custom_figure()
        if test_target:
            predicted, = plt.plot(self.y_predict_test, label='Predicted')
            actual, = plt.plot(self.y_test, label='Actual')
        else:
            predicted, = plt.plot(self.y_predict_train, label='Predicted')
            actual, = plt.plot(self.y_train, label='Actual')
        plt.xlabel('Day')
        plt.ylabel('New Cases')
        plt.title(self.model_name)
        plt.legend = ['Predict', 'Actual']
        plt.legend = [predicted, actual]
        plt.show()

    def train_model(self, custom_figure=plt.figure, custom_column='total_cars', test_target=True, plot_input=False,
                    plot_output=True):
        """
        :param custom_figure: to change figure in Plot
        :param custom_column: Column for input Plot
        :param test_target: if true plot train data and predict
        :param plot_input: if true plot input target and custom_column
        :param plot_output: if true plot output predict and actual base on test_target
        :return None:
        """
        if plot_input:
            self.plot_input(custom_figure=custom_figure, custom_column=custom_column)
        self.fit()
        if plot_output:
            self.plot_output(custom_figure=custom_figure, test_target=test_target)
