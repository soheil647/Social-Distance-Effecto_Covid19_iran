import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, BayesianRidge, RidgeCV, LassoLars, \
    ElasticNet, TheilSenRegressor, ARDRegression, RANSACRegressor, HuberRegressor
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, VotingRegressor
from sklearn.ensemble import AdaBoostRegressor


class ModelCreator:
    """
    Models Are; svr, knn, tree, logistic, linear, ridge, lasso, bayesian, ridgecv, LassoLars, ElasticNetو
    BaggingClassifier(base_estimator=estimator, n_estimators=number_of_estimator, max_features=0.5)

    Set test_split_available to split test and train
    """

    def __init__(self, x_train, y_train, test_split_available=False, test_size=0.1, shuffle=True, number_of_estimator=10, estimator=None, estimators=None, random_state=None):
        if test_split_available:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x_train, y_train,
                                                                                    test_size=test_size,
                                                                                    shuffle=shuffle,
                                                                                    random_state=random_state)
        else:
            self.x_test = x_train
            self.y_test = y_train
            self.x_train = x_train
            self.y_train = y_train
        self.y_predict_test = {}
        self.y_predict_train = {}
        self.models = {'svr': SVR(), 'knn': KNeighborsRegressor(), 'tree': DecisionTreeRegressor(),
                       'logistic': LogisticRegression(), 'linear': LinearRegression(), 'ridge': Ridge(),
                       'ridgecv': RidgeCV(), 'lasso': Lasso(), 'lassolars': LassoLars(alpha=0.1),
                       'bayesian': BayesianRidge(), 'ElasticNet': ElasticNet(),
                       'TheilSenRegressor': TheilSenRegressor(),
                       'ARDRegression': ARDRegression(), 'RANSACRegressor': RANSACRegressor(),
                       'HuberRegressor': HuberRegressor(), 'randomForest': RandomForestRegressor(n_estimators=50),
                       'boost': AdaBoostRegressor(random_state=0, n_estimators=100)}

        self.estimator = self.models[estimator]
        estimators_list = []
        for i in range(len(estimators)):
            estimators_list.append((estimators[i], self.models[estimators[i]]))

        self.models = {'svr': SVR(), 'knn': KNeighborsRegressor(), 'tree': DecisionTreeRegressor(),
                       'logistic': LogisticRegression(), 'linear': LinearRegression(), 'ridge': Ridge(),
                       'ridgecv': RidgeCV(), 'lasso': Lasso(), 'lassolars': LassoLars(alpha=0.1),
                       'bayesian': BayesianRidge(), 'ElasticNet': ElasticNet(),
                       'TheilSenRegressor': TheilSenRegressor(),
                       'ARDRegression': ARDRegression(), 'RANSACRegressor': RANSACRegressor(),
                       'HuberRegressor': HuberRegressor(), 'randomForest': RandomForestRegressor(n_estimators=50),
                       'bagging': BaggingRegressor(base_estimator=self.estimator, n_estimators=number_of_estimator, max_features=0.8),
                       'voting': VotingRegressor(estimators=estimators_list), 'boost': AdaBoostRegressor(random_state=0, n_estimators=100)}

    def fit(self, model_name, show_train_error=False, show_output=False):
        regr = self.models[model_name]
        regr.fit(self.x_train, self.y_train)
        self.y_predict_test = regr.predict(self.x_test)
        self.y_predict_train = regr.predict(self.x_train)
        if show_output:
            df = pd.DataFrame({'Actual': self.y_test, 'Predicted': self.y_predict_test})
            print(df)

        print("########### Test Error for Model name: ", model_name, " ###########")
        accuracy = regr.score(self.x_test, self.y_test)
        print('Accuracy is : ', accuracy * 100, '%')
        print('Mean Absolute Error:', mean_absolute_error(self.y_test, self.y_predict_test))
        print('Mean Squared Error:', mean_squared_error(self.y_test, self.y_predict_test))
        print('Root Mean Squared Error:', np.sqrt(mean_squared_error(self.y_test, self.y_predict_test)))

        scores = cross_val_score(regr, self.x_test, self.y_test, cv=5)
        print('Cross Score is: ', scores.mean())

        if show_train_error:
            print("########### Train Error for ###########")
            accuracy = regr.score(self.x_train, self.y_train)
            print('Accuracy is : ', accuracy * 100, '%')
            print('Mean Absolute Error:', mean_absolute_error(self.y_train, self.y_predict_train))
            print('Mean Squared Error:', mean_squared_error(self.y_train, self.y_predict_train))
            print('Root Mean Squared Error:', np.sqrt(mean_squared_error(self.y_train, self.y_predict_train)))
            scores = cross_val_score(regr, self.x_test, self.y_test, cv=5)
            print('Cross Score is: ', scores.mean())

    def plot_input(self, custom_figure, custom_column):
        custom_figure()
        plt.title('Daily New Corona Virus Cases')
        plt.xlabel('Day')
        plt.ylabel('New Cases')
        plt.plot(list(self.y_train))
        plt.show()
        plt.title(custom_column)
        plt.xlabel('Day')
        plt.ylabel(custom_column)
        plt.plot(list(self.x_train[custom_column]))
        plt.show()

    def plot_output(self, model_name, custom_figure, test_target):
        custom_figure()
        if test_target:
            predicted, = plt.plot(self.y_predict_test, label='Predicted')
            actual, = plt.plot(list(self.y_test), label='Actual')
        else:
            predicted, = plt.plot(self.y_predict_train, label='Predicted')
            actual, = plt.plot(list(self.y_train), label='Actual')
        plt.xlabel('Day')
        plt.ylabel('New Cases')
        plt.title(model_name)
        plt.legend(['Predicted', 'Actual'])
        plt.show()

    def train_model(self, model_name, custom_figure=plt.figure, custom_column='total_cars', test_target=True,
                    plot_input=False,
                    plot_output=True, show_train_error=False, show_output=False):
        """
        :param model_name: define model
        :param show_output: if true show outputs and predicted data:
        :param show_train_error: if true show train error:
        :param custom_figure: to change figure in Plot
        :param custom_column: Column for input Plot
        :param test_target: if true plot train data and predict
        :param plot_input: if true plot input target and custom_column
        :param plot_output: if true plot output predict and actual base on test_target
        :return None:
        """
        if plot_input:
            self.plot_input(custom_figure=custom_figure, custom_column=custom_column)
        self.fit(model_name=model_name, show_train_error=show_train_error, show_output=show_output)
        if plot_output:
            self.plot_output(model_name=model_name, custom_figure=custom_figure, test_target=test_target)
        print("END MODEL\n")
