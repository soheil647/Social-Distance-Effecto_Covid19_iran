import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def vector_regression(X_train, y_train):
    regressor = SVR(kernel='rbf')
    X_train = X_train.reshape(-1, 1)
    regressor.fit(X_train, y_train)
    # 5 Predicting a new result
    y_pred = regressor.predict(6.5)
    df = pd.DataFrame({'Actual': y_train, 'Predicted': y_pred})
    print(df)


def linear_regression(X_train, X_test, y_train, y_test):
    logistic = linear_model.LinearRegression()
    logistic.fit(X_train, y_train)

    y_predict_train = logistic.predict(X_train)
    y_predict = logistic.predict(X_test)
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_predict})
    print(df)

    # print("Logistic Classifier")
    # print("For Train Predicts\n", sklearn.metrics.classification_report(y_train, y_predict_train))
    # print("Accuracy: ", sklearn.metrics.accuracy_score(y_train, y_predict_train))
    # print("For Test Predicts\n", sklearn.metrics.classification_report(y_test, y_predict))
    # print("Accuracy: ", sklearn.metrics.accuracy_score(y_test, y_predict))
    # print("\n")

    plt.plot(y_test,  color='gray')
    # plt.plot(X_test, y_predict, color='red', linewidth=2)
    plt.show()
    print('Mean Absolute Error:', sklearn.metrics.mean_absolute_error(y_test, y_predict))
    print('Mean Squared Error:', sklearn.metrics.mean_squared_error(y_test, y_predict))
    print('Root Mean Squared Error:', np.sqrt(sklearn.metrics.mean_squared_error(y_test, y_predict)))


df = pd.read_excel('Travel_data.xlsx', index_col='Date')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
y_train = df['Daily New Cases']
del df['Daily New Cases']
x_train = df
# x_train = df['total_cars']
# y_train = np.reshape(list(y_train), (-1,1))
# # x_train = np.reshape(list(x_train), (-1,1))
# sc_X = StandardScaler()
# sc_y = StandardScaler()
# X = sc_X.fit_transform(x_train)
# y = sc_y.fit_transform(y_train)

# Create a minimum and maximum processor object
min_max_scaler = MinMaxScaler()

# Create an object to transform the data to fit minmax processor
x_scaled = min_max_scaler.fit_transform(x)

# Run the normalizer on the dataframe
df_normalized = pd.DataFrame(x_scaled)

# print(X)
# X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
# print(x_train)
linear_regression(x_train, x_train, y_train, y_train)
# vector_regression(X, y)