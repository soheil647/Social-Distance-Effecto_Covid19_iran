from scipy.optimize import minimize, optimize
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from scipy import integrate, optimize

from PreProsses import PreProcess


def fitFunc(sir_values, time, beta, gamma, k):
    s = sir_values[0]
    i = sir_values[1]
    r = sir_values[2]

    res = np.zeros((3))
    res[0] = - beta * s * i
    res[1] = beta * s * i - gamma * i
    res[2] = gamma * i
    return res

def lsq(model, xdata, ydata, n):
    """least squares"""
    time_total = xdata
    # original record data
    data_record = ydata
    # normalize train data
    k = 1.0/sum(data_record)
    # init t = 0 values + normalized
    I0 = data_record[0]*k
    S0 = 1 - I0
    R0 = 0
    N0 = [S0,I0,R0]
    # Set initial parameter values
    param_init = [0.75, 0.75]
    param_init.append(k)
    # fitting
    param = minimize(sse(model, N0, time_total, k, data_record, n), param_init, method="nelder-mead").x
    # get the fitted model
    Nt = integrate.odeint(model, N0, time_total, args=tuple(param))
    # scale out
    Nt = np.divide(Nt, k)
    # Get the second column of data corresponding to I
    return Nt[:,1]

def sse(model, N0, time_total, k, data_record, n):
    """sum of square errors"""
    def result(x):
        Nt = integrate.odeint(model, N0, time_total[:n], args=tuple(x))
        INt = [row[1] for row in Nt]
        INt = np.divide(INt, k)
        difference = data_record[:n] - INt
        # square the difference
        diff = np.dot(difference, difference)
        return diff
    return result

def sir_model(y, x, beta, gamma):
    S = -beta * y[0] * y[1] / N
    R = gamma * y[1]
    I = -(S + R)
    return S, I, R

def fit_odeint(x, beta, gamma):
    return integrate.odeint(sir_model, (S0, I0, R0), x, args=(beta, gamma))[:,1]


train, target = PreProcess('Tehran').process_input_data()
N = 1.0
I0 = target[0]
S0 = N - I0
R0 = 0.0

popt, pcov = optimize.curve_fit(fit_odeint, train, target)
fitted = fit_odeint(train, *popt)

plt.plot(train, target, 'o')
plt.plot(train, fitted)
plt.show()


# result = lsq(fitFunc, train, target, 60)
#
# # Plot data and fit
# pl.clf()
# pl.plot(train, target, "o")
# pl.plot(train, result)
# pl.show()