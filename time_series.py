import copy
import pandas as pd
from pylab import rcParams
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA


def group_date(df_with_date_time):
    df = copy.deepcopy(df_with_date_time)
    df = df[['date']]
    df['date'] = pd.to_datetime(df.date)
    df = df.groupby(['date']).size().reset_index(name='count')
    # df[['date', 'count']].plot(x='date', linestyle='-', marker='o')

    df = df.set_index('date')
    y = df['count'].resample('D').mean()
    y = y.fillna(y.bfill())
    return y


def perodicity(y):
    rcParams['figure.figsize'] = 9, 5
    decomposition = sm.tsa.seasonal_decompose(y, model='additive')
    fig = decomposition.plot()
    plt.show()


def plot_data(y, y_hat, graph, error):
    y.plot()
    y_hat.plot(color='orange')
    plt.title('%s Error: %f'%(graph, error))
    plt.ylabel('Log of number of accidents')
    plt.show()


def last_observed(y):
    y_hat = y.shift(1)
    y_hat.ix[0] = y.ix[0]
    error = mean_squared_error(y, y_hat)
    plot_data(y, y_hat, 'Last Observed', error)
    # Last Observed Error = 13401
    return error


def fixed_average(y):
    y_hat = copy.deepcopy(y)
    for t in range(1,len(y)):
        y_hat.ix[t] = np.mean(y[0:t])
    error = mean_squared_error(y, y_hat)
    plot_data(y, y_hat, 'Fixed Average', error)
    # Fixed Average Error = 18608
    return error


def moving_average(y, w_size, plot):
    end = len(y) - 1
    rolling = y.rolling(window=w_size)
    y_hat = rolling.mean()
    y_hat = y_hat.shift(1)
    error = mean_squared_error(y[w_size:end], y_hat[w_size:end])
    if plot:
        plot_data(y, y_hat, 'Moving Average', error)
    return error


def get_window_size(y):
    err = []
    end = len(y) - 1
    for i in range(1, end):
        err.append(moving_average(y, i, 0))
    error = min(err)
    w_size = err.index(error)+1
    # plt.plot(err)
    # plt.ylabel('Error')
    # plt.show()
    # w_size, MA Error = (7, 12261.495888790296)
    return w_size, error


def seasonal(y, w_size):
    y_hat = copy.deepcopy(y)
    for t in range(w_size, len(y)):
        y_hat.ix[t] = y[t-w_size]
    error = mean_squared_error(y, y_hat)
    plot_data(y, y_hat, 'Seasonal', error)
    # Seasonal Error = 19518.549575070821
    return error


def EWMA(y, span, plot):
    end = len(y) - 1
    y_hat = pd.ewma(y, span=span, freq="D")
    y_hat = y_hat.shift(1)
    error = mean_squared_error(y[1:end], y_hat[1:end])
    plot_data(y, y_hat, 'EWMA', error)
    return error


def get_span(y):
    err = []
    end = len(y) - 1
    for i in range(1, end):
        err.append(EWMA(y, i, 0))
    error = min(err)
    span = err.index(error)+1
    plt.plot(err)
    plt.ylabel('Error')
    plt.show()
    # span, EWMA Error = (4, 11668.316300768642)
    return span, error


def AR_model(y):
    X = y.values
    train, test = X[1:len(X) - 40], X[len(X) - 40:]
    model = AR(train)
    model_fit = model.fit()
    # print('Lag: %s' % model_fit.k_ar)
    # print('Coefficients: %s' % model_fit.params)
    predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)
    # for i in range(len(predictions)):
    #     print('predicted=%f, expected=%f' % (predictions[i], test[i]))
    error = mean_squared_error(test, predictions)
    # print('Test MSE: %.3f' % error)
    plt.plot(test)
    plt.plot(predictions, color='orange')
    plt.title('AR Error: %f' % (error))
    plt.ylabel('Log of number of accidents')
    plt.show()
    # Lag: 16
    # Test MSE: 59472.904
    # [model_fit.k_ar, error] = 16, 59472.904165063774
    return model_fit.k_ar, error


def ARIMA_model(y, p, d, q):
    X = y.values
    train, test = X[1:len(X) - 40], X[len(X) - 40:]
    model = ARIMA(train, order=(p,d,q))
    model_fit = model.fit()
    # print('Lag: %s' % model_fit.k_ar)
    # print('Coefficients: %s' % model_fit.params)
    predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)
    for i in range(len(predictions)):
        print('predicted=%f, expected=%f' % (predictions[i], test[i]))
    error = mean_squared_error(test, predictions)
    # print('Test MSE: %.3f' % error)
    plt.plot(test)
    plt.plot(predictions, color='orange')
    plt.title('ARMA Error: %f' % (error))
    plt.ylabel('Log of number of accidents')
    plt.show()
    # error = 59941.62250510782
    return error


def run():
    df_with_date_time = pd.read_csv('cleaned_data.csv', sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
    y = group_date(df_with_date_time)
    y = np.log(y)
    perodicity(y)
    last_observed(y)
    fixed_average(y)
    [w_size, error_MA] = get_window_size(y)
    seasonal(y, w_size)
    [span, error_EWMA] = get_span(y)
    [lag, error_AR] = AR_model(y)
    ARIMA_model(y, lag, 0, span)


if __name__ == "__main__":
    run()