import yfinance as yf
import numpy as np
import pandas as pd
from datetime import date,timedelta
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from dash.dependencies import Input, Output, State
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, LSTM


def forecast_indicator(start_date,end_date,input1,input2):
    # df = yf.download(input2,start_date,end_date)
    # scaler = MinMaxScaler(feature_range=(0,1))
    # scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1,1))

    # prediction_days = 60

    # x_train = []
    # y_train = []

    # for x in range(prediction_days, len(scaled_data)):
    #     x_train.append(scaled_data[x-prediction_days:x, 0])
    #     y_train.append(scaled_data[x, 0])

    # x_train, y_train = np.array(x_train), np.array(y_train)
    # x_train = np.atleast_2d(x_train)
    # x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

    # model = Sequential()

    # model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    # model.add(Dropout(0.2))
    # model.add(LSTM(units=50, return_sequences=True))
    # model.add(Dropout(0.2))
    # model.add(LSTM(units=50))
    # model.add(Dropout(0.2))
    # model.add(Dense(units=1))

    # model.compile(optimizer='adam', loss='mean_squared_error')
    # model.fit(x_train, y_train, epochs=60, batch_size=32)

    # test_start = end_date
    # test_end = pd.to_datetime(end_date) + pd.DateOffset(days=input1)

    # test_data = yf.download(input2,test_start,test_end)
    # actual_prices = test_data['Close'].values

    # total_dataset = pd.concat((df['Close'], test_data['Close']), axis=0)

    # model_inputs = total_dataset[len(total_dataset) -len(test_data) - prediction_days:].values
    # model_inputs = model_inputs.reshape(-1, 1)
    # model_inputs = scaler.transform(model_inputs)

    # x_test =[]

    # for x in range(prediction_days, len(model_inputs)):
    #     x_test.append(model_inputs[x-prediction_days:x, 0])

    # x_test = np.array(x_test)
    # x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # predicted_prices = model.predict(x_test)
    # predicted_prices = scaler.inverse_transform(predicted_prices)

    df = yf.download(input2,start_date,end_date)

    # df.head()
    df = df[['Close']]
    
    forecast_out = input1

    df['Prediction'] = df[['Close']].shift(-forecast_out)
     
    X = np.array(df.drop(['Prediction'],1))

    X = X[:-forecast_out]

    y = np.array(df['Prediction'])

    y = y[:-forecast_out]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # gsc = GridSearchCV(
    #     estimator=SVR(kernel='rbf'),
    #     param_grid={
    #         'C': [0.1, 1, 100, 1000],
    #         'epsilon': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
    #         'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5]
    #     },
    #     cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
    
    # grid_result = gsc.fit(X, y)
    # best_params = grid_result.best_params_
    # best_svr = SVR(kernel='rbf', C=best_params["C"], epsilon=best_params["epsilon"], gamma=best_params["gamma"],
    #                 coef0=0.1, shrinking=True,
    #                 tol=0.001, cache_size=200, verbose=False, max_iter=-1)

    # print(grid_result)
    test_start = end_date
    test_end = pd.to_datetime(end_date) + pd.tseries.offsets.DateOffset(days=input1)
    df2 = yf.download(input2,test_start,test_end)

    # svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    # svr_rbf.fit(x_train, y_train)

    gsc = GridSearchCV(
        estimator=SVR(kernel='rbf'),
        param_grid={
            'C': [0.1, 1, 100, 1000],
            'epsilon': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
            'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5]
        },
        cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
    
    grid_result = gsc.fit(X, y)
    best_params = grid_result.best_params_
    best_svr = SVR(kernel='rbf', C=best_params["C"], epsilon=best_params["epsilon"], gamma=best_params["gamma"],
                    coef0=0.1, shrinking=True,
                    tol=0.001, cache_size=200, verbose=False, max_iter=-1)

    best_result = best_svr.fit(X,y)
    x_forecast = np.array(df.drop(['Prediction'],1))[-forecast_out:]

    svr_prediction = best_result.predict(x_forecast)
    print(svr_prediction)


    return svr_prediction