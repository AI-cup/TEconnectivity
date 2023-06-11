import pandas as pd
from pmdarima.arima import auto_arima
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import numpy as np
import matplotlib.pyplot as plt
from darts import TimeSeries
from datetime import datetime
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from darts.models import ExponentialSmoothing, TBATS, AutoARIMA, Theta
import joblib


def classify_data(data):
    # Calculate the mean and standard deviation of the data
    mean = sum(data) / len(data)
    std_dev = (sum((x - mean) ** 2 for x in data) / len(data)) ** 0.5

    # Classify the data based on its mean and standard deviation
    if std_dev < mean / 10:
        return "smooth"
    elif std_dev > mean / 5:
        return "lumpy"
    else:
        return "intermittent"


def create_timeseries(data, t_index):
    pd.to_datetime(data[t_index], infer_datetime_format=True)
    df.head()


def part_indexing(data):
    p_index = data.iloc[:0, 1:]
    return p_index

def load_model(model_path):
    model = joblib.load(model_path)

df = pd.read_csv("data/data_final.csv")
df.head()
part_indexing(df)
create_timeseries(df, 'Date')

def n_beats_model(series1, train, val):
    from darts.models import NBEATSModel
    model = NBEATSModel(input_chunk_length=3, output_chunk_length=3, random_state=42)
    model.fit(train, epochs=50, verbose=True);
    filename = 'saved_models/finalized_model'+str(part)+'.sav'
    joblib.dump(model, filename)
    pred = model.predict(series=train, n=6)
    # plt.figure(figsize=(10, 6))
    # series1.plot(label="actual")
    # pred.plot(label="forecast")
    pred.to_csv('prediction/prediction_nbeats_'+str(part)+'.csv')
    # error =[mape(val,pred), r2_score(val, pred)]
    # df13 = pd.DataFrame(error)
    # df13.to_csv('prediction/error_nbeats_'+str(part)+'.csv')


def croston(ts, extra_periods=1, alpha=0.1):
    """Implementation of Croston's method for intermittent demand forecasting.
    Parameters:
    ts: pandas.DataFrame, the input time series data, with the first column being the demand data.
    extra_periods: int, optional (default=1), the number of periods to forecast beyond the end of the input time series.
    alpha: float, optional (default=0.1), the smoothing parameter for the EWMA method.
    Returns:
    forecast: pandas.DataFrame, the forecasted demand values and probabilities.
    """
    # Initialize variables
    demand = np.array(ts.iloc[:, 0])  # demand data
    periods = len(demand)
    d_hat = np.zeros(periods)  # forecasted demand
    p = np.zeros(periods)  # probability of demand occurrence
    f = np.zeros(periods)  # forecast error
    q = np.zeros(periods)  # inter-demand intervals
    # Calculate forecasted demand and probability
    for i in range(periods):
        if demand[i] > 0:
            p[i] = 1
            d_hat[i] = demand[i]
        q[i] = 1 + (1 - p[i]) * q[i - 1]
        d_hat[i] = alpha * demand[i] + (1 - alpha) * d_hat[i - 1]
    # Calculate forecast error and adjusted alpha
    for i in range(periods):
        if demand[i] == 0:
            f[i] = 0
        else:
            f[i] = demand[i] - d_hat[i]
    nonzero_f = f[np.nonzero(f)]
    if len(nonzero_f) > 0:
        alpha_adj = 1 - np.exp(np.log(1 - alpha) / (1 + (np.mean(q[1:][demand[1:] > 0]) / np.mean(nonzero_f))))
    else:
        alpha_adj = alpha
    # Calculate forecast for extra periods
    d_hat_extra = np.zeros(extra_periods)
    p_extra = np.zeros(extra_periods)
    q_extra = np.zeros(extra_periods)
    for i in range(extra_periods):
        q_extra[i] = 1 + (1 - p[-1]) * q[-1]
        d_hat_extra[i] = alpha_adj * d_hat[-1]
        p_extra[i] = alpha_adj * p[-1] + (1 - alpha_adj)
    # Combine results
    forecast = np.concatenate((d_hat, d_hat_extra))
    probability = np.concatenate((p, p_extra))
    result = pd.DataFrame({'Demand': forecast, 'Probability': probability})

    return result


for part in part_indexing(df):
    data_pd = df[part]
    summary1 = df[part].describe()
    string = classify_data(data_pd)
    df11 = pd.DataFrame([string], columns=['type of data'])
    summary1 = pd.concat([summary1, df11])
    train_data = df[part].iloc[:-6]  # Use all data except the last 12 months for training
    test_data = df[part].iloc[-6:]  # Use the last 12 months for testing
    model = auto_arima(train_data, seasonal=True, stepwise=True, suppress_warnings=True)
    model.fit(train_data)
    predictions = model.predict(n_periods=len(test_data))
    filename = 'saved_models/finalized_model' + str(part) + '.sav'
    joblib.dump(model, filename)
    predictions = predictions.round(decimals=0)
    print(predictions)
    df[part+'_predicted'] = predictions
    error = [mean_squared_error(test_data, predictions),mean_absolute_percentage_error(test_data, predictions), r2_score(test_data, predictions)]
    df13 = pd.DataFrame(error)
    df13.to_csv('data/error/'+str(part)+'_error.csv')

    # result1 = pd.concat([predictions, result1])
    # result1.to_csv('data/predict/predicted_data'+part+'.csv')
df.to_csv("result.csv")







