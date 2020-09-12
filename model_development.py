import datetime

from data_handling import get_synthetic_dataset, preprocess_for_supervised_learning, convert_time_series_to_relative


df = get_synthetic_dataset()
df = convert_time_series_to_relative(df)
df = preprocess_for_supervised_learning(df)


start_forecast = 2007
t_train = df.index[df.index < datetime.datetime(start_forecast,12,31)]
t_forecast = df.index[df.index >= datetime.datetime(start_forecast,12,31)]

df_training = df.loc[t_train,:]
df_prediction = df.loc[t_forecast,:]
y_predict = df_prediction['y'].values