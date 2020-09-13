import datetime
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
plt.style.use('seaborn-dark')


from data_handling import get_synthetic_dataset, preprocess_for_supervised_learning, convert_time_series_to_relative


def forecast(model, df_training, df_prediction):
    # Trains the model with df_training and performs
    # prediction on df_prediction
    
    X_train = df_training.iloc[:,1:].values
    y_train = df_training.iloc[:,0].values
    
    X_test = df_prediction.iloc[:,1:].values
    # y_test = df_prediction.iloc[:,0].values
    
    model.fit(X_train, y_train)
    
    y_predicted = model.predict(X_test)
  
    result = pd.Series(data=y_predicted, index = df_prediction.index)
    
    return result


df = get_synthetic_dataset()
df = convert_time_series_to_relative(df)
df = preprocess_for_supervised_learning(df)

start = df.index[0] 
end = df.index[-1]
start_forecast = datetime.datetime(2100, 12, 31)
t_train = df.index[df.index < start_forecast]
t_forecast = df.index[df.index >= start_forecast]

df_training = df.loc[t_train,:]
df_prediction = df.loc[t_forecast,:]
y_predict = df_prediction['y'].values


model = LinearRegression()
#model = GradientBoostingRegressor(n_estimators = 70, max_depth = 2, min_samples_split=2, learning_rate = 0.5)

y_forecast = forecast(model, df_training, df_prediction)
y_forecast = y_forecast.values


mse = mean_squared_error(y_forecast, y_predict)



# Cast datetimes for matplotlib

fig, ax = plt.subplots()

ax.plot(df['y'], label='real')
ax.plot(t_forecast,y_forecast, label='OLS - mse: ' + str(round(mse,2)), alpha=0.7)

ax.axvline(x=start_forecast, ymin=0, ymax=1, color='black',linestyle='--', alpha=0.5)

ax.set_xlabel('year') 
ax.set_ylabel('change in %') 
ax.set_title("GDP growth - real vs. forecast")
ax.legend()
fig.autofmt_xdate()
plt.grid()


#plt.savefig('forecast_out_of_time.png', dpi = 350)
#plt.close()