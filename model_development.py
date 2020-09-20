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


# =============================================================================
# # Get, preprocess and split data 
# =============================================================================

df = get_synthetic_dataset()
df = convert_time_series_to_relative(df)
df = preprocess_for_supervised_learning(df)

start = df.index[0] 
end = df.index[-1]
start_forecast = datetime.datetime(2000, 12, 31)
t_train = df.index[df.index < start_forecast]
t_forecast = df.index[df.index >= start_forecast]

df_training = df.loc[t_train,:]
df_prediction = df.loc[t_forecast,:]
y_predict = df_prediction['y'].values



# =============================================================================
# # Fitting Models - a list of models, each is a tuple
# =============================================================================

models = []


name = 'OLS'
model = LinearRegression()
y_forecast = forecast(model, df_training, df_prediction)
y_forecast = y_forecast.values
mse = mean_squared_error(np.exp(y_forecast), np.exp(y_predict))
models.append( (name, y_forecast, mse))

name = 'GBM1'
model = GradientBoostingRegressor(n_estimators = 5, max_depth = 3, 
                                  min_samples_split=2, learning_rate = 0.8)
y_forecast = forecast(model, df_training, df_prediction)
y_forecast = y_forecast.values
mse = mean_squared_error(np.exp(y_forecast), np.exp(y_predict))
models.append( (name, y_forecast, mse))


name = 'GBM2'
model = GradientBoostingRegressor(n_estimators = 20, max_depth = 2, 
                                  min_samples_split=2, learning_rate = 0.5)
y_forecast = forecast(model, df_training, df_prediction)
y_forecast = y_forecast.values
mse = mean_squared_error(np.exp(y_forecast), np.exp(y_predict))
models.append( (name, y_forecast, mse))

name = 'GBM3'
model = GradientBoostingRegressor(n_estimators = 15, max_depth = 2, 
                                  min_samples_split=3, learning_rate = 0.5)
y_forecast = forecast(model, df_training, df_prediction)
y_forecast = y_forecast.values
mse = mean_squared_error(np.exp(y_forecast), np.exp(y_predict))
models.append( (name, y_forecast, mse))


name = 'GBM4'
model = GradientBoostingRegressor(n_estimators = 10, max_depth = 3, 
                                  min_samples_split=2, learning_rate = 0.5)
y_forecast = forecast(model, df_training, df_prediction)
y_forecast = y_forecast.values
mse = mean_squared_error(np.exp(y_forecast), np.exp(y_predict))
models.append( (name, y_forecast, mse))


name = 'GBM5'
model = GradientBoostingRegressor(n_estimators = 5, max_depth = 10, 
                                  min_samples_split=2, learning_rate = 0.9)
y_forecast = forecast(model, df_training, df_prediction)
y_forecast = y_forecast.values
mse = mean_squared_error(np.exp(y_forecast), np.exp(y_predict))
models.append( (name, y_forecast, mse))


# =============================================================================
# # Create Plot
# =============================================================================

fig, ax = plt.subplots()

ax.plot(np.exp(df['y']), label='real')

for model in models:
    name = model[0]
    y_forecast = model[1]
    mse = model[2]
    
    label = name + ' ' + str(round(mse,2))
    
    ax.plot(t_forecast, np.exp(y_forecast), label=label, alpha=0.5)

ax.axvline(x=start_forecast, ymin=0, ymax=1, color='black',linestyle='--', alpha=0.5)

ax.set_xlabel('year') 
ax.set_ylabel('GDP growth change in %') 
ax.set_title("GDP growth - real vs. forecast")
ax.legend()
fig.autofmt_xdate()
plt.grid()


#plt.savefig('forecast_out_of_time.png', dpi = 400)
#plt.close()