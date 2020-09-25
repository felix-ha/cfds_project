import datetime
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.metrics import mean_squared_error
from pmdarima.arima import auto_arima


    
from torch import nn, no_grad, save, load
from torch import from_numpy, zeros
from torch.optim import SGD


from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import os


import matplotlib.pyplot as plt
plt.style.use('seaborn-dark')


from data_handling import get_synthetic_dataset, preprocess_for_supervised_learning, convert_time_series_to_relative, transform_index_to_datetime

from gao_data import get_PairSampleDf

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

# Synthetic dummy data set
#df = get_synthetic_dataset()

# Data from goa
df = get_PairSampleDf()  

df = transform_index_to_datetime(df)
df = convert_time_series_to_relative(df)
df = preprocess_for_supervised_learning(df)

start = df.index[0] 
end = df.index[-1]
start_forecast = datetime.datetime(2040, 12, 31)
t_train = df.index[df.index < start_forecast]
t_forecast = df.index[df.index >= start_forecast]

df_training = df.loc[t_train,:]
df_prediction = df.loc[t_forecast,:]
y_predict = df_prediction['y'].values



# =============================================================================
# # Fitting Models - a list of models, each is a tuple
# =============================================================================

models = []

# =============================================================================
# # ToDo: implment WEO
# name = 'WEO'
# model = GradientBoostingRegressor(n_estimators = 20, max_depth = 2, 
#                                   min_samples_split=2, learning_rate = 0.5)
# y_forecast = forecast(model, df_training, df_prediction)
# y_forecast = y_forecast.values
# mse = mean_squared_error(np.exp(y_forecast), np.exp(y_predict))
# models.append( (name, y_forecast, mse))
# =============================================================================


name = 'OLS'
model = LinearRegression()
y_forecast = forecast(model, df_training, df_prediction)
y_forecast = y_forecast.values
mse = mean_squared_error(np.exp(y_forecast), np.exp(y_predict))
models.append( (name, y_forecast, mse))



name = 'ARIMA'

y_train = df_training.iloc[:, 0]
X_train = df_training.iloc[:, 1:]
y_test = df_prediction.iloc[:, 0]
X_test = df_prediction.iloc[:, 1:]

model = auto_arima(y = y_train,
                   trace=True, 
                   start_p=0,
                   max_p=3,
                   start_q=0,
                   max_q=3,
                   seasonal = False,
                   stepwise= True,
                   exogenous=X_train) 

model.fit(y= y_train, exogenous=X_train)

y_forecast = model.predict(n_periods=y_test.shape[0],
                      exogenous = X_test)
mse = mean_squared_error(np.exp(y_forecast), np.exp(y_predict))
models.append( (name, y_forecast, mse))


name = 'GBM'
model = GradientBoostingRegressor(n_estimators = 50, max_depth = 2, 
                                  min_samples_split=10, learning_rate = 0.03)
y_forecast = forecast(model, df_training, df_prediction)
y_forecast = y_forecast.values
mse = mean_squared_error(np.exp(y_forecast), np.exp(y_predict))
models.append( (name, y_forecast, mse))






# =============================================================================
# RNN start
# =============================================================================


# =============================================================================
# # Prepare Data for RNN
# =============================================================================

name = 'RNN'

N, dummy_dim = df_training.shape
dummy_dim -= 1

time_steps = 25
horizon = 1
sequence_length = time_steps + horizon 


max_index = N - sequence_length + 1

X = np.empty([max_index, sequence_length,dummy_dim])
y = np.empty([max_index, sequence_length])

for i in range(max_index):

    X[i] = df_training.iloc[i:i+sequence_length,1:].values
    y[i] = df_training.iloc[i:i+sequence_length,0].values
        

    
# =============================================================================
# # Rnn Model    
# =============================================================================
    



class RNN(nn.Module):
    def __init__(self, input_size, seq_len, output_size, hidden_dim, n_layers):
        super(RNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):
        r_out, hidden = self.rnn(x, hidden)
        r_out = self.fc(r_out)
        
        return r_out
        
    def initHidden(self):
        return zeros(1, self.seq_len, self.hidden_dim)
    
    

N, seq_len, dummy_dim = X.shape

input_size=dummy_dim
hidden_dim=5
n_layers=1
output_size=1

n_epochs = 75
batch_size = 25
lr = 0.33
test_size = 0.0001


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=123)




X_train_T = from_numpy(X_train).float()
y_train_T = from_numpy(y_train).float()
X_val_T = from_numpy(X_val).float()
y_val_T = from_numpy(y_val).float()



train_ds = TensorDataset(X_train_T, y_train_T)
train_dl = DataLoader(train_ds, batch_size=batch_size)  

valid_ds = TensorDataset(X_val_T, y_val_T)
valid_dl = DataLoader(valid_ds, batch_size=batch_size * 2)




model = RNN(input_size, seq_len, output_size=output_size, hidden_dim=hidden_dim, n_layers=n_layers)


hidden_0 = zeros(1, seq_len, hidden_dim)
training_losses = np.empty(n_epochs)
valid_losses = np.empty(n_epochs)

loss_func = nn.MSELoss()
optimizer = SGD(model.parameters(), lr = lr)  



    
# =============================================================================
# # Training loop 
# =============================================================================

for epoch in range(n_epochs):
    model.train()
    training_loss = 0
    for X_batch, y_batch in train_dl:
        optimizer.zero_grad()
        
        y_pred = model(X_batch, hidden_0)
        
        loss = loss_func(y_pred.squeeze(), y_batch)
        
        training_loss += loss.item()
       

        loss.backward()
        optimizer.step()
   

    model.eval()
    valid_loss = 0
    with no_grad():
        for X_batch, y_batch in valid_dl:
            y_pred = model(X_batch, hidden_0)
            loss = loss_func(y_pred.squeeze(), y_batch.squeeze()) 
            valid_loss += loss.item()
    
    
    training_loss_epoch = training_loss * 100
    valid_loss_epoch = valid_loss * 100
    
    training_losses[epoch] = training_loss_epoch
    valid_losses[epoch] = valid_loss_epoch
    
    print('Epoch {}: train loss: {:.4} valid loss: {:.4}'
          .format(epoch, training_loss_epoch, valid_loss_epoch))   
    
    

# =============================================================================
# # Serializing model 
# =============================================================================

wdir= r'C:/Users/hauer/Documents/Repositories/cfds_project'
save_dir = os.path.join(wdir, 'pytorch_models')
model_name = 'rnn.torch'

if(not os.path.isdir(save_dir)):
    os.mkdir(save_dir)
    
save(model.state_dict(), os.path.join(save_dir, model_name))

model = RNN(input_size, seq_len, output_size=output_size, hidden_dim=hidden_dim, n_layers=n_layers)
model.load_state_dict(load( os.path.join(save_dir, model_name)))
model.eval()

# =============================================================================
# # Evaluation / Plotting 
# =============================================================================


# Run RNN with whole df, only selecting the outputs that are wanted for prediction
X_eval = df.iloc[:,1:].values
y_eval = df.iloc[:,0].values
X_eval_T = from_numpy(X_eval).float()
N, _ = X_eval_T.shape
X_eval_T = X_eval_T.view([-1, N, dummy_dim])

hidden_0 = zeros(1, N, hidden_dim)
model.eval()
with no_grad():
    y_hat = model(X_eval_T, hidden_0)
    
y_hat =  y_hat.view(-1).numpy()
y_forecast = y_hat[-len(t_forecast):]



mse = mean_squared_error(np.exp(y_forecast), np.exp(y_predict))
models.append((name, y_forecast, mse))


# =============================================================================
# # RNN end
# =============================================================================


















# =============================================================================
# name = 'RL'
# model = GradientBoostingRegressor(n_estimators = 5, max_depth = 10, 
#                                   min_samples_split=2, learning_rate = 0.9)
# y_forecast = forecast(model, df_training, df_prediction)
# y_forecast = y_forecast.values
# mse = mean_squared_error(np.exp(y_forecast), np.exp(y_predict))
# models.append( (name, y_forecast, mse))
# =============================================================================


# =============================================================================
# # Create Plots
# =============================================================================

fig, ax = plt.subplots()

ax.plot(np.exp(df['y']), label='real')

for model in models:
    name = model[0]
    y_forecast = model[1]
    mse = model[2]
    
    label = name + ' ' + str(round(mse,3))
    
    ax.plot(t_forecast, np.exp(y_forecast), label=label, alpha=0.5)

ax.axvline(x=start_forecast, ymin=0, ymax=1, color='black',linestyle='--', alpha=0.5)

ax.set_xlabel('year') 
ax.set_ylabel('GDP growth change in %') 
ax.set_title("GDP growth - real vs. forecast")
legend  = ax.legend(bbox_to_anchor=(1.05, 1))
fig.autofmt_xdate()
plt.grid()


#plt.savefig('forecast_out_of_time.png', dpi = 500, bbox_extra_artists=(legend,), bbox_inches='tight')
#plt.close()