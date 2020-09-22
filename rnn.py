import numpy as np
from data_handling import get_synthetic_dataset, preprocess_for_supervised_learning, convert_time_series_to_relative
import datetime
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import os


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



# Prepare Data for RNN


N, dummy_dim = df_training.shape
dummy_dim -= 1

time_steps = 15
horizon = 1
sequence_length = time_steps + horizon 


max_index = N - sequence_length + 1

X = np.empty([max_index, sequence_length,dummy_dim])
y = np.empty([max_index, sequence_length])

for i in range(max_index):

    X[i] = df_training.iloc[i:i+sequence_length,1:].values
    y[i] = df_training.iloc[i:i+sequence_length,0].values
        

    
# =============================================================================
# # Rnn Debug    
# =============================================================================
    
    
from torch import nn, no_grad, save, load
from torch import from_numpy, zeros
from torch.optim import SGD


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

n_epochs = 10
batch_size = 25
lr = 0.25
test_size = 0.25


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


    
from matplotlib import pyplot as plt

x = range(0,len(y_eval))
plt.plot(x,y_eval)
plt.plot(x, y_hat)
plt.show()
