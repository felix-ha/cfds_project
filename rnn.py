import numpy as np
from data_handling import get_synthetic_dataset, preprocess_for_supervised_learning, convert_time_series_to_relative
import datetime

def generate_subsequences(x, sequence_length):
    """
    Given a time series x, returns a prepared data set for a
    sequence-to-sequence model. 
    
    x = (x_0, x_1, ..., x_N) -->
     
    x_batch = (x_i+1, x_i+2, ..., x_sequence_length)
    y_batch = (x_i+2, x_i+3, ..., x_sequence_length+1)
    
    returns a tuple X, y with dimensions 
    [len(x) - sequence_length, sequence_length]
    
    """

    X, y = [], []
    
    for i in range(len(x)):
        x_start = i
        x_end = i+sequence_length     
       
        y_start = x_start+1
        y_end = x_end+1
        
        if y_end > len(x):
            break
        
        x_batch = x[x_start:x_end] 
        y_batch = x[y_start:y_end]
        
        X.append(x_batch)
        y.append(y_batch)
    
    return np.stack(X), np.stack(y)


# =============================================================================
# # Get, preprocess and split data 
# =============================================================================

df = get_synthetic_dataset()
df = convert_time_series_to_relative(df)
#df = preprocess_for_supervised_learning(df)

start = df.index[0] 
end = df.index[-1]
start_forecast = datetime.datetime(1904, 12, 31)
t_train = df.index[df.index < start_forecast]
t_forecast = df.index[df.index >= start_forecast]

df_training = df.loc[t_train,:]
df_prediction = df.loc[t_forecast,:]
y_predict = df_prediction['y'].values

N, dummy_dim = df_training.shape
dummy_dim -= 1



time_steps = 1
horizon = 1
sequence_length = time_steps + horizon 


max_index = N - sequence_length + 1

X_training = np.empty([max_index, sequence_length,dummy_dim])
y_training = np.empty([max_index, sequence_length])

for i in range(max_index):

    X_training[i] = df_training.iloc[i:i+sequence_length,1:].values
    y_training[i] = df_training.iloc[i:i+sequence_length,0].values
        

    
# =============================================================================
# # Rnn Debug    
# =============================================================================
    
    
from torch import nn
from torch import from_numpy, zeros


class RNN_debug(nn.Module):
    def __init__(self, input_size, seq_len, output_size, hidden_dim, n_layers):
        super(RNN_debug, self).__init__()

        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
                # get RNN outputs, hidden is  unused
        r_out, hidden = self.rnn(x, hidden)
        
        print("output of rnn cell: ")
        print(r_out)
        print("")
        
        print(" last hidden state of rnn cell: ")
        print(hidden)
        print("")
        
       
        
        # get final output
        r_out = self.fc(r_out)
        
        print("fully connected layer: ")
        print(r_out)
        print("")
        
    def initHidden(self):
        return zeros(1, self.seq_len, self.hidden_dim)
    
    

N, seq_len, dummy_dim = X_training.shape

X_T = from_numpy(X_training).float()
y_T = from_numpy(y_training).float()


input_size=dummy_dim
hidden_dim=5
n_layers=1
output_size=1

model = RNN_debug(input_size, seq_len, output_size=output_size, hidden_dim=hidden_dim, n_layers=n_layers)


hidden = zeros(1, seq_len, hidden_dim)


print("N: ", N)
print("seq_len: ", seq_len)
print("dummy_dim: ", dummy_dim)
print("hidden_dim: ", hidden_dim)
print("")

print("input to network: ")
print(X_T)
print("")

print("initial hidden state")
print(hidden)
print("")

y = model.forward(X_T, hidden)
    
    