import numpy as np
import pandas as pd

    

def convert_time_series_to_relative(df):
    # Assings each t the Values of ln(X_t / X_(t-1))
    # X_0 will be dropped
    
    df_new = df.iloc[1:, :].copy()
    
    for variable in df.columns:
        df_new[variable] = np.log(df[variable].iloc[:-1].values / df[variable].iloc[1:].values)
        
    return df_new
 
    
def preprocess_for_supervised_learning(df):
   
   
    # filter and replace inf and - inf
    
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    
    # shifting index for supervised learning, tuples look like (x_t-1, y_t)
    # NOTE: y needs to be the first column of the df!!
    
    df_variables = df.iloc[:,1:]
    df_y = df.iloc[:,0]
    
    df_variables.index = df_variables.index.shift(periods = 1,  freq ='Y')
   
    df = pd.DataFrame(df_y).join(df_variables, how='inner')
    
    return df


def get_synthetic_dataset():
    t = np.arange(0, 2 * np.pi, step = 0.05)
    y = 2 * np.sin(t) + 2.5
    x = 1.5 * np.cos(t) + 2
    z = np.exp(np.cos(t))
    
    
    df = pd.DataFrame(data = {'t': t,
                              'y': y,
                              'x': x,
                              'z': z
                              })
    
    
    
    
    
    # creating timestamps as index
    
    index = pd.date_range('1900', periods=len(t), freq='Y')
    
    # higher frequency
    # pd.date_range('2015-06-30', periods=8, freq='6M')
    
    # setting timestamp as index and dropping column t
    df = df.set_index(index)
    del df['t']
  
    return df

    
    
if __name__ == '__main__':
    df = get_synthetic_dataset()
    df = convert_time_series_to_relative(df)
    df = preprocess_for_supervised_learning(df)
    
    
    
    # transposing data frame
    
    #df_T = df.transpose()
    #df_T_back = df_T.transpose()
    
    
    
    
    



