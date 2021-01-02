import numpy as np
import pandas as pd


def get_predictions_weo(df_weo, country, start_forecast, end_forecast):
       
    df = df_weo[df_weo['country'] == country]
    
    
    for col in df.columns:
        if 'S' in col:
            del df[col] 
            
    del df['WEO_Country_Code']     
    
    
    df = df[df['year'] >= start_forecast]
    
    
    predictions_weo = []
    years = np.arange(start_forecast, end_forecast+1, 1)
    
    for year in years:
       
        df_curr = df[df['year'] == year]
        
        year_WEO = year - 1 
        column = 'F' + str(year_WEO) + 'ngdp_rpch'
        y_pred_year = df_curr[column].values[0]
        
        predictions_weo.append(y_pred_year)
    
    predictions_weo = pd.Series(data = predictions_weo, index = years)
    
    return predictions_weo



# Not used method atm    

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
    t = np.arange(0, 8 * np.pi, step = 0.1)
    y = 1 * np.sin(t) + 2
    x = 1 * np.cos(t + 0.1) + 2
    z = 2 * np.cos(t - 0.5) + 2.2
    
    
    df = pd.DataFrame(data = {'t': t,
                              'y': y,
                              'x': x,
                              'z': z
                              })
    

    del df['t']
  
    return df

    # higher frequency
    # pd.date_range('2015-06-30', periods=8, freq='6M')
def transform_index_to_datetime(df):

    index = pd.date_range('1900', periods=len(df), freq='Y').to_pydatetime()   
    df = df.set_index(index)
    
    return df
    
    
if __name__ == '__main__':
    path = r'C:\Users\hauer\Dropbox\CFDS\Project\data\WEOhistorical.xlsx'
    df_weo =  pd.read_excel(path,sheet_name='ngdp_rpch')
    df_result = get_predictions_weo(df_weo, 'Germany', 2010, 2018)
    df_weo['country'].unique()
    
    
    
    
    
   # df = get_synthetic_dataset()
   # df = transform_index_to_datetime(df)
   # df = convert_time_series_to_relative(df)
   # df = preprocess_for_supervised_learning(df)
    
    
    
    # transposing data frame
    
    #df_T = df.transpose()
    #df_T_back = df_T.transpose()
    
    
    
    
    



