import pandas as pd
import sqlite3 as sl
import numpy as np



def get_country_id(con, country):
    # returns id of the country
    # if the country does not exists, 
    # a new entry in database is added 
    
    df_countries = pd.read_sql("""SELECT * FROM countries;""", con)
    
    if not country in df_countries['name'].values:
        query ="INSERT INTO countries (name, code) VALUES ('" + country + "', '" + country[0:3].upper() + "');"
        
        with con:
            con.execute(query)
            
        df_countries = pd.read_sql("""SELECT * FROM countries;""", con)

    return df_countries[df_countries['name'] == country]['id'].values[0]


def get_type_id(con, type_name):
    # returns id of the data_type
    # if the data_type does not exists, 
    # a new entry in database is added 
    
    df = pd.read_sql("""SELECT * FROM types;""", con)
    
    if not type_name in df['name'].values:
        query ="INSERT INTO types (name) VALUES ('" + type_name + "');"
        
        with con:
            con.execute(query)
            
        df = pd.read_sql("""SELECT * FROM types;""", con)

    return df[df['name'] == type_name]['id'].values[0] 



# Dataset to write in database

t = np.arange(0, 20 * np.pi, step = 0.18)
y =  np.sin(t) + 1.1
x =  np.cos(t) + 1.1
z = np.exp(np.cos(t))


index = pd.date_range('1900', periods=len(t), freq='Y').to_pydatetime()

df = pd.DataFrame(data = {'t': t,
                          'y': y,
                          'x': x,
                          'z': z, 'date': index
                          })



# parameters

country = 'germany'
type_name = 'gdp'


con = sl.connect('my-test.db')
           

country_id = get_country_id(con, country)
type_id = get_type_id(con, type_name)




# create new info: 

query = "INSERT INTO info (typeID, countryID) VALUES ('" + str(type_id) + "', '" + str(country_id) + "');"                

with con:
    con.execute(query)
    
    
# get infoID
    
df_info = pd.read_sql("""select * from info""", con)   
info_id = np.max(df_info['id'])



#prepare df to format of database

df_to_write = df[['date', 'x']]
df_to_write = df_to_write.rename(columns={"x": "value"})
df_to_write['infoID'] = info_id

df_to_write.to_sql("data", con, index = False, if_exists='append')

























