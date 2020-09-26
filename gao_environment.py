from gao_data import get_PairSampleDf


class Environment():
    
    
    def __init__(self, n_games, window_size_observation, size_time_series):
        self.n_games = n_games
        self.window_size_observation = window_size_observation 
        self.size_time_series = size_time_series
        self.current_game = 0
        self.t_current = 0
        
        self.list_of_games = self.get_list_of_games()
        self.t_max = self.size_time_series - self.window_size_observation - 1
      
        
    def get_list_of_games(self):
        list_of_games = []
        
        for i in range(self.n_games):
            list_of_games.append(get_PairSampleDf())
                   
        return list_of_games
    
    
    def reset(self):
        if self.current_game == self.n_games - 1:
            self.current_game = 0
        else:
            self.current_game += 1
            
        self.t_current = 0
        
        observation = self.get_observation()
        
        return observation
        
    
    def step(self, action):
        
        done = False
        if action == 0:		# wait/close
            reward = 0.
            self.empty = True
        elif action == 1:	# open
            reward = self.get_noncash_reward()
            self.empty = False
        elif action == 2:	# keep
            reward = self.get_noncash_reward()
        else:
            raise ValueError('no valid action: ' + str(action))
        
        self.t_current += 1
        #return self.get_state(), reward, self.t == self.t_max, self.get_valid_actions()
        
        
        done = self.t_current == self.t_max
        info = None
        observation = self.get_observation()
        
        return observation, reward, done, info
    
    
    def get_observation(self):
        df_current = self.list_of_games[self.current_game]
        
        observation = df_current.iloc[self.t_current:(self.t_current + self.window_size_observation), :]
        
        return observation
        
    
    def render(self):
        pass

    
    
    
if __name__ == '__main__':
    n_games = 2
    window_size_observation = 40
    size_time_series = 180
    env = Environment(n_games, window_size_observation, size_time_series)
    


    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = 0
            observation_, reward, done, info = env.step(action)
    
