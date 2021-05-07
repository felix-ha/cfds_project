import gym

env = gym.make('MountainCarContinuous-v0')
env.reset()
for _ in range(2):
    env.render()
    
    action = env.action_space.sample()
    print('action: ', action)
    
    observation, reward, done, info = env.step(action) # take a random action
    
    print('observation: ', observation)
env.close()