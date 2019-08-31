import gym
env = gym.make('CartPole-v0')
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from collections import deque
import random
import numpy as np

# ======PARAMETERS=======# 
EPISODES = 10000
SHOW_EVERY = 1
SHOW_PREVIEW = True
QUALIFYING_THRESH = 120

def main():
    epsilon = 0
    successful_times = 0
    failed_times = 0
    network = load_model('models/CartPole-v15.model')
    for episode in range(EPISODES):
        ep_reward = 0
        done = False
        current_state = env.reset()
        while not done:
            action = np.argmax(network.predict(np.array(current_state).reshape(-1,len(current_state)))[0])
            new_state, reward, done, info = env.step(action)

            ep_reward += reward
            
            current_state = new_state

            if SHOW_PREVIEW and episode % SHOW_EVERY == 0:
                env.render()

        if ep_reward > QUALIFYING_THRESH:
            successful_times += 1
            print('nailed it!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        else:
            failed_times += 1
            print('failed!!!!!!!!!!')
            
        print('failed times: {}, successful times: {}, episode reward: {}'.format(failed_times,successful_times,ep_reward))
        
if __name__ == '__main__':
    main()
