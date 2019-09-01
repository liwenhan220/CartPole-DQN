import gym
env = gym.make('CartPole-v0')
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from collections import deque
import random
import numpy as np
import matplotlib.pyplot as plt

MIN_REPLAY_SIZE = 1_000
MINIBATCH_SIZE = 32
LR = 0.001
gamma = 0.99
INIT_EPSILON = 1.0
FINAL_EPSILON = 0.0
EPISODES = 10000
RANDOM_STEPS = 250
replay_memory = deque(maxlen=50_000)
SHOW_EVEY = 1
SHOW_PREVIEW = True
MODEL = 'models/CartPole-v15.model'
record = deque(maxlen=1)
record.append(200)

def dqn():
    
    model = Sequential()

    model.add(Dense(128, activation='relu',input_shape=[4]))
##    model.add(Dropout(0.2))
    
    model.add(Dense(256, activation='relu'))
##    model.add(Dropout(0.2))
    
    model.add(Dense(512, activation='relu'))
##    model.add(Dropout(0.2))
    
    model.add(Dense(256, activation='relu'))
##    model.add(Dropout(0.2))
    
    model.add(Dense(128, activation='relu'))
##    model.add(Dropout(0.2))

    model.add(Dense(2, activation='relu'))

    model.compile(loss='mse',optimizer=RMSprop(lr=LR),metrics=['accuracy'])
    return model

def train(model, transition):
    if len(transition) < MIN_REPLAY_SIZE:
        return
    minibatch = random.sample(transition, MINIBATCH_SIZE)
    X = []
    y = []
    for i in range(len(minibatch)):
        state = minibatch[i][0]
        reward = minibatch[i][1]
        next_state = minibatch[i][2]
        is_terminal = minibatch[i][3]
        action = minibatch[i][4]
        current_qs = model.predict(np.array(state).reshape(-1,len(state)))[0]
        if is_terminal:
            new_q = reward
        else:
            future_qs = model.predict(np.array(next_state).reshape(-1,len(next_state)))[0]
            new_q = reward + gamma * np.max(future_qs)
        current_qs[action] = new_q
        X.append(state)
        y.append(current_qs)
    model.fit(np.array(X).reshape(-1,len(state)),np.array(y),batch_size=MINIBATCH_SIZE)
   

def main(): 
    epsilon = 1
    start_value = 1
    network = dqn()
    if MODEL is not None:
        trained_model = load_model(MODEL)
        network.set_weights(trained_model.get_weights())
    for episode in range(EPISODES):
        ep_reward = 0
        done = False
        current_state = env.reset()
        while not done:
            if np.random.random() > epsilon:
                action = np.argmax(network.predict(np.array(current_state).reshape(-1,len(current_state)))[0])
            else:
                action = np.random.randint(0,env.action_space.n)
            new_state, reward, done, info = env.step(action)

            ep_reward += reward
            
            replay_memory.append([current_state, reward, new_state, done, action])
            train(network, replay_memory)
            current_state = new_state

            if SHOW_PREVIEW and episode % SHOW_EVEY == 0:
                env.render()

        if epsilon > FINAL_EPSILON:
            epsilon = INIT_EPSILON + (FINAL_EPSILON - INIT_EPSILON)/RANDOM_STEPS * episode
        print('episode: {}, epsilon: {}, ep_reward: {}'.format(episode,epsilon,ep_reward))
        if ep_reward >= record[0]:
            record.append(ep_reward)
            network.save('models/CartPole-v{}.model'.format(start_value))
            start_value += 1
            print('saving model!!!!!')
##        plt.xlim = (0,episode)
##        plt.ylim = (0,ep_reward)
##        plt.plot(episode,ep_reward,c='b')
##        plt.show()
        
if __name__ == '__main__':
    main()
