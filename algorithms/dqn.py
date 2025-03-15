import random
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import deque
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
import datetime
import logging
import os 

## Log variables
timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H.%M.%S")
name_file = ".\\logs\\DQN-" + timestamp + ".txt"

log = logging.basicConfig(filename = name_file, 
                    level = logging.DEBUG,
                    format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')

# Deep Reinforcement Network Algorithm [DQN]
# Hyperparameters
#Size of minibatches used during learning
#MINI_BATCH_SIZE = 32
REPLAY_MEMORY_SIZE = 1000
DISCOUNT_FACTOR  = 0.9
LEARNING_RATE = 0.001
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.1
EXPLORATION_DECAY = 0.9

class DQN_Solver:
    def __init__(self, observation_space, policies_space):
        self.policies_space = policies_space  #5 acciones si no es jerarquico
        self.observation_space = observation_space #8
        self.memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.exploration_rate = 0
        self.saved_network_scores = deque(maxlen=10)

    def get_initialize_model_network(self):
        self.exploration_rate = EXPLORATION_MAX

        asd = keras.initializers.VarianceScaling(scale=2)

        model = Sequential()
        #Capa input a capa oculta 1
        model.add(
            Dense(
                256, 
                input_shape=(self.observation_space,), 
                activation="relu", 
                kernel_initializer=asd
            )
        )

        #Capa oculta 1 a capa oculta 2
        model.add(
            Dense(
                256, 
                input_shape=(256,), 
                activation="relu", 
                kernel_initializer=asd
            )
        )

        #Capa oculta 2 a capa oculta 3
        model.add(
            Dense(
                256,
                activation="relu", 
                kernel_initializer=asd
            )
        )

        #Capa oculta 3 a capa oculta 4
        model.add(
            Dense(
                256,
                activation="relu", 
                kernel_initializer=asd
            )
        )

        #Capa oculta 4 a capa output
        model.add(
            Dense(
                self.policies_space, 
                activation="linear", 
                kernel_initializer=asd
            )
        )

        model.compile(loss="logcosh", optimizer=Adam(learning_rate=LEARNING_RATE))
        print(model.summary())

        return model

    def get_model_network(self):
        return self.model_network

    def remember(self, state, policy, reward, next_state, done):
        self.memory.append((state, policy, reward, next_state, done))

        i = 0
        for state, policy, reward, next_state, terminal in self.memory:
            i = i + 1
            #print('i:{} : policy: {} reward = {} state: {} next_state: {} terminal: {}'.format(i,policy,reward,state,next_state, terminal))

    def choose_action(self, state, exploration_rate):
        #if np.random.rand() < self.exploration_rate:
        if np.random.rand() < exploration_rate:
            return (random.randrange(self.policies_space), True)

        q_values = self.model_network.predict(state)
        return (np.argmax(q_values[0]), False)

    def experience_replay(self):
        if len(self.memory) < MINI_BATCH_SIZE:
            return
        batch = random.sample(self.memory, MINI_BATCH_SIZE)
        for state, policy, reward, state_next, terminal in batch:
            #os.system('pause')
            q_update = reward
            print(state_next)
            if not terminal:
                #os.system('pause')
                q_predict = self.model_network.predict(state_next)
                print(q_predict)

                #print('q_predict[0]\n')
                print(q_predict[0])

                q_update = (reward + DISCOUNT_FACTOR * np.amax(q_predict[0]))

            #os.system('pause')
            q_values = self.model_network.predict(state)
            #print('q_values\n')
            #print(q_values)
            q_values[0][policy] = q_update
            #print('\n\nq_update\n')
            #print(q_update)
            print('Q_table: {} policy: {} Q_update = {} state: {} state_next: {}'.format(q_values,policy,q_update,state,state_next))
            self.model_network.fit(state, q_values, verbose=0)
            
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)
        print("exploration_rate :",self.exploration_rate)

    def compare_results_and_save(score, reward):
        # Save network
        filePrefix="testNetwork"
        if(len(saved_network_scores) == 0 or max(saved_network_scores) < score):
            if len(saved_network_scores) < saved_network_scores.maxlen:
                saved_network_scores.append(score)
                self.save_model(filePrefix+str(score))
            else:
                poppedScore = saved_network_scores.popleft()
                saved_network_scores.append(score)

                os.remove(filePrefix+str(poppedScore)+'.h5')
                self.save_model(filePrefix+str(score))
        print(saved_network_scores)

    def save_model(self, name):
        self.model_network.save(name+'.h5')
