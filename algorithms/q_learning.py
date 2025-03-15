import random
import numpy as np
import pandas as pd
import datetime
import logging
import os
import sys

## Log variables
timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H.%M.%S")
name_file = ".\\logs\\Qlearning-" + timestamp + ".txt"

log = logging.basicConfig(filename = name_file, 
                    level = logging.DEBUG,
                    format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')

# Reinforcement Learning Algorithm
class QLearningTable: 
    def __init__(self, policies, total_policies):
        self.learning_rate = 0
        self.reward_decay = 0
        self.policies = policies
        self.count = 0
        self.q_table = self.create_model()
        self.total_policies = total_policies
        self.columns = self.q_table.columns[0:self.total_policies]

    def setup_hyperparameters(self, condition):
        print("learning_rate: ", self.learning_rate)
        print("reward_decay : ", self.reward_decay)
        if condition:
            self.learning_rate = 0.001
            self.reward_decay = 0.8
        else:
            """
            El factor de aprendizaje original del agente es 0.001, 
            luego establezco en 0.01 al volver a entrenar los caminos (recorrido por cada episodio). 
            Buscando que el agente actualice sus valores Q de forma más agresiva, 
            lo que le ayudaría a aprender más rápidamente de sus errores.
            """

            self.learning_rate = 0.01
            """
            El valor de gamma original del agente es 0.8, utilizo 0.99 para volver a entrenar los caminos (recorrido por cada episodio).
            buscando que el agente tenga en cuenta las recompensas futuras aún más. Esto podría ayudar al agente a aprender 
            de sus errores y mejorar su rendimiento en el futuro.
            """
            self.reward_decay = 0.99

    def get_qtable(self):
        return self.q_table

    def exponential_decay(self, episode, EXPLORATION_DECAY, EXPLORATION_MAX):
        epsilon = EXPLORATION_MAX * np.exp(-EXPLORATION_DECAY * episode)
        return epsilon

    def choose_action(self, observation, epsilon, game_time, trainning=False):
        self.check_state_exist(observation)

        if trainning:
            if np.random.uniform() < epsilon: #EXPLORATION 
                action = np.random.choice(self.policies[ 0 : self.total_policies ])
                action_tuple = (action, True)
            else: #EXPLOTATION
                Qactions = self.q_table.loc[observation, self.columns]
                action = np.random.choice(Qactions[ Qactions == np.max(Qactions)].index )
                action_tuple = (action, False)
        else:
                Qactions = self.q_table.loc[observation, self.columns]
                print("Qactions: ", Qactions)
                if Qactions.shape[0] > 0:
                    action = np.random.choice(Qactions[Qactions == np.max(Qactions)].index)
                else: #Significa que la observacion no existe, entonces no realizara nada
                    action = 'do_nothing'

                print("action  : ", action)
                #os.system("pause")
                action_tuple = (action, False)
        return action_tuple  #String

    def propagate_rewards(self, episode_rewards, final_reward):
        self.setup_hyperparameters(condition=False)
        """
        Propaga las recompensas finales del episodio desde el principio al final actualizando los valores Q.
        """
        if len(episode_rewards) == 0:
            return

        index = len(episode_rewards)
        for i in range(len(episode_rewards)):
            state           = episode_rewards[i][0]
            action          = episode_rewards[i][1]
            instant_reward  = episode_rewards[i][2]
            next_state      = episode_rewards[i][3]

            if not(state) == None and not(action) == None and not(instant_reward) == None:
                print("Iteración {} ->".format(i))
                print("        Estado actual         : {}".format(state))
                print("        Siguiente estado      : {}".format(next_state))
                print("        Acción elegida        : {}".format(action))
                print("        Recompensa instantánea: {}".format(instant_reward))
                print("        Recompensa final      : {}".format(final_reward))
                print("        Recompensa total      : {}".format(instant_reward + final_reward))
                print("ANTES   Valor Q (estado actual, acción elegida): {}".format(self.q_table.loc[state, action]))
                self.learn( state,
                            action, 
                            #(instant_reward + final_reward),
                            final_reward,
                            next_state)
                print("DESPUES Valor Q (estado actual, acción elegida): {}".format(self.q_table.loc[state, action]))
                print(50*"_")
        self.setup_hyperparameters(condition=True) #Para setear en estado normal despues de ejecutar todo el
        #os.system("pause")

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]

        if s_ != 'terminal':
          q_max_next_state = self.q_table.loc[s_, self.columns].max()
          print("******* Valor Q (estado siguiente)             : {}".format(q_max_next_state))
          q_target = r + self.reward_decay * q_max_next_state
        else:
          q_target = r 

        error = q_target - q_predict
        self.q_table.loc[s, a] += self.learning_rate * error

    def check_state_exist(self, state): # Check to see if the state is in the QTable already, and if not it will add it with a value of 0 for all possible policies.
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(pd.Series([0] * len(self.policies), index=self.q_table.columns, name=state))

    def load_train_csv(self):
        df = pd.read_csv('old_qlearning_table_train.csv')  
        df = df.rename(columns = {'Unnamed: 0':'status'})
        df = df.set_index(['status'])
        return df

    def _valid_csv_exists(self):
        try:
            file = open('old_qlearning_table_train.csv')
            file.close()
            return True
        except FileNotFoundError:
            return False

    def create_model(self):
        exist_csv = self._valid_csv_exists()
        print("exist_csv: ", exist_csv )
        os.system("pause")
        if exist_csv:
            q_table = self.load_train_csv()
            logging.info('Load table qlearning...  ' +  str(q_table.shape))
        else:
            q_table = pd.DataFrame(columns=self.policies, dtype=np.float64)
        return q_table
