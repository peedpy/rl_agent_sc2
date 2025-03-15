from pysc2.lib import features, actions, units
from general_agent import TerranAgent
from algorithms.q_learning import QLearningTable
from algorithms.rewards import Reward
import pandas as pd
import numpy as np
import datetime
import logging
import os
import random
import chardet

## Log variables
timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H.%M.%S")
name_file = ".\\logs\\Agent - Qlearning-" + timestamp + ".txt"

log = logging.basicConfig(filename = name_file, 
                          level = logging.DEBUG,
                          format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')

multiActions = []
def executeActions():
    global multiActions
    if len(multiActions) > 0:
        action = multiActions[0]
        multiActions = multiActions[ 1 : len(multiActions) ]
        return action
    else:
        return False

#Va a tener siempre la ultima tabla actualizada
qtable = None

#Definicion de los hiperparametros
EXPLORATION_MAX     = 1.0
EXPLORATION_MIN     = 0.1
EXPLORATION_DECAY   = 0.0003

# Diccionario que mapea cada acción a su índice en self.totals
ACTION_TO_INDEX = {
    'harvest_minerals': 0,
    'harvest_gas': 1,
    'build_command_center': 2,
    'build_scv': 3,
    'build_supply_depot': 4,
    'build_barracks': 5,
    'build_tech_lab': 6,
    'build_bunker': 7,
    'explore_csv': 8,
    'train_marine': 9,
    'train_marauder': 10,
    'attack_with_marine': 11,
    'defense_with_marine': 12,
    'attack_with_marauder': 13,
    'do_nothing': 14
}

class AgentQlearning(TerranAgent):
    def __init__(self, step_mul, train_mode=True):
        super(AgentQlearning, self).__init__()
        #Hiperparametros
        self.epsilon = 0
        self.episodes = 0
        self.step_mul = step_mul
        self.total_game_time = 0
        #-------------------------------------------
        self.prev_minerals = 0
        self.prev_gas      = 0
        self.prev_supply   = 0
        #-------------------------------------------
        self.train_mode = train_mode
        print("train_mode: ", self.train_mode)
        os.system('pause')
        #-------------------------------------------
        #Definicion de Politicas(acciones)
        self.policies = self.get_all_policies()
        self.total_policies = len(self.policies)
        #print(self.policies )
        #print(self.total_policies )
        #-------------------------------------------
        #Relacionados con la recompensa
        self.reward_actions = 0
        self.execute_action = None
        self.instant_action = ''
        self.total_rewards_by_episode = 0
        self.total_rewards_by_policy = 0
        self.reward_function = Reward()
        #-------------------------------------------
        #Control de tiempo
        self.start = 0
        self.finish = 0
        #-------------------------------------------
        #Control de construcciones por episodio
        #self.totals = (self.total_policies + 1)*[0]
        self.totals = (20)*[0]
        #-------------------------------------------
        #Varios
        self.count_exploration = 0
        self.count_explotation = 0
        self.reward_final = 0
        #-------------------------------------------
        #El propósito de episode_rewards es acumular las toda la información(s, a,  r, s) a lo 
        #largo de un episodio para luego utilizarlas en la actualización de los valores Q, 
        #generalmente al final de cada episodio.
        self.episode_rewards = []
        #-------------------------------------------
        global qtable
        qtable = QLearningTable(self.policies, 
                                self.total_policies)
        #-------------------------------------------
        self.data_stats_train, last_episodes, self.new_name_file = self.check_file_stats_game(agent_name='qlearning')
        self.episodes = last_episodes
        self.new_game()

    def get_encoding_and_separator(self, name_file):
        # Reemplaza 'tu_archivo.csv' con la ruta a tu archivo
        with open(name_file, 'rb') as f:
            result = chardet.detect(f.read())
        encoding = result['encoding']
        with open(name_file, 'r', encoding=encoding) as f:
            first_lines = [next(f) for _ in range(5)]  # Lee las primeras 5 líneas (ajusta según tu necesidad)

        possible_separators = [',', ';', '\t']  # Agrega otros posibles separadores si es necesario
        separator_counts = {}

        for line in first_lines:
            for separator in possible_separators:
                count = line.count(separator)
                if separator in separator_counts:
                    separator_counts[separator] += count
                else:
                    separator_counts[separator] = count

        # Encuentra el separador con la cuenta más alta
        best_separator = max(separator_counts, key=separator_counts.get)
        return encoding, best_separator

    def _valid_csv_exists(self, agent_name):
        try:
            name_file = 'old_'+str(agent_name)+'_stats_train.csv'
            file = open(name_file)
            file.close()
            return True, name_file
        except FileNotFoundError:
            return False, ''

    def check_file_stats_game(self, agent_name='qlearning'):
        keys_list = [
            'episode',
            'epsilon',
            'total_steps',
            'total_game_time',
            'harvest_minerals',
            'harvest_gas',
            'build_command_center',
            'build_scv',
            'build_supply_depot',
            'build_barracks',
            'build_tech_lab',
            'build_bunker',
            'explore_csv',
            'train_marine',
            'train_marauder',
            'attack_with_marine',
            'defense_with_marine',
            'attack_with_marauder',
            'minerals_used',
            'gas_used',
            'supply_used',
            'detectable_enemy_units',
            'do_nothing',
            'total_fail',
            'count_exploration',
            'count_explotation',
            'total_rewards',
            'reward_final',
            'scores',
            'start_datetime',
            'fin_datetime',
            'diff_time_min',
            'len_enemy_units',
            'len_my_units'
        ]

        exist_csv, name_file = self._valid_csv_exists(agent_name)
        last_episodes = 0
        if exist_csv:
            encoding, best_separator = self.get_encoding_and_separator(name_file)
            print("encoding:{} - best_separator:{}".format(encoding, best_separator))
            data_stats_train = pd.read_csv(name_file, encoding=encoding, delimiter=best_separator)
            data_stats_train = data_stats_train[keys_list]
            name_file = 'old_'+str(agent_name)+'_stats_train.csv'
            print(name_file)
            print(data_stats_train.head())
            print('Load status stats table {}... : {}'.format(agent_name, data_stats_train.shape ))
            last_episodes = data_stats_train.shape[0] - 1 #Porque siempre la primera ejecucion no sirve
            print("last_episodes: ", last_episodes)
            os.system('pause')
        else:
            data_stats_train = pd.DataFrame(columns=keys_list, dtype=np.float64)
            name_file = 'new_'+str(agent_name)+'_stats_train.csv'
        return data_stats_train, last_episodes, name_file

    def update_final_reward_and_retrain(self, obs, final_state):
        # Extraer unidades de la observación
        raw_units = obs.observation.raw_units
        enemy_units = [unit for unit in raw_units if unit.alliance == features.PlayerRelative.ENEMY]
        my_units = [unit for unit in raw_units if unit.alliance == features.PlayerRelative.SELF]
        len_enemy_units = len(enemy_units)
        len_my_units = len(my_units)

        # Inicializar recompensa final y constantes
        self.reward_final = 0
        c = 0.1
        r_win = 100
        r_loss = -50

        # Calcular la recompensa final según el resultado del juego
        if obs.reward == 1:
            print("¡Has ganado!")
            self.reward_final = r_win
        elif obs.reward == -1:
            print("Has perdido.")
            self.reward_final = r_loss
        elif obs.reward == 0:
            """
            Una recompensa positiva (hasta un máximo de 10) cuando tienes más 
            o igual cantidad de unidades que el enemigo, y una penalización negativa
            (pero limitada a -1) cuando tienes menos unidades que el enemigo. 
            Esto incentiva al agente a mantener o superar el número de unidades enemigas y,
            en caso contrario, aplica una penalización controlada.
            """
            print("¡Has empatado!")
            # Evitar división por cero: si es 0, usar 1 como mínimo
            len_my = len_enemy_units if len_enemy_units > 0 else 1
            len_enemy = len_enemy_units if len_enemy_units > 0 else 1
            if len_my >= len_enemy:
                self.reward_final = min(int(c * (len_my / len_enemy)), 10)
            else:
                self.reward_final = max(int(-c * (len_enemy / len_my)), -1)

        # Imprimir la recompensa final y actualizar la suma total
        print(f"reward_final: {self.reward_final}")
        #os.system("pause")
        self.total_rewards_by_episode += self.reward_final

        # Agrega una tupla más con información del tiempo t_final a episode_rewards
        self.episode_rewards.append((
            self.previous_state, 
            self.previous_policy,
            self.reward_final,
            final_state
        ))

        qtable.learn(
            self.previous_state, 
            self.previous_policy, 
            self.reward_final,
            final_state
        )

        qtable.propagate_rewards(self.episode_rewards, self.reward_final)
        #os.system("pause")

    def get_state(self, obs):
        scvs                        = self.helpers.get_my_units_by_type(obs, units.Terran.SCV)
        marines                     = self.helpers.get_my_units_by_type(obs, units.Terran.Marine)
        marauders                   = self.helpers.get_my_units_by_type(obs, units.Terran.Marauder)
        completed_command_center    = self.helpers.get_my_completed_units_by_type(obs, units.Terran.CommandCenter)
        completed_supply_depots     = self.helpers.get_my_completed_units_by_type(obs, units.Terran.SupplyDepot)
        completed_barrackses        = self.helpers.get_my_completed_units_by_type(obs, units.Terran.Barracks)
        completed_refineries        = self.helpers.get_my_completed_units_by_type(obs, units.Terran.Refinery)
        completed_barrack_tech_lab  = self.helpers.get_my_completed_units_by_type(obs, units.Terran.BarracksTechLab)
        completed_bunker            = self.helpers.get_my_completed_units_by_type(obs, units.Terran.Bunker)
        
        states = obs.observation.raw_units
        len_enemy_units             = len([unit for unit in states if unit.alliance == features.PlayerRelative.ENEMY])
        len_enemy_units             = min(len_enemy_units, 1000)
        #len_my_units                = len([unit for unit in states if unit.alliance == features.PlayerRelative.SELF])
        #len_my_units                = min(len_my_units, 200)

        total_minerals_collected    = obs.observation.player.minerals
        minerals_used               = max(0, self.prev_minerals - total_minerals_collected)
        self.prev_minerals          = total_minerals_collected

        total_gas_vespene_collected = obs.observation.player.vespene
        gas_used                    = max(0, self.prev_gas - total_gas_vespene_collected)
        self.prev_gas               = total_gas_vespene_collected

        free_supply                 = (obs.observation.player.food_cap - obs.observation.player.food_used)
        supply_used                 = max(0, self.prev_supply - free_supply)
        self.prev_supply            = free_supply
    
        game_time                   = int(obs.observation["game_loop"]  / (16 * self.step_mul))

        """ 
        Toma todos los valores del juego que consideramos importantes 
        y luego los devuelve en una tupla para alimentar nuestro algoritmo de aprendizaje automático

        Observaciones:
            1. Cantidad de trabajadores SCVs
            2. Cantidad de marines entrenados
            3. Cantidad de Marauders entrenados
            4. Cantidad de command Center completados
            5. Cantidad de supplyDepots completados
            6. Cantidad de Barracks completados
            7. Cantidad de Refinerias completados
            8. Cantidad de tech labs completados
            9. Cantidad de Bunkers completados
            10. Cantidad total de Estructuras y Ejercitos del enemigo detectado
            11. Cantidad de minerales utilizados
            12. Cantidad de Gases utilizados
            13. Cantidad de Recursos utilizados
            14. El instante de tiempo t
         """
        return (
            self.normalize_to_float(len(scvs),100, 1),
            self.normalize_to_float(len(marines),200, 1),
            self.normalize_to_float(len(marauders),100, 1),
            self.normalize_to_float(len(completed_command_center),50, 1),
            self.normalize_to_float(len(completed_supply_depots),50, 1),
            self.normalize_to_float(len(completed_barrackses),50, 1),
            self.normalize_to_float(len(completed_refineries),50, 1),
            self.normalize_to_float(len(completed_barrack_tech_lab),50, 1),
            self.normalize_to_float(len(completed_bunker),50, 1),
            self.normalize_to_float(len_enemy_units,1000, 1),
            self.normalize_to_float(minerals_used,20000, 1),
            self.normalize_to_float(gas_used,5000, 1),
            self.normalize_to_float(supply_used,500, 1),
            self.normalize_to_float(game_time,100, 1)
        )

    def normalize_to_float(self, value, factor, digit):
        format_decimal = '{:.'+str(digit)+'f}'
        return float(format_decimal.format(value/factor))    

    def step(self, obs):    
        super(AgentQlearning, self).step(obs)
        #-------------------------------------------
        #Obtener nuevo estado
        state = str(self.get_state(obs))
        print(100*'-')
        logging.info('State                     :' + str(state))
        #-------------------------------------------
        if self.train_mode and obs.last():
            self.update_final_reward_and_retrain(obs, 'terminal')
        #-------------------------------------------
        global multiActions
        if multiActions:
            # Ejecutar la acción instantánea y obtener detalles de la acción
            instant_action = executeActions()
            specify_action, execute_action, positions = self.get_specific_action(obs, instant_action)

            # Actualizar el contador de acciones según se haya ejecutado o no la acción
            if execute_action:
                index = ACTION_TO_INDEX.get(instant_action)
                if index is not None:
                    self.totals[index] += 1
            else:
                self.totals[15] += 1

            # Calcular y acumular la recompensa si estamos en modo entrenamiento
            if self.train_mode:
                reward_actions = self.reward_function.get_specific_reward(
                    instant_action,
                    execute_action, 
                    obs
                )

                self.total_rewards_by_policy  += reward_actions
                self.total_rewards_by_episode += reward_actions

                logging.info(f"Reward instantaneo        : {reward_actions}")
                logging.info(f"Total Rewards por política: {self.total_rewards_by_policy}")
            return specify_action
        else:
            if self.train_mode:
                # Actualiza la Q-table con la política previa
                next_state = 'terminal' if obs.last() else state

                # Agrega la información del episodio a episode_rewards para actualizar
                # la Q-table posteriormente al finalizar el episodio
                if self.previous_policy is not None:
                    self.episode_rewards.append((
                        self.previous_state, 
                        self.previous_policy, 
                        self.total_rewards_by_policy,
                        next_state
                    ))

                    qtable.learn(
                        self.previous_state,
                        self.previous_policy,
                        self.total_rewards_by_policy,
                        next_state
                    )

                # Actualiza epsilon usando decaimiento exponencial y asegura que no sea menor al mínimo
                self.epsilon = qtable.exponential_decay(self.episodes, EXPLORATION_DECAY, EXPLORATION_MAX)
                self.epsilon = max(EXPLORATION_MIN, self.epsilon)
                logging.info(f"epsilon seleccionado: {self.epsilon}")
            #-------------------------------------------
            # Actualiza el tiempo total de juego
            # Representa el número de "ticks" o frames que han transcurrido desde el inicio del juego en StarCraft II
            game_time = int(obs.observation["game_loop"]  / (16 * self.step_mul))
            self.total_game_time += game_time

            # Selección de la acción según la política
            policy_selected, random_action = qtable.choose_action(
                state,
                self.epsilon,
                game_time, 
                self.train_mode
            )

            if random_action:
                self.count_exploration += 1
            else:
                self.count_explotation += 1

            key, policy_select      = self.get_specific_policy( -1, policy_selected )
            #-------------------------------------------
            if policy_select != None:  
                multiActions = policy_select
                self.total_actions_by_policy = len(multiActions)
                logging.info('politica(accion/acciones) seleccionada/s          : ' + str(policy_select))
            #-------------------------------------------
            self.previous_state             = state
            self.previous_policy            = policy_selected
            self.total_rewards_by_policy    = 0
            return actions.FUNCTIONS.no_op()

    def reset(self):
        self.finish     = datetime.datetime.now()
        difference_time = (self.finish - self.start).total_seconds()

        super(AgentQlearning, self).reset()

        #logging.info(qtable.q_table)
        qtable.count += 1
        qtable.q_table.to_csv('new_qlearning_table_train.csv', mode='w')
        #-----------------------------------------------------------------------------------------
        new_row = { 'episode'             :self.episodes,
                    'epsilon'             :self.epsilon,
                    'total_steps'         :self.steps,
                    'total_game_time'     :self.total_game_time,
                    'harvest_minerals'    :self.totals[0],
                    'harvest_gas'         :self.totals[1],
                    'build_command_center':self.totals[2],
                    'build_scv'           :self.totals[3], 
                    'build_supply_depot'  :self.totals[4], 
                    'build_barracks'      :self.totals[5],
                    'build_tech_lab'      :self.totals[6],
                    'build_bunker'        :self.totals[7],
                    'explore_csv'         :self.totals[8],
                    'train_marine'        :self.totals[9],
                    'train_marauder'      :self.totals[10], 
                    'attack_with_marine'  :self.totals[11],
                    'defense_with_marine' :self.totals[12],
                    'attack_with_marauder':self.totals[13],
                    'minerals_used'       :None,
                    'gas_used'            :None,
                    'supply_used'         :None,
                    'detectable_enemy_units':None,
                    'do_nothing'          :self.totals[14],
                    'total_fail'          :self.totals[15],
                    'count_exploration'   :self.count_exploration,
                    'count_explotation'   :self.count_explotation,
                    'total_rewards'       :self.total_rewards_by_episode,
                    'reward_final'        :self.reward_final,               
                    'start_datetime'      :self.start, 
                    'fin_datetime'        :self.finish, 
                    'diff_time_min'       :difference_time,
                    'len_enemy_units'     :None,
                    'len_my_units'        :None
                    }
        name_file = 'new_qlearning_stats_train.csv'
        self.data_stats_train = self.data_stats_train.append(new_row, ignore_index=True)
        #data = pd.concat([data, new_row])
        self.data_stats_train.to_csv(name_file, mode='w')

        self.new_game()
 
    # Start the new game and store actions and states for the reinforcement learning
    def new_game(self):
        self.count_exploration          = 0
        self.count_explotation          = 0
        self.start                      = datetime.datetime.now()
        #self.totals                    = (self.total_policies + 1)*[0]
        self.totals                     = (20)*[0]
        self.episode_rewards            = []
        self.total_game_time            = 0
        self.previous_state             = None
        self.previous_policy            = None
        self.total_rewards_by_episode   = 0
        self.reward_final               = 0 
        self.prev_minerals              = 0
        self.prev_gas                   = 0
        self.prev_supply                = 0
        self.helpers.initialize_used_positions()

  
