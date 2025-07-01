"""
Agente Aleatorio para StarCraft II con PySC2.

Este módulo implementa un agente que selecciona acciones de forma aleatoria
para jugar StarCraft II, sirviendo como línea base (baseline) para comparar
el rendimiento del agente Q-Learning. Incluye:
- Selección aleatoria de acciones
- Cálculo de recompensas para análisis
- Persistencia de estadísticas de rendimiento
- Métricas de comparación con otros agentes

Este agente es útil para establecer un punto de referencia y evaluar
la efectividad del aprendizaje por refuerzo.

Referencias:
- https://github.com/deepmind/pysc2
- https://liquipedia.net/starcraft2/Terran_Strategy

Autor: Pablo Escobar
Fecha: 2022
"""

from pysc2.lib import features, actions, units
from general_agent import TerranAgent
from algorithms.q_learning import QLearningTable
from algorithms.rewards import Reward
from libs.logging_config import get_logger
import pandas as pd
import numpy as np
import datetime
import os
import random
import chardet

# Configuración del logger para este módulo
logger = get_logger('agent_random')

# Variables globales para manejo de acciones múltiples
multiActions = []

def executeActions():
    """
    Ejecuta la siguiente acción en la cola de acciones múltiples.
    
    Returns:
        str or False: Próxima acción a ejecutar o False si no hay acciones pendientes
    """
    global multiActions
    if len(multiActions) > 0:
        action = multiActions[0]
        multiActions = multiActions[1:len(multiActions)]
        logger.debug(f"Ejecutando acción: {action}")
        return action
    else:
        logger.debug("No hay acciones pendientes")
        return False

# Tabla Q global (no se usa para aprendizaje, solo para compatibilidad)
qtable = None

# Semilla para reproducibilidad
random.seed(42)

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

class AgentRandom(TerranAgent):
    """
    Agente aleatorio para StarCraft II como línea base de comparación.
    
    Esta clase implementa un agente que selecciona acciones de forma completamente
    aleatoria, sin ningún aprendizaje. Sirve como punto de referencia para
    evaluar la efectividad del agente Q-Learning.
    
    Attributes:
        epsilon (float): No se usa en este agente (siempre 0)
        episodes (int): Número de episodios completados
        step_mul (int): Multiplicador de pasos del juego
        total_game_time (int): Tiempo total de juego acumulado
        policies (list): Lista de políticas/acciones disponibles
        total_policies (int): Número total de políticas
        reward_function (Reward): Función de cálculo de recompensas
        episode_rewards (list): Recompensas acumuladas del episodio actual
        totals (list): Contador de acciones ejecutadas por tipo
    """
    
    def __init__(self, step_mul):
        """
        Inicializa el agente aleatorio.
        
        Args:
            step_mul (int): Multiplicador de pasos del juego
            
        Example:
            >>> agent = AgentRandom(step_mul=8)
        """
        super(AgentRandom, self).__init__()
        
        logger.info("Inicializando agente aleatorio")
        
        # Hiperparámetros (no se usan para selección aleatoria)
        self.epsilon = 0
        self.episodes = 0
        self.step_mul = step_mul
        self.total_game_time = 0
        
        # Variables de seguimiento de recursos
        self.prev_minerals = 0
        self.prev_gas = 0
        self.prev_supply = 0
        
        # Definición de políticas (acciones)
        self.policies = self.get_all_policies()
        self.total_policies = len(self.policies)
        logger.info(f"Políticas disponibles: {self.total_policies}")
        
        # Sistema de recompensas (para análisis)
        self.reward_actions = 0
        self.execute_action = None
        self.instant_action = ''
        self.total_rewards_by_episode = 0
        self.total_rewards_by_policy = 0
        self.reward_function = Reward()
        
        # Control de tiempo
        self.start = 0
        self.finish = 0
        
        # Control de construcciones por episodio
        self.totals = [0] * 20  # Contador para cada tipo de acción
        
        # Variables de control
        self.count_exploration = 0
        self.count_explotation = 0
        self.reward_final = 0
        
        # Acumulador de recompensas del episodio (para análisis)
        self.episode_rewards = []
        
        # Inicializar tabla Q global (no se usa para aprendizaje)
        global qtable
        qtable = QLearningTable(self.policies, self.total_policies)
        
        # Cargar estadísticas de entrenamiento previo
        self.data_stats_train, last_episodes, self.new_name_file = self.check_file_stats_game(agent_name='random')
        self.episodes = last_episodes
        
        # Iniciar nuevo juego
        self.new_game()
        
        logger.info("Agente aleatorio inicializado correctamente")

    def get_encoding_and_separator(self, name_file):
        """
        Detecta la codificación y separador de un archivo CSV.
        
        Args:
            name_file (str): Ruta al archivo CSV
            
        Returns:
            tuple: (encoding, separator) donde:
                - encoding: Codificación detectada del archivo
                - separator: Separador de columnas detectado
        """
        logger.debug(f"Detectando codificación y separador para: {name_file}")
        
        # Detectar codificación
        with open(name_file, 'rb') as f:
            result = chardet.detect(f.read())
        encoding = result['encoding']
        
        # Leer primeras líneas para detectar separador
        with open(name_file, 'r', encoding=encoding) as f:
            first_lines = [next(f) for _ in range(5)]

        # Posibles separadores
        possible_separators = [',', ';', '\t']
        separator_counts = {}

        for line in first_lines:
            for separator in possible_separators:
                count = line.count(separator)
                separator_counts[separator] = separator_counts.get(separator, 0) + count

        # Encontrar el separador más frecuente
        best_separator = max(separator_counts, key=separator_counts.get)
        
        logger.debug(f"Codificación detectada: {encoding}, Separador: {best_separator}")
        return encoding, best_separator

    def _valid_csv_exists(self, agent_name):
        """
        Verifica si existe un archivo CSV de estadísticas previo.
        
        Args:
            agent_name (str): Nombre del agente
            
        Returns:
            tuple: (exists, filename) donde:
                - exists: True si el archivo existe
                - filename: Nombre del archivo
        """
        name_file = f'old_{agent_name}_stats_train.csv'
        try:
            with open(name_file) as file:
                file.close()
            logger.debug(f"Archivo CSV encontrado: {name_file}")
            return True, name_file
        except FileNotFoundError:
            logger.debug(f"Archivo CSV no encontrado: {name_file}")
            return False, ''

    def check_file_stats_game(self, agent_name='random'):
        """
        Verifica y carga estadísticas de entrenamiento previo.
        
        Args:
            agent_name (str): Nombre del agente
            
        Returns:
            tuple: (data_stats_train, last_episodes, new_name_file) donde:
                - data_stats_train: DataFrame con estadísticas
                - last_episodes: Número del último episodio
                - new_name_file: Nombre del archivo para nuevas estadísticas
        """
        logger.info("Verificando archivo de estadísticas de entrenamiento")
        
        # Lista de columnas esperadas
        keys_list = [
            'episode', 'epsilon', 'total_steps', 'total_game_time',
            'harvest_minerals', 'harvest_gas', 'build_command_center',
            'build_scv', 'build_supply_depot', 'build_barracks',
            'build_tech_lab', 'build_bunker', 'explore_csv',
            'train_marine', 'train_marauder', 'attack_with_marine',
            'defense_with_marine', 'attack_with_marauder', 'minerals_used',
            'gas_used', 'supply_used', 'detectable_enemy_units',
            'do_nothing', 'total_fail', 'count_exploration',
            'count_explotation', 'total_rewards', 'reward_final',
            'scores', 'start_datetime', 'fin_datetime', 'diff_time_min',
            'len_enemy_units', 'len_my_units'
        ]

        exist_csv, name_file = self._valid_csv_exists(agent_name)
        last_episodes = 0
        
        if exist_csv:
            # Cargar estadísticas existentes
            encoding, best_separator = self.get_encoding_and_separator(name_file)
            logger.info(f"Codificación: {encoding}, Separador: {best_separator}")
            
            data_stats_train = pd.read_csv(name_file, encoding=encoding, delimiter=best_separator)
            data_stats_train = data_stats_train[keys_list]
            
            logger.info(f"Archivo: {name_file}")
            logger.info(f"Estadísticas cargadas: {data_stats_train.shape}")
            logger.info(f"Último episodio: {data_stats_train.shape[0] - 1}")
            
            last_episodes = data_stats_train.shape[0] - 1
        else:
            # Crear nuevo DataFrame de estadísticas
            data_stats_train = pd.DataFrame(columns=keys_list, dtype=np.float64)
            name_file = f'new_{agent_name}_stats_train.csv'
            logger.info("Nuevo archivo de estadísticas creado")
        
        return data_stats_train, last_episodes, name_file

    def update_final_reward(self, obs):
        """
        Actualiza la recompensa final del episodio.
        
        Args:
            obs: Observación final del juego
        """
        logger.info("Actualizando recompensa final del episodio")
        
        # Extraer información de unidades
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

        # Calcular recompensa final según resultado del juego
        if obs.reward == 1:
            logger.info("¡Victoria! Recompensa: +100")
            self.reward_final = r_win
        elif obs.reward == -1:
            logger.info("Derrota. Penalización: -50")
            self.reward_final = r_loss
        elif obs.reward == 0:
            logger.info("Empate. Calculando recompensa basada en unidades")
            # Evitar división por cero
            len_my = len_enemy_units if len_enemy_units > 0 else 1
            len_enemy = len_enemy_units if len_enemy_units > 0 else 1
            
            if len_my >= len_enemy:
                self.reward_final = min(int(c * (len_my / len_enemy)), 10)
            else:
                self.reward_final = max(int(-c * (len_enemy / len_my)), -1)

        logger.info(f"Recompensa final calculada: {self.reward_final}")
        self.total_rewards_by_episode += self.reward_final

    def get_state(self, obs):
        """
        Extrae y normaliza el estado del juego (para compatibilidad).
        
        Args:
            obs: Observación del estado actual del juego
            
        Returns:
            tuple: Estado normalizado con 14 características del juego
            
        Note:
            Este método se mantiene para compatibilidad, aunque el agente
            aleatorio no usa el estado para tomar decisiones.
        """
        logger.debug("Extrayendo estado del juego (agente aleatorio)")
        
        # Obtener unidades propias
        scvs = self.helpers.get_my_units_by_type(obs, units.Terran.SCV)
        marines = self.helpers.get_my_units_by_type(obs, units.Terran.Marine)
        marauders = self.helpers.get_my_units_by_type(obs, units.Terran.Marauder)
        
        # Obtener estructuras completadas
        completed_command_center = self.helpers.get_my_completed_units_by_type(obs, units.Terran.CommandCenter)
        completed_supply_depots = self.helpers.get_my_completed_units_by_type(obs, units.Terran.SupplyDepot)
        completed_barrackses = self.helpers.get_my_completed_units_by_type(obs, units.Terran.Barracks)
        completed_refineries = self.helpers.get_my_completed_units_by_type(obs, units.Terran.Refinery)
        completed_barrack_tech_lab = self.helpers.get_my_completed_units_by_type(obs, units.Terran.BarracksTechLab)
        completed_bunker = self.helpers.get_my_completed_units_by_type(obs, units.Terran.Bunker)
        
        # Contar unidades enemigas
        states = obs.observation.raw_units
        len_enemy_units = len([unit for unit in states if unit.alliance == features.PlayerRelative.ENEMY])
        len_enemy_units = min(len_enemy_units, 1000)  # Limitar a 1000

        # Calcular recursos utilizados
        total_minerals_collected = obs.observation.player.minerals
        minerals_used = max(0, self.prev_minerals - total_minerals_collected)
        self.prev_minerals = total_minerals_collected

        total_gas_vespene_collected = obs.observation.player.vespene
        gas_used = max(0, self.prev_gas - total_gas_vespene_collected)
        self.prev_gas = total_gas_vespene_collected

        free_supply = (obs.observation.player.food_cap - obs.observation.player.food_used)
        supply_used = max(0, self.prev_supply - free_supply)
        self.prev_supply = free_supply
    
        # Calcular tiempo de juego
        game_time = int(obs.observation["game_loop"] / (16 * self.step_mul))

        # Retornar estado normalizado con 14 características
        state = (
            self.normalize_to_float(len(scvs), 100, 1),           # SCVs
            self.normalize_to_float(len(marines), 200, 1),        # Marines
            self.normalize_to_float(len(marauders), 100, 1),      # Marauders
            self.normalize_to_float(len(completed_command_center), 50, 1),  # Command Centers
            self.normalize_to_float(len(completed_supply_depots), 50, 1),    # Supply Depots
            self.normalize_to_float(len(completed_barrackses), 50, 1),       # Barracks
            self.normalize_to_float(len(completed_refineries), 50, 1),       # Refineries
            self.normalize_to_float(len(completed_barrack_tech_lab), 50, 1), # Tech Labs
            self.normalize_to_float(len(completed_bunker), 50, 1),           # Bunkers
            self.normalize_to_float(len_enemy_units, 1000, 1),               # Enemy units
            self.normalize_to_float(minerals_used, 20000, 1),                # Minerals used
            self.normalize_to_float(gas_used, 5000, 1),                      # Gas used
            self.normalize_to_float(supply_used, 500, 1),                    # Supply used
            self.normalize_to_float(game_time, 100, 1)                       # Game time
        )
        
        logger.debug(f"Estado extraído: {state}")
        return state

    def normalize_to_float(self, value, factor, digit):
        """
        Normaliza un valor dividiéndolo por un factor y formateándolo.
        
        Args:
            value (int): Valor a normalizar
            factor (int): Factor de normalización
            digit (int): Número de decimales
            
        Returns:
            float: Valor normalizado
            
        Example:
            >>> normalized = agent.normalize_to_float(50, 100, 1)
            >>> print(f"Valor normalizado: {normalized}")  # 0.5
        """
        format_decimal = '{:.' + str(digit) + 'f}'
        return float(format_decimal.format(value / factor))    

    def step(self, obs):    
        """
        Ejecuta un paso del agente aleatorio en el juego.
        
        Args:
            obs: Observación del estado actual del juego
            
        Returns:
            action: Acción a ejecutar en el juego
        """
        super(AgentRandom, self).step(obs)
        
        # Obtener nuevo estado (para compatibilidad)
        state = str(self.get_state(obs))
        logger.info('=' * 100)
        logger.info(f'Estado actual: {state}')
        
        # Manejar final de episodio
        if obs.last():
            self.update_final_reward(obs)
        
        # Ejecutar acciones múltiples si hay pendientes
        global multiActions
        if multiActions:
            # Ejecutar acción instantánea
            instant_action = executeActions()
            specify_action, execute_action, positions = self.get_specific_action(obs, instant_action)

            # Actualizar contador de acciones
            if execute_action:
                index = ACTION_TO_INDEX.get(instant_action)
                if index is not None:
                    self.totals[index] += 1
            else:
                self.totals[15] += 1  # Contador de fallos

            # Calcular recompensa (para análisis)
            reward_actions = self.reward_function.get_specific_reward(
                instant_action, execute_action, obs
            )

            self.total_rewards_by_policy += reward_actions
            self.total_rewards_by_episode += reward_actions

            logger.info(f"Recompensa instantánea: {reward_actions}")
            logger.info(f"Total recompensas por política: {self.total_rewards_by_policy}")

            return specify_action
        else:
            # Seleccionar nueva acción de forma aleatoria
            # Actualizar tiempo total de juego
            game_time = int(obs.observation["game_loop"] / (16 * self.step_mul))
            self.total_game_time += game_time

            # Selección completamente aleatoria (sin aprendizaje)
            policy_selected = random.randint(0, 13)
            logger.debug(f"Política aleatoria seleccionada: {policy_selected}")

            key, policy_select = self.get_specific_policy(policy_selected, '')
            
            if policy_select is not None:  
                multiActions = policy_select
                self.total_actions_by_policy = len(multiActions)
                logger.info(f'Política aleatoria seleccionada: {policy_select}')
            
            # Actualizar estado anterior (para compatibilidad)
            self.previous_state = state
            self.previous_policy = policy_selected
            self.total_rewards_by_policy = 0
            
            return actions.FUNCTIONS.no_op()

    def reset(self):
        """
        Reinicia el agente al final de un episodio.
        
        Guarda estadísticas y prepara el agente para el siguiente episodio.
        """
        logger.info("Reiniciando agente aleatorio al final del episodio")
        
        self.finish = datetime.datetime.now()
        difference_time = (self.finish - self.start).total_seconds()

        super(AgentRandom, self).reset()

        # Guardar tabla Q (para compatibilidad)
        qtable.count += 1
        qtable.q_table.to_csv('new_random_table_train.csv', mode='w')
        
        # Crear nueva fila de estadísticas
        new_row = {
            'episode': self.episodes,
            'epsilon': self.epsilon,
            'total_steps': self.steps,
            'total_game_time': self.total_game_time,
            'harvest_minerals': self.totals[0],
            'harvest_gas': self.totals[1],
            'build_command_center': self.totals[2],
            'build_scv': self.totals[3], 
            'build_supply_depot': self.totals[4], 
            'build_barracks': self.totals[5],
            'build_tech_lab': self.totals[6],
            'build_bunker': self.totals[7],
            'explore_csv': self.totals[8],
            'train_marine': self.totals[9],
            'train_marauder': self.totals[10], 
            'attack_with_marine': self.totals[11],
            'defense_with_marine': self.totals[12],
            'attack_with_marauder': self.totals[13],
            'minerals_used': None,
            'gas_used': None,
            'supply_used': None,
            'detectable_enemy_units': None,
            'do_nothing': self.totals[14],
            'total_fail': self.totals[15],
            'count_exploration': self.count_exploration,
            'count_explotation': self.count_explotation,
            'total_rewards': self.total_rewards_by_episode,
            'reward_final': self.reward_final,               
            'start_datetime': self.start, 
            'fin_datetime': self.finish, 
            'diff_time_min': difference_time,
            'len_enemy_units': None,
            'len_my_units': None
        }
        
        # Guardar estadísticas
        name_file = 'new_random_stats_train.csv'
        self.data_stats_train = self.data_stats_train.append(new_row, ignore_index=True)
        self.data_stats_train.to_csv(name_file, mode='w')

        # Iniciar nuevo juego
        self.new_game()
 
    def new_game(self):
        """
        Inicializa un nuevo juego/episodio.
        
        Reinicia todas las variables de estado y contadores
        necesarios para comenzar un nuevo episodio.
        """
        logger.info("Iniciando nuevo juego/episodio (agente aleatorio)")
        
        # Reiniciar contadores
        self.count_exploration = 0
        self.count_explotation = 0
        self.start = datetime.datetime.now()
        self.totals = [0] * 20
        self.episode_rewards = []
        self.total_game_time = 0
        
        # Reiniciar estado
        self.previous_state = None
        self.previous_policy = None
        self.total_rewards_by_episode = 0
        self.reward_final = 0 
        
        # Reiniciar recursos
        self.prev_minerals = 0
        self.prev_gas = 0
        self.prev_supply = 0
        
        # Reiniciar posiciones utilizadas
        self.helpers.initialize_used_positions()
        
        logger.info("Nuevo juego inicializado correctamente (agente aleatorio)")
  
