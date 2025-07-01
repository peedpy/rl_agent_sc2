"""
Módulo de algoritmo Q-Learning para StarCraft II con PySC2.

Este módulo implementa el algoritmo Q-Learning para el agente de aprendizaje por refuerzo,
incluyendo:
- Tabla Q para almacenar valores de estado-acción
- Selección de acciones con exploración vs explotación
- Actualización de valores Q con propagación de recompensas
- Gestión de hiperparámetros y decaimiento de exploración
- Persistencia y carga de tablas Q desde archivos CSV

Referencias:
- https://en.wikipedia.org/wiki/Q-learning
- https://liquipedia.net/starcraft2/Strategy

Autor: Pablo Escobar
Fecha: 2022
"""

import random
import numpy as np
import pandas as pd
import datetime
import os
import sys
from libs.logging_config import get_logger

# Configuración del logger para este módulo
logger = get_logger('algorithms.q_learning')

# Reinforcement Learning Algorithm
class QLearningTable: 
    """
    Implementación del algoritmo Q-Learning para StarCraft II.
    
    Esta clase maneja el aprendizaje por refuerzo del agente, incluyendo:
    - Gestión de la tabla Q (estado-acción)
    - Selección de acciones con estrategia epsilon-greedy
    - Actualización de valores Q con propagación de recompensas
    - Persistencia de datos de entrenamiento
    
    Attributes:
        learning_rate (float): Tasa de aprendizaje (alpha)
        reward_decay (float): Factor de descuento (gamma)
        policies (list): Lista de políticas/acciones disponibles
        count (int): Contador de episodios
        q_table (DataFrame): Tabla Q con valores estado-acción
        total_policies (int): Número total de políticas
        columns (Index): Columnas de la tabla Q
    """
    
    def __init__(self, policies, total_policies):
        """
        Inicializa la tabla Q-Learning.
        
        Args:
            policies (list): Lista de políticas/acciones disponibles
            total_policies (int): Número total de políticas
            
        Example:
            >>> policies = ['build_scv', 'train_marine', 'attack']
            >>> q_table = QLearningTable(policies, len(policies))
        """
        logger.info("Inicializando tabla Q-Learning")
        
        # Hiperparámetros iniciales
        self.learning_rate = 0
        self.reward_decay = 0
        
        # Configuración de políticas
        self.policies = policies
        self.total_policies = total_policies
        self.count = 0
        
        # Crear tabla Q
        self.q_table = self.create_model()
        self.columns = self.q_table.columns[0:self.total_policies]
        
        logger.info(f"Tabla Q-Learning inicializada con {len(policies)} políticas")

    def setup_hyperparameters(self, condition):
        """
        Configura los hiperparámetros del algoritmo Q-Learning.
        
        Args:
            condition (bool): True para configuración normal, False para reentrenamiento
            
        Example:
            >>> q_table.setup_hyperparameters(True)  # Configuración normal
            >>> q_table.setup_hyperparameters(False)  # Reentrenamiento
        """
        logger.debug(f"Configurando hiperparámetros - condición: {condition}")
        
        if condition:
            # Configuración normal para entrenamiento estándar
            self.learning_rate = 0.001
            self.reward_decay = 0.9
            logger.info("Hiperparámetros configurados para entrenamiento normal")
        else:
            # Configuración para reentrenamiento con valores más agresivos
            self.learning_rate = 0.01
            self.reward_decay = 0.99
            logger.info("Hiperparámetros configurados para reentrenamiento")
        
        logger.debug(f"Learning rate: {self.learning_rate}, Reward decay: {self.reward_decay}")

    def get_qtable(self):
        """
        Obtiene la tabla Q actual.
        
        Returns:
            DataFrame: Tabla Q con valores estado-acción
        """
        return self.q_table

    def exponential_decay(self, episode, EXPLORATION_DECAY, EXPLORATION_MAX):
        """
        Calcula el valor de epsilon usando decaimiento exponencial.
        
        Args:
            episode (int): Número del episodio actual
            EXPLORATION_DECAY (float): Factor de decaimiento de exploración
            EXPLORATION_MAX (float): Valor máximo de exploración
            
        Returns:
            float: Valor de epsilon para el episodio actual
            
        Example:
            >>> epsilon = q_table.exponential_decay(100, 0.0003, 1.0)
            >>> print(f"Epsilon: {epsilon:.4f}")
        """
        epsilon = EXPLORATION_MAX * np.exp(-EXPLORATION_DECAY * episode)
        logger.debug(f"Epsilon calculado: {epsilon:.4f} para episodio {episode}")
        return epsilon

    def choose_action(self, observation, epsilon, game_time, training=False):
        """
        Selecciona una acción usando la estrategia epsilon-greedy.
        
        Args:
            observation (str): Estado actual del juego
            epsilon (float): Probabilidad de exploración
            game_time (int): Tiempo actual del juego
            training (bool): True si está en modo entrenamiento
            
        Returns:
            tuple: (acción, es_exploración) donde:
                - acción: Acción seleccionada
                - es_exploración: True si fue exploración, False si fue explotación
                
        Example:
            >>> action, is_exploration = q_table.choose_action(state, 0.1, 100, True)
            >>> print(f"Acción: {action}, Exploración: {is_exploration}")
        """
        logger.debug(f"Seleccionando acción - epsilon: {epsilon:.4f}, training: {training}")
        
        # Verificar que el estado existe en la tabla Q
        self.check_state_exist(observation)

        if training:
            # Modo entrenamiento: usar epsilon-greedy
            if np.random.uniform() < epsilon:
                # EXPLORACIÓN: acción aleatoria
                action = np.random.choice(self.policies[0:self.total_policies])
                action_tuple = (action, True)
                logger.debug(f"Exploración: acción aleatoria '{action}' seleccionada")
            else:
                # EXPLOTACIÓN: mejor acción según tabla Q
                Qactions = self.q_table.loc[observation, self.columns]
                action = np.random.choice(Qactions[Qactions == np.max(Qactions)].index)
                action_tuple = (action, False)
                logger.debug(f"Explotación: mejor acción '{action}' seleccionada")
        else:
            # Modo evaluación: siempre explotación
            Qactions = self.q_table.loc[observation, self.columns]
            logger.debug(f"Qactions disponibles: {Qactions}")
            
            if Qactions.shape[0] > 0:
                action = np.random.choice(Qactions[Qactions == np.max(Qactions)].index)
                logger.debug(f"Acción seleccionada en evaluación: '{action}'")
            else:
                # Estado no existe, no hacer nada
                action = 'do_nothing'
                logger.warning(f"Estado '{observation}' no encontrado, ejecutando 'do_nothing'")
            
            action_tuple = (action, False)
        
        return action_tuple

    def propagate_rewards(self, episode_rewards, final_reward):
        """
        Propaga las recompensas finales del episodio actualizando los valores Q.
        
        Esta función implementa la propagación de recompensas hacia atrás,
        actualizando los valores Q de todas las acciones del episodio con
        la recompensa final como señal de retroalimentación.
        
        Args:
            episode_rewards (list): Lista de tuplas (estado, acción, recompensa, siguiente_estado)
            final_reward (float): Recompensa final del episodio
            
        Example:
            >>> episode_data = [(state1, action1, reward1, state2), ...]
            >>> q_table.propagate_rewards(episode_data, 100.0)
        """
        logger.info("Iniciando propagación de recompensas")
        
        # Configurar hiperparámetros para reentrenamiento
        self.setup_hyperparameters(condition=False)
        
        if len(episode_rewards) == 0:
            logger.warning("No hay recompensas de episodio para propagar")
            return

        # Procesar cada paso del episodio
        for i, (state, action, instant_reward, next_state) in enumerate(episode_rewards):
            if state is not None and action is not None and instant_reward is not None:
                logger.debug(f"Iteración {i}:")
                logger.debug(f"  Estado actual: {state}")
                logger.debug(f"  Siguiente estado: {next_state}")
                logger.debug(f"  Acción elegida: {action}")
                logger.debug(f"  Recompensa instantánea: {instant_reward}")
                logger.debug(f"  Recompensa final: {final_reward}")
                logger.debug(f"  Recompensa total: {instant_reward + final_reward}")
                
                # Mostrar valor Q antes de la actualización
                old_q_value = self.q_table.loc[state, action]
                logger.debug(f"  Valor Q antes: {old_q_value}")
                
                # Aprender con la recompensa final
                self.learn(state, action, final_reward, next_state)
                
                # Mostrar valor Q después de la actualización
                new_q_value = self.q_table.loc[state, action]
                logger.debug(f"  Valor Q después: {new_q_value}")
                logger.debug("  " + 50*"_")
        
        # Restaurar hiperparámetros normales
        self.setup_hyperparameters(condition=True)
        logger.info("Propagación de recompensas completada")

    def learn(self, s, a, r, s_):
        """
        Actualiza el valor Q para un par estado-acción específico.
        
        Implementa la ecuación de actualización Q-Learning:
        Q(s,a) = Q(s,a) + α[r + γ * max(Q(s',a')) - Q(s,a)]
        
        Args:
            s (str): Estado actual
            a (str): Acción tomada
            r (float): Recompensa recibida
            s_ (str): Estado siguiente
            
        Example:
            >>> q_table.learn("state1", "build_scv", 10.0, "state2")
        """
        logger.debug(f"Aprendiendo: estado={s}, acción={a}, recompensa={r}, siguiente_estado={s_}")
        
        # Verificar que el estado siguiente existe
        self.check_state_exist(s_)
        
        # Obtener valor Q actual
        q_predict = self.q_table.loc[s, a]
        logger.debug(f"Valor Q actual: {q_predict}")

        if s_ != 'terminal':
            # Calcular valor máximo del estado siguiente
            q_max_next_state = self.q_table.loc[s_, self.columns].max()
            logger.debug(f"Valor Q máximo del siguiente estado: {q_max_next_state}")
            
            # Calcular valor objetivo
            q_target = r + self.reward_decay * q_max_next_state
        else:
            # Estado terminal: no hay estado siguiente
            q_target = r
            logger.debug("Estado terminal detectado")

        # Calcular error y actualizar valor Q
        error = q_target - q_predict
        self.q_table.loc[s, a] += self.learning_rate * error
        
        logger.debug(f"Error: {error:.4f}, Nuevo valor Q: {self.q_table.loc[s, a]:.4f}")

    def check_state_exist(self, state):
        """
        Verifica si un estado existe en la tabla Q y lo agrega si no existe.
        
        Args:
            state (str): Estado a verificar
            
        Example:
            >>> q_table.check_state_exist("nuevo_estado")
        """
        if state not in self.q_table.index:
            logger.debug(f"Agregando nuevo estado a la tabla Q: {state}")
            self.q_table = self.q_table.append(
                pd.Series([0] * len(self.policies), 
                         index=self.q_table.columns, 
                         name=state)
            )

    def load_train_csv(self):
        """
        Carga la tabla Q desde un archivo CSV de entrenamiento previo.
        
        Returns:
            DataFrame: Tabla Q cargada desde el archivo
            
        Example:
            >>> q_table = q_table.load_train_csv()
        """
        logger.info("Cargando tabla Q desde archivo CSV")
        
        try:
            df = pd.read_csv('old_qlearning_table_train.csv')  
            df = df.rename(columns={'Unnamed: 0': 'status'})
            df = df.set_index(['status'])
            
            logger.info(f"Tabla Q cargada exitosamente: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error al cargar tabla Q: {str(e)}")
            raise

    def _valid_csv_exists(self):
        """
        Verifica si existe un archivo CSV de entrenamiento previo.
        
        Returns:
            bool: True si el archivo existe, False en caso contrario
        """
        try:
            with open('old_qlearning_table_train.csv') as file:
                file.close()
            logger.debug("Archivo CSV de entrenamiento encontrado")
            return True
        except FileNotFoundError:
            logger.debug("Archivo CSV de entrenamiento no encontrado")
            return False

    def create_model(self):
        """
        Crea o carga el modelo de tabla Q.
        
        Si existe un archivo CSV de entrenamiento previo, lo carga.
        En caso contrario, crea una nueva tabla Q vacía.
        
        Returns:
            DataFrame: Tabla Q inicializada
            
        Example:
            >>> q_table = q_table.create_model()
        """
        logger.info("Creando modelo de tabla Q")
        
        exist_csv = self._valid_csv_exists()
        logger.info(f"Archivo CSV existe: {exist_csv}")
        
        if exist_csv:
            # Cargar tabla Q existente
            q_table = self.load_train_csv()
            logger.info(f'Tabla Q cargada: {q_table.shape}')
        else:
            # Crear nueva tabla Q vacía
            q_table = pd.DataFrame(columns=self.policies, dtype=np.float64)
            logger.info("Nueva tabla Q creada")
        
        return q_table
