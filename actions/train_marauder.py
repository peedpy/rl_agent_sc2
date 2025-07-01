"""
Módulo para el entrenamiento de Marauders en StarCraft II.

Los Marauders son unidades avanzadas de combate de la raza Terran, especializadas
en daño contra unidades blindadas. Este módulo maneja la lógica para entrenar
Marauders desde cuarteles con laboratorio tecnológico.

Referencias:
- https://liquipedia.net/starcraft2/Marauder
- https://liquipedia.net/starcraft2/Barracks

Autor: Pablo Escobar
Fecha: 2022
"""

import logging
from pysc2.lib import actions, features, units
import random
import numpy as np
import os

# Configuración del logger para este módulo
logger = logging.getLogger(__name__)

class TrainMarauder:
    """
    Clase responsable del entrenamiento de Marauders Terran.
    
    Los Marauders son unidades avanzadas que requieren un laboratorio tecnológico
    en el cuartel para ser entrenados. Son efectivos contra unidades blindadas
    y proporcionan fuego de apoyo de largo alcance.
    
    Attributes:
        MARAUDER_COST (int): Costo en minerales para entrenar un Marauder (100)
        MARAUDER_GAS_COST (int): Costo en gas vespene para entrenar un Marauder (25)
        MAX_MARAUDERS (int): Límite máximo de Marauders permitidos (100)
        MAX_QUEUE_LENGTH (int): Máximo de órdenes pendientes en cuartel (10)
    """
    
    # Constantes de configuración
    MARAUDER_COST = 100
    MARAUDER_GAS_COST = 25
    MAX_MARAUDERS = 100
    MAX_QUEUE_LENGTH = 10
    
    def __init__(self):
        """
        Inicializa la clase TrainMarauder.
        
        No requiere parámetros de inicialización ya que toda la lógica
        se ejecuta en el método train_marauder.
        """
        logger.debug("Inicializando módulo TrainMarauder")
        pass

    def train_marauder(self, obs, helper):
        """
        Entrena un nuevo Marauder desde el cuartel con laboratorio tecnológico más cercano.
        
        Esta función implementa la lógica para entrenar Marauders de manera eficiente:
        1. Verifica que existan cuarteles con laboratorio tecnológico completados
        2. Comprueba que haya suficientes recursos (minerales y gas)
        3. Verifica que haya supply disponible
        4. Controla que no se haya alcanzado el límite de Marauders
        5. Filtra cuarteles que no estén sobrecargados
        6. Selecciona el cuartel más cercano al centro de mando
        7. Ejecuta la acción de entrenamiento
        
        Args:
            obs: Observación del estado actual del juego
            helper: Objeto Helper con funciones auxiliares
            
        Returns:
            tuple: (acción, flag_éxito, posición) donde:
                - acción: Función RAW de PySC2 a ejecutar
                - flag_éxito: 1 si se puede ejecutar, 0 en caso contrario
                - posición: Coordenadas del cuartel (x, y) o (None, None)
                
        Example:
            >>> train_marauder = TrainMarauder()
            >>> action, success, pos = train_marauder.train_marauder(obs, helper)
            >>> if success:
            ...     print(f"Entrenando Marauder en cuartel más cercano en posición {pos}")
        """
        logger.debug("Iniciando proceso de entrenamiento de Marauder")
        
        # Obtener cuarteles con laboratorio tecnológico completados
        completed_barrackses_tech_lab = helper.get_my_completed_units_by_type(obs, units.Terran.BarracksTechLab)
        if not completed_barrackses_tech_lab:
            logger.debug("No hay cuarteles con laboratorio tecnológico completados disponibles")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        
        logger.debug(f"Cuarteles con laboratorio tecnológico disponibles: {len(completed_barrackses_tech_lab)}")
        
        # Verificar recursos disponibles
        minerals = obs.observation.player.minerals
        vespene = obs.observation.player.vespene
        
        if minerals < self.MARAUDER_COST:
            logger.debug(f"Minerales insuficientes para Marauder: {minerals}/{self.MARAUDER_COST}")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        
        if vespene < self.MARAUDER_GAS_COST:
            logger.debug(f"Gas vespene insuficiente para Marauder: {vespene}/{self.MARAUDER_GAS_COST}")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        
        # Verificar supply disponible
        free_supply = (obs.observation.player.food_cap - obs.observation.player.food_used)
        if free_supply <= 0:
            logger.debug(f"No hay supply disponible: {free_supply}")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        
        # Verificar límite de Marauders
        marauders = helper.get_my_units_by_type(obs, units.Terran.Marauder)
        current_marauder_count = len(marauders)
        if current_marauder_count >= self.MAX_MARAUDERS:
            logger.info(f"Límite de Marauders alcanzado ({self.MAX_MARAUDERS}). No se puede entrenar más.")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        
        logger.debug(f"Marauders actuales: {current_marauder_count}/{self.MAX_MARAUDERS}, Supply disponible: {free_supply}")
        
        # Obtener ubicación del centro de mando para calcular distancias
        command_center_location = helper.get_command_center_location(obs, units.Terran.CommandCenter)
        if command_center_location is None:
            logger.warning("No se encontró centro de mando para calcular distancias")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        
        # Filtrar cuarteles que no estén sobrecargados
        available_barracks = []
        for barrack in completed_barrackses_tech_lab:
            if barrack.order_length <= self.MAX_QUEUE_LENGTH:
                available_barracks.append(barrack)
        
        if not available_barracks:
            logger.debug("No hay cuarteles con laboratorio tecnológico disponibles (todos sobrecargados)")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        
        logger.debug(f"Cuarteles disponibles sin sobrecarga: {len(available_barracks)}")
        
        # Calcular distancias al centro de mando y seleccionar el más cercano
        cc_position = (command_center_location.x, command_center_location.y)
        distances = helper.get_distances(obs, available_barracks, cc_position)
        
        if not distances:
            logger.warning("No se pudieron calcular distancias a los cuarteles")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        
        # Seleccionar el cuartel más cercano al centro de mando
        closest_barrack_index = np.argmin(distances)
        barrack = available_barracks[closest_barrack_index]
        barrack_position = (barrack.x, barrack.y)
        distance_to_cc = distances[closest_barrack_index]
        
        logger.debug(f"Cuartel seleccionado en {barrack_position}, distancia al CC: {distance_to_cc:.2f}")
        
        # Ejecutar entrenamiento
        try:
            action = actions.RAW_FUNCTIONS.Train_Marauder_quick("now", barrack.tag)
            
            logger.info(f"Entrenando Marauder en cuartel más cercano en posición {barrack_position}")
            logger.debug(f"Cuartel tag: {barrack.tag}, órdenes pendientes: {barrack.order_length}, distancia al CC: {distance_to_cc:.2f}")
            
            return (action, 1, barrack_position)
            
        except Exception as e:
            logger.error(f"Error al entrenar Marauder: {str(e)}")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))

    def has_tech_lab(self, obs, completed_barrackses):
        for barrack in completed_barrackses:
            for addon in obs.observation.raw_units:
                if addon.unit_type == units.Terran.BarracksTechLab and \
                    addon.alliance == features.PlayerRelative.SELF and \
                    abs(addon.x - barrack.x) < 5 and abs(addon.y - barrack.y) < 5:
                    print("La Barrack con posicion ({},{}) tiene un Tech Lab.".format(barrack.x, barrack.y))
                    return True, (barrack.x+1, barrack.y+1), barrack.tag
                    #os.system("pause")
        return False, (0,0), ''
