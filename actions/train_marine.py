"""
Módulo para el entrenamiento de Marines en StarCraft II.

Los Marines son las unidades básicas de combate de la raza Terran, versátiles
y efectivas contra unidades terrestres y aéreas. Este módulo maneja la lógica
para entrenar Marines de manera eficiente desde los cuarteles.

Referencias:
- https://liquipedia.net/starcraft2/Marine
- https://liquipedia.net/starcraft2/Barracks

Autor: Pablo Escobar
Fecha: 2022
"""

import logging
from pysc2.lib import actions, features, units
import random
import numpy as np

# Configuración del logger para este módulo
logger = logging.getLogger(__name__)

class TrainMarine:
    """
    Clase responsable del entrenamiento de Marines Terran.
    
    Los Marines son unidades fundamentales para la estrategia militar Terran,
    proporcionando fuego de apoyo y siendo efectivos contra una amplia variedad
    de unidades enemigas. Esta clase maneja la lógica para entrenar Marines
    de manera eficiente, considerando recursos, supply y capacidad de producción.
    
    Attributes:
        MARINE_COST (int): Costo en minerales para entrenar un Marine (50)
        MAX_MARINES (int): Límite máximo de Marines permitidos (200)
        MAX_QUEUE_LENGTH (int): Máximo de órdenes pendientes en cuartel (10)
    """
    
    # Constantes de configuración
    MARINE_COST = 50
    MAX_MARINES = 200
    MAX_QUEUE_LENGTH = 10
    
    def __init__(self):
        """
        Inicializa la clase TrainMarine.
        
        No requiere parámetros de inicialización ya que toda la lógica
        se ejecuta en el método train_marine.
        """
        logger.debug("Inicializando módulo TrainMarine")
        pass

    def train_marine(self, obs, helper):
        """
        Entrena un nuevo Marine desde un cuartel disponible.
        
        Esta función implementa la lógica para entrenar Marines de manera eficiente:
        1. Verifica que existan cuarteles completados
        2. Comprueba que haya suficientes recursos (minerales)
        3. Verifica que haya supply disponible
        4. Controla que no se haya alcanzado el límite de Marines
        5. Selecciona un cuartel con capacidad de producción
        6. Ejecuta la acción de entrenamiento
        
        Args:
            obs: Observación del estado actual del juego
            helper: Objeto Helper con funciones auxiliares
            
        Returns:
            tuple: (acción, flag_éxito, posición) donde:
                - acción: Función RAW de PySC2 a ejecutar
                - flag_éxito: 1 si se puede ejecutar, 0 en caso contrario
                - posición: Coordenadas del cuartel (x, y) o (None, None)
                
        Example:
            >>> train_marine = TrainMarine()
            >>> action, success, pos = train_marine.train_marine(obs, helper)
            >>> if success:
            ...     print(f"Entrenando Marine en cuartel en posición {pos}")
        """
        logger.debug("Iniciando proceso de entrenamiento de Marine")
        
        # Obtener cuarteles completados
        completed_barrackses = helper.get_my_completed_units_by_type(obs, units.Terran.Barracks)
        if not completed_barrackses:
            logger.debug("No hay cuarteles completados disponibles para entrenar Marines")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        
        logger.debug(f"Cuarteles completados disponibles: {len(completed_barrackses)}")
        
        # Verificar recursos disponibles
        minerals = obs.observation.player.minerals
        if minerals < self.MARINE_COST:
            logger.debug(f"Minerales insuficientes para Marine: {minerals}/{self.MARINE_COST}")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        
        # Verificar supply disponible
        free_supply = (obs.observation.player.food_cap - obs.observation.player.food_used)
        if free_supply <= 0:
            logger.debug(f"No hay supply disponible: {free_supply}")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        
        # Verificar límite de Marines
        marines = helper.get_my_units_by_type(obs, units.Terran.Marine)
        current_marine_count = len(marines)
        if current_marine_count >= self.MAX_MARINES:
            logger.info(f"Límite de Marines alcanzado ({self.MAX_MARINES}). No se puede entrenar más.")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        
        logger.debug(f"Marines actuales: {current_marine_count} de {self.MAX_MARINES}, Supply disponible: {free_supply}")
        
        # Obtener todos los cuarteles (incluyendo los en construcción)
        barracks = helper.get_my_units_by_type(obs, units.Terran.Barracks)
        if not barracks:
            logger.warning("No se encontraron cuarteles para entrenar Marines")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        
        # Seleccionar cuartel aleatorio
        barrack = random.choice(barracks)
        barrack_position = (barrack.x, barrack.y)
        
        # Verificar que el cuartel no esté sobrecargado
        if barrack.order_length > self.MAX_QUEUE_LENGTH:
            logger.debug(f"Cuartel en {barrack_position} sobrecargado: {barrack.order_length} órdenes pendientes")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        
        # Ejecutar entrenamiento
        try:
            action = actions.RAW_FUNCTIONS.Train_Marine_quick("now", barrack.tag)
            
            logger.info(f"Entrenando Marine en cuartel en posición {barrack_position}")
            logger.debug(f"Cuartel tag: {barrack.tag}, órdenes pendientes: {barrack.order_length}")
            
            return (action, 1, barrack_position)
            
        except Exception as e:
            logger.error(f"Error al entrenar Marine: {str(e)}")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
