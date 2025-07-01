"""
Módulo para la construcción de cuarteles en StarCraft II.

Los cuarteles son edificios fundamentales para la producción de unidades militares
Terran. Este módulo maneja la lógica para construir cuarteles de manera eficiente,
considerando la disponibilidad de recursos, SCVs y posicionamiento estratégico.

Referencias:
- https://liquipedia.net/starcraft2/Barracks
- https://liquipedia.net/starcraft2/SCV

Autor: Pablo Escobar
Fecha: 2022
"""

import logging
from pysc2.lib import actions, features, units
import random
import numpy as np # Mathematical functions
import os

# Configuración del logger para este módulo
logger = logging.getLogger(__name__)

class BuildBarracks:
    """
    Clase responsable de la construcción de cuarteles Terran.
    
    Los cuarteles son edificios esenciales para la producción de unidades militares
    como Marines y Marauders. Esta clase maneja la lógica para construir cuarteles
    de manera eficiente, considerando recursos, trabajadores disponibles y
    posicionamiento estratégico cerca del centro de mando.
    
    Attributes:
        BARRACKS_COST (int): Costo en minerales para construir un cuartel (150)
        MAX_BARRACKS (int): Límite máximo de cuarteles permitidos (50)
        BUILD_RANGE (int): Rango de construcción alrededor del centro de mando (±5)
    """
    
    # Constantes de configuración
    BARRACKS_COST = 150
    MAX_BARRACKS = 50
    BUILD_RANGE = 5
    
    def __init__(self):
        """
        Inicializa la clase BuildBarracks.
        
        No requiere parámetros de inicialización ya que toda la lógica
        se ejecuta en el método build_barracks.
        """
        logger.debug("Inicializando módulo BuildBarracks")
        pass

    def build_barracks(self, obs, helper):
        """
        Construye un nuevo cuartel en una ubicación estratégica.
        
        Esta función implementa la lógica para construir cuarteles de manera eficiente:
        1. Verifica que existan depósitos de suministro completados
        2. Comprueba que haya suficientes recursos (minerales)
        3. Verifica que haya SCVs disponibles
        4. Controla que no se haya alcanzado el límite de cuarteles
        5. Selecciona una ubicación estratégica cerca del centro de mando
        6. Asigna el SCV más cercano para la construcción
        
        Args:
            obs: Observación del estado actual del juego
            helper: Objeto Helper con funciones auxiliares
            
        Returns:
            tuple: (acción, flag_éxito, posición) donde:
                - acción: Función RAW de PySC2 a ejecutar
                - flag_éxito: 1 si se puede ejecutar, 0 en caso contrario
                - posición: Coordenadas del cuartel (x, y) o (None, None)
                
        Example:
            >>> build_barracks = BuildBarracks()
            >>> action, success, pos = build_barracks.build_barracks(obs, helper)
            >>> if success:
            ...     print(f"Construyendo cuartel en posición {pos}")
        """
        logger.debug("Iniciando proceso de construcción de cuartel")
        
        # Obtener ubicación del centro de mando
        unit_type = units.Terran.CommandCenter
        command_center_location = helper.get_command_center_location(obs, unit_type)
        if command_center_location is None:
            logger.warning("No se encontró centro de mando para construir cuartel")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        
        # Verificar depósitos de suministro completados
        completed_supply_depots = helper.get_my_completed_units_by_type(obs, units.Terran.SupplyDepot)
        if len(completed_supply_depots) == 0:
            logger.debug("No hay depósitos de suministro completados. Se requiere al menos uno.")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        
        # Verificar recursos disponibles
        minerals = obs.observation.player.minerals
        if minerals < self.BARRACKS_COST:
            logger.debug(f"Minerales insuficientes para cuartel: {minerals} de {self.BARRACKS_COST}")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        
        # Verificar SCVs disponibles
        scvs = helper.get_my_units_by_type(obs, units.Terran.SCV)
        if len(scvs) == 0:
            logger.debug("No hay SCVs disponibles para construir cuartel")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        
        # Verificar límite de cuarteles
        barrackses = helper.get_my_units_by_type(obs, units.Terran.Barracks)
        current_barracks_count = len(barrackses)
        if current_barracks_count >= self.MAX_BARRACKS:
            logger.info(f"Límite de cuarteles alcanzado ({self.MAX_BARRACKS}). No se puede construir más.")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        
        logger.debug(f"Cuarteles actuales: {current_barracks_count}/{self.MAX_BARRACKS}")
        logger.debug(f"Depósitos completados: {len(completed_supply_depots)}, SCVs disponibles: {len(scvs)}")
        
        # Calcular ubicación para el cuartel
        li = -self.BUILD_RANGE
        ls = self.BUILD_RANGE
        positions = (command_center_location.x, command_center_location.y)
        
        # Generar ubicación aleatoria cerca del centro de mando
        tuple_location = helper.random_location(positions, li, ls)
        x, y = tuple_location[0], tuple_location[1]
        
        # Validar ubicación (evitar colisiones)
        barracks_xy = helper.validate_random_location(obs, x, y, positions, li, ls)
        
        if barracks_xy:
            # Encontrar el SCV más cercano para la construcción
            distances = helper.get_distances(obs, scvs, barracks_xy)
            if distances is not None and np.size(distances) > 0:
                scv = scvs[np.argmin(distances)]
                distance = min(distances)
                
                logger.info(f"Construyendo cuartel en posición {barracks_xy}")
                logger.debug(f"SCV asignado en ({scv.x}, {scv.y}), distancia: {distance:.2f}")
                
                # Ejecutar construcción
                try:
                    action = actions.RAW_FUNCTIONS.Build_Barracks_pt("now", scv.tag, barracks_xy)
                    return (action, 1, barracks_xy)
                    
                except Exception as e:
                    logger.error(f"Error al construir cuartel: {str(e)}")
                    return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
            else:
                logger.warning("No se pudieron calcular distancias a SCVs")
                return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        else:
            logger.debug("No se pudo encontrar ubicación válida para el cuartel")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
