"""
Módulo para la construcción de búnkeres en StarCraft II.

Los búnkeres son estructuras defensivas que pueden albergar unidades terrestres
y proporcionar protección adicional. Este módulo maneja la lógica para construir
búnkeres de manera estratégica para defensa de la base.

Referencias:
- https://liquipedia.net/starcraft2/Bunker
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

class BuildBunker:
    """
    Clase responsable de la construcción de búnkeres Terran.
    
    Los búnkeres son estructuras defensivas que pueden contener hasta 4 unidades
    terrestres, proporcionando protección adicional y aumentando el daño de las
    unidades alojadas. Esta clase maneja la lógica para construir búnkeres
    de manera estratégica cerca del centro de mando.
    
    Attributes:
        BUNKER_COST (int): Costo en minerales para construir un búnker (100)
        MAX_BUNKERS (int): Límite máximo de búnkeres permitidos (50)
        BUILD_RANGE (int): Rango de construcción alrededor del centro de mando (±5)
        used_positions (list): Lista de posiciones ya utilizadas
    """
    
    # Constantes de configuración
    BUNKER_COST = 100
    MAX_BUNKERS = 50
    BUILD_RANGE = 5
    
    def __init__(self):
        """
        Inicializa la clase BuildBunker.
        
        Inicializa la lista de posiciones utilizadas para evitar
        construir búnkeres en ubicaciones ya ocupadas.
        """
        logger.debug("Inicializando módulo BuildBunker")
        self.used_positions = []

    def build_bunker(self, obs, helper):
        """
        Construye un nuevo búnker en una ubicación estratégica.
        
        Esta función implementa la lógica para construir búnkeres de manera eficiente:
        1. Verifica que haya suficientes recursos (minerales)
        2. Comprueba que haya SCVs disponibles
        3. Controla que no se haya alcanzado el límite de búnkeres
        4. Selecciona una ubicación estratégica cerca del centro de mando
        5. Asigna el SCV más cercano para la construcción
        
        Args:
            obs: Observación del estado actual del juego
            helper: Objeto Helper con funciones auxiliares
            
        Returns:
            tuple: (acción, flag_éxito, posición) donde:
                - acción: Función RAW de PySC2 a ejecutar
                - flag_éxito: 1 si se puede ejecutar, 0 en caso contrario
                - posición: Coordenadas del búnker (x, y) o (None, None)
                
        Example:
            >>> build_bunker = BuildBunker()
            >>> action, success, pos = build_bunker.build_bunker(obs, helper)
            >>> if success:
            ...     print(f"Construyendo búnker en posición {pos}")
        """
        logger.debug("Iniciando proceso de construcción de búnker")
        
        # Obtener ubicación del centro de mando
        unit_type = units.Terran.CommandCenter
        command_center_location = helper.get_command_center_location(obs, unit_type)
        if command_center_location is None:
            logger.warning("No se encontró centro de mando para construir búnker")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        
        # Verificar recursos disponibles
        minerals = obs.observation.player.minerals
        if minerals < self.BUNKER_COST:
            logger.debug(f"Minerales insuficientes para búnker: {minerals}/{self.BUNKER_COST}")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        
        # Verificar SCVs disponibles
        scvs = helper.get_my_units_by_type(obs, units.Terran.SCV)
        if len(scvs) == 0:
            logger.debug("No hay SCVs disponibles para construir búnker")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        
        # Verificar límite de búnkeres
        bunkers = helper.get_my_units_by_type(obs, units.Terran.Bunker)
        current_bunkers_count = len(bunkers)
        if current_bunkers_count >= self.MAX_BUNKERS:
            logger.info(f"Límite de búnkeres alcanzado ({self.MAX_BUNKERS}). No se puede construir más.")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        
        logger.debug(f"Búnkeres actuales: {current_bunkers_count}/{self.MAX_BUNKERS}")
        logger.debug(f"SCVs disponibles: {len(scvs)}")
        
        # Calcular ubicación para el búnker
        li = -self.BUILD_RANGE
        ls = self.BUILD_RANGE
        positions = (command_center_location.x, command_center_location.y)
        
        # Generar ubicación aleatoria cerca del centro de mando
        tuple_location = helper.random_location(positions, li, ls)
        x, y = tuple_location[0], tuple_location[1]
        
        # Validar ubicación (evitar colisiones)
        bunker_xy = helper.validate_random_location(obs, x, y, positions, li, ls)
        
        if bunker_xy:
            # Encontrar el SCV más cercano para la construcción
            distances = helper.get_distances(obs, scvs, bunker_xy)
            if distances is not None and np.size(distances) > 0:
                scv = scvs[np.argmin(distances)]
                distance = min(distances)
                
                logger.info(f"Construyendo búnker en posición {bunker_xy}")
                logger.debug(f"SCV asignado en ({scv.x}, {scv.y}), distancia: {distance:.2f}")
                
                # Ejecutar construcción
                try:
                    action = actions.RAW_FUNCTIONS.Build_Bunker_pt("now", scv.tag, bunker_xy)
                    return (action, 1, bunker_xy)
                    
                except Exception as e:
                    logger.error(f"Error al construir búnker: {str(e)}")
                    return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
            else:
                logger.warning("No se pudieron calcular distancias a SCVs")
                return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        else:
            logger.debug("No se pudo encontrar ubicación válida para el búnker")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
