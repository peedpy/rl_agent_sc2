"""
Módulo para la construcción de centros de mando en StarCraft II.

Los centros de mando son las estructuras principales de la raza Terran, 
proporcionando supply, producción de SCVs y siendo el núcleo de la base.
Este módulo maneja la lógica para construir centros de mando adicionales
para expandir la economía y producción.

Referencias:
- https://liquipedia.net/starcraft2/Command_Center
- https://liquipedia.net/starcraft2/SCV

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

class BuildCommandCenter:
    """
    Clase responsable de la construcción de centros de mando Terran.
    
    Los centros de mando son estructuras fundamentales que proporcionan:
    - Supply para unidades
    - Producción de SCVs
    - Capacidad de mando y control
    - Base para expansiones económicas
    
    Esta clase maneja la lógica para construir centros de mando adicionales
    de manera estratégica para expandir la economía.
    
    Attributes:
        COMMAND_CENTER_COST (int): Costo en minerales para construir un centro de mando (400)
        MAX_COMMAND_CENTERS (int): Límite máximo de centros de mando permitidos (10)
        EXPANSION_RANGE (int): Rango de expansión desde el centro principal (±15)
    """
    
    # Constantes de configuración
    COMMAND_CENTER_COST = 400
    MAX_COMMAND_CENTERS = 10
    EXPANSION_RANGE = 15
    
    def __init__(self):
        """
        Inicializa la clase BuildCommandCenter.
        
        No requiere parámetros de inicialización ya que toda la lógica
        se ejecuta en el método build_command_center.
        """
        logger.debug("Inicializando módulo BuildCommandCenter")
        pass

    def build_command_center(self, obs, helper):
        """
        Construye un nuevo centro de mando en una ubicación estratégica.
        
        Esta función implementa la lógica para construir centros de mando de manera eficiente:
        1. Verifica que haya suficientes recursos (minerales)
        2. Comprueba que haya SCVs disponibles
        3. Controla que no se haya alcanzado el límite de centros de mando
        4. Selecciona una ubicación estratégica para la expansión
        5. Asigna el SCV más cercano para la construcción
        
        Args:
            obs: Observación del estado actual del juego
            helper: Objeto Helper con funciones auxiliares
            
        Returns:
            tuple: (acción, flag_éxito, posición) donde:
                - acción: Función RAW de PySC2 a ejecutar
                - flag_éxito: 1 si se puede ejecutar, 0 en caso contrario
                - posición: Coordenadas del centro de mando (x, y) o (None, None)
                
        Example:
            >>> build_cc = BuildCommandCenter()
            >>> action, success, pos = build_cc.build_command_center(obs, helper)
            >>> if success:
            ...     print(f"Construyendo centro de mando en posición {pos}")
        """
        logger.debug("Iniciando proceso de construcción de centro de mando")
        
        # Obtener ubicación del centro de mando principal
        unit_type = units.Terran.CommandCenter
        command_center_location = helper.get_command_center_location(obs, unit_type)
        if command_center_location is None:
            logger.warning("No se encontró centro de mando principal para expansión")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        
        # Verificar recursos disponibles
        minerals = obs.observation.player.minerals
        if minerals < self.COMMAND_CENTER_COST:
            logger.debug(f"Minerales insuficientes para centro de mando: {minerals} de {self.COMMAND_CENTER_COST}")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        
        # Verificar SCVs disponibles
        scvs = helper.get_my_units_by_type(obs, units.Terran.SCV)
        if len(scvs) == 0:
            logger.debug("No hay SCVs disponibles para construir centro de mando")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        
        # Verificar límite de centros de mando
        command_centers = helper.get_my_units_by_type(obs, units.Terran.CommandCenter)
        current_cc_count = len(command_centers)
        if current_cc_count >= self.MAX_COMMAND_CENTERS:
            logger.info(f"Límite de centros de mando alcanzado ({self.MAX_COMMAND_CENTERS}). No se puede construir más.")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        
        logger.debug(f"Centros de mando actuales: {current_cc_count}/{self.MAX_COMMAND_CENTERS}")
        logger.debug(f"SCVs disponibles: {len(scvs)}")
        
        # Calcular ubicación para la expansión
        li = -self.EXPANSION_RANGE
        ls = self.EXPANSION_RANGE
        positions = (command_center_location.x, command_center_location.y)
        
        # Generar ubicación aleatoria para la expansión
        tuple_location = helper.random_location(positions, li, ls)
        x, y = tuple_location[0], tuple_location[1]
        
        # Validar ubicación (evitar colisiones)
        command_center_xy = helper.validate_random_location(obs, x, y, positions, li, ls)
        
        if command_center_xy:
            # Encontrar el SCV más cercano para la construcción
            distances = helper.get_distances(obs, scvs, command_center_xy)
            if distances is not None and np.size(distances) > 0:
                scv = scvs[np.argmin(distances)]
                distance = min(distances)
                
                logger.info(f"Construyendo centro de mando en posición {command_center_xy}")
                logger.debug(f"SCV asignado en ({scv.x}, {scv.y}), distancia: {distance:.2f}")
                
                # Ejecutar construcción
                try:
                    action = actions.RAW_FUNCTIONS.Build_CommandCenter_pt("now", scv.tag, command_center_xy)
                    return (action, 1, command_center_xy)
                    
                except Exception as e:
                    logger.error(f"Error al construir centro de mando: {str(e)}")
                    return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
            else:
                logger.warning("No se pudieron calcular distancias a SCVs")
                return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        else:
            logger.debug("No se pudo encontrar ubicación válida para el centro de mando")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
