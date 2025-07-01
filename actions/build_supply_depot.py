"""
Módulo para la construcción de depósitos de suministro en StarCraft II.

Los depósitos de suministro son edificios esenciales que aumentan la capacidad
de población del jugador. Este módulo maneja la lógica para construir depósitos
de manera eficiente, considerando recursos, trabajadores y posicionamiento.

Referencias:
- https://liquipedia.net/starcraft2/Supply_Depot
- https://liquipedia.net/starcraft2/Resources#Supply

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

class SupplyDepot:
    """
    Clase responsable de la construcción de depósitos de suministro Terran.
    
    Los depósitos de suministro son edificios fundamentales que aumentan la
    capacidad de población del jugador, permitiendo entrenar más unidades.
    Esta clase maneja la lógica para construir depósitos de manera eficiente,
    considerando recursos, trabajadores disponibles y posicionamiento estratégico.
    
    Attributes:
        SUPPLY_DEPOT_COST (int): Costo en minerales para construir un depósito (100)
        MAX_SUPPLY_DEPOTS (int): Límite máximo de depósitos permitidos (50)
        BUILD_RANGE (int): Rango de construcción alrededor del centro de mando (±5)
        used_positions (list): Lista de posiciones ya utilizadas
    """
    
    # Constantes de configuración
    SUPPLY_DEPOT_COST = 100
    MAX_SUPPLY_DEPOTS = 50
    BUILD_RANGE = 5
    
    def __init__(self):
        """
        Inicializa la clase SupplyDepot.
        
        Inicializa la lista de posiciones utilizadas para evitar
        construir depósitos en ubicaciones ya ocupadas.
        """
        logger.debug("Inicializando módulo SupplyDepot")
        self.used_positions = []

    def build_supply_depot(self, obs, helper):
        """
        Construye un nuevo depósito de suministro en una ubicación estratégica.
        
        Esta función implementa la lógica para construir depósitos de suministro:
        1. Verifica que haya suficientes recursos (minerales)
        2. Comprueba que haya SCVs disponibles
        3. Controla que no se haya alcanzado el límite de depósitos
        4. Selecciona una ubicación estratégica cerca del centro de mando
        5. Asigna el SCV más cercano para la construcción
        
        Args:
            obs: Observación del estado actual del juego
            helper: Objeto Helper con funciones auxiliares
            
        Returns:
            tuple: (acción, flag_éxito, posición) donde:
                - acción: Función RAW de PySC2 a ejecutar
                - flag_éxito: 1 si se puede ejecutar, 0 en caso contrario
                - posición: Coordenadas del depósito (x, y) o (None, None)
                
        Example:
            >>> supply_depot = SupplyDepot()
            >>> action, success, pos = supply_depot.build_supply_depot(obs, helper)
            >>> if success:
            ...     print(f"Construyendo depósito de suministro en posición {pos}")
        """
        logger.debug("Iniciando proceso de construcción de depósito de suministro")
        
        # Obtener ubicación del centro de mando
        unit_type = units.Terran.CommandCenter
        command_center_location = helper.get_command_center_location(obs, unit_type)
        if command_center_location is None:
            logger.warning("No se encontró centro de mando para construir depósito de suministro")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        
        # Verificar recursos disponibles
        minerals = obs.observation.player.minerals
        if minerals < self.SUPPLY_DEPOT_COST:
            logger.debug(f"Minerales insuficientes para depósito: {minerals}/{self.SUPPLY_DEPOT_COST}")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        
        # Verificar SCVs disponibles
        scvs = helper.get_my_units_by_type(obs, units.Terran.SCV)
        if len(scvs) == 0:
            logger.debug("No hay SCVs disponibles para construir depósito de suministro")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        
        # Verificar límite de depósitos
        supply_depots = helper.get_my_units_by_type(obs, units.Terran.SupplyDepot)
        current_supply_depots_count = len(supply_depots)
        if current_supply_depots_count >= self.MAX_SUPPLY_DEPOTS:
            logger.info(f"Límite de depósitos de suministro alcanzado ({self.MAX_SUPPLY_DEPOTS}). No se puede construir más.")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        
        logger.debug(f"Depósitos actuales: {current_supply_depots_count}/{self.MAX_SUPPLY_DEPOTS}")
        logger.debug(f"SCVs disponibles: {len(scvs)}")
        
        # Calcular ubicación para el depósito
        li = -self.BUILD_RANGE
        ls = self.BUILD_RANGE
        positions = (command_center_location.x, command_center_location.y)
        
        # Generar ubicación aleatoria cerca del centro de mando
        tuple_location = helper.random_location(positions, li, ls)
        x, y = tuple_location[0], tuple_location[1]
        
        # Validar ubicación (evitar colisiones)
        supply_depot_xy = helper.validate_random_location(obs, x, y, positions, li, ls)
        
        if supply_depot_xy:
            # Encontrar el SCV más cercano para la construcción
            distances = helper.get_distances(obs, scvs, supply_depot_xy)
            if distances is not None and np.size(distances) > 0:
                scv = scvs[np.argmin(distances)]
                distance = min(distances)
                
                logger.info(f"Construyendo depósito de suministro en posición {supply_depot_xy}")
                logger.debug(f"SCV asignado en ({scv.x}, {scv.y}), distancia: {distance:.2f}")
                
                # Ejecutar construcción
                try:
                    action = actions.RAW_FUNCTIONS.Build_SupplyDepot_pt("now", scv.tag, supply_depot_xy)
                    return (action, 1, supply_depot_xy)
                    
                except Exception as e:
                    logger.error(f"Error al construir depósito de suministro: {str(e)}")
                    return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
            else:
                logger.warning("No se pudieron calcular distancias a SCVs")
                return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        else:
            logger.debug("No se pudo encontrar ubicación válida para el depósito de suministro")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
