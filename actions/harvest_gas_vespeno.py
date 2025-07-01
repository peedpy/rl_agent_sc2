"""
Módulo para la recolección de gas vespene en StarCraft II.

El gas vespene es un recurso esencial para entrenar unidades avanzadas y construir
estructuras tecnológicas. Este módulo maneja la lógica para construir refinerías
y asignar SCVs a la recolección de gas vespene.

Referencias:
- https://liquipedia.net/starcraft2/Refinery_(Legacy_of_the_Void)
- https://liquipedia.net/starcraft2/Vespene_Gas

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

class HarvestGasVespeno:
    """
    Clase responsable de la recolección de gas vespene Terran.
    
    El gas vespene es un recurso fundamental para:
    - Entrenar unidades avanzadas (Marauders, etc.)
    - Construir estructuras tecnológicas
    - Investigar mejoras y tecnologías
    
    Esta clase maneja la lógica para construir refinerías en geysers
    de vespene y asignar SCVs para la recolección.
    
    Attributes:
        units (list): Lista de tipos de geysers de vespene disponibles
        MAX_REFINERIES (int): Límite máximo de refinerías permitidas (16)
    """
    
    # Constantes de configuración
    MAX_REFINERIES = 16
    
    def __init__(self):
        """
        Inicializa la clase HarvestGasVespeno.
        
        Define los tipos de geysers de vespene que pueden ser utilizados
        para construir refinerías.
        """
        logger.debug("Inicializando módulo HarvestGasVespeno")
        self.units = [
            units.Neutral.VespeneGeyser,
            units.Neutral.ProtossVespeneGeyser,
            units.Neutral.PurifierVespeneGeyser,
            units.Neutral.RichVespeneGeyser,
            units.Neutral.ShakurasVespeneGeyser
        ]

    def harvest_gas_vespeno(self, obs, helper):
        """
        Construye una refinería en un geyser de vespene y asigna SCV para recolección.
        
        Esta función implementa la lógica para recolectar gas vespene de manera eficiente:
        1. Verifica que haya SCVs disponibles (sin órdenes pendientes)
        2. Comprueba que no se haya alcanzado el límite de refinerías
        3. Identifica geysers de vespene disponibles en el mapa
        4. Selecciona el SCV más cercano al geyser
        5. Construye la refinería en el geyser seleccionado
        
        Args:
            obs: Observación del estado actual del juego
            helper: Objeto Helper con funciones auxiliares
            
        Returns:
            tuple: (acción, flag_éxito, posición) donde:
                - acción: Función RAW de PySC2 a ejecutar
                - flag_éxito: 1 si se puede ejecutar, 0 en caso contrario
                - posición: Coordenadas del SCV (x, y) o (None, None)
                
        Example:
            >>> harvest_gas = HarvestGasVespeno()
            >>> action, success, pos = harvest_gas.harvest_gas_vespeno(obs, helper)
            >>> if success:
            ...     print(f"Construyendo refinería con SCV en posición {pos}")
        """
        logger.debug("Iniciando proceso de recolección de gas vespene")
        
        # Obtener SCVs disponibles (sin órdenes pendientes)
        scvs = helper.get_my_units_by_type(obs, units.Terran.SCV)
        idle_scvs = [scv for scv in scvs if scv.order_length == 0]
        
        if not idle_scvs:
            logger.debug("No hay SCVs disponibles para construir refinería")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        
        # Verificar límite de refinerías
        completed_refineries = helper.get_my_completed_units_by_type(obs, units.Terran.Refinery)
        current_refineries_count = len(completed_refineries)
        if current_refineries_count >= self.MAX_REFINERIES:
            logger.info(f"Límite de refinerías alcanzado ({self.MAX_REFINERIES}). No se puede construir más.")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        
        logger.debug(f"SCVs disponibles: {len(idle_scvs)}, Refinerías actuales: {current_refineries_count} de {self.MAX_REFINERIES}")
        
        # Identificar geysers de vespene disponibles
        gas_patches = [unit for unit in obs.observation.raw_units if unit.unit_type in self.units]
        
        if not gas_patches:
            logger.debug("No se encontraron geysers de vespene en el mapa")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        
        logger.debug(f"Geysers de vespene disponibles: {len(gas_patches)}")
        
        # Seleccionar SCV aleatorio
        scv = random.choice(idle_scvs)
        scv_position = (scv.x, scv.y)
        
        # Calcular distancias del SCV a los geysers
        distances = helper.get_distances(obs, gas_patches, scv_position)
        
        if distances is None or np.size(distances) == 0:
            logger.warning("No se pudieron calcular distancias a los geysers")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        
        # Seleccionar el geyser más cercano
        closest_gas_index = np.argmin(distances)
        gas_patch = gas_patches[closest_gas_index]
        distance_to_gas = distances[closest_gas_index]
        
        logger.info(f"Construyendo refinería en geyser en posición ({gas_patch.x}, {gas_patch.y})")
        logger.debug(f"SCV en {scv_position}, distancia al geyser: {distance_to_gas:.2f}")
        
        # Ejecutar construcción de refinería
        try:
            action = actions.RAW_FUNCTIONS.Build_Refinery_pt("now", scv.tag, gas_patch.tag)
            return (action, 1, scv_position)
            
        except Exception as e:
            logger.error(f"Error al construir refinería: {str(e)}")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
