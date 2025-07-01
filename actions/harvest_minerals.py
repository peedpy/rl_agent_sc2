"""
Módulo para la recolección de minerales en StarCraft II.

Este módulo maneja la lógica para asignar SCVs a la recolección de minerales,
optimizando la distribución de trabajadores para maximizar la eficiencia económica.
La estrategia implementada sigue las mejores prácticas de StarCraft II.

Referencias:
- https://liquipedia.net/starcraft2/Resources#Supply
- https://liquipedia.net/starcraft2/SCV

Notas de optimización:
- Tener dos trabajadores por campo mineral generalmente se considera óptimo
- Cuando hay más de dos trabajadores por campo mineral, es mejor agregar bases adicionales
- Con dos trabajadores por campo de minerales, una base con 8 campos cosechará ~925 minerales/min

Autor: Pablo Escobar
Fecha: 2022
"""

import logging
from pysc2.lib import actions, features, units
import random
import numpy as np

# Configuración del logger para este módulo
logger = logging.getLogger(__name__)

#Tener dos trabajadores por campo mineral generalmente se considera óptimo.
#Cuando hay más de dos trabajadores disponibles por campo mineral, generalmente es mejor agregar base(s) adicional(es).
#https://liquipedia.net/starcraft2/Resources#Supply
#Con dos trabajadores por campo de minerales, una base con 8 campos de minerales cosechará alrededor de 925 minerales por minuto.
class HarvestMinerals:
    """
    Clase responsable de la recolección eficiente de minerales.
    
    Esta clase implementa la lógica para asignar SCVs a campos de minerales
    de manera óptima, considerando la distancia y la distribución de trabajadores.
    La estrategia se basa en las mejores prácticas de StarCraft II para maximizar
    la eficiencia económica.
    
    Attributes:
        units (list): Lista de tipos de unidades que representan campos de minerales
        OPTIMAL_WORKERS_PER_PATCH (int): Número óptimo de trabajadores por campo (2)
    """
    
    # Constante de optimización
    OPTIMAL_WORKERS_PER_PATCH = 2
    
    def __init__(self):
        """
        Inicializa la clase HarvestMinerals con los tipos de campos de minerales.
        
        Define todos los tipos de campos de minerales disponibles en diferentes
        mapas de StarCraft II, incluyendo campos normales, ricos y variantes específicas.
        """
        logger.debug("Inicializando módulo HarvestMinerals")
        
        # Definir todos los tipos de campos de minerales disponibles
        self.units = [
            # Campos de minerales estándar
            units.Neutral.BattleStationMineralField,
            units.Neutral.BattleStationMineralField750,
            units.Neutral.LabMineralField,
            units.Neutral.LabMineralField750,
            units.Neutral.MineralField,
            units.Neutral.MineralField750,
            
            # Campos de minerales Purifier
            units.Neutral.PurifierMineralField,
            units.Neutral.PurifierMineralField750,
            units.Neutral.PurifierRichMineralField,
            units.Neutral.PurifierRichMineralField750,
            
            # Campos de minerales ricos
            units.Neutral.RichMineralField,
            units.Neutral.RichMineralField750,
            
            # Campos de gas vespene (incluidos para completitud)
            units.Neutral.VespeneGeyser,
            units.Neutral.ProtossVespeneGeyser,
            units.Neutral.PurifierVespeneGeyser,
            units.Neutral.RichVespeneGeyser,
            units.Neutral.ShakurasVespeneGeyser
        ]
        
        logger.debug(f"Definidos {len(self.units)} tipos de campos de recursos")

    def get_scv_harvest_minerals(self, obs, helper):
        """
        Asigna SCVs disponibles a la recolección de minerales.
        
        Esta función implementa la lógica para asignar SCVs a campos de minerales
        de manera eficiente:
        1. Identifica SCVs disponibles (sin órdenes pendientes)
        2. Localiza campos de minerales en el mapa
        3. Calcula distancias para optimizar asignación
        4. Asigna el SCV al campo más cercano
        
        Args:
            obs: Observación del estado actual del juego
            helper: Objeto Helper con funciones auxiliares
            
        Returns:
            tuple: (acción, flag_éxito, posición) donde:
                - acción: Función RAW de PySC2 a ejecutar
                - flag_éxito: 1 si se puede ejecutar, 0 en caso contrario
                - posición: Coordenadas del SCV asignado (x, y) o (None, None)
                
        Example:
            >>> harvest = HarvestMinerals()
            >>> action, success, pos = harvest.get_scv_harvest_minerals(obs, helper)
            >>> if success:
            ...     print(f"SCV asignado a minerales en posición {pos}")
        """
        logger.debug("Iniciando asignación de SCV a recolección de minerales")
        
        # Obtener SCVs disponibles (sin órdenes pendientes)
        scvs = helper.get_my_units_by_type(obs, units.Terran.SCV)
        free_supply = [scv for scv in scvs if scv.order_length == 0]
        
        if len(free_supply) == 0:
            logger.debug("No hay SCVs disponibles para asignar a minerales")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        
        logger.debug(f"SCVs disponibles para minerales: {len(free_supply)}")
        
        # Identificar campos de minerales en el mapa
        mineral_patches = [unit for unit in obs.observation.raw_units if unit.unit_type in self.units]
        
        if len(mineral_patches) == 0:
            logger.warning("No se encontraron campos de minerales en el mapa")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        
        logger.debug(f"Campos de minerales encontrados: {len(mineral_patches)}")
        
        # Seleccionar SCV aleatorio para asignar
        scv = random.choice(free_supply)
        scv_position = (scv.x, scv.y)
        
        # Calcular distancias a todos los campos de minerales
        distances = helper.get_distances(obs, mineral_patches, scv_position)
        
        if distances is None or np.size(distances) == 0:
            logger.warning("No se pudieron calcular distancias a campos de minerales")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        
        # Seleccionar el campo de minerales más cercano
        closest_patch_index = np.argmin(distances)
        mineral_patch = mineral_patches[closest_patch_index]
        distance = distances[closest_patch_index]
        
        logger.info(f"Asignando SCV en {scv_position} a minerales en ({mineral_patch.x}, {mineral_patch.y}) - Distancia: {distance:.2f}")
        logger.debug(f"SCV tag: {scv.tag}, Mineral patch tag: {mineral_patch.tag}")
        
        # Ejecutar acción de recolección
        try:
            action = actions.RAW_FUNCTIONS.Harvest_Gather_unit("now", scv.tag, mineral_patch.tag)
            return (action, 1, scv_position)
            
        except Exception as e:
            logger.error(f"Error al asignar SCV a minerales: {str(e)}")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
