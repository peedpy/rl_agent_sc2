"""
Módulo para la configuración y gestión de acciones en StarCraft II.

Este módulo centraliza la configuración de todas las acciones disponibles
para el agente de aprendizaje por refuerzo, proporcionando una interfaz
unificada para acceder a las diferentes acciones del juego.

Referencias:
- https://liquipedia.net/starcraft2/Terran
- https://liquipedia.net/starcraft2/Units

Autor: Pablo Escobar
Fecha: 2022
"""

import logging
from pysc2.lib import actions, features, units
import random
import numpy as np

# Importar todas las acciones disponibles
from .build_scv import BuildSCV
from .harvest_minerals import HarvestMinerals
from .build_supply_depot import SupplyDepot
from .build_barracks import BuildBarracks
from .train_marine import TrainMarine
from .build_tech_lab import BuildTechLab
from .train_marauder import TrainMarauder
from .build_bunker import BuildBunker
from .build_command_center import BuildCommandCenter
from .harvest_gas_vespeno import HarvestGasVespeno
from .attack_army import AttackArmy
from .explore_csv import ExploreCSV

# Configuración del logger para este módulo
logger = logging.getLogger(__name__)

class SetActions:
    """
    Clase responsable de la configuración y gestión de todas las acciones disponibles.
    
    Esta clase centraliza todas las acciones del agente, proporcionando:
    - Inicialización de todas las clases de acciones
    - Configuración de parámetros de acciones
    - Interfaz unificada para acceder a las acciones
    - Gestión de prioridades y condiciones de ejecución
    
    Attributes:
        actions_list (list): Lista de todas las acciones disponibles
        action_instances (dict): Instancias de las clases de acciones
        action_weights (dict): Pesos/prioridades de cada acción
    """
    
    def __init__(self):
        """
        Inicializa la clase SetActions.
        
        Crea instancias de todas las clases de acciones y configura
        los parámetros iniciales para cada una.
        """
        logger.debug("Inicializando módulo SetActions")
        
        # Lista de todas las acciones disponibles
        self.actions_list = [
            "do_nothing",
            "build_scv",
            "harvest_minerals", 
            "build_supply_depot",
            "build_barracks",
            "train_marine",
            "build_tech_lab",
            "train_marauder",
            "build_bunker",
            "build_command_center",
            "harvest_gas_vespeno",
            "attack_with_marine",
            "defense_with_marine", 
            "attack_with_marauder",
            "explore_csv"
        ]
        
        # Crear instancias de todas las clases de acciones
        self.action_instances = {
            "do_nothing": None,  # do_nothing no requiere instancia de clase
            "build_scv": BuildSCV(),
            "harvest_minerals": HarvestMinerals(),
            "build_supply_depot": SupplyDepot(),
            "build_barracks": BuildBarracks(),
            "train_marine": TrainMarine(),
            "build_tech_lab": BuildTechLab(),
            "train_marauder": TrainMarauder(),
            "build_bunker": BuildBunker(),
            "build_command_center": BuildCommandCenter(),
            "harvest_gas_vespeno": HarvestGasVespeno(),
            "attack_with_marine": AttackArmy(),
            "defense_with_marine": AttackArmy(),
            "attack_with_marauder": AttackArmy(),
            "explore_csv": ExploreCSV()
        }
        
        # Configurar pesos/prioridades de acciones (pueden ser ajustados)
        self.action_weights = {
            "do_nothing": 0.1,  # Baja prioridad para do_nothing
            "build_scv": 1.0,
            "harvest_minerals": 1.0,
            "build_supply_depot": 1.0,
            "build_barracks": 1.0,
            "train_marine": 1.0,
            "build_tech_lab": 1.0,
            "train_marauder": 1.0,
            "build_bunker": 1.0,
            "build_command_center": 1.0,
            "harvest_gas_vespeno": 1.0,
            "attack_with_marine": 1.0,
            "defense_with_marine": 1.0,
            "attack_with_marauder": 1.0,
            "explore_csv": 1.0
        }
        
        logger.info(f"Configuradas {len(self.actions_list)} acciones disponibles")
        logger.debug(f"Acciones: {', '.join(self.actions_list)}")

    def get_action(self, action_name, obs, helper):
        """
        Ejecuta una acción específica por nombre.
        
        Esta función proporciona una interfaz unificada para ejecutar
        cualquier acción disponible en el sistema.
        
        Args:
            action_name (str): Nombre de la acción a ejecutar
            obs: Observación del estado actual del juego
            helper: Objeto Helper con funciones auxiliares
            
        Returns:
            tuple: (acción, flag_éxito, posición) donde:
                - acción: Función RAW de PySC2 a ejecutar
                - flag_éxito: 1 si se puede ejecutar, 0 en caso contrario
                - posición: Coordenadas de la acción (x, y) o (None, None)
                
        Raises:
            KeyError: Si la acción especificada no existe
            
        Example:
            >>> set_actions = SetActions()
            >>> action, success, pos = set_actions.get_action("build_scv", obs, helper)
            >>> if success:
            ...     print(f"Acción ejecutada en posición {pos}")
        """
        logger.debug(f"Ejecutando acción: {action_name}")
        
        if action_name not in self.action_instances:
            logger.error(f"Acción no encontrada: {action_name}")
            raise KeyError(f"Acción '{action_name}' no está disponible")
        
        try:
            # Obtener la instancia de la acción
            action_instance = self.action_instances[action_name]
            
            # Manejar acciones de ataque específicas
            if action_name == "do_nothing":
                result = (actions.RAW_FUNCTIONS.no_op(), 1, (None, None))
            elif action_name == "attack_with_marine":
                result = action_instance.send_to_attack_opposite(obs, helper, 'marine_attack')
            elif action_name == "defense_with_marine":
                result = action_instance.send_to_attack_opposite(obs, helper, 'marine_defense')
            elif action_name == "attack_with_marauder":
                result = action_instance.send_to_attack_opposite(obs, helper, 'marauder')
            else:
                # Mapear nombres de acciones a métodos correspondientes para acciones normales
                if action_name == "build_scv":
                    result = action_instance.train_scv(obs, helper)
                elif action_name == "harvest_minerals":
                    result = action_instance.get_scv_harvest_minerals(obs, helper)
                elif action_name == "build_supply_depot":
                    result = action_instance.build_supply_depot(obs, helper)
                elif action_name == "build_barracks":
                    result = action_instance.build_barracks(obs, helper)
                elif action_name == "train_marine":
                    result = action_instance.train_marine(obs, helper)
                elif action_name == "build_tech_lab":
                    result = action_instance.build_tech_lab(obs, helper)
                elif action_name == "train_marauder":
                    result = action_instance.train_marauder(obs, helper)
                elif action_name == "build_bunker":
                    result = action_instance.build_bunker(obs, helper)
                elif action_name == "build_command_center":
                    result = action_instance.build_command_center(obs, helper)
                elif action_name == "harvest_gas_vespeno":
                    result = action_instance.harvest_gas_vespeno(obs, helper)
                elif action_name == "explore_csv":
                    result = action_instance.explore_csv(obs, helper)
                else:
                    logger.error(f"Método no encontrado para acción: {action_name}")
                    return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
            
            logger.debug(f"Acción {action_name} ejecutada con resultado: {result[1]}")
            return result
            
        except Exception as e:
            logger.error(f"Error al ejecutar acción {action_name}: {str(e)}")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))

    def get_available_actions(self):
        """
        Obtiene la lista de todas las acciones disponibles.
        
        Returns:
            list: Lista de nombres de acciones disponibles
            
        Example:
            >>> set_actions = SetActions()
            >>> actions = set_actions.get_available_actions()
            >>> print(f"Acciones disponibles: {actions}")
        """
        return self.actions_list.copy()

    def get_action_weight(self, action_name):
        """
        Obtiene el peso/prioridad de una acción específica.
        
        Args:
            action_name (str): Nombre de la acción
            
        Returns:
            float: Peso de la acción (1.0 por defecto)
            
        Example:
            >>> set_actions = SetActions()
            >>> weight = set_actions.get_action_weight("build_scv")
            >>> print(f"Peso de build_scv: {weight}")
        """
        return self.action_weights.get(action_name, 1.0)

    def set_action_weight(self, action_name, weight):
        """
        Establece el peso/prioridad de una acción específica.
        
        Args:
            action_name (str): Nombre de la acción
            weight (float): Nuevo peso de la acción
            
        Example:
            >>> set_actions = SetActions()
            >>> set_actions.set_action_weight("attack_army", 2.0)
        """
        if action_name in self.action_weights:
            self.action_weights[action_name] = weight
            logger.debug(f"Peso de acción {action_name} actualizado a {weight}")
        else:
            logger.warning(f"No se puede establecer peso para acción inexistente: {action_name}")
