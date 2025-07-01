"""
Módulo para la construcción y entrenamiento de SCVs (trabajadores) en StarCraft II.

Los SCVs son las unidades básicas de trabajo de la raza Terran, responsables de
construir edificios, recolectar recursos y reparar estructuras. Este módulo maneja
la lógica para entrenar nuevos SCVs desde los centros de mando.

Referencias:
- https://liquipedia.net/starcraft2/SCV
- https://liquipedia.net/starcraft2/Resources#Supply

Autor: Pablo Escobar
Fecha: 2022
"""

import logging
from pysc2.lib import actions, features, units
import random

# Configuración del logger para este módulo
logger = logging.getLogger(__name__)

class BuildSCV:
    """
    Clase responsable del entrenamiento de SCVs (trabajadores) Terran.
    
    Los SCVs son fundamentales para la economía y construcción de la raza Terran.
    Esta clase maneja la lógica para determinar cuándo y dónde entrenar nuevos SCVs,
    considerando factores como la disponibilidad de recursos y límites de población.
    
    Attributes:
        MAX_SCVS (int): Límite máximo de SCVs permitidos (100)
        SCV_COST (int): Costo en minerales para entrenar un SCV (50)
    """
    
    # Constantes de configuración
    MAX_SCVS = 100
    SCV_COST = 50
    
    def __init__(self):
        """
        Inicializa la clase BuildSCV.
        
        No requiere parámetros de inicialización ya que toda la lógica
        se ejecuta en el método train_scv.
        """
        logger.debug("Inicializando módulo BuildSCV")
        pass

    def train_scv(self, obs, helper):
        """
        Entrena un nuevo SCV desde el centro de mando disponible.
        
        Esta función implementa la lógica para entrenar SCVs de manera eficiente:
        1. Verifica que exista un centro de mando disponible
        2. Comprueba que no se haya alcanzado el límite de SCVs
        3. Verifica que haya suficientes recursos
        4. Ejecuta la acción de entrenamiento
        
        Args:
            obs: Observación del estado actual del juego
            helper: Objeto Helper con funciones auxiliares
            
        Returns:
            tuple: (acción, flag_éxito, posición) donde:
                - acción: Función RAW de PySC2 a ejecutar
                - flag_éxito: 1 si se puede ejecutar, 0 en caso contrario
                - posición: Coordenadas del centro de mando (x, y) o (None, None)
                
        Example:
            >>> build_scv = BuildSCV()
            >>> action, success, pos = build_scv.train_scv(obs, helper)
            >>> if success:
            ...     print(f"Entrenando SCV en posición {pos}")
        """
        logger.debug("Iniciando proceso de entrenamiento de SCV")
        
        # Obtener el centro de mando disponible
        command_center = helper.get_command_center_location(obs, units.Terran.CommandCenter)
        if command_center is None:
            logger.warning("No se encontró centro de mando disponible para entrenar SCV")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        
        # Obtener SCVs existentes
        scvs = helper.get_my_units_by_type(obs, units.Terran.SCV)
        current_scv_count = len(scvs)
        
        logger.debug(f"SCVs actuales: {current_scv_count}/{self.MAX_SCVS}")
        
        # Verificar límite de SCVs
        if current_scv_count >= self.MAX_SCVS:
            logger.info(f"Límite de SCVs alcanzado ({self.MAX_SCVS}). No se puede entrenar más.")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        
        # Verificar recursos disponibles
        minerals = obs.observation.player.minerals
        if minerals < self.SCV_COST:
            logger.debug(f"Minerales insuficientes: {minerals}/{self.SCV_COST}")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        
        # Verificar que el centro de mando no esté sobrecargado
        if command_center.order_length > 5:
            logger.debug(f"Centro de mando sobrecargado: {command_center.order_length} órdenes pendientes")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        
        # Ejecutar entrenamiento
        try:
            action = actions.RAW_FUNCTIONS.Train_SCV_quick("now", command_center.tag)
            position = (command_center.x, command_center.y)
            
            logger.info(f"Entrenando SCV en centro de mando en posición {position}")
            logger.debug(f"Centro de mando tag: {command_center.tag}, órdenes pendientes: {command_center.order_length}")
            
            return (action, 1, position)
            
        except Exception as e:
            logger.error(f"Error al entrenar SCV: {str(e)}")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
