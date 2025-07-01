"""
Módulo para la exploración del mapa en StarCraft II.

La exploración es fundamental para descubrir recursos, ubicaciones del enemigo
y puntos estratégicos en el mapa. Este módulo maneja la lógica para enviar
SCVs a explorar diferentes áreas del mapa de manera eficiente.

Referencias:
- https://liquipedia.net/starcraft2/SCV
- https://liquipedia.net/starcraft2/Map_Exploration

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

class ExploreCSV:
    """
    Clase responsable de la exploración del mapa usando SCVs.
    
    La exploración es una actividad estratégica que permite:
    - Descubrir recursos adicionales
    - Localizar unidades y estructuras enemigas
    - Identificar puntos estratégicos del mapa
    - Mejorar la información disponible para el agente
    
    Esta clase maneja la lógica para enviar SCVs a explorar
    diferentes áreas del mapa evitando colisiones.
    
    Attributes:
        positions (list): Lista de posiciones ya exploradas
        MAP_WIDTH (int): Ancho del mapa (64)
        MAP_HEIGHT (int): Alto del mapa (64)
    """
    
    # Constantes de configuración
    MAP_WIDTH = 64
    MAP_HEIGHT = 64
    
    def __init__(self):
        """
        Inicializa la clase ExploreCSV.
        
        Inicializa la lista de posiciones exploradas para evitar
        explorar las mismas áreas repetidamente.
        """
        logger.debug("Inicializando módulo ExploreCSV")
        self.positions = []

    def explore_csv(self, obs, helper):
        """
        Envía un SCV a explorar una ubicación del mapa.
        
        Esta función implementa la lógica para explorar el mapa de manera eficiente:
        1. Verifica que haya SCVs disponibles
        2. Selecciona una ubicación aleatoria o de la lista de posiciones
        3. Valida que la ubicación no colisione con estructuras propias
        4. Genera una ubicación objetivo con pequeña variación
        5. Envía el SCV a la ubicación seleccionada
        
        Args:
            obs: Observación del estado actual del juego
            helper: Objeto Helper con funciones auxiliares
            
        Returns:
            tuple: (acción, flag_éxito, posición) donde:
                - acción: Función RAW de PySC2 a ejecutar
                - flag_éxito: 1 si se puede ejecutar, 0 en caso contrario
                - posición: Coordenadas objetivo (x, y) o (None, None)
                
        Example:
            >>> explore = ExploreCSV()
            >>> action, success, pos = explore.explore_csv(obs, helper)
            >>> if success:
            ...     print(f"Enviando SCV a explorar en posición {pos}")
        """
        logger.debug("Iniciando proceso de exploración del mapa")
        
        # Obtener SCVs disponibles
        scvs = helper.get_my_units_by_type(obs, units.Terran.SCV)
        free_supply = [scv for scv in scvs if scv.order_length >= 0]
        
        if not free_supply:
            logger.debug("No hay SCVs disponibles para exploración")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        
        logger.debug(f"SCVs disponibles para exploración: {len(free_supply)}")
        
        # Seleccionar SCV aleatorio
        scv = random.choice(free_supply)
        scv_position = (scv.x, scv.y)
        
        # Seleccionar estrategia de exploración
        selection = random.randint(1, 2)
        
        if selection == 1 or len(self.positions) == 0:
            # Explorar nueva ubicación aleatoria
            x = random.randint(0, self.MAP_WIDTH - 1)
            y = random.randint(0, self.MAP_HEIGHT - 1)
            logger.debug("Seleccionada nueva ubicación aleatoria para exploración")
        else:
            # Usar ubicación de la lista de posiciones exploradas
            x, y = self.positions[random.randint(0, len(self.positions) - 1)]
            logger.debug("Seleccionada ubicación de lista de posiciones exploradas")
        
        logger.debug(f"Posición inicial seleccionada: ({x}, {y})")
        
        # Verificar colisiones con estructuras propias
        collision_attempts = 0
        max_attempts = 10
        
        while self.collides_with_my_structures(x, y, obs) and collision_attempts < max_attempts:
            x = random.randint(0, self.MAP_WIDTH - 1)
            y = random.randint(0, self.MAP_HEIGHT - 1)
            collision_attempts += 1
            logger.debug(f"Colisión detectada, intento {collision_attempts}: nueva posición ({x}, {y})")
        
        if collision_attempts >= max_attempts:
            logger.warning("No se pudo encontrar posición válida después de múltiples intentos")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        
        # Agregar posición a la lista si es nueva
        if selection == 1 or (x, y) not in self.positions:
            self.positions.append((x, y))
            logger.debug(f"Posición agregada a lista de exploradas: ({x}, {y})")
        
        # Generar ubicación objetivo con pequeña variación
        target_location = (x + random.randint(1, 3), y + random.randint(1, 3))
        
        # Guardar última posición de exploración en el helper
        helper.set_last_explore_position(target_location)
        
        logger.info(f"Enviando SCV a explorar en posición {target_location}")
        logger.debug(f"SCV en {scv_position}, objetivo: {target_location}")
        logger.debug(f"Posiciones exploradas totales: {len(self.positions)}")
        
        # Ejecutar movimiento de exploración
        try:
            action = actions.RAW_FUNCTIONS.Move_pt("now", scv.tag, target_location)
            return (action, 1, target_location)
            
        except Exception as e:
            logger.error(f"Error al enviar SCV a explorar: {str(e)}")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))

    def collides_with_my_structures(self, x, y, obs):
        """
        Verifica si una posición colisiona con estructuras propias.
        
        Esta función verifica si la posición (x, y) está ocupada por
        alguna estructura u obstáculo propio en el mapa.
        
        Args:
            x (int): Coordenada X de la posición a verificar
            y (int): Coordenada Y de la posición a verificar
            obs: Observación del estado actual del juego
            
        Returns:
            bool: True si hay colisión, False en caso contrario
        """
        # Obtener todas las unidades visibles en el mapa
        all_units = obs.observation.raw_units
        
        # Verificar si la posición (x, y) colisiona con alguna estructura u obstáculo
        for unit in all_units:
            if unit.alliance == features.PlayerRelative.SELF:  # Verificar solo las unidades propias
                if abs(unit.x - x) <= unit.radius and abs(unit.y - y) <= unit.radius:
                    return True  # La posición colisiona con una estructura
        
        return False  # La posición no colisiona con ninguna estructura
