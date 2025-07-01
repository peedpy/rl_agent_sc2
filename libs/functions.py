"""
Módulo de funciones auxiliares para StarCraft II con PySC2.

Este módulo proporciona una clase Helper que centraliza todas las funciones auxiliares
necesarias para el agente de aprendizaje por refuerzo, incluyendo:
- Gestión de unidades y estructuras
- Cálculo de distancias y posicionamiento
- Análisis de cuadrantes y zonas calientes
- Validación de ubicaciones para construcción
- Gestión de recursos y estados del juego

Referencias:
- https://github.com/deepmind/pysc2
- https://liquipedia.net/starcraft2/Terran

Autor: Pablo Escobar
Fecha: 2022
"""

from pysc2.lib import features, units
import numpy as np # Mathematical functions
import random
import os
from libs.logging_config import get_logger

# Configuración del logger para este módulo
logger = get_logger('libs.functions')

class Helper:
    """
    Clase auxiliar que proporciona funciones de soporte para el agente de StarCraft II.
    
    Esta clase centraliza todas las operaciones auxiliares necesarias para:
    - Obtener y filtrar unidades del juego
    - Calcular distancias y posiciones
    - Analizar la distribución del enemigo en el mapa
    - Validar ubicaciones para construcción
    - Gestionar recursos y estados del juego
    
    Attributes:
        total_send_units (int): Contador total de unidades enviadas
        base_top_left (bool): Indica si la base está en la esquina superior izquierda
        quadrant (int): Cuadrante actual del mapa (0-3)
        list_quadrants (list): Lista de cuadrantes utilizados
        last_position (tuple): Última posición explorada (x, y)
        used_positions (list): Lista de posiciones ya utilizadas
    """
    
    def __init__(self):
        """
        Inicializa la clase Helper con valores por defecto.
        
        Configura todas las variables de estado necesarias para el funcionamiento
        del agente, incluyendo contadores, posiciones y configuraciones de mapa.
        """
        logger.debug("Inicializando clase Helper")
        
        # Contador de unidades enviadas
        self.total_send_units = 0
        
        # Configuración de posición de la base
        self.base_top_left = None
        self.quadrant = 0
        self.list_quadrants = [0, 0, 0, 0]
        
        # Última posición explorada
        self.last_position = (0, 0)
        
        # Lista de posiciones utilizadas
        self.used_positions = []
        
        logger.info("Clase Helper inicializada correctamente")

    def get_enemy_units_by_type(self, obs, unit_type):
        """
        Obtiene todas las unidades enemigas de un tipo específico.
        
        Args:
            obs: Observación del estado actual del juego
            unit_type: Tipo de unidad a buscar
            
        Returns:
            list: Lista de unidades enemigas del tipo especificado
            
        Example:
            >>> enemy_marines = helper.get_enemy_units_by_type(obs, units.Terran.Marine)
            >>> print(f"Enemigos marines detectados: {len(enemy_marines)}")
        """
        logger.debug(f"Buscando unidades enemigas de tipo: {unit_type}")
        
        enemy_units = [
            unit for unit in obs.observation.raw_units
            if unit.unit_type == unit_type 
            and unit.alliance == features.PlayerRelative.ENEMY
        ]
        
        logger.debug(f"Encontradas {len(enemy_units)} unidades enemigas de tipo {unit_type}")
        return enemy_units

    def get_quadrant(self, x, y):
        """
        Determina el cuadrante del mapa basado en las coordenadas (x, y).
        
        El mapa se divide en 4 cuadrantes:
        - Cuadrante 0: Superior izquierdo (x < 32, y < 32)
        - Cuadrante 1: Superior derecho (x >= 32, y < 32)
        - Cuadrante 2: Inferior izquierdo (x < 32, y >= 32)
        - Cuadrante 3: Inferior derecho (x >= 32, y >= 32)
        
        Args:
            x (int): Coordenada X
            y (int): Coordenada Y
            
        Returns:
            list: Lista de 4 elementos con 1 en el cuadrante correspondiente
            
        Example:
            >>> quadrant = helper.get_quadrant(40, 20)
            >>> print(f"Cuadrante: {quadrant}")  # [0, 1, 0, 0]
        """
        if x < 32:
            return [1, 0, 0, 0] if y < 32 else [0, 0, 1, 0]
        else:
            return [0, 1, 0, 0] if y < 32 else [0, 0, 0, 1]

    def reset_list_quadrants(self):
        """
        Reinicia la lista de cuadrantes a valores por defecto.
        
        Utilizado para limpiar el estado de cuadrantes antes de un nuevo análisis.
        """
        logger.debug("Reiniciando lista de cuadrantes")
        self.list_quadrants = [0, 0, 0, 0]

    def get_calculate_hot_zone(self, obs):
        """
        Calcula las zonas calientes del mapa basándose en la distribución del enemigo.
        
        Esta función analiza la distribución de unidades enemigas en el mapa y
        determina qué cuadrantes tienen mayor concentración de ejército y estructuras.
        Es útil para tomar decisiones tácticas sobre dónde atacar o defender.
        
        Args:
            obs: Observación del estado actual del juego
            
        Returns:
            tuple: (army_positions, structure_positions, max_quadrants) donde:
                - army_positions: Posiciones de unidades de ejército enemigas
                - structure_positions: Posiciones de estructuras enemigas
                - max_quadrants: Índices de cuadrantes con mayor concentración
                
        Example:
            >>> army_pos, struct_pos, max_quads = helper.get_calculate_hot_zone(obs)
            >>> print(f"Cuadrante con más ejército: {max_quads[0]}")
        """
        logger.debug("Calculando zonas calientes del mapa")
        
        # Reiniciar la lista de cuadrantes
        self.reset_list_quadrants()

        # Conjunto de tipos de unidades de interés
        desired_types = {
            units.Terran.SCV, 
            units.Terran.Marine,
            units.Terran.Marauder,
            units.Terran.Reaper,
            units.Terran.Raven, 
            units.Terran.Medivac,
            units.Terran.SupplyDepot, 
            units.Terran.Barracks, 
            units.Terran.CommandCenter, 
            units.Terran.Refinery,
            units.Terran.Starport,
            units.Terran.Armory,
            units.Terran.OrbitalCommand
        }

        # Diccionario para mapear unit_type a un nombre legible
        UNIT_NAMES = {
            units.Terran.SCV: "SCV",
            units.Terran.Marine: "Marine",
            units.Terran.Marauder: "Marauder",
            units.Terran.Reaper: "Reaper",
            units.Terran.Raven: "Raven",
            units.Terran.Medivac: "Medivac",
            units.Terran.SupplyDepot: "Supply Depot",
            units.Terran.Barracks: "Barracks",
            units.Terran.CommandCenter: "Command Center",
            units.Terran.Refinery: "Refinery",
            units.Terran.Starport: "Starport",
            units.Terran.Armory: "Armory",
            units.Terran.OrbitalCommand: "Orbital Command"
        }

        # Conjunto de tipos considerados estructuras
        structure_types = {
            units.Terran.SupplyDepot,
            units.Terran.Barracks,
            units.Terran.CommandCenter,
            units.Terran.Refinery,
            units.Terran.Starport,
            units.Terran.Armory,
            units.Terran.OrbitalCommand
        }

        # Filtrar todas las unidades enemigas de interés
        enemy_units = [
            unit for unit in obs.observation.raw_units 
            if unit.alliance == features.PlayerRelative.ENEMY and unit.unit_type in desired_types
        ]
        
        logger.info(f"Total de unidades enemigas de interés: {len(enemy_units)}")

        # Inicializar acumuladores de cuadrantes para cada grupo
        army_quadrants = np.array([0, 0, 0, 0])
        structure_quadrants = np.array([0, 0, 0, 0])
        
        # Listas para almacenar las posiciones con el nombre de la unidad
        army_positions = []      # Para unidades de ejército
        structure_positions = [] # Para estructuras

        # Procesar cada unidad enemiga
        for enemy in enemy_units:
            x, y = enemy.x, enemy.y
            enemy_name = UNIT_NAMES.get(enemy.unit_type, "Unknown")
            logger.debug(f"Enemigo: {enemy_name:<20} --> (x,y): ({x:3d}, {y:3d})")
            
            # Calcular el cuadrante para esta unidad
            quadrant = self.get_quadrant(x, y)
            
            # Clasificar la unidad en ejército o estructura y acumular el cuadrante
            if enemy.unit_type in structure_types:
                structure_quadrants = np.add(structure_quadrants, quadrant)
                structure_positions.append((enemy_name, (x, y)))
            else:
                army_quadrants = np.add(army_quadrants, quadrant)
                army_positions.append((enemy_name, (x, y)))

        # Determinar el cuadrante con mayor concentración para cada grupo
        max_army_quadrant_index = int(np.argmax(army_quadrants))
        max_structure_quadrant_index = int(np.argmax(structure_quadrants))

        quadrant_names = {
            0: "Cuadrante 1 (Superior Izquierdo)",
            1: "Cuadrante 2 (Superior Derecho)",
            2: "Cuadrante 3 (Inferior Izquierdo)",
            3: "Cuadrante 4 (Inferior Derecho)"
        }
        
        logger.info(f"Cuadrantes de ejército: {army_quadrants}")
        logger.info(f"Cuadrantes de estructuras: {structure_quadrants}")
        logger.info(f"Cuadrante con mayor concentración de ejército: {quadrant_names.get(max_army_quadrant_index, 'Unknown')}")
        logger.info(f"Cuadrante con mayor concentración de estructuras: {quadrant_names.get(max_structure_quadrant_index, 'Unknown')}")

        # Convertir las listas de posiciones a tuplas para la salida final
        army_positions = tuple(army_positions)
        structure_positions = tuple(structure_positions)
        
        return army_positions, structure_positions, (max_army_quadrant_index, max_structure_quadrant_index)

    def set_last_explore_position(self, last_position):
        """
        Establece la última posición explorada.
        
        Args:
            last_position (tuple): Coordenadas (x, y) de la última posición explorada
        """
        logger.debug(f"Estableciendo última posición explorada: {last_position}")
        self.last_position = last_position

    def initialize_used_positions(self):
        """
        Inicializa la lista de posiciones utilizadas.
        
        Utilizado al comenzar un nuevo juego para limpiar el historial
        de posiciones ya ocupadas.
        """
        logger.debug("Inicializando lista de posiciones utilizadas")
        self.used_positions = []

    def get_my_units_by_type(self, obs, unit_type):
        """
        Obtiene todas las unidades propias de un tipo específico.
        
        Args:
            obs: Observación del estado actual del juego
            unit_type: Tipo de unidad a buscar
            
        Returns:
            list: Lista de unidades propias del tipo especificado
            
        Example:
            >>> my_marines = helper.get_my_units_by_type(obs, units.Terran.Marine)
            >>> print(f"Marines propios: {len(my_marines)}")
        """
        logger.debug(f"Buscando unidades propias de tipo: {unit_type}")
        
        my_units = [
            unit for unit in obs.observation.raw_units
            if unit.unit_type == unit_type 
            and unit.alliance == features.PlayerRelative.SELF
        ]
        
        logger.debug(f"Encontradas {len(my_units)} unidades propias de tipo {unit_type}")
        return my_units

    def get_my_completed_units_by_type(self, obs, unit_type):
        """
        Obtiene todas las unidades propias completadas de un tipo específico.
        
        Args:
            obs: Observación del estado actual del juego
            unit_type: Tipo de unidad a buscar
            
        Returns:
            list: Lista de unidades propias completadas del tipo especificado
            
        Example:
            >>> completed_barracks = helper.get_my_completed_units_by_type(obs, units.Terran.Barracks)
            >>> print(f"Cuarteles completados: {len(completed_barracks)}")
        """
        logger.debug(f"Buscando unidades propias completadas de tipo: {unit_type}")
        
        completed_units = [
            unit for unit in obs.observation.raw_units
            if unit.unit_type == unit_type 
            and unit.build_progress == 100
            and unit.alliance == features.PlayerRelative.SELF
        ]
        
        logger.debug(f"Encontradas {len(completed_units)} unidades propias completadas de tipo {unit_type}")
        return completed_units

    def get_distances(self, obs, units, xy):
        """
        Calcula las distancias desde una lista de unidades a un punto específico.
        
        Args:
            obs: Observación del estado actual del juego
            units (list): Lista de unidades
            xy (tuple): Coordenadas del punto objetivo (x, y)
            
        Returns:
            numpy.ndarray: Array con las distancias calculadas
            
        Example:
            >>> distances = helper.get_distances(obs, scvs, (50, 50))
            >>> print(f"Distancias: {distances}")
        """
        logger.debug(f"Calculando distancias desde {len(units)} unidades a punto {xy}")
        
        units_xy = [(unit.x, unit.y) for unit in units]
        if len(units_xy) > 0:
            distances = np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1)
            logger.debug(f"Distancia mínima: {np.min(distances):.2f}, máxima: {np.max(distances):.2f}")
            return distances
        else:
            logger.warning("No hay unidades para calcular distancias")
            return []

    def get_command_center_top_left(self, obs):
        """
        Determina si el centro de mando está en la esquina superior izquierda del mapa.
        
        Args:
            obs: Observación del estado actual del juego
            
        Returns:
            bool: True si está en la esquina superior izquierda, False en caso contrario
        """
        logger.debug("Determinando posición del centro de mando")
        
        unit_type = units.Terran.CommandCenter
        command_center = self.get_my_units_by_type(obs, unit_type)
        
        if len(command_center) > 0:
            cc = command_center[0]
            self.base_top_left = (cc.x < 32)
            logger.info(f"Centro de mando en esquina superior izquierda: {self.base_top_left}")
            return self.base_top_left
        else:
            logger.warning("No se encontró centro de mando")
            return None

    def get_command_center_location(self, obs, unit_type):
        """
        Obtiene la ubicación de un centro de mando disponible.
        
        Selecciona aleatoriamente entre todos los centros de mando disponibles
        que no estén sobrecargados (menos de 10 órdenes pendientes).
        
        Args:
            obs: Observación del estado actual del juego
            unit_type: Tipo de unidad (ignorado, siempre usa CommandCenter)
            
        Returns:
            unit or None: Centro de mando seleccionado o None si no hay disponibles
            
        Example:
            >>> cc = helper.get_command_center_location(obs, units.Terran.CommandCenter)
            >>> if cc:
            ...     print(f"Centro de mando en: ({cc.x}, {cc.y})")
        """
        logger.debug("Buscando centro de mando disponible")
        
        unit_type = units.Terran.CommandCenter
        command_centers = self.get_my_units_by_type(obs, unit_type)
        
        logger.debug(f"Centros de mando encontrados: {len(command_centers)}")
        
        if len(command_centers) > 0:
            # Elección aleatoria entre todos los centros de mando
            random_index = random.randint(0, len(command_centers) - 1)
            command_center = command_centers[random_index]

            logger.debug(f"Centro de mando seleccionado: {random_index}")
            
            if command_center.order_length <= 10:
                logger.info(f"Centro de mando disponible en: ({command_center.x}, {command_center.y})")
                return command_center
            else:
                logger.warning(f"Centro de mando sobrecargado: {command_center.order_length} órdenes pendientes")
                return None
        
        logger.warning("No hay centros de mando disponibles")
        return None

    def get_base_top_left(self):
        """
        Obtiene el valor de base_top_left.
        
        Returns:
            bool: True si la base está en la esquina superior izquierda
        """
        return self.base_top_left

    def set_base_top_left(self):
        """
        Reinicia el valor de base_top_left.
        """
        logger.debug("Reiniciando base_top_left")
        self.base_top_left = None

    def random_location(self, positions=(0, 0), li=0, ls=0):
        """
        Genera una posición aleatoria cerca de un punto de referencia.
        
        Args:
            positions (tuple): Punto de referencia (x, y)
            li (int): Límite inferior del rango
            ls (int): Límite superior del rango
            
        Returns:
            tuple: Coordenadas aleatorias (x, y)
            
        Example:
            >>> pos = helper.random_location((50, 50), -5, 5)
            >>> print(f"Posición aleatoria: {pos}")
        """
        x = positions[0] + random.randint(li, ls)
        y = positions[1] + random.randint(li, ls)
        
        logger.debug(f"Posición aleatoria generada: ({x}, {y}) desde referencia {positions}")
        return (x, y)

    def validate_random_location(self, obs, x, y, positions=(0, 0), li=0, ls=0):
        """
        Valida y ajusta una posición para evitar colisiones con unidades existentes.
        
        Intenta encontrar una posición válida que no esté ocupada por unidades
        propias. Si no encuentra una posición válida después de 10 intentos,
        genera una posición aleatoria.
        
        Args:
            obs: Observación del estado actual del juego
            x (int): Coordenada X inicial
            y (int): Coordenada Y inicial
            positions (tuple): Punto de referencia para ajustes
            li (int): Límite inferior del rango
            ls (int): Límite superior del rango
            
        Returns:
            tuple: Coordenadas válidas (x, y)
            
        Example:
            >>> valid_pos = helper.validate_random_location(obs, 50, 50, (40, 40), -5, 5)
            >>> print(f"Posición válida: {valid_pos}")
        """
        logger.debug(f"Validando posición: ({x}, {y})")
        
        # Generar un array con todas las posiciones de mis unidades
        positions_list = []
        unit_types = [
            units.Terran.SCV, 
            units.Terran.Marine,
            units.Terran.SupplyDepot, 
            units.Terran.Barracks, 
            units.Terran.CommandCenter
        ]
        
        for unit_type in unit_types:
            my_units = self.get_my_units_by_type(obs, unit_type)
            for unit in my_units:
                positions_list.append((unit.x, unit.y))
        
        logger.debug(f"Posiciones ocupadas encontradas: {len(positions_list)}")
        
        # Intentar encontrar una posición válida
        for i in range(10):
            logger.debug(f"Intento {i+1}: Posición a ocupar: ({x}, {y})")
            
            if (x, y) not in positions_list:
                logger.info(f"Posición válida encontrada: ({x}, {y})")
                return (x, y)
            else:
                logger.debug(f"Posición ocupada, generando nueva posición")
                tuple_location = self.random_location((x, y), li, ls)
                x, y = tuple_location[0], tuple_location[1]
        
        # Si no se encontró posición válida, generar una aleatoria
        logger.warning("No se encontró posición válida después de 10 intentos")
        tuple_location = self.random_location((x + 1, y + 1), li, ls)
        x, y = tuple_location[0], tuple_location[1]

        # Asegurar coordenadas positivas
        if x < 0:
            x *= -1
        if y < 0:
            y *= -1
            
        logger.info(f"Posición final generada: ({x}, {y})")
        return (x, y)
     
    def select_rand_quadrant(self, base_top_left):
        """
        Selecciona un cuadrante aleatorio basado en la posición de la base.
        
        Args:
            base_top_left (bool): Indica si la base está en la esquina superior izquierda
            
        Returns:
            int: Índice del cuadrante seleccionado (1-4)
        """
        if self.base_top_left == True:
            rand = random.randint(2, 4)  # 3 cuadrantes --- 2,3,4
        elif self.base_top_left == False:
            rand = random.randint(1, 3)  # 3 cuadrantes --- 1,2,3
            
        logger.debug(f"Cuadrante aleatorio seleccionado: {rand}")
        return rand

    def get_xy(self):
        """
        Obtiene coordenadas específicas basadas en la posición de la base y el cuadrante.
        
        Returns:
            tuple: Coordenadas (x, y) calculadas
        """
        # Ajustar cuadrante si es necesario
        if self.base_top_left == True and self.quadrant == 1:
            self.quadrant = self.select_rand_quadrant(self.base_top_left)
        elif self.base_top_left == False and self.quadrant == 4:
            self.quadrant = self.select_rand_quadrant(self.base_top_left)
            
        # Calcular coordenadas según la posición de la base y cuadrante
        if self.base_top_left == True and self.quadrant != 0:
            if self.quadrant == 2:  # RIGHT HIGHER   31<= x <=63     0<= y <=31 
                x, y = 39, 23
            elif self.quadrant == 3:  # LL = LEFT LOWER      0<= x <=31    32<= y <=63 
                x, y = 19, 44
            elif self.quadrant == 4:  # RL = RIGHT LOWER    32<= x <=63    32<= y <=63
                x, y = 39, 44
        elif self.base_top_left == False and self.quadrant != 0:
            if self.quadrant == 1:   # LH = LEFT HIGHER     0<= x <=31     0<= y <=31
                x, y = 19, 23
            elif self.quadrant == 2:  # RIGHT HIGHER   31<= x <=63     0<= y <=31 
                x, y = 39, 23
            elif self.quadrant == 3:  # LL = LEFT LOWER      0<= x <=31    32<= y <=63
                x, y = 19, 44
        else:  # Por defecto
            x, y = 30, 30

        logger.debug(f"Coordenadas calculadas: ({x}, {y}) para cuadrante {self.quadrant}")
        return (x, y)

    def get_terran_unit(self, unit_type, army_label=''):
        """
        Obtiene el nombre de una unidad Terran basado en su tipo y etiqueta de ejército.
        
        Args:
            unit_type (int): Tipo de unidad
            army_label (str): Etiqueta del ejército ('marine_attack', 'marine_defense', 'marauder')
            
        Returns:
            str: Nombre de la unidad
        """
        # Diccionario de tipos de unidades que el Marine atacará
        unit_type_to_name__that_marine_will_attack = {
            18: "CommandCenter", 19: "Barracks", 20: "EngineeringBay", 21: "Barracks",
            22: "EngineeringBay", 23: "MissileTurret", 24: "Bunker", 25: "SensorTower",
            26: "GhostAcademy", 27: "Factory", 28: "Starport", 29: "Armory", 30: "FusionCore",
            31: "AutoTurret", 32: "SiegeTankSieged", 33: "SiegeTank", 34: "VikingAssault",
            35: "VikingFighter", 36: "CommandCenterFlying", 37: "BarracksTechLab",
            38: "BarracksReactor", 39: "FactoryTechLab", 40: "FactoryReactor",
            41: "StarportTechLab", 42: "StarportReactor", 43: "FactoryFlying",
            44: "StarportFlying", 45: "SCV", 46: "BarracksFlying", 47: "SupplyDepotLowered",
            48: "Marine", 49: "Reaper", 50: "Ghost", 51: "Marauder", 52: "Thor",
            53: "Hellion", 54: "Medivac", 55: "Banshee", 56: "Raven", 57: "Battlecruiser",
            58: "Nuke", 130: "PlanetaryFortress", 132: "OrbitalCommand",
            134: "OrbitalCommandFlying", 144: "GhostAlternate", 145: "GhostNova",
            268: "MULE", 484: "Hellbat", 498: "WidowMine", 500: "WidowMineBurrowed",
            692: "Cyclone", 734: "LiberatorAG", 689: "Liberator", 830: "KD8Charge",
            1913: "RepairDrone", 1960: "RefineryRich"
        }

        # Diccionario de tipos de unidades que el Marauder atacará (modo defensivo)
        unit_type_to_name__that_marauder_will_attack = {
            49: "Reaper", 50: "Ghost", 51: "Marauder", 48: "Marine", 54: "Medivac",
            56: "Raven", 32: "SiegeTankSieged", 33: "SiegeTank", 52: "Thor",
            53: "Hellion", 268: "MULE", 45: "SCV"
        }

        # Seleccionar el diccionario apropiado según la etiqueta del ejército
        if army_label in ['marine_attack', 'marine_defense']:
            unit_name = unit_type_to_name__that_marine_will_attack.get(unit_type, "Unknown")
        elif army_label == 'marauder':
            unit_name = unit_type_to_name__that_marauder_will_attack.get(unit_type, "Unknown")
        else:
            unit_name = unit_type_to_name__that_marine_will_attack.get(unit_type, "Unknown")

        logger.debug(f"Unidad Terran: {unit_name} (tipo: {unit_type}, ejército: {army_label})")
        return unit_name






