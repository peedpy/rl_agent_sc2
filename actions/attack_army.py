"""
Módulo para el control táctico de unidades militares en StarCraft II.

Este módulo implementa la lógica de combate para Marines y Marauders, incluyendo
estrategias de ataque y defensa basadas en análisis de posiciones enemigas,
distancias y priorización de objetivos.

Estrategias implementadas:
- Marines: Efectivos contra unidades a distancia y para ataques rápidos
- Marauders: Mejores para combate cercano y contra unidades con alta salud
- Análisis de "hot zones": Identificación de zonas con mayor concentración enemiga

Referencias:
- https://liquipedia.net/starcraft2/Marine
- https://liquipedia.net/starcraft2/Marauder

Autor: Pablo Escobar
Fecha: 2022
"""

import logging
from pysc2.lib import actions, features, units
import random
import numpy as np # Mathematical functions
import os
from collections import Counter

# Configuración del logger para este módulo
logger = logging.getLogger(__name__)

"""
DEFINICIONES
1. Se solicita la accion de atacar con Marine o Marauder
2. Se verifica que exista al menos un Marine o Marauder libre para prepararse a atacar
3. Identificar los enemigos mas cercanos, y los enemigos validos a atacar
4. Verifico su estado de salud, el que tenga menos salud ataco
5. Verifico su posicion, el que esta mas cerca ataco
6. Punto 4 y 5 se seleccionara de forma aleatoria

Atacando con Marines:
Unidades distantes y saludables:
* Si el enemigo tiene unidades a larga distancia y con una cantidad significativa de salud, es eficaz usar Marines 
debido a su rango y mayor velocidad de ataque. Pueden debilitar al enemigo antes de que lleguen al combate cuerpo a cuerpo.
Ataque rápido y golpe y huida:

* Para ataques rápidos y para debilitar rápidamente unidades enemigas, los Marines pueden ser más efectivos ya que pueden disparar 
y retirarse antes de recibir mucho daño.

Atacando con Marauders:
Unidades cercanas y saludables:

Si las unidades enemigas están más cerca y tienen una cantidad decente de salud, los Marauders son una mejor opción debido a su 
mayor salud y daño por disparo. Pueden resistir más tiempo en el combate cuerpo a cuerpo.
Unidades debilitadas:

Si las unidades enemigas están debilitadas y tienen poca salud, los Marauders pueden eliminarlas rápidamente debido a su daño por disparo más alto. 
Es más eficiente para acabar con unidades que están a punto de ser derrotadas

"""

class AttackArmy:
    """
    Clase responsable del control táctico de unidades militares.
    
    Esta clase implementa estrategias de combate inteligentes para Marines y Marauders,
    incluyendo análisis de posiciones enemigas, cálculo de distancias óptimas y
    priorización de objetivos basada en salud y proximidad.
    
    Attributes:
        total_send_units (int): Contador total de unidades enviadas a combate
        marines (list): Lista de Marines actualmente en combate
        marauders (list): Lista de Marauders actualmente en combate
        army_positions (tuple): Posiciones de unidades enemigas de combate
        structure_positions (tuple): Posiciones de estructuras enemigas
    """
    
    # Constantes de configuración táctica
    MIN_DISTANCE_ATTACK = 20  # Distancia mínima para ataques con Marines
    MAX_DISTANCE_DEFENSE = 20  # Distancia máxima para defensa con Marauders
    
    def __init__(self): 
        """
        Inicializa la clase AttackArmy con contadores y listas de seguimiento.
        """
        logger.debug("Inicializando módulo AttackArmy")
        
        self.total_send_units = 0
        self.marines = []  # Total de marinos que están atacando actualmente
        self.marauders = []  # Total de marauders que están atacando actualmente

        self.army_positions = None
        self.structure_positions = None

    def send_to_attack_opposite(self, obs, helper, army_label):
        """
        Envía unidades militares a atacar objetivos enemigos.
        
        Esta función implementa la lógica de combate inteligente:
        1. Analiza las "hot zones" (zonas calientes) del mapa
        2. Identifica unidades enemigas válidas para atacar
        3. Calcula distancias y prioriza objetivos
        4. Selecciona la mejor unidad para el ataque
        5. Ejecuta la acción de ataque
        
        Args:
            obs: Observación del estado actual del juego
            helper: Objeto Helper con funciones auxiliares
            army_label (str): Tipo de ataque ('marine_attack', 'marine_defense', 'marauder')
            
        Returns:
            tuple: (acción, flag_éxito, posición) donde:
                - acción: Función RAW de PySC2 a ejecutar
                - flag_éxito: 1 si se puede ejecutar, 0 en caso contrario
                - posición: Coordenadas del objetivo (x, y) o (None, None)
                
        Example:
            >>> attack = AttackArmy()
            >>> action, success, pos = attack.send_to_attack_opposite(obs, helper, 'marine_attack')
            >>> if success:
            ...     print(f"Atacando objetivo en posición {pos}")
        """
        logger.info(f"Iniciando ataque con {army_label}")
        logger.debug("=" * 80)
        logger.debug("ANÁLISIS DE ZONAS CALIENTES")
        logger.debug("=" * 80)

        # Analizar zonas calientes del mapa
        army_positions, structure_positions, max_quadrants_index = helper.get_calculate_hot_zone(obs)
        
        logger.debug(f"Posiciones de ejército enemigo: {len(army_positions)} unidades")
        logger.debug(f"Posiciones de estructuras enemigas: {len(structure_positions)} estructuras")
        logger.debug(f"Cuadrantes con mayor concentración (ejército/estructuras): {max_quadrants_index}")

        # Determinar tipo de unidad a usar
        unit = None
        if army_label in ['marine_attack', 'marine_defense']:
            unit = units.Terran.Marine
            logger.debug("Usando Marines para el ataque")
        elif army_label == 'marauder':
            unit = units.Terran.Marauder
            logger.debug("Usando Marauders para el ataque")
        else:
            logger.warning(f"Tipo de ejército no reconocido: {army_label}")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))

        # Obtener unidades disponibles
        army_units = helper.get_my_units_by_type(obs, unit)
        free_army_units = [_unit for _unit in army_units if _unit.order_length == 0]
        enemy_army_units = [_unit for _unit in obs.observation.raw_units if _unit.alliance == features.PlayerRelative.ENEMY]
        
        logger.debug(f"Unidades {army_label} disponibles: {len(free_army_units)}")
        logger.debug(f"Unidades enemigas detectadas: {len(enemy_army_units)}")
        
        if not free_army_units or not enemy_army_units:
            logger.debug("No hay unidades disponibles para ataque o no hay enemigos detectados")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))

        # Filtrar unidades enemigas válidas
        target_units = []
        enemy_units_data = []
        
        for enemy_unit in enemy_army_units:
            unit_type = enemy_unit.unit_type
            enemy_tag = helper.get_terran_unit(unit_type, army_label)
            
            if enemy_tag != "Unknown":
                health = enemy_unit.health
                position = (enemy_unit.x, enemy_unit.y)
                enemy_units_data.append((army_label, unit_type, enemy_tag, health, position))
                target_units.append(enemy_unit)

        logger.debug(f"Unidades enemigas válidas para ataque: {len(enemy_units_data)}")
        
        if not enemy_units_data:
            logger.debug("No se encontraron unidades enemigas válidas para atacar")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))

        # Mostrar información detallada de unidades enemigas
        logger.debug("Detalles de unidades enemigas:")
        for i, data in enumerate(enemy_units_data, 1):
            logger.debug(f"  {i}. Tipo: {data[2]}, Salud: {data[3]}, Posición: {data[4]}")

        # Calcular distancias y priorizar objetivos
        closest_units = []
        for my_unit in free_army_units:
            distances = [((my_unit.x - unit.x) ** 2 + (my_unit.y - unit.y) ** 2) ** 0.5 for unit in target_units]

            # Filtrar unidades según la estrategia
            filtered_units = []
            if army_label in ["marauder", "marine_defense"]:
                # Para Marauders y defensa: atacar unidades cercanas
                filtered_units = [(unit, distance, unit.health) for unit, distance in zip(target_units, distances) 
                                 if distance <= self.MAX_DISTANCE_DEFENSE]
            elif army_label == "marine_attack":
                # Para ataque con Marines: atacar unidades a distancia
                filtered_units = [(unit, distance, unit.health) for unit, distance in zip(target_units, distances) 
                                 if distance >= self.MIN_DISTANCE_ATTACK]

            # Ordenar por distancia y salud (priorizar cercanos con poca salud)
            closest_units_sorted = sorted(filtered_units, key=lambda x: (x[1], x[2]))
            
            if closest_units_sorted:
                closest_units.append(closest_units_sorted[0])

        logger.debug(f"Unidades candidatas para ataque: {len(closest_units)}")

        # Seleccionar el objetivo prioritario
        if closest_units:
            priority_unit = min(closest_units, key=lambda x: (x[1], x[2]))
            enemy_unit = priority_unit[0]
            distance = priority_unit[1]
            health = priority_unit[2]
            
            x, y = enemy_unit.x, enemy_unit.y
            random_my_unit = random.choice(free_army_units)
            selected_tag = random_my_unit.tag
            
            logger.info(f"Atacando con {army_label} desde ({random_my_unit.x}, {random_my_unit.y}) a objetivo en ({x}, {y})")
            logger.debug(f"Distancia al objetivo: {distance:.2f}, Salud del objetivo: {health}")
            
            # Ejecutar ataque
            try:
                action = actions.RAW_FUNCTIONS.Attack_pt("now", selected_tag, (x, y))
                return (action, 1, (x, y))
                
            except Exception as e:
                logger.error(f"Error al ejecutar ataque: {str(e)}")
                return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        else:
            logger.debug("No se encontraron objetivos válidos para el ataque")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
