"""
Módulo de sistema de recompensas para StarCraft II con PySC2.

Este módulo implementa un sistema de cálculo de recompensas para el agente de aprendizaje
por refuerzo, incluyendo:
- Monitoreo de cambios en unidades y estructuras
- Cálculo de penalizaciones por pérdida de unidades
- Evaluación de eficiencia en el uso de recursos
- Recompensas por acciones de ataque
- Análisis de la relación entre gasto y producción militar

El objetivo es penalizar la pérdida de estructuras y ejércitos según su importancia,
incentivando el incremento de la capacidad militar y penalizando el mal uso de recursos.

Referencias:
- https://liquipedia.net/starcraft2/Terran_Strategy
- https://github.com/deepmind/pysc2

Autor: Pablo Escobar
Fecha: 2022
"""

from pysc2.lib import actions, features
from pysc2.lib import units as sc2_units
import random
import numpy as np
import pandas as pd
import datetime
import os
import math
from libs.logging_config import get_logger

# Configuración del logger para este módulo
logger = get_logger('algorithms.rewards')

class Reward:
    """
    Sistema de cálculo de recompensas para el agente de StarCraft II.
    
    Esta clase implementa un sistema de recompensas que monitorea los cambios en
    unidades, recursos y acciones de ataque, asignando recompensas o penalizaciones
    según el comportamiento del agente.
    
    Estrategia principal:
    - Penalización por pérdida de unidades según su importancia
    - Penalización por mal uso de recursos (gasto sin producción militar)
    - Recompensa por acciones de ataque proactivas
    - Evaluación de eficiencia en la producción de unidades militares
    
    Attributes:
        prev_minerals (int): Minerales en el paso anterior
        prev_gas (int): Gas en el paso anterior
        prev_supply (int): Suministro en el paso anterior
        previous_unit_counts (dict): Conteo de unidades en el paso anterior
    """
    
    def __init__(self):
        """
        Inicializa el sistema de recompensas.
        
        Configura las variables de seguimiento para recursos y unidades
        del paso anterior, necesarias para calcular cambios y recompensas.
        """
        logger.info("Inicializando sistema de recompensas")
        
        # Recursos del paso anterior
        self.prev_minerals = 0
        self.prev_gas = 0
        self.prev_supply = 0
        
        # Conteo de unidades del paso anterior
        self.previous_unit_counts = {}
        
        logger.info("Sistema de recompensas inicializado correctamente")

    def count_units_by_type(self, obs, unit_type):
        """
        Cuenta las unidades completadas de un tipo específico.
        
        Args:
            obs: Observación del estado actual del juego
            unit_type: Tipo de unidad a contar
            
        Returns:
            int: Número de unidades completadas del tipo especificado
            
        Example:
            >>> marine_count = reward.count_units_by_type(obs, units.Terran.Marine)
            >>> print(f"Marines completados: {marine_count}")
        """
        count = sum(
            1
            for unit in obs.observation.raw_units
            if unit.unit_type == unit_type
            and unit.alliance == features.PlayerRelative.SELF
            and unit.build_progress == 100
        )
        
        logger.debug(f"Unidades de tipo {unit_type} completadas: {count}")
        return count

    def get_unit_counts(self, obs):
        """
        Calcula recompensas basadas en cambios en el conteo de unidades.
        
        Esta función compara el conteo actual de unidades con el anterior
        y asigna recompensas o penalizaciones según los cambios observados.
        
        Args:
            obs: Observación del estado actual del juego
            
        Returns:
            float: Recompensa total calculada por cambios en unidades
            
        Example:
            >>> reward = reward.get_unit_counts(obs)
            >>> print(f"Recompensa por unidades: {reward}")
        """
        logger.debug("Calculando recompensas por cambios en unidades")
        
        # Definir niveles de penalización según importancia de unidades
        first_level = -0.1   # Unidades básicas (SCV, Bunker, etc.)
        second_level = -0.3  # Estructuras importantes (Barracks, Tech Lab)
        third_level = -1.0   # Unidades críticas (Command Center, Marine, Marauder)

        # Configurar penalizaciones por tipo de unidad
        UNIT_PENALTIES = {
            'command_center': third_level,  # Mayor penalización por perder Centros de Comandos
            'scv': first_level,             # Penalización menor por SCVs
            'marine': third_level,          # Mayor penalización por perder Marines
            'marauder': third_level,        # Mayor penalización por perder Marauders
            'barracks': second_level,       # Penalización media por Barracks
            'tech_lab': second_level,       # Penalización media por Tech Labs
            'bunker': first_level,          # Penalización menor por Bunkers
            'refinery': first_level,        # Penalización menor por Refinerías
            'supply_depot': first_level     # Penalización menor por Supply Depots
        }

        # Configurar recompensas por producción de unidades militares
        UNIT_REWARDS = {
            'marine': 1.0,      # Recompensa por producir Marines
            'marauder': 1.0,    # Recompensa por producir Marauders
        }

        # Mapear tipos de unidades a sus identificadores
        UNIT_TYPES = {
            'command_center': sc2_units.Terran.CommandCenter,
            'scv': sc2_units.Terran.SCV,
            'supply_depot': sc2_units.Terran.SupplyDepot,
            'barracks': sc2_units.Terran.Barracks,
            'marine': sc2_units.Terran.Marine,
            'marauder': sc2_units.Terran.Marauder,
            'tech_lab': sc2_units.Terran.BarracksTechLab,
            'bunker': sc2_units.Terran.Bunker,
            'refinery': sc2_units.Terran.Refinery
        }

        # Obtener conteo actual de unidades
        current_unit_counts = {
            key: self.count_units_by_type(obs, value)
            for key, value in UNIT_TYPES.items()
        }
        
        reward1 = 0
        logger.debug(f"Conteo anterior de unidades: {self.previous_unit_counts}")
        logger.debug(f"Conteo actual de unidades: {current_unit_counts}")

        # Comparar con el estado anterior y calcular recompensas
        for unit_type, current_count in current_unit_counts.items():
            previous_count = self.previous_unit_counts.get(unit_type, 0)
            difference = current_count - previous_count

            # Determinar si es penalización o recompensa
            if difference < 0:
                # Pérdida de unidades: aplicar penalización
                applied_type = "PENALIZACIÓN"
                amount = UNIT_PENALTIES.get(unit_type, -1.0)
                value_applied = amount * abs(difference)
            elif difference > 0:
                # Ganancia de unidades: aplicar recompensa
                applied_type = "RECOMPENSA"
                amount = UNIT_REWARDS.get(unit_type, 0.001)
                value_applied = amount * difference
            else:
                # Sin cambios: continuar al siguiente
                logger.debug(
                    f"Unidad: {unit_type:15} | Prev: {previous_count:3d} | Curr: {current_count:3d} | "
                    f"Diff: {difference:+3d} | Sin cambios aplicados."
                )
                continue

            # Registrar el cambio y la recompensa aplicada
            logger.info(
                f"Unidad: {unit_type:15} | "
                f"Prev: {previous_count:3d} | "
                f"Curr: {current_count:3d} | "
                f"Diff: {difference:+3d} | "
                f"Aplicado: {applied_type}({amount:.3f}) | "
                f"Valor: {value_applied:.3f}"
            )
            reward1 += value_applied

        # Cálculo de penalización por infraestructuras innecesarias
        # que no se traducen en incremento de Ejército
        reward2 = self._calculate_infrastructure_penalty(current_unit_counts)
        
        # Actualizar conteo anterior para el próximo cálculo
        self.previous_unit_counts = current_unit_counts
        
        total_reward = reward1 + reward2
        logger.info(f"Recompensa por unidades (reward1): {reward1:.3f}")
        logger.info(f"Penalización por infraestructura (reward2): {reward2:.3f}")
        logger.info(f"Recompensa total por unidades: {total_reward:.3f}")
        
        return total_reward

    def _calculate_infrastructure_penalty(self, current_unit_counts):
        """
        Calcula penalización por infraestructuras que no producen unidades militares.
        
        Args:
            current_unit_counts (dict): Conteo actual de unidades
            
        Returns:
            float: Penalización calculada
        """
        barracks = current_unit_counts.get('barracks', 0)
        tech_labs = current_unit_counts.get('tech_lab', 0)
        marines = current_unit_counts.get('marine', 0)
        marauders = current_unit_counts.get('marauder', 0)

        penalty = 0
        
        # Penalización agresiva si hay Barracks sin producir unidades
        if barracks > 0 and marines + marauders == 0:
            penalty -= 2.0
            logger.warning(f"Penalización: {barracks} Barracks sin producir unidades militares")
        
        # Penalización si hay Tech Labs sin producir Marauders
        if barracks > 0 and tech_labs > 0 and marauders == 0:
            penalty -= 1.0
            logger.warning(f"Penalización: {tech_labs} Tech Labs sin producir Marauders")
        
        return penalty

    def penalize_resource_misuse(self, obs):
        """
        Penaliza el mal uso de recursos que no se traduce en producción militar.
        
        Esta función evalúa la eficiencia del gasto de recursos comparando
        los recursos gastados con las unidades militares producidas.
        
        Args:
            obs: Observación del estado actual del juego
            
        Returns:
            float: Penalización por mal uso de recursos
            
        Example:
            >>> penalty = reward.penalize_resource_misuse(obs)
            >>> print(f"Penalización por mal uso de recursos: {penalty}")
        """
        logger.debug("Evaluando eficiencia en el uso de recursos")
        
        player_info = obs.observation.player
        current_minerals = player_info.minerals
        current_gas = player_info.vespene

        # Calcular los recursos gastados desde el último paso
        minerals_used = self.prev_minerals - current_minerals
        gas_used = self.prev_gas - current_gas

        # Obtener cantidad anterior de unidades militares
        marines = self.previous_unit_counts.get('marine', 0)
        marauders = self.previous_unit_counts.get('marauder', 0)

        # Costo estimado de las unidades
        marine_cost = 50      # Minerales por Marine
        marauder_cost = 100   # Minerales por Marauder
        marauder_gas_cost = 25 # Gas por Marauder

        # Estimación del costo de producción de las unidades
        expected_resource_cost = (
            (marines * marine_cost) +
            (marauders * marauder_cost) +
            (marauders * marauder_gas_cost)
        )

        logger.debug(f"Costo esperado de recursos: {expected_resource_cost}")

        # Recursos totales gastados
        total_resources_spent = max(0, minerals_used + gas_used)
        logger.debug(f"Recursos totales gastados: {total_resources_spent}")

        # Evaluar la eficiencia: cuánto gastamos vs cuánto producimos
        efficiency_ratio = 0
        if total_resources_spent > 0:
            efficiency_ratio = round(expected_resource_cost / total_resources_spent, 5)

        logger.debug(f"Ratio de eficiencia: {efficiency_ratio}")
        
        # Penalización si la eficiencia es baja
        penalty = 0
        if efficiency_ratio < 1:
            penalty = -0.2 * (1 - efficiency_ratio)
            logger.warning(f"Penalización por baja eficiencia: {penalty:.3f}")

        # Actualizar los recursos previos para el próximo cálculo
        self.prev_minerals = current_minerals
        self.prev_gas = current_gas
        
        logger.debug(f"Minerales previos actualizados: {self.prev_minerals}")
        logger.debug(f"Gas previo actualizado: {self.prev_gas}")
        
        return penalty

    def get_reward_for_attack(self, action):
        """
        Obtiene recompensa por acciones de ataque específicas.
        
        Args:
            action (str): Acción de ataque ejecutada
            
        Returns:
            float: Recompensa por la acción de ataque
            
        Example:
            >>> attack_reward = reward.get_reward_for_attack("attack_with_marine")
            >>> print(f"Recompensa por ataque: {attack_reward}")
        """
        # Configurar recompensas por acciones de ataque
        reward_for_attack = {
            "attack_with_marauder": 1.0,
            "attack_with_marine": 1.0,
            "defense_with_marine": 1.0,
        }
        
        reward = reward_for_attack.get(action, 0.0)
        logger.debug(f"Recompensa por acción de ataque '{action}': {reward}")
        return reward

    def get_specific_reward(self, action, execute_action, obs):
        """
        Calcula la recompensa específica para una acción en un momento dado.
        
        Esta es la función principal que coordina todos los cálculos de recompensas,
        incluyendo penalizaciones por mal uso de recursos, recompensas por cambios
        en unidades, y recompensas por acciones de ataque.
        
        Args:
            action (str): Acción ejecutada
            execute_action (bool): True si la acción se ejecutó correctamente
            obs: Observación del estado actual del juego
            
        Returns:
            float: Recompensa total calculada
            
        Example:
            >>> total_reward = reward.get_specific_reward("build_scv", True, obs)
            >>> print(f"Recompensa total: {total_reward}")
        """
        logger.info("=" * 100)
        logger.info("CÁLCULO DE RECOMPENSA EN TIEMPO t")
        logger.info("=" * 100)
        
        # Recompensa base (ligeramente negativa para incentivar acción)
        total_rewards = -0.001
        penalize_resource = 0

        if execute_action:
            # Calcular penalización por mal uso de recursos
            penalize_resource = self.penalize_resource_misuse(obs)
            logger.info(f"Penalización por mal uso de recursos: {penalize_resource:.3f}")

            # Calcular recompensas por cambios en unidades
            total_rewards_unit_counts = self.get_unit_counts(obs)
            logger.info(f"Recompensa por cambios en unidades: {total_rewards_unit_counts:.3f}")

            # Acumular recompensas
            total_rewards += penalize_resource

            # Calcular recompensa por acción de ataque
            total_reward_for_attack = self.get_reward_for_attack(action)
            logger.info(f"Recompensa por acción de ataque: {total_reward_for_attack:.3f}")

            total_rewards += total_reward_for_attack
        else:
            logger.warning("La acción no puede ejecutarse porque no se cumplen los requisitos del juego")

        logger.info(f"Recompensa total calculada: {total_rewards:.3f}")
        return total_rewards
