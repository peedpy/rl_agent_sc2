from pysc2.lib import actions, features
from pysc2.lib import units as sc2_units

import random
import numpy as np
import pandas as pd
import datetime
import logging
import os
import math

"""
Este bloque forma parte de un sistema de cálculo de recompensas dentro de un entorno de aprendizaje.
Monitorea los cambios en unidades, recursos y acciones de ataque, y asigna recompensas o penalizaciones ofensivas.
El objetivo es penalizar la pérdida de estructuras y ejércitos en cada paso de tiempo 𝑡, según su nivel de importancia,
 es decir, se busca incrementar la cantidad de ejército.

Además, se penaliza específicamente cuando el gasto de recursos no se traduce en un aumento efectivo de la capacidad 
militar, de modo que la penalización sea directamente proporcional a la diferencia entre los recursos gastados y las
unidades militares producidas.

Estrategia:
Penalización: Si el gasto en recursos (minerales y gas) no se refleja en un aumento en el número de unidades ofensivas
(Marines y Marauders), se aplica una penalización.

Cálculo de eficiencia: Se compara la cantidad de recursos gastados (minerales y gas) con las unidades producidas
(Marines y Marauders) y se penaliza si los recursos no se utilizan de manera eficiente.

Recompensa por ataque: Se otorgan incentivos adicionales cuando se ejecutan acciones de ataque con unidades ofensivas,
promoviendo estrategias proactivas en el combate.
"""

class Reward:
    def __init__(self):
        self.prev_minerals = 0
        self.prev_gas = 0
        self.prev_supply = 0
        self.previous_unit_counts = {}

    def count_units_by_type(self, obs, unit_type):
        return sum(
            1
            for unit in obs.observation.raw_units
            if unit.unit_type == unit_type
            and unit.alliance == features.PlayerRelative.SELF
            and unit.build_progress == 100
        )

    def get_unit_counts(self, obs):
        first_level = -0.1
        second_level = -0.3
        third_level = -1

        UNIT_PENALTIES = {
            'command_center': third_level,  # Mayor penalización por perder Centros de Comandos
            'scv': first_level,
            'marine': third_level,          # Mayor penalización por perder Marines
            'marauder': third_level,        # Mayor penalización por perder Marauders
            'barracks': second_level,
            'tech_lab': second_level,
            'bunker': first_level,
            'refinery': first_level,
            'supply_depot': first_level
        }

        UNIT_REWARDS = {
            'marine': 1,
            'marauder': 1,
        }

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

        current_unit_counts = {
            key: self.count_units_by_type(obs, value)
            for key, value in UNIT_TYPES.items()
        }
        reward1 = 0

        print(f"=> previous_unit_counts      : {self.previous_unit_counts}")
        print(f"=> current_unit_counts       : {current_unit_counts}")

        # Comparar con el estado anterior
        for unit_type, current_count in current_unit_counts.items():
            previous_count = self.previous_unit_counts.get(unit_type, 0)
            difference = current_count - previous_count

            # Determinamos si es penalización o recompensa
            if difference < 0:
                applied_type = "PENALTY"
                amount = UNIT_PENALTIES.get(unit_type, -1)
                value_applied = amount * abs(difference)
            elif difference > 0:
                applied_type = "REWARD"
                amount = UNIT_REWARDS.get(unit_type, 0.001)
                value_applied = amount * difference
            else:
                # Caso sin cambio
                print(
                    f"Unit: {unit_type:15} | Prev: {previous_count:3d} | Curr: {current_count:3d} | "
                    f"Diff: {difference:+3d} | No penalty/reward applied."
                )
                continue  # Pasa al siguiente

            # Construimos la línea de texto con formato
            print(
                f"Unit: {unit_type:15} | "
                f"Prev: {previous_count:3d} | "
                f"Curr: {current_count:3d} | "
                f"Diff: {difference:+3d} | "
                f"Applied: {applied_type}({amount:.3f}) | "
                f"Value: {value_applied:.3f}"
            )
            reward1 += value_applied

        # Cálculo de Penalización por infraestructuras innecesarias
        # que no se traducen en incremento de Ejército
        reward2 = 0
        barracks = current_unit_counts.get('barracks', 0)
        tech_labs = current_unit_counts.get('tech_lab', 0)
        marines = current_unit_counts.get('marine', 0)
        marauders = current_unit_counts.get('marauder', 0)

        if barracks > 0 and marines + marauders == 0:
            # Penalización agresiva si hay Barracks sin producir unidades
            reward2 -= 2
        if barracks > 0 and tech_labs > 0 and marauders == 0:
            # Penalización si hay Tech Labs sin producir Marauders
            reward2 -= 1

        self.previous_unit_counts = current_unit_counts
        print(f"Current rewards 1           : {reward1}")
        print(f"Current rewards 2           : {reward2}")
        return reward1 + reward2

    def penalize_resource_misuse(self, obs):
        player_info = obs.observation.player
        current_minerals = player_info.minerals
        current_gas = player_info.vespene

        # Calcular los recursos gastados desde el último paso
        minerals_used = self.prev_minerals - current_minerals
        gas_used = self.prev_gas - current_gas

        # Unidades clave: Cantidad anterior de Marines y Marauders
        marines = self.previous_unit_counts.get('marine', 0)
        marauders = self.previous_unit_counts.get('marauder', 0)

        # Costo estimado de las unidades (Marine: 50 minerales, Marauder: 100 minerales, 25 gas)
        marine_cost = 50
        marauder_cost = 100
        marauder_gas_cost = 25

        # Estimación del costo de producción de las unidades
        expected_resource_cost = (
            (marines * marine_cost)
            + (marauders * marauder_cost)
            + (marauders * marauder_gas_cost)
        )

        print(f"expected_resource_cost      : {expected_resource_cost}")

        # Recursos totales gastados
        total_resources_spent = max(0, minerals_used + gas_used)
        print(f"total_resources_spent       : {total_resources_spent}")

        # Evaluamos la eficiencia: cuánto gastamos vs cuánto producimos
        efficiency_ratio = 0
        if total_resources_spent > 0:
            efficiency_ratio = round(expected_resource_cost / total_resources_spent, 5)

        print(f"efficiency_ratio            : {efficiency_ratio}")
        # Penalización si la eficiencia es baja (el gasto no se traduce en más unidades)
        penalty = 0
        if efficiency_ratio < 1:
            penalty = -0.2 * (1 - efficiency_ratio)

        # Actualizar los recursos previos para el próximo cálculo
        self.prev_minerals = current_minerals
        self.prev_gas = current_gas
        print(f"prev_minerals               : {self.prev_minerals}")
        print(f"prev_gas                    : {self.prev_gas}")
        return penalty

    def get_reward_for_attack(self, action):
        reward_for_attack = {
            "attack_with_marauder": 1,
            "attack_with_marine": 1,
            "defense_with_marine": 1,
        }
        return reward_for_attack.get(action, 0)

    def get_specific_reward(self, action, execute_action, obs):
        print("=" * 100)
        print("REWARD AT TIME t")
        print("=" * 100)
        total_rewards = -0.001
        penalize_resource = 0

        if execute_action:
            penalize_resource = self.penalize_resource_misuse(obs)
            print(f"***penalize_resource_misuse    : {penalize_resource}")

            total_rewards_unit_counts = self.get_unit_counts(obs)
            print(f"***total_rewards_unit_counts   : {total_rewards_unit_counts}")

            total_rewards += penalize_resource

            total_reward_for_attack = self.get_reward_for_attack(action)
            print(f"***total_reward_for_attack     : {total_reward_for_attack}")

            total_rewards += total_reward_for_attack
        else:
            print("The action cannot be executed because the necessary game requirements for construction/training are not met.")

        print(f"***total_rewards               : {total_rewards}")

        #os.system("pause")
        return total_rewards
