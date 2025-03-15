from pysc2.lib import actions, features, units
import random
import numpy as np # Mathematical functions
import os
from collections import Counter

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
  def __init__(self): 
    self.total_send_units = 0
    self.marines   = []  #Total de marinos que estan atacando actualmente
    self.marauders = []

    self.army_positions = None
    self.structure_positions = None


  def send_to_attack_opposite(self, obs, helper, army_label):
    #-------------------------------------------
    print("=" * 100)
    print("CALCULATE HOT ZONE")
    print("=" * 100)

    army_positions, structure_positions, max_quadrants_index = helper.get_calculate_hot_zone(obs)
    print(f"***army_positions                           : {army_positions}")
    print(f"***structure_positions                      : {structure_positions}")
    print(f"***max_quadrants_index(army/structure)      : {max_quadrants_index}")
    #print(f"***min_quadrants_index(army/structure)      : {min_quadrants_index}")
    #os.system("pause")

    unit = None
    if army_label == 'marine_attack' or army_label == 'marine_defense':
      unit = units.Terran.Marine
    elif army_label == 'marauder':
      unit = units.Terran.Marauder

    army_units = helper.get_my_units_by_type(obs, unit)
    free_army_units  = [_unit for _unit in army_units if _unit.order_length == 0]
    enemy_army_units = [_unit for _unit in obs.observation.raw_units if _unit.alliance == features.PlayerRelative.ENEMY]
    
    if len(free_army_units) > 0 and len(enemy_army_units) > 0:
      target_units = []
      enemy_units_data = []
      for enemy_unit in enemy_army_units:
          unit_type = enemy_unit.unit_type
          enemy_tag = helper.get_terran_unit(unit_type, army_label)
          
          if enemy_tag != "Unknown":
            health = enemy_unit.health
            position = (enemy_unit.x, enemy_unit.y)
            enemy_units_data.append(( army_label ,unit_type, enemy_tag, health, position ))
            target_units.append(enemy_unit)

      print(50*'_')
      c = 0
      for data in enemy_units_data:
        c+=1
        print("***{} - found data: {}".format(c, data))
      print(50*'_')

      if not enemy_units_data:
          return (actions.RAW_FUNCTIONS.no_op(), 0 , (None, None))

      #Por cada unidad libre del ejercito, verifica la distancia con respecto a todas las unidades enemigas
      # visibles en el rango de vision
      # Priorizar las unidades enemigas más cercanas con menor cantidad de vida.
      MIN_DISTANCE = 20
      MAX_DISTANCE = 20
      closest_units = []
      for m in free_army_units:
        distances = [((m.x - unit.x) ** 2 + (m.y - unit.y) ** 2) ** 0.5 for unit in target_units]

        filtered_units = []
        if army_label == "marauder" or army_label == "marine_defense":
          filtered_units = [(unit, distance, unit.health) for unit, distance in zip(target_units, distances) if distance <= MAX_DISTANCE]
        elif army_label == "marine_attack":
          filtered_units = [(unit, distance, unit.health) for unit, distance in zip(target_units, distances) if distance >= MIN_DISTANCE]

        # Sort the filtered list of enemy units by distance and health
        closest_units_sorted = sorted(filtered_units, key=lambda x: (x[1], x[2]))
        
        # Print the distances and health of filtered enemy units
        #print("{} => ({},{}) - Distance to nearby enemy units: ".format(army_label, m.x, m.y, closest_units_sorted))
        
        # Select the closest enemy unit with the lowest health
        if closest_units_sorted:
            closest_units.append(closest_units_sorted[0])

      #print("closest_units => {}".format(closest_units))
      # Select the most prioritized enemy unit (the first one in the list)
      if closest_units:
        priority_unit = min(closest_units, key=lambda x: (x[1], x[2]))
        enemy_unit = priority_unit[0]       
        x, y = enemy_unit.x, enemy_unit.y
        random_my_unit = random.choice(free_army_units)
        selected_tag = random_my_unit.tag
        print("({},{}) -- Posicion a atacar: {}".format( random_my_unit.x, random_my_unit.y , (x,y) ))
        
        #os.system('pause')
        try:
          return (actions.RAW_FUNCTIONS.Attack_pt("now", selected_tag, (x, y) ), 1,   (x, y) )
        except Exception as e:
          return (actions.RAW_FUNCTIONS.no_op(), 0 , (None, None))
      else:
        return (actions.RAW_FUNCTIONS.no_op(), 0 , (None, None))
    else:
      return (actions.RAW_FUNCTIONS.no_op(), 0 , (None, None))
