from pysc2.lib import actions, features, units
import random
import numpy as np
import os

class ExploreCSV:
  def __init__(self):
    self.positions = []

  def explore_csv(self, obs, helper):
    scvs = helper.get_my_units_by_type(obs, units.Terran.SCV)
    free_supply = [scv for scv in scvs if scv.order_length >= 0]

    if len(free_supply) > 0:
      scv = random.choice(free_supply)

      map_width  = 64  # Ancho del mapa
      map_height = 64  # Alto del mapa

      #Elijo una posicion aleatoria o elijo de forma aleatoria una posicion valida
      selection =  random.randint(1, 2)
      if selection == 1 or len(self.positions) == 0:
        x = random.randint(0, map_width  - 1)
        y = random.randint(0, map_height - 1)
      else:
        x,y = self.positions[random.randint(0, len(self.positions) - 1)]

      print("(x, y): {}".format((x, y)))
      #Verificar si es un punto que no colisiona con mis construcciones
      while self.collides_with_my_structures(x, y, obs):
        x = random.randint(0, map_width  - 1)
        y = random.randint(0, map_height - 1)

      #Porque en caso de que haya extraido del array, ya no hace falta volver a agregarlo
      if selection == 1 or (x,y) not in self.positions:
        self.positions.append((x,y))

      #Validar si la posicion no colisiona con estructuras y obstaculos en el mapa
      target_location = ( x + random.randint(1,3) , y + random.randint(1,3) )

      #Para ubicar la posible ultima ubicacion obtenida de la exploracion
      helper.set_last_explore_position(target_location)
      print("target_location (x, y): {}".format(target_location))
      #print("quadrant: {}".format(quadrant))
      #print("self.positions:  {}".format(self.positions))
      #print("target_location: {}".format(target_location))
      #print(" helper.get_base_top_left(): {}".format( helper.get_base_top_left()))
      #os.system("pause")      
      return (actions.RAW_FUNCTIONS.Move_pt("now", scv.tag, target_location), 1, target_location)
    else:
      return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))

  def collides_with_my_structures(self, x, y, obs):
      # Obtener todas las unidades visibles en el mapa
      all_units = obs.observation.raw_units
      
      # Verificar si la posición (x, y) colisiona con alguna estructura u obstáculo
      for unit in all_units:
          if unit.alliance == features.PlayerRelative.SELF:  # Verificar solo las unidades propias (estructuras propias)
              if abs(unit.x - x) <= unit.radius and abs(unit.y - y) <= unit.radius:
                  return True  # La posición colisiona con una estructura
      return False  # La posición no colisiona con ninguna estructura
