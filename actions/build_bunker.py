from pysc2.lib import actions, features, units
import random
import numpy as np # Mathematical functions
import os

class BuildBunker:
    def __init__(self):
        self.used_positions = []
        pass

    def build_bunker(self, obs, helper):
      #Selecciona el command Center
      unit_type = units.Terran.CommandCenter
      command_center_location = helper.get_command_center_location(obs, unit_type)
      
      bunkers = helper.get_my_units_by_type(obs, units.Terran.Bunker) 
      scvs = helper.get_my_units_by_type(obs, units.Terran.SCV)
        
      if (obs.observation.player.minerals >= 100
        and len(scvs) > 0
        and command_center_location is not None
        and len(bunkers) <= 50):
        #Identificar cuales son los rangos permitidos para X e Y
        li = -5
        ls =  5
        positions = (command_center_location.x ,command_center_location.y)
        tuple_location = helper.random_location(positions, li, ls)

        x = tuple_location[0]
        y = tuple_location[1]
        bunker_xy = helper.validate_random_location(obs, x, y, positions, li, ls)

        if bunker_xy != False:
          distances = helper.get_distances(obs, scvs, bunker_xy)
          scv = scvs[np.argmin(distances)]

          return (actions.RAW_FUNCTIONS.Build_Bunker_pt("now", scv.tag, bunker_xy), 1, bunker_xy)
        else:
          return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
      else:
        return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
