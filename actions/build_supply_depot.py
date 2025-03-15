from pysc2.lib import actions, features, units
import random
import numpy as np
import os

class SupplyDepot:
    def __init__(self):
        self.used_positions = []
        pass

    def build_supply_depot(self, obs, helper):
      #Selecciona el command Center
      unit_type = units.Terran.CommandCenter
      command_center_location = helper.get_command_center_location(obs, unit_type)
      
      supply_depots = helper.get_my_units_by_type(obs, units.Terran.SupplyDepot) 
      scvs = helper.get_my_units_by_type(obs, units.Terran.SCV)
        
      if (obs.observation.player.minerals >= 100
        and len(scvs) > 0
        and command_center_location is not None
        and len(supply_depots) <= 50):
        #Identificar cuales son los rangos permitidos para X e Y
        li = -5
        ls =  5
        positions = (command_center_location.x ,command_center_location.y)
        tuple_location = helper.random_location(positions, li, ls)

        x = tuple_location[0]
        y = tuple_location[1]
        supply_depot_xy = helper.validate_random_location(obs, x, y, positions, li, ls)

        if supply_depot_xy != False:
          distances = helper.get_distances(obs, scvs, supply_depot_xy)
          scv = scvs[np.argmin(distances)]

          return (actions.RAW_FUNCTIONS.Build_SupplyDepot_pt( "now", scv.tag, supply_depot_xy), 1, supply_depot_xy)
        else:
          return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
      else:
        return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
