from pysc2.lib import actions, features, units
import random

class BuildSCV:
    def __init__(self):
        pass

    def train_scv(self, obs, helper):
      #Selecciona el command Center
      command_center = helper.get_command_center_location(obs, units.Terran.CommandCenter)
      scvs           = helper.get_my_units_by_type(obs, units.Terran.SCV)
      
      if command_center is not None and len(scvs) <= 100:
          return ( actions.RAW_FUNCTIONS.Train_SCV_quick("now",command_center.tag), 1, (command_center.x, command_center.y) )
      else:
        return ( actions.RAW_FUNCTIONS.no_op(), 0, (None, None) )
