from pysc2.lib import actions, features, units
import random
import numpy as np # Mathematical functions
import os

class BuildBarracks:
  def __init__(self):
    pass

  def build_barracks(self, obs, helper):
    #Selecciona el command Center
    unit_type = units.Terran.CommandCenter
    command_center_location = helper.get_command_center_location(obs, unit_type)
    completed_supply_depots = helper.get_my_completed_units_by_type(obs, units.Terran.SupplyDepot)
    barrackses = helper.get_my_units_by_type(obs, units.Terran.Barracks)
    scvs = helper.get_my_units_by_type(obs, units.Terran.SCV)

    if (len(completed_supply_depots) > 0
      and obs.observation.player.minerals >= 150 
      and len(scvs) > 0 
      and command_center_location is not None
      and len(barrackses) <= 50 ):

      #Identificar cuales son los rangos permitidos para X e Y
      li = -5
      ls =  5
      positions = (command_center_location.x ,command_center_location.y)
      tuple_location = helper.random_location(positions, li, ls)

      x = tuple_location[0]
      y = tuple_location[1]
      barracks_xy = helper.validate_random_location(obs, x, y, positions, li, ls)

      if barracks_xy != False:
        distances =  helper.get_distances(obs, scvs, barracks_xy)
        scv = scvs[np.argmin(distances)]
        return (actions.RAW_FUNCTIONS.Build_Barracks_pt("now", scv.tag, barracks_xy), 1, barracks_xy)
      else:
        return (actions.RAW_FUNCTIONS.no_op(), 0, (None,None))
    else:  
      return (actions.RAW_FUNCTIONS.no_op(), 0, (None,None))
