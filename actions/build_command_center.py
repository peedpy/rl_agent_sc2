from pysc2.lib import actions, features, units
import random
import numpy as np
import os

class BuildCommandCenter:
  def __init__(self):
      pass

  def build_command_center(self, obs, helper):
    scvs = helper.get_my_units_by_type(obs, units.Terran.SCV)
    free_scvs = [scv for scv in scvs if scv.order_length == 0]
    cc = helper.get_my_completed_units_by_type(obs, units.Terran.CommandCenter)

    if len(free_scvs) > 0 and len(cc) <= 50:
      scv = random.choice(free_scvs)

      quadrant = random.randint(1, 3)

      #target_position = helper.get_last_explore_position()
      #if target_position == (0,0):
      if helper.get_base_top_left():
        if quadrant == 1:   #LH = LEFT HIGHER    0<= x <=31     0<= y <=31
            x = 19
            y = 23
        elif quadrant == 2: #RH = RIGHT HIGHER        31<= x <=63     0<= y <=31 
            x = 39
            y = 23
        elif quadrant == 3: #LL = LEFT LOWER          0<= x <=31    32<= y <=63 
            x = 19
            y = 44
      else:
        if quadrant == 1: #RL = RIGHT LOWER         32<= x <=63    32<= y <=63
            x = 39
            y = 44
        elif quadrant == 2: #LL = LEFT LOWER          0<= x <=31    32<= y <=63 
            x = 19
            y = 44
        elif quadrant == 3: #RH = RIGHT HIGHER        31<= x <=63     0<= y <=31 
            x = 39
            y = 23
      target_location = helper.random_location((x,y), -5, 5)

      #os.system("pause")
      try:
        return (actions.RAW_FUNCTIONS.Build_CommandCenter_pt("now", scv.tag, target_location), 1, target_location)
      except Exception as e:
        return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
    else:
      return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
