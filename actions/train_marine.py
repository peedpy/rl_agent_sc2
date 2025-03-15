from pysc2.lib import actions, features, units
import random
import numpy as np

class TrainMarine:
    def __init__(self):
        pass

    def train_marine(self, obs, helper):
      completed_barrackses = helper.get_my_completed_units_by_type(obs, units.Terran.Barracks)
      free_supply = (obs.observation.player.food_cap - obs.observation.player.food_used)
      marines     = helper.get_my_units_by_type(obs, units.Terran.Marine)

      if (len(completed_barrackses) > 0
        and obs.observation.player.minerals >= 50
        and free_supply > 0
        and len(marines) <= 200):

        barracks = helper.get_my_units_by_type(obs, units.Terran.Barracks)
        barrack = random.choice(barracks)

        #Se puede entrenar un marino si el barrack se encuentra con no mas de hasta max 10 ordenes pendientes
        if barrack.order_length <= 10:
          return (actions.RAW_FUNCTIONS.Train_Marine_quick("now", barrack.tag), 1, (barrack.x, barrack.y))
        else:
          return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
      return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
