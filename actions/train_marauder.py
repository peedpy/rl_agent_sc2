from pysc2.lib import actions, features, units
import random
import numpy as np
import os

class TrainMarauder:
    def __init__(self):
        pass

    def train_marauder(self, obs, helper):
        completed_barrackses = helper.get_my_completed_units_by_type(obs, units.Terran.Barracks)
        free_supply = (obs.observation.player.food_cap - obs.observation.player.food_used)
        marauder = helper.get_my_units_by_type(obs, units.Terran.Marauder)  
        
        #print("len(completed_barrackses): ", len(completed_barrackses))
        #print("len(marauder): ", len(marauder))
        if (len(completed_barrackses) > 0 
            and obs.observation.player.minerals >= 100
            and obs.observation.player.vespene >= 25            
            and free_supply >= 2
            and len(marauder) <= 100
        ):
            #Se necesita identificar un barrack con tech lab
            target_point = ()
            found, target_point, barrack_tag = self.has_tech_lab(obs, completed_barrackses)
            if found:
                return (actions.RAW_FUNCTIONS.Train_Marauder_quick("now", barrack_tag), 1, target_point)
        return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))

    def has_tech_lab(self, obs, completed_barrackses):
        for barrack in completed_barrackses:
            for addon in obs.observation.raw_units:
                if addon.unit_type == units.Terran.BarracksTechLab and \
                    addon.alliance == features.PlayerRelative.SELF and \
                    abs(addon.x - barrack.x) < 5 and abs(addon.y - barrack.y) < 5:
                    print("La Barrack con posicion ({},{}) tiene un Tech Lab.".format(barrack.x, barrack.y))
                    return True, (barrack.x+1, barrack.y+1), barrack.tag
                    #os.system("pause")
        return False, (0,0), ''
