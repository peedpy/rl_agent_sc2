from pysc2.lib import actions, features, units
import random
import numpy as np # Mathematical functions
import os

class BuildTechLab:
    def __init__(self):
        self.positions = [] #Tendra las posiciones de los barracks que ya tienen el TechLab, para no volver a repetir

    def build_tech_lab(self, obs, helper):
        completed_barrackses = helper.get_my_completed_units_by_type(obs, units.Terran.Barracks)
        scvs = helper.get_my_units_by_type(obs, units.Terran.SCV)
        completed_barrack_tech_lab = helper.get_my_completed_units_by_type(obs, units.Terran.BarracksTechLab)

        if (len(completed_barrackses) > 0
            and obs.observation.player.minerals >= 50
            and obs.observation.player.vespene >= 25
            and len(completed_barrack_tech_lab) <= 50):
   
            target_point = ()
            success, target_point, barrack_tag = self.has_tech_lab(obs, completed_barrackses)
            if success:
                distances =  helper.get_distances(obs, scvs, target_point)
                if len(distances) > 0:
                    scv = scvs[np.argmin(distances)]
                    return (actions.RAW_FUNCTIONS.Build_TechLab_Barracks_pt("now", [barrack_tag], target_point), 1, target_point)  
        return (actions.RAW_FUNCTIONS.no_op(), 0, (None,None))


    def has_tech_lab(self, obs, completed_barrackses):
        for barrack in completed_barrackses:
            for addon in obs.observation.raw_units:
                if addon.unit_type == units.Terran.BarracksTechLab and \
                    addon.alliance == features.PlayerRelative.SELF and \
                    abs(addon.x - barrack.x) < 5 and abs(addon.y - barrack.y) < 5:
                    #print("La Barrack con posicion ({},{}) tiene un Tech Lab.".format(barrack.x, barrack.y))
                    #os.system("pause")
                    #return True
                    break
                else:
                    #print("La Barrack con posicion ({},{}) es la seleccionada.".format(barrack.x, barrack.y))
                    return True, (barrack.x+1, barrack.y+1), barrack.tag
        return False, (0,0), None
