#https://liquipedia.net/starcraft2/Refinery_(Legacy_of_the_Void)
import os 
from pysc2.lib import actions, features, units
import random
import numpy as np

#1. Construye una refineria sobre el Gas Vespeno para su posterior recoleccion
class HarvestGas:
    def __init__(self):
        self.units = [
                       units.Neutral.VespeneGeyser,
                       units.Neutral.ProtossVespeneGeyser,
                       units.Neutral.PurifierVespeneGeyser,
                       units.Neutral.RichVespeneGeyser,
                       units.Neutral.ShakurasVespeneGeyser
                  ]

    def get_scv_harvest_gas(self, obs, helper):
        scvs = helper.get_my_units_by_type(obs, units.Terran.SCV)
        idle_scvs = [scv for scv in scvs if scv.order_length == 0]
        #print('gas_patch  --> idle_scvs: ', idle_scvs)
        completed_refineries = helper.get_my_completed_units_by_type(obs, units.Terran.Refinery)
        if len(idle_scvs) > 0 and len(completed_refineries) <= 16:
          gas_patches = [unit for unit in obs.observation.raw_units
                             if unit.unit_type in self.units ]

          scv = random.choice(idle_scvs)
          distances = helper.get_distances(obs, gas_patches, (scv.x, scv.y))
          gas_patch = gas_patches[np.argmin(distances)]
          #print('gas_patch(x,y): ', gas_patch.x, gas_patch.y)

          return (actions.RAW_FUNCTIONS.Build_Refinery_pt("now", scv.tag, gas_patch.tag), 1, (scv.x, scv.y))
        return (actions.RAW_FUNCTIONS.no_op(), 0, (None,None))
