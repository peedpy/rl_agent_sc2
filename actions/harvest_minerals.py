from pysc2.lib import actions, features, units
import random
import numpy as np

#Tener dos trabajadores por campo mineral generalmente se considera óptimo.
#Cuando hay más de dos trabajadores disponibles por campo mineral, generalmente es mejor agregar base(s) adicional(es).
#https://liquipedia.net/starcraft2/Resources#Supply
#Con dos trabajadores por campo de minerales, una base con 8 campos de minerales cosechará alrededor de 925 minerales por minuto.
class HarvestMinerals:
  def __init__(self):
    self.units = [
      units.Neutral.BattleStationMineralField,
      units.Neutral.BattleStationMineralField750,
      units.Neutral.LabMineralField,
      units.Neutral.LabMineralField750,
      units.Neutral.MineralField,
      units.Neutral.MineralField750,
      units.Neutral.PurifierMineralField,
      units.Neutral.PurifierMineralField750,
      units.Neutral.PurifierRichMineralField,
      units.Neutral.PurifierRichMineralField750,
      units.Neutral.RichMineralField,
      units.Neutral.RichMineralField750,
      units.Neutral.VespeneGeyser,
      units.Neutral.ProtossVespeneGeyser,
      units.Neutral.PurifierVespeneGeyser,
      units.Neutral.RichVespeneGeyser,
      units.Neutral.ShakurasVespeneGeyser
    ]

  def get_scv_harvest_minerals(self, obs, helper):
    scvs = helper.get_my_units_by_type(obs, units.Terran.SCV)
    free_supply = [scv for scv in scvs if scv.order_length == 0]

    if len(free_supply) > 0:
      mineral_patches = [unit for unit in obs.observation.raw_units if unit.unit_type in self.units ]

      scv = random.choice(free_supply)
      distances = helper.get_distances(obs, mineral_patches, (scv.x, scv.y))
      mineral_patch = mineral_patches[np.argmin(distances)] 

      return (actions.RAW_FUNCTIONS.Harvest_Gather_unit("now", scv.tag, mineral_patch.tag), 1, (scv.x, scv.y))
    return (actions.RAW_FUNCTIONS.no_op(), 0, (None,None))
