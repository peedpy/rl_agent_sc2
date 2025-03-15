from pysc2.lib import actions
from actions.build_scv import BuildSCV
from actions.harvest_minerals import HarvestMinerals
from actions.build_supply_depot import SupplyDepot
from actions.build_barracks import BuildBarracks
from actions.train_marine import TrainMarine
from actions.attack_army import AttackArmy
from actions.build_command_center import BuildCommandCenter
from actions.explore_csv import ExploreCSV
from actions.harvest_gas_vespeno import HarvestGas
from actions.build_tech_lab import BuildTechLab
from actions.build_bunker import BuildBunker
from actions.train_marauder import TrainMarauder

class Action:
	def __init__(self):
		self.do_nothing         	= actions.RAW_FUNCTIONS.no_op()
		self.build_scv          	= BuildSCV()
		self.harvest_minerals   	= HarvestMinerals()
		self.build_supply_depot 	= SupplyDepot()
		self.build_barracks     	= BuildBarracks()
		self.train_marine       	= TrainMarine()
		self.attack_army      		= AttackArmy()
		self.build_command_center   = BuildCommandCenter()
		self.explore_csv   			= ExploreCSV()
		self.harvest_gas 			= HarvestGas()
		self.build_tech_lab 		= BuildTechLab()
		self.build_bunker 			= BuildBunker()
		self.train_marauder         = TrainMarauder()

	def get_object_actions(self):
		return {
			'do_nothing': self.do_nothing,
			'build_scv': self.build_scv,
			'harvest_minerals': self.harvest_minerals,
			'build_supply_depot': self.build_supply_depot,
			'build_barracks': self.build_barracks,
			'train_marine': self.train_marine,
			'attack_with_marine': self.attack_army,
			'defense_with_marine': self.attack_army,
			'attack_with_marauder': self.attack_army,
			'build_command_center': self.build_command_center,
			'explore_csv': self.explore_csv,
			'harvest_gas': self.harvest_gas,
			'build_tech_lab': self.build_tech_lab,
			'build_bunker': self.build_bunker,
			'train_marauder': self.train_marauder
		}
