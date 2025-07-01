from pysc2.agents import base_agent
from pysc2.lib import actions, features, units

from libs.functions import Helper
from actions.set_actions import SetActions
import os

class TerranAgent(base_agent.BaseAgent):
    def __init__(self):
        super(TerranAgent, self).__init__()

        self.dict_policies = {}  
        self._set_policies()
        self.total_policies = len(self.dict_policies)

        self.helpers  = Helper()
        
        #Se crea la instancia con todos los objetos necesarios para la ejecucion de una accion determinada
        self.actions = SetActions()  

    def get_helpers(self):
        return self.helpers

    def get_actions(self):
        return self.actions

    def step(self, obs):
        super(TerranAgent, self).step(obs)

    def get_specific_action(self, obs, key):
        """
        Ejecuta una acción específica usando el sistema SetActions.
        
        Args:
            obs: Observación del estado actual del juego
            key (str): Nombre de la acción a ejecutar
            
        Returns:
            tuple: (acción, flag_éxito, posición) donde:
                - acción: Función RAW de PySC2 a ejecutar
                - flag_éxito: 1 si se puede ejecutar, 0 en caso contrario
                - posición: Coordenadas de la acción (x, y) o (None, None)
        """
        try:
            print(f"action: {key}")
            return self.actions.get_action(key, obs, self.helpers)
        except Exception as e:
            print(f"Error ejecutando acción {key}: {str(e)}")
            return (actions.FUNCTIONS.no_op(), 1, (None, None))

    def _set_policies(self):
        """
        Define un diccionario que mapea los nombres de políticas a listas de acciones.
        """
        # Diversidad de Estrategias: El conjunto cubre aspectos cruciales del juego
        # Enfoque Híbrido con Políticas Atómicas y Compuestas:
        self.dict_policies = {
            "policy_0":   ["do_nothing"], #Acción atómica para controlar

            # Economía y recolección (build_scv, harvest_minerals, harvest_gas_vespeno)
            "policy_1": ["build_scv"], #Acción atómica
            "policy_2": ["harvest_minerals"], #Acción atómica
            "policy_3": ["harvest_gas_vespeno"], #Acción atómica

            # Expansión (build_command_center, build_supply_depot, explore_csv)
            "policy_4": ["build_command_center","harvest_minerals","harvest_minerals"], #Construir base + optimizar el inicio
            "policy_5": ["explore_csv"], #Explorar es atómico y sirve de apoyo a EXPANSION
            "policy_6": ["build_supply_depot"], #Accion atomica para soporte

            # Infraestructura militar (build_barracks, build_bunker, build_tech_lab) y 
            # Producción de unidades (train_marine, train_marauder)         
            "policy_7": ["build_barracks"] + 5 * ["train_marine"], #Acción compuesta MILITARY_PRODUCTION
            "policy_8": ["build_bunker"],  #Acción atómica - Soporte defensivo
            "policy_9": ["build_tech_lab"]+ 2 * ["train_marauder","train_marauder"], #TECH_UPGRADE

            #Tácticas de combate (attack_with_marine, defense_with_marine, attack_with_marauder)
            "policy_10": 5 * ["attack_with_marine"], #Compuesta de ataque - ATTACK_MARINE
            "policy_11": 5 * ["defense_with_marine"],   #Compuesta de defensa - DEFENSE_MARINE
            "policy_12": 5 * ["attack_with_marine"] + 2 * ["attack_with_marauder"], #Compuesta de ataque - ATTACK_MARAUDER
            "policy_13": 3 * ["attack_with_marauder"] #Compuesta de ataque - ATTACK_MARAUDER
        }

    def get_specific_policy(self, num=-1, text=''):
        """
        Retorna una política específica en base a un índice numérico o una clave de texto.

        Parámetros:
          num (int): Identificador numérico de la política. Si num >= 0, se usa "policy_<num>".
          text (str): Alternativamente, se puede proporcionar directamente la clave de la política.

        Retorna:
          tuple: (clave_política, lista_de_acciones) si se encuentra la política, o None si no se especifica.
        """
        if num > -1:
            key = f"policy_{num}"
            return key, self.dict_policies.get(key)
        elif text:
            return text, self.dict_policies.get(text)
        else:
            return None

    def get_all_policies(self):
        key_tuple = ()
        for num in range(0, self.total_policies):
            key_tuple = key_tuple + ("policy_"+str(num),)
        return key_tuple