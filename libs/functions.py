from pysc2.lib import features, units
import numpy as np # Mathematical functions
import random
import os

class Helper:
    def __init__(self):
      self.used_positions = []

      self.total_send_units = 0
      self.position = 0
      
      #Todas las acciones que necesitan ubicarse dentro del mapa utilizaran estos dos datos generados por el estado
      self.base_top_left = None
      self.quadrant = 0
      self.list_quadrants  = [0,0,0,0]

      #Sirve para la construccion de command center
      self.last_position = (0,0)

    def get_enemy_units_by_type(self, obs, unit_type):
      return [unit for unit in obs.observation.raw_units
              if unit.unit_type == unit_type 
              and unit.alliance == features.PlayerRelative.ENEMY]

    def get_quadrant(self, x, y):
        if x < 32:
            return [1,0,0,0] if y < 32 else [0,0,1,0]
        else:
            return [0,1,0,0] if y < 32 else [0,0,0,1]

    def reset_list_quadrants(self):
        self.list_quadrants  = [0,0,0,0]

    def get_calculate_hot_zone(self, obs):
        # Reiniciar la lista de cuadrantes (si es que se usa para otros fines)
        self.reset_list_quadrants()

        # Conjunto de tipos de unidades de interés
        desired_types = {
            units.Terran.SCV, 
            units.Terran.Marine,
            units.Terran.Marauder,
            units.Terran.Reaper,
            units.Terran.Raven, 
            units.Terran.Medivac,
            units.Terran.SupplyDepot, 
            units.Terran.Barracks, 
            units.Terran.CommandCenter, 
            units.Terran.Refinery,
            units.Terran.Starport,
            units.Terran.Armory,
            units.Terran.OrbitalCommand
        }

        # Diccionario para mapear unit_type a un nombre legible
        UNIT_NAMES = {
            units.Terran.SCV: "SCV",
            units.Terran.Marine: "Marine",
            units.Terran.Marauder: "Marauder",
            units.Terran.Reaper: "Reaper",
            units.Terran.Raven: "Raven",
            units.Terran.Medivac: "Medivac",
            units.Terran.SupplyDepot: "Supply Depot",
            units.Terran.Barracks: "Barracks",
            units.Terran.CommandCenter: "Command Center",
            units.Terran.Refinery: "Refinery",
            units.Terran.Starport: "Starport",
            units.Terran.Armory: "Armory",
            units.Terran.OrbitalCommand: "Orbital Command"
        }

        # Conjunto de tipos considerados estructuras
        structure_types = {
            units.Terran.SupplyDepot,
            units.Terran.Barracks,
            units.Terran.CommandCenter,
            units.Terran.Refinery,
            units.Terran.Starport,
            units.Terran.Armory,
            units.Terran.OrbitalCommand
        }

        # Filtrar todas las unidades enemigas de interés
        enemy_units = [
            unit for unit in obs.observation.raw_units 
            if unit.alliance == features.PlayerRelative.ENEMY and unit.unit_type in desired_types
        ]
        print(f"Total enemy units of interest: {len(enemy_units)}")

        # Inicializar acumuladores de cuadrantes para cada grupo
        army_quadrants = np.array([0, 0, 0, 0])
        structure_quadrants = np.array([0, 0, 0, 0])
        
        # Listas para almacenar las posiciones con el nombre de la unidad
        army_positions = []      # Para unidades de ejército
        structure_positions = [] # Para estructuras

        # Procesar cada unidad enemiga
        for enemy in enemy_units:
            x, y = enemy.x, enemy.y
            enemy_name = UNIT_NAMES.get(enemy.unit_type, "Unknown")
            print(f"Enemy: {enemy_name:<20} --> (x,y): ({x:3d}, {y:3d})")
            
            # Calcular el cuadrante para esta unidad
            quadrant = self.get_quadrant(x, y)
            
            # Clasificar la unidad en ejército o estructura y acumular el cuadrante
            if enemy.unit_type in structure_types:
                structure_quadrants = np.add(structure_quadrants, quadrant)
                structure_positions.append((enemy_name, (x, y)))
            else:
                army_quadrants = np.add(army_quadrants, quadrant)
                army_positions.append((enemy_name, (x, y)))

        # Determinar el cuadrante con mayor/menor concentración para cada grupo
        max_army_quadrant_index = int(np.argmax(army_quadrants))
        max_structure_quadrant_index = int(np.argmax(structure_quadrants))

        min_army_quadrant_index = int(np.argmin(army_quadrants))
        min_structure_quadrant_index = int(np.argmin(structure_quadrants))

        quadrant_names = {
            0: "Cuadrante 1 (Superior Izquierdo)",
            1: "Cuadrante 2 (Superior Derecho)",
            2: "Cuadrante 3 (Inferior Izquierdo)",
            3: "Cuadrante 4 (Inferior Derecho)"
        }
        
        print(f"Army quadrants          : {army_quadrants}")
        print(f"Structure quadrants     : {structure_quadrants}")
        print(f"Quadrant with highest army concentration      : {quadrant_names.get(max_army_quadrant_index, 'Unknown')}")
        print(f"Quadrant with highest structure concentration : {quadrant_names.get(max_structure_quadrant_index, 'Unknown')}")
        print(f"Quadrant with lowest army concentration       : {quadrant_names.get(min_army_quadrant_index, 'Unknown')}")
        print(f"Quadrant with lowest structure concentration  : {quadrant_names.get(min_structure_quadrant_index, 'Unknown')}")

        # Convertir las listas de posiciones a tuplas para la salida final
        army_positions = tuple(army_positions)
        structure_positions = tuple(structure_positions)
        
        # Retorna dos cosas: 
        # 1) Una tupla con las posiciones (nombre y coordenadas) de unidades de ejército y estructuras.
        # 2) Una tupla con el índice del cuadrante con mayor concentración para army y para structures.
        # 2) Una tupla con el índice del cuadrante con menor concentración para army y para structures.
        return army_positions, structure_positions, (max_army_quadrant_index, max_structure_quadrant_index)#, (min_army_quadrant_index, min_structure_quadrant_index)

###############################################################################################





    def set_last_explore_position(self, last_position):
        self.last_position = last_position

    def get_last_explore_position(self):
        return self.last_position

    def initialize_used_positions(self):
      self.used_positions = []
    
    def get_used_positions(self):
      return self.used_positions

    def get_my_units_by_type(self, obs, unit_type):
      return [unit for unit in obs.observation.raw_units
              if unit.unit_type == unit_type 
              and unit.alliance == features.PlayerRelative.SELF]
  


    def get_my_completed_units_by_type(self, obs, unit_type):
      return [unit for unit in obs.observation.raw_units
              if unit.unit_type == unit_type 
              and unit.build_progress == 100
              and unit.alliance == features.PlayerRelative.SELF]

    def get_enemy_completed_units_by_type(self, obs, unit_type):
      return [unit for unit in obs.observation.raw_units
              if unit.unit_type == unit_type 
              and unit.build_progress == 100
              and unit.alliance == features.PlayerRelative.ENEMY]

    def get_distances(self, obs, units, xy):
        units_xy = [(unit.x, unit.y) for unit in units]
        if len(units_xy) > 0:
          #print("units_xy: ", units_xy)
          #print("xy: ", xy)
          return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1) # Normalize the array
        return []

    def get_command_center_top_left(self, obs):
        #Selecciona el command Center
        unit_type = units.Terran.CommandCenter
        command_center = self.get_my_units_by_type(obs, unit_type)
        if len(command_center):
           cc = command_center[0]
           self.base_top_left = (cc.x < 32)
        return self.base_top_left

    def get_command_center_location(self, obs, unit_type):
        #print(70*"_-")
        #print("get_command_center_location:")
        unit_type = units.Terran.CommandCenter
        command_centers = self.get_my_units_by_type(obs, unit_type)
        
        #print("len(command_centers): {}".format(len(command_centers)))
        if len(command_centers) > 0:
            #Eleccion aleatoria entre todos los command centers
            random_ = random.randint(0, len(command_centers) - 1)
            command_center = command_centers[random_]

            #print("random_: {}".format(random_))
            #print("command_center: {}".format(command_center))
            #os.system("pause")
            if command_center.order_length <= 10:
                #print("command_center: ({}, {})".format(command_center.x, command_center.y))
                return command_center
            else:
               return None
        
        return None

    def get_base_top_left(self):
        return self.base_top_left

    def set_base_top_left(self):
        self.base_top_left = None

    #Permite generar una posicion cerca del command center
    #unit: build_supply_depot            ---->             self.used_positions_buildSupplyDepot
    def random_location(self, positions=(0,0), li=0, ls=0):
        #Identificar cuales son los rangos permitidos para X e Y
        x = positions[0] + random.randint(li, ls)
        y = positions[1] + random.randint(li, ls)
        return (x,y)

    def validate_random_location(self,obs, x, y, positions=(0,0), li=0, ls=0):
        #Generar un array con todas las posiciones de mis unidades
        positions_list = self.get_units_positions(obs)
        #print(150*"/")
        #print("def validate_random_location")
        #print("POSICIONES OCUPADAS: ", positions)
        for i in range(10):
            print('---> Posicion a ocupar: (',x,',',y,')')
            if (x, y) not in positions_list:
                #Como esta posicion a ocupar no existe en la lista de ocupados lo usare
                return (x,y)
            else:
                #print(i,':   USADO EN UNA BUSQUEDA ANTERIOR')
                tuple_location = self.random_location((x,y), li, ls)
                x,y = tuple_location[0], tuple_location[1]
        
        tuple_location = self.random_location((x+1,y+1), li, ls)
        x,y = tuple_location[0], tuple_location[1]

        if x < 0:
            x*=-1
        if y < 0:
            y*=-1
        return (x,y)

    ###A probarrrrrrrr no se usaaaaaaaaaaaaaaaaaaaaaaaaaa
    def validate_random_location_(self, obs, x, y, positions=(0,0), li=0, ls=0):
        #Para determinar por cada paso de tiempo las posiciones de mis unidades estaticas
        for unit in [   units.Terran.SupplyDepot, 
                        units.Terran.Barracks, 
                        units.Terran.CommandCenter]:
            my_unit = self.get_my_units_by_type(obs, unit)
            #print('\n----:'+str(unit))
            #print('enemy type: ', type(enemy_unit))
            longitud = len(my_unit)
            #print('my_unit total: ', longitud)
            if longitud > 0:
                for my in my_unit:
                    xx = my.x
                    yy = my.y
                    if (x, y) in (xx, yy):
                        tuple_location = self.random_location(positions, li, ls)
                        x,y = tuple_location[0], tuple_location[1]
            else:
                return (x,y)

            #print('self.list_quadrants    : ', self.list_quadrants)
        return self.list_quadrants
     
    def select_rand_quadrant(self, base_top_left):
        if self.base_top_left == True:
            rand = random.randint(2, 4)  #3 cuadrantes --- 2,3,4
        elif self.base_top_left == False:
            rand = random.randint(1, 3)  #3 cuadrantes --- 1,2,3
        #print('select_rand()')
        #print('rand                    : {}'.format(rand))
        #print('base_top_left           : {}'.format(base_top_left))
        return rand

    def get_xy(self):
        #Aveces explota, pero no identifique todavia el porque, por eso agregue esta porcion de code
        if self.base_top_left == True and self.quadrant == 1:
            self.quadrant = self.select_rand_quadrant(self.base_top_left)
        elif self.base_top_left == False and self.quadrant == 4:
            self.quadrant = self.select_rand_quadrant(self.base_top_left)
            
        if self.base_top_left == True and self.quadrant != 0:
            if self.quadrant == 2: #RIGHT HIGHER   31<= x <=63     0<= y <=31 
                x = 39
                y = 23
            elif self.quadrant == 3: #LL = LEFT LOWER      0<= x <=31    32<= y <=63 
                x = 19
                y = 44
            elif self.quadrant == 4: #RL = RIGHT LOWER    32<= x <=63    32<= y <=63
                x = 39
                y = 44
        elif self.base_top_left == False and self.quadrant != 0:
            if self.quadrant == 1:   #LH = LEFT HIGHER     0<= x <=31     0<= y <=31
                x = 19
                y = 23
            elif self.quadrant == 2: #RIGHT HIGHER   31<= x <=63     0<= y <=31 
                x = 39
                y = 23
            elif self.quadrant == 3: #LL = LEFT LOWER      0<= x <=31    32<= y <=63
                x = 19
                y = 44
        else: #Por defecto
            x = 30
            y = 30

        return (x,y)

    #Detectar los cuadrantes donde mas cantidad de enemigos hay
    #def hot_space(self, obs, unit_type):
        #self.get_enemy_units_by_type(self, )
    
      
    def get_found_positions_to_attack(self):
        quadrant_index = 0
        if sum(self.list_quadrants) > 0:
            #Encontrar el mayor de los cuadrantes
            max_ = max(self.list_quadrants)
            quadrant_index = int(np.where(self.list_quadrants == max_)[0][0] + 1)
            #Buscar por cuadrante las posiciones estaticas, del punto  medio
            if quadrant_index == 1:
                positions = (16,16)
            elif quadrant_index == 2:
                positions = (48,16)
            elif quadrant_index == 3:
                positions = (16,48)
            elif quadrant_index == 4:
                positions = (48,48)
        else:
            base_top_left = self.get_base_top_left()
            if base_top_left:
                positions = (38, 44)
            else:
                positions = (19, 23)
        
        #print("Quadrant win                     : ", quadrant_index)
        #print("Position to attack with Marine   : ", positions)
        return positions
    

    
    """
    Sirve para obtener por cada paso de tiempo la ubicacion de cada unidad creada,
    para asi no utilizar esa informacion para ubicar una proxima unidad
    """
    def get_units_positions(self, obs):
        positions = []
        for unit in [   units.Terran.SCV, 
                        units.Terran.Marine,
                        units.Terran.SupplyDepot, 
                        units.Terran.Barracks, 
                        units.Terran.CommandCenter]:

            my_unit = self.get_my_units_by_type(obs, unit)
            #print('\n----:'+str(unit))
            #print('my type: ', type(my_unit))
            longitud = len(my_unit)
            #print('my_unit total: ', longitud)
            if longitud > 0:
                for my in my_unit:
                    x = my.x
                    y = my.y
                    #print('my[i] --> (x,y): ','(',x, y,')')
                    positions.append((x,y))
        return positions

    def get_terran_unit(self, unit_type, army_label=''):
        unit_type_to_name__that_marine_will_attack = {
            18: "CommandCenter",
            19: "Barracks",
            20: "EngineeringBay",
            21: "Barracks",
            22: "EngineeringBay",
            23: "MissileTurret",
            24: "Bunker",
            25: "SensorTower",
            26: "GhostAcademy",
            27: "Factory",
            28: "Starport",
            29: "Armory",
            30: "FusionCore",
            31: "AutoTurret",
            32: "SiegeTankSieged",
            33: "SiegeTank",
            34: "VikingAssault",
            35: "VikingFighter",
            36: "CommandCenterFlying",
            37: "BarracksTechLab",
            38: "BarracksReactor",
            39: "FactoryTechLab",
            40: "FactoryReactor",
            41: "StarportTechLab",
            42: "StarportReactor",
            43: "FactoryFlying",
            44: "StarportFlying",
            45: "SCV",
            46: "BarracksFlying",
            47: "SupplyDepotLowered",
            48: "Marine",
            49: "Reaper",
            50: "Ghost",
            51: "Marauder",
            52: "Thor",
            53: "Hellion",
            54: "Medivac",
            55: "Banshee",
            56: "Raven",
            57: "Battlecruiser",
            58: "Nuke",
            130: "PlanetaryFortress",
            132: "OrbitalCommand",
            134: "OrbitalCommandFlying",
            144: "GhostAlternate",
            145: "GhostNova",
            268: "MULE",
            484: "Hellbat",
            498: "WidowMine",
            500: "WidowMineBurrowed",
            692: "Cyclone",
            734: "LiberatorAG",
            689: "Liberator",
            830: "KD8Charge",
            1913: "RepairDrone",
            1960: "RefineryRich"
        }

        #Marauder estara modo defensivo
        unit_type_to_name__that_marauder_will_attack = {
            49: "Reaper",
            50: "Ghost",
            51: "Marauder",
            48: "Marine",
            54: "Medivac",
            56: "Raven",
            32: "SiegeTankSieged",
            33: "SiegeTank",
            52: "Thor",
            53: "Hellion",
            268: "MULE",  #Similar al SCV, sirve para recolectar minerales
            45: "SCV",
        }

        unit_name = ""
        if army_label == 'marine_attack' or 'marine_defense':
            unit_name = unit_type_to_name__that_marine_will_attack.get(unit_type, "Unknown")
        elif army_label == 'marauder':
            unit_name = unit_type_to_name__that_marauder_will_attack.get(unit_type, "Unknown")
        else:
            unit_name = unit_type_to_name__that_marine_will_attack.get(unit_type, "Unknown")

        return unit_name


    """
    En el entorno de PySC2, las unidades enemigas se identifican por su etiqueta (tag), que es un número único asignado a cada unidad en el juego.
    En StarCraft 2, solo puedes detectar unidades enemigas dentro de tu rango de visión. 
    Si una unidad enemiga está fuera de tu rango de visión, no recibirás actualizaciones sobre su posición ni sobre su existencia.

    """
    def get_detectable_enemy_units(self, obs, enemy_tag_units):
        states = obs.observation.raw_units
        enemy_units = [unit for unit in states if unit.alliance == features.PlayerRelative.ENEMY]
        for unit in enemy_units:
            unit_name = self.get_terran_unit(unit.unit_type, '')
            print("unit.unit_type: {:10}  --  unit.name : {:30} ".format(unit.unit_type, unit_name))

            if unit.tag not in enemy_tag_units:
                enemy_tag_units.append(unit.tag)
        
        length = len(enemy_tag_units)
        print(f"detectable_enemy_units length: {length}")

        return enemy_tag_units

            






