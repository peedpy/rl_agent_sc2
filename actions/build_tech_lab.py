"""
Módulo para la construcción de laboratorios tecnológicos en StarCraft II.

Los laboratorios tecnológicos son adiciones a los cuarteles que permiten
entrenar unidades avanzadas como Marauders y Marauders. Este módulo maneja
la lógica para construir laboratorios tecnológicos en cuarteles existentes.

Referencias:
- https://liquipedia.net/starcraft2/Tech_Lab
- https://liquipedia.net/starcraft2/Barracks

Autor: Pablo Escobar
Fecha: 2022
"""

import logging
from pysc2.lib import actions, features, units
import random
import numpy as np # Mathematical functions
import os

# Configuración del logger para este módulo
logger = logging.getLogger(__name__)

class BuildTechLab:
    """
    Clase responsable de la construcción de laboratorios tecnológicos Terran.
    
    Los laboratorios tecnológicos son adiciones a los cuarteles que permiten:
    - Entrenar unidades avanzadas (Marauders, Marauders)
    - Investigar mejoras tecnológicas
    - Mejorar la capacidad de combate del ejército
    
    Esta clase maneja la lógica para construir laboratorios tecnológicos
    en cuarteles existentes de manera estratégica.
    
    Attributes:
        TECH_LAB_COST (int): Costo en minerales para construir un laboratorio tecnológico (50)
        MAX_TECH_LABS (int): Límite máximo de laboratorios tecnológicos permitidos (20)
    """
    
    # Constantes de configuración
    TECH_LAB_COST = 50
    MAX_TECH_LABS = 20
    
    def __init__(self):
        """
        Inicializa la clase BuildTechLab.
        
        No requiere parámetros de inicialización ya que toda la lógica
        se ejecuta en el método build_tech_lab.
        """
        logger.debug("Inicializando módulo BuildTechLab")
        self.positions = [] #Tendra las posiciones de los barracks que ya tienen el TechLab, para no volver a repetir

    def build_tech_lab(self, obs, helper):
        """
        Construye un laboratorio tecnológico en un cuartel existente.
        
        Esta función implementa la lógica para construir laboratorios tecnológicos de manera eficiente:
        1. Verifica que haya suficientes recursos (minerales y gas)
        2. Comprueba que haya cuarteles completados disponibles
        3. Controla que no se haya alcanzado el límite de laboratorios tecnológicos
        4. Selecciona un cuartel sin laboratorio tecnológico usando la función has_tech_lab
        5. Ejecuta la construcción del laboratorio tecnológico
        
        Args:
            obs: Observación del estado actual del juego
            helper: Objeto Helper con funciones auxiliares
            
        Returns:
            tuple: (acción, flag_éxito, posición) donde:
                - acción: Función RAW de PySC2 a ejecutar
                - flag_éxito: 1 si se puede ejecutar, 0 en caso contrario
                - posición: Coordenadas del cuartel (x, y) o (None, None)
                
        Example:
            >>> build_tech_lab = BuildTechLab()
            >>> action, success, pos = build_tech_lab.build_tech_lab(obs, helper)
            >>> if success:
            ...     print(f"Construyendo laboratorio tecnológico en cuartel en posición {pos}")
        """
        logger.debug("Iniciando proceso de construcción de laboratorio tecnológico")
        
        # Obtener cuarteles completados
        completed_barrackses = helper.get_my_completed_units_by_type(obs, units.Terran.Barracks)
        if len(completed_barrackses) == 0:
            logger.debug("No hay cuarteles completados disponibles para construir laboratorio tecnológico")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        
        # Verificar recursos disponibles
        minerals = obs.observation.player.minerals
        vespene = obs.observation.player.vespene
        
        if minerals < self.TECH_LAB_COST:
            logger.debug(f"Minerales insuficientes para laboratorio tecnológico: {minerals}/{self.TECH_LAB_COST}")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        
        if vespene < 25:  # Costo en gas vespene
            logger.debug(f"Gas vespene insuficiente para laboratorio tecnológico: {vespene} de 25")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        
        # Verificar límite de laboratorios tecnológicos
        completed_barrack_tech_lab = helper.get_my_completed_units_by_type(obs, units.Terran.BarracksTechLab)
        current_tech_labs_count = len(completed_barrack_tech_lab)
        if current_tech_labs_count >= self.MAX_TECH_LABS:
            logger.info(f"Límite de laboratorios tecnológicos alcanzado ({self.MAX_TECH_LABS}). No se puede construir más.")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        
        logger.debug(f"Laboratorios tecnológicos actuales: {current_tech_labs_count}/{self.MAX_TECH_LABS}")
        logger.debug(f"Cuarteles completados disponibles: {len(completed_barrackses)}")
        
        # Usar la función has_tech_lab para encontrar un cuartel disponible
        success, target_point, barrack_tag = self.has_tech_lab(obs, completed_barrackses)
        
        if success:
            # Encontrar SCV más cercano para la construcción
            scvs = helper.get_my_units_by_type(obs, units.Terran.SCV)
            if len(scvs) > 0:
                distances = helper.get_distances(obs, scvs, target_point)
                if distances is not None and np.size(distances) > 0:
                    scv = scvs[np.argmin(distances)]
                    distance = min(distances)
                    
                    logger.info(f"Construyendo laboratorio tecnológico en cuartel en posición {target_point}")
                    logger.debug(f"SCV asignado en ({scv.x}, {scv.y}), distancia: {distance:.2f}")
                    
                    # Ejecutar construcción
                    try:
                        action = actions.RAW_FUNCTIONS.Build_TechLab_Barracks_pt("now", [barrack_tag], target_point)
                        return (action, 1, target_point)
                        
                    except Exception as e:
                        logger.error(f"Error al construir laboratorio tecnológico: {str(e)}")
                        return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
                else:
                    logger.warning("No se pudieron calcular distancias a SCVs")
                    return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
            else:
                logger.debug("No hay SCVs disponibles para construir laboratorio tecnológico")
                return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))
        else:
            logger.debug("No se encontró cuartel disponible sin laboratorio tecnológico")
            return (actions.RAW_FUNCTIONS.no_op(), 0, (None, None))

    def has_tech_lab(self, obs, completed_barrackses):
        """
        Verifica si un cuartel tiene laboratorio tecnológico y encuentra uno disponible.
        
        Esta función busca entre los cuarteles completados para encontrar uno que
        no tenga laboratorio tecnológico, verificando las unidades en el mapa.
        
        Args:
            obs: Observación del estado actual del juego
            completed_barrackses: Lista de cuarteles completados
            
        Returns:
            tuple: (éxito, posición, tag_cuartel) donde:
                - éxito: True si se encontró un cuartel disponible
                - posición: Coordenadas (x, y) para construir el laboratorio
                - tag_cuartel: Tag del cuartel seleccionado
        """
        logger.debug("Verificando cuarteles disponibles para laboratorio tecnológico")
        
        for barrack in completed_barrackses:
            has_tech_lab = False
            
            # Verificar si el cuartel ya tiene laboratorio tecnológico
            for addon in obs.observation.raw_units:
                if (addon.unit_type == units.Terran.BarracksTechLab and 
                    addon.alliance == features.PlayerRelative.SELF and 
                    abs(addon.x - barrack.x) < 5 and abs(addon.y - barrack.y) < 5):
                    
                    logger.debug(f"Cuartel en ({barrack.x}, {barrack.y}) ya tiene laboratorio tecnológico")
                    has_tech_lab = True
                    break
            
            # Si no tiene laboratorio tecnológico, este cuartel está disponible
            if has_tech_lab == False:
                target_position = (barrack.x + 1, barrack.y + 1)
                logger.debug(f"Cuartel en ({barrack.x}, {barrack.y}) disponible para laboratorio tecnológico")
                return True, target_position, barrack.tag
        
        logger.debug("No se encontraron cuarteles disponibles sin laboratorio tecnológico")
        return False, (0, 0), None
