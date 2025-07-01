"""
Script principal para ejecutar el agente Q-Learning contra IA en StarCraft II.

Este script configura el entorno de PySC2 y ejecuta el agente Q-Learning
contra un bot de dificultad muy fácil para entrenamiento y evaluación.

Autor: Pablo Escobar
Fecha: 2022
"""

import logging
from agent_qlearning import AgentQlearning
from absl import app, flags
from pysc2.env import sc2_env, run_loop
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
import os

# Importar logging centralizado al inicio
from libs.logging_config import get_logger

# Configurar logger para este módulo
logger = get_logger('main_qlearning')

def main(unused_argv):
    """
    Función principal que ejecuta el agente Q-Learning.
    
    Configura el entorno de StarCraft II y ejecuta el agente en modo de entrenamiento
    contra un bot de dificultad muy fácil.
    """
    # Configurar logging
    logger.info("Iniciando agente Q-Learning vs IA")
    
    # Configuración del agente
    train_mode = True
    step_mul = 16  # Este valor permite que el agente tome decisiones con mayor frecuencia
    
    logger.info(f"Modo de entrenamiento: {train_mode}")
    logger.info(f"Step multiplier: {step_mul}")
    
    # Crear instancia del agente
    agent = AgentQlearning(step_mul, train_mode)
    logger.info("Agente Q-Learning inicializado correctamente")
    
    # Bucle principal de entrenamiento
    while True:
        try:
            logger.info("Iniciando nuevo episodio de entrenamiento")
            
            with sc2_env.SC2Env(
                map_name="Simple64",
                players=[
                    sc2_env.Agent(sc2_env.Race.terran, 'Q-learning'),
                    sc2_env.Bot(sc2_env.Race.terran, sc2_env.Difficulty.medium)
                ],
                agent_interface_format=features.AgentInterfaceFormat(
                    action_space=actions.ActionSpace.RAW,
                    use_raw_units=True,
                    raw_resolution=64,
                ),
                step_mul=step_mul,  # Velocidad de ejecución del juego
                disable_fog=False,
            ) as env:
                logger.info("Entorno SC2 configurado correctamente")
                logger.info(f"Mapa: Simple64, Raza: Terran, Dificultad: Very Easy")
                
                # Ejecutar bucle de entrenamiento
                run_loop.run_loop([agent], env, max_episodes=1000)
                
        except KeyboardInterrupt:
            logger.info("Entrenamiento interrumpido por el usuario")
            break
        except Exception as e:
            logger.error(f"Error durante el entrenamiento: {str(e)}")
            logger.info("Reintentando en 5 segundos...")
            import time
            time.sleep(5)
            continue

if __name__ == "__main__":
    app.run(main)
