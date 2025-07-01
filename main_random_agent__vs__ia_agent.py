# Importar logging centralizado al inicio
from libs.logging_config import get_logger

# Configurar logger para este módulo
logger = get_logger('main_random')

from agent_random import AgentRandom
from absl import app, flags
from pysc2.env import sc2_env, run_loop
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
import os

def main(unused_argv):
  # Este valor permitirá que el agente tome decisiones con mayor frecuencia. 
  # Elegí para equilibrar el tiempo de procesamiento del agente con la velocidad del juego.
  step_mul = 16
  agent = AgentRandom(step_mul)
  while True:
    try:
      with sc2_env.SC2Env(
        map_name="Simple64",
        players=[ sc2_env.Agent(sc2_env.Race.terran,'Random_Agent'),
                  sc2_env.Bot(sc2_env.Race.terran, sc2_env.Difficulty.easy)
                ],
        agent_interface_format=features.AgentInterfaceFormat(
            action_space = actions.ActionSpace.RAW,
            use_raw_units = True,
            raw_resolution = 64,
        ),
        step_mul=step_mul, # How fast it runs the game
        disable_fog=False,
      ) as env:
        run_loop.run_loop([agent], env, max_episodes=10000)
    except KeyboardInterrupt:
      pass

if __name__ == "__main__":
    app.run(main)
