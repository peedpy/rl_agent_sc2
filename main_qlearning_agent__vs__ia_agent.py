from agent_qlearning import AgentQlearning
from absl import app, flags
from pysc2.env import sc2_env, run_loop
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
import os

def main(unused_argv):
  train_mode = True
  # Este valor permitirá que el agente tome decisiones con mayor frecuencia. 
  # Elegí para equilibrar el tiempo de procesamiento del agente con la velocidad del juego.
  step_mul = 16
  agent = AgentQlearning(step_mul, train_mode)
  while True:
    try:
      with sc2_env.SC2Env(
        map_name="Simple64",
        players=[ sc2_env.Agent(sc2_env.Race.terran,'Q-learning'),
                  sc2_env.Bot(sc2_env.Race.terran, sc2_env.Difficulty.very_easy)
                ],
        agent_interface_format=features.AgentInterfaceFormat(
            action_space = actions.ActionSpace.RAW,
            use_raw_units = True,
            raw_resolution = 64,
        ),
        step_mul=step_mul, # How fast it runs the game
        disable_fog=False,
      ) as env:
        run_loop.run_loop([agent], env, max_episodes=1000)
    except KeyboardInterrupt:
      pass

if __name__ == "__main__":
    app.run(main)
