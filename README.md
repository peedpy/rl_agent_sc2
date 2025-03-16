# rl_agent_sc2
Q-learning Agent for the SC2 Game

> **Note:** The following instructions are intended for Windows environments only.
## Game Setup
1. Navigate to **[battle.net](https://download.battle.net/en-us/desktop)**.
2. Download and install the battle.net application.
3. Log in or create a new account.
4. Locate StarCraft II and download the game (approximately 50 GB).

## Setting Up the Virtual Environment
For guidance, refer to [Virtual Environments](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/). To create a virtual environment in your project directory, execute the following commands. This process creates a new virtual environment in a local folder named `.venv`.

```bash
$ py -m venv .venv
$ .venv\Scripts\activate
$ py -m pip install --upgrade pip
```

## Installing Dependencies
Install the required packages by running:
```bash
$ pip install -r requirements.txt
```

Alternatively, to install specific package versions manually, use the following commands:
```bash
#Install a specific package version
$ py -m pip install "pysc2==3.0.0"
$ py -m pip install "protobuf==3.19.4"
$ py -m pip install "pandas==1.4.3"
$ py -m pip install "numpy==1.22.4"
$ py -m pip install "chardet==4.0.0"
```

## Running the Agent
Launch the Q-Learning agent by executing:
```bash
$ python main_qlearning_agent__vs__ia_agent.py
```
