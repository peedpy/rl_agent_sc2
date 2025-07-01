# 🤖 Agente Q-Learning para StarCraft II

Un agente de inteligencia artificial basado en Q-Learning para jugar StarCraft II, implementado usando PySC2.

## 📋 Descripción

Este proyecto implementa un agente de aprendizaje por refuerzo que aprende a jugar StarCraft II usando el algoritmo Q-Learning. El agente está diseñado para la raza Terran y puede realizar acciones básicas como:

- **Economía**: Construir SCVs, recolectar minerales y gas
- **Infraestructura**: Construir edificios (cuarteles, depósitos, etc.)
- **Producción militar**: Entrenar Marines y Marauders
- **Combate**: Atacar unidades enemigas con estrategias tácticas
- **Exploración**: Explorar el mapa para encontrar recursos y enemigos

## 🏗️ Arquitectura del Proyecto

```
rl_agent_sc2/
├── actions/                 # Módulos de acciones del juego
│   ├── set_actions.py      # Configuración centralizada de acciones
│   ├── build_scv.py        # Entrenamiento de trabajadores
│   ├── harvest_minerals.py # Recolección de minerales
│   ├── train_marine.py     # Entrenamiento de Marines
│   ├── attack_army.py      # Lógica de combate
│   └── ...                 # Otras acciones
├── algorithms/             # Algoritmos de aprendizaje
│   ├── q_learning.py       # Implementación Q-Learning
│   └── rewards.py          # Sistema de recompensas
├── libs/                   # Utilidades y helpers
│   ├── functions.py        # Funciones auxiliares
│   └── logging_config.py   # Configuración de logging
├── agent_qlearning.py      # Agente principal Q-Learning
├── agent_random.py         # Agente de referencia aleatorio
└── main_qlearning_agent__vs__ia_agent.py  # Script principal
```

## 🚀 Instalación

### Requisitos Previos

1. **StarCraft II**: Descarga e instala StarCraft II desde [Battle.net](https://download.battle.net/en-us/desktop)
2. **Python 3.7+**: Asegúrate de tener Python instalado

### Configuración del Entorno

> **Nota**: Las siguientes instrucciones están destinadas para entornos Windows.

#### 1. Configurar Entorno Virtual

```bash
# Crear entorno virtual
py -m venv .venv

# Activar entorno virtual
.venv\Scripts\activate

# Actualizar pip
py -m pip install --upgrade pip
```

#### 2. Instalar Dependencias

```bash
# Instalar todas las dependencias
pip install -r requirements.txt
```

O instalar manualmente:

```bash
# Instalar versiones específicas
py -m pip install "pysc2==3.0.0"
py -m pip install "protobuf==3.19.4"
py -m pip install "pandas==1.4.3"
py -m pip install "numpy==1.22.4"
py -m pip install "chardet==4.0.0"
```

## 🎮 Ejecución

### Agente Q-Learning

```bash
python main_qlearning_agent__vs__ia_agent.py
```

### Agente Aleatorio (Referencia)

```bash
python main_random_agent__vs__ia_agent.py
```

## 📊 Sistema de Logging

El proyecto incluye un sistema de logging estructurado que proporciona:

- **Logs en consola** con colores para mejor legibilidad
- **Logs en archivo** con rotación automática
- **Diferentes niveles** de logging (DEBUG, INFO, WARNING, ERROR)
- **Formato estructurado** con timestamp, nivel, módulo y mensaje

### Configuración de Logging

```python
from libs.logging_config import setup_logging

# Configurar logging para desarrollo
setup_logging(log_level=logging.DEBUG, log_to_file=True, log_to_console=True)

# Configurar logging para producción
setup_logging(log_level=logging.INFO, log_to_file=True, log_to_console=False)
```

### Ejemplo de Uso

```python
import logging
logger = logging.getLogger(__name__)

logger.debug("Información detallada para debugging")
logger.info("Información general del proceso")
logger.warning("Advertencia sobre una situación")
logger.error("Error que requiere atención")
```

## 🧠 Algoritmo Q-Learning

### Características

- **Exploración vs Explotación**: Epsilon-greedy con decay exponencial
- **Persistencia**: Guardado y carga de tablas Q en CSV
- **Recompensas adaptativas**: Sistema de recompensas basado en múltiples métricas
- **Políticas compuestas**: Combinaciones inteligentes de acciones

### Hiperparámetros

```python
EXPLORATION_MAX = 1.0      # Exploración inicial
EXPLORATION_MIN = 0.1      # Exploración mínima
EXPLORATION_DECAY = 0.0009 # Tasa de decay de exploración
LEARNING_RATE = 0.01       # Tasa de aprendizaje
REWARD_DECAY = 0.8         # Factor de descuento
```

## 🎯 Acciones Implementadas

### Acciones de Economía
- `build_scv`: Entrenar trabajadores
- `harvest_minerals`: Recolectar minerales
- `harvest_gas`: Construir refinerías y recolectar gas

### Acciones de Infraestructura
- `build_supply_depot`: Construir depósitos de suministro
- `build_barracks`: Construir cuarteles
- `build_command_center`: Construir centros de mando
- `build_tech_lab`: Construir laboratorios tecnológicos
- `build_bunker`: Construir búnkeres

### Acciones Militares
- `train_marine`: Entrenar Marines
- `train_marauder`: Entrenar Marauders
- `attack_with_marine`: Atacar con Marines
- `defense_with_marine`: Defender con Marines
- `attack_with_marauder`: Atacar con Marauders

### Acciones Estratégicas
- `explore_csv`: Explorar el mapa

## 📈 Políticas de Acciones

El agente utiliza 14 políticas predefinidas que combinan acciones de manera estratégica:

```python
"policy_0": ["do_nothing"]                    # Control
"policy_1": ["build_scv"]                     # Economía básica
"policy_2": ["harvest_minerals"]              # Recolección
"policy_3": ["harvest_gas"]                   # Gas
"policy_4": ["build_command_center", "harvest_minerals", "harvest_minerals"]  # Expansión
"policy_5": ["explore_csv"]                   # Exploración
"policy_6": ["build_supply_depot"]            # Infraestructura
"policy_7": ["build_barracks"] + 5 * ["train_marine"]  # Producción militar
"policy_8": ["build_bunker"]                  # Defensa
"policy_9": ["build_tech_lab"] + 2 * ["train_marauder"]  # Tecnología
"policy_10": 5 * ["attack_with_marine"]       # Ataque masivo
"policy_11": 5 * ["defense_with_marine"]      # Defensa masiva
"policy_12": 5 * ["attack_with_marine"] + 2 * ["attack_with_marauder"]  # Combinado
"policy_13": 3 * ["attack_with_marauder"]     # Ataque especializado
```

## 🔧 Configuración del Juego

### Entorno PySC2

```python
sc2_env.SC2Env(
    map_name="Simple64",                    # Mapa de entrenamiento
    players=[
        sc2_env.Agent(sc2_env.Race.terran, 'Q-learning'),
        sc2_env.Bot(sc2_env.Race.terran, sc2_env.Difficulty.very_easy)
    ],
    agent_interface_format=features.AgentInterfaceFormat(
        action_space=actions.ActionSpace.RAW,  # Acceso directo a unidades
        use_raw_units=True,                    # Información detallada
        raw_resolution=64,                     # Resolución del mapa
    ),
    step_mul=16,                              # Velocidad del juego
    disable_fog=False,                        # Niebla de guerra activa
)
```

## 📁 Estructura de Datos

### Archivos de Entrenamiento

- `new_qlearning_stats_train.csv`: Estadísticas de entrenamiento
- `new_qlearning_table_train.csv`: Tabla Q actual
- `new_qlearning_status_game_train.csv`: Estado del juego por episodio

### Logs

- `logs/sc2_agent_YYYY-MM-DD_HH-MM-SS.log`: Archivos de log con timestamp

## 🤝 Contribución

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 👨‍💻 Autor

**Tu Nombre** - [tu-email@ejemplo.com](mailto:tu-email@ejemplo.com)

## 🙏 Agradecimientos

- [PySC2](https://github.com/deepmind/pysc2) - API de Python para StarCraft II
- [DeepMind](https://deepmind.com/) - Desarrollo de PySC2
- [Blizzard Entertainment](https://www.blizzard.com/) - StarCraft II

## 📚 Referencias

- [PySC2 Documentation](https://github.com/deepmind/pysc2)
- [StarCraft II Wiki](https://liquipedia.net/starcraft2/Main_Page)
- [Q-Learning Algorithm](https://en.wikipedia.org/wiki/Q-learning)
