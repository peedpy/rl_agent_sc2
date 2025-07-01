"""
Configuración centralizada de logging para el proyecto StarCraft II RL Agent.

Este módulo proporciona una configuración unificada de logging que es utilizada
por todos los agentes y módulos del proyecto, asegurando consistencia en el
formato y ubicación de los logs.

Características:
- Logging unificado en un solo archivo
- Formato consistente con timestamp
- Niveles de logging configurables
- Rotación de archivos de log
- Colores en consola para mejor legibilidad

Autor: Pablo Escobar
Fecha: 2022
"""

import logging
import datetime
import os
from logging.handlers import RotatingFileHandler
import sys

# Configuración global de logging
def setup_logging(log_level=logging.DEBUG, max_file_size=10*1024*1024, backup_count=5):
    """
    Configura el sistema de logging centralizado para todo el proyecto.
    
    Args:
        log_level: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        max_file_size: Tamaño máximo del archivo de log en bytes (default: 10MB)
        backup_count: Número de archivos de backup a mantener
        
    Returns:
        logging.Logger: Logger configurado
        
    Example:
        >>> logger = setup_logging(logging.INFO)
        >>> logger.info("Sistema de logging inicializado")
    """
    
    # Crear directorio de logs si no existe
    logs_dir = ".\\logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
        print(f"Directorio de logs creado: {logs_dir}")
    
    # Generar nombre de archivo con timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H.%M.%S")
    log_filename = f"{logs_dir}\\rl_agent_sc2_{timestamp}.log"
    
    # Configurar formato del log
    log_format = '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Crear formateador
    formatter = logging.Formatter(log_format, date_format)
    
    # Configurar handler para archivo con rotación
    file_handler = RotatingFileHandler(
        log_filename,
        maxBytes=max_file_size,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    
    # Configurar handler para consola con colores
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Formateador con colores para consola
    class ColoredFormatter(logging.Formatter):
        """Formateador con colores para la consola."""
        
        COLORS = {
            'DEBUG': '\033[36m',    # Cyan
            'INFO': '\033[32m',     # Green
            'WARNING': '\033[33m',  # Yellow
            'ERROR': '\033[31m',    # Red
            'CRITICAL': '\033[35m', # Magenta
            'RESET': '\033[0m'      # Reset
        }
        
        def format(self, record):
            # Agregar color al nivel de log
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
            
            return super().format(record)
    
    console_formatter = ColoredFormatter(log_format, date_format)
    console_handler.setFormatter(console_formatter)
    
    # Configurar logger raíz
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Limpiar handlers existentes para evitar duplicación
    root_logger.handlers.clear()
    
    # Agregar handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Log inicial
    root_logger.info("=" * 80)
    root_logger.info("SISTEMA DE LOGGING CENTRALIZADO INICIALIZADO")
    root_logger.info(f"Archivo de log: {log_filename}")
    root_logger.info(f"Nivel de logging: {logging.getLevelName(log_level)}")
    root_logger.info("=" * 80)
    
    return root_logger

def get_logger(name):
    """
    Obtiene un logger configurado para un módulo específico.
    
    Args:
        name (str): Nombre del módulo (ej: 'agent_qlearning', 'algorithms.q_learning')
        
    Returns:
        logging.Logger: Logger configurado para el módulo
        
    Example:
        >>> logger = get_logger('agent_qlearning')
        >>> logger.info("Agente Q-Learning inicializado")
    """
    return logging.getLogger(name)

# Configuración por defecto
DEFAULT_LOG_LEVEL = logging.DEBUG
DEFAULT_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
DEFAULT_BACKUP_COUNT = 5

# Inicializar logging por defecto al importar el módulo
setup_logging(DEFAULT_LOG_LEVEL, DEFAULT_MAX_FILE_SIZE, DEFAULT_BACKUP_COUNT) 