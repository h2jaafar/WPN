
from main import MainRunner
from typing import Tuple, Callable, Type, List, Optional, Dict, Any, Union
from algorithms.configuration.configuration import Configuration
from algorithms.algorithm import Algorithm
from algorithms.basic_testing import BasicTesting
from algorithms.configuration.maps.map import Map
from algorithms.lstm.LSTM_tile_by_tile import BasicLSTMModule
from algorithms.lstm.ML_model import MLModel
from simulator.services.debug import DebugLevel
from analyzer.analyzer import Analyzer
from generator.generator import Generator as generator

from structures import Size
from simulator.services import services

config = type(Configuration)
config=Configuration()
MainRunner(config).run()
