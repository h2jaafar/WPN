from typing import Tuple, Callable, Type, List, Optional, Dict, Any, Union

from algorithms.algorithm import Algorithm
from algorithms.basic_testing import BasicTesting
from algorithms.configuration.maps.map import Map
from maps import Maps
from algorithms.lstm.LSTM_tile_by_tile import BasicLSTMModule
from algorithms.lstm.ML_model import MLModel
from simulator.services.debug import DebugLevel
from structures import Size
from algorithms.lstm.LSTM_CAE_tile_by_tile import LSTMCAEModel


class Configuration:
    simulator_grid_display: bool
    simulator_initial_map: Optional[Union[str, Map]]
    simulator_algorithm_type: Optional[Type[Algorithm]]
    simulator_testing_type: Optional[Type[BasicTesting]]
    simulator_algorithm_parameters: Tuple[List, Dict]
    simulator_graphics: bool
    simulator_key_frame_speed: int
    simulator_key_frame_skip: int
    simulator_write_debug_level: DebugLevel
    simulator_window_size: Size
    generator: bool
    generator_labelling_atlases: List[Any]
    generator_gen_type: str
    generator_nr_of_examples: int
    generator_labelling_features: List[str]
    generator_labelling_labels: List[str]
    generator_single_labelling_features: List[str]
    generator_single_labelling_labels: List[str]
    generator_modify: Callable[[], Tuple[str, Callable[[Map], Map]]]
    trainer: bool
    trainer_model: Type[MLModel]
    trainer_custom_config: Optional[Dict[str, Any]]
    trainer_pre_process_data_only: bool
    analyzer: bool
    load_simulator: bool
    clear_cache: bool

    def __init__(self) -> None:
        # Simulator settings
        self.simulator_grid_display = True
        self.simulator_initial_map = Maps.grid_map_labyrinth
        self.simulator_testing_type = None
        self.simulator_algorithm_type = None
        self.simulator_algorithm_parameters = [], {}
        self.simulator_graphics = True
        self.simulator_key_frame_speed = 20
        self.simulator_key_frame_skip = 3
        self.simulator_write_debug_level = DebugLevel.MEDIUM
        self.simulator_window_size = Size(200, 200)

        # Generator (The ones with none are optional)
        self.generator = True
        self.generator_labelling_atlases = None #["block_map_3"]
        self.generator_nr_of_examples = 5
        self.generator_gen_type = "block_map"
        self.generator_labelling_features = []
        self.generator_labelling_labels = []
        self.generator_single_labelling_features = []
        self.generator_single_labelling_labels = []
        self.generator_aug_labelling_features = []
        self.generator_aug_labelling_labels = []
        self.generator_aug_single_labelling_features = []
        self.generator_aug_single_labelling_labels = []
        self.generator_modify = None

        # Trainer
        self.trainer = True
        self.trainer_model = LSTMCAEModel
        self.trainer_custom_config = None
        self.trainer_pre_process_data_only = True
        self.trainer_bypass_and_replace_pre_processed_cache = True

        # Custom behaviour settings
        self.analyzer = True

        # Simulator
        self.load_simulator = True

        # Cache
        self.clear_cache = False
