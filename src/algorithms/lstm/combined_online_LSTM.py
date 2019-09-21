import copy
from threading import Thread
from typing import List, Tuple, Optional, Set

from algorithms.algorithm import Algorithm
from algorithms.basic_testing import BasicTesting
from algorithms.configuration.maps.map import Map
from algorithms.lstm.LSTM_tile_by_tile import OnlineLSTM
from simulator.services.algorithm_runner import AlgorithmRunner
from simulator.services.services import Services
from simulator.views.map_displays.entities_map_display import EntitiesMapDisplay
from simulator.views.map_displays.online_lstm_map_display import OnlineLSTMMapDisplay
from simulator.views.map_displays.solid_color_map_display import SolidColorMapDisplay
from structures import Point


class CombinedOnlineLSTM(Algorithm):
    kernel_names: List[str]
    _max_it: float
    _threaded: bool

    __active_kernel: Optional[AlgorithmRunner]
    __total_path: Set[Point]

    def __init__(self, services: Services, testing: BasicTesting = None, kernel_names: List["str"] = None,
                 max_it: float = float('inf'), threaded: bool = False):
        super().__init__(services, testing)

        if not kernel_names:
            self.kernel_names = [
                "caelstm_section_lstm_training_block_map_10000_model",
                "caelstm_section_lstm_training_uniform_random_fill_10000_model",
                "caelstm_section_lstm_training_house_10000_model",
                "caelstm_section_lstm_training_uniform_random_fill_10000_block_map_10000_house_10000_model",
                "caelstm_section_lstm_training_uniform_random_fill_10000_block_map_10000_model",
                "tile_by_tile_training_uniform_random_fill_10000_model",
                "tile_by_tile_training_block_map_10000_model",
                "tile_by_tile_training_house_10000_model",
                "tile_by_tile_training_uniform_random_fill_10000_block_map_10000_model",
                "tile_by_tile_training_uniform_random_fill_10000_block_map_10000_house_10000_model",
            ]

        self._max_it = max_it
        self._threaded = threaded
        self.__active_kernel = None
        self.__total_path = set()

        # testing fields
        self.kernel_call_idx = None

    def set_display_info(self):
        active_kernel_displays = []

        if self.__active_kernel:
            active_kernel_displays = [
                # *self.__active_kernel.instance.set_display_info(),
                OnlineLSTMMapDisplay(self._services, custom_map=self.__active_kernel.map),
                EntitiesMapDisplay(self._services, custom_map=self.__active_kernel.map),
            ]

        return super().set_display_info() + [
            *active_kernel_displays,
            SolidColorMapDisplay(self._services, self.__total_path, (0, 150, 0), z_index=80),
        ]

    # TODO when max_it is inf take the solution where we are closer to the goal or implement special case
    def _find_path_internal(self) -> None:
        kernels: List[Tuple[int, AlgorithmRunner]] = list(map(lambda kernel: (kernel[0],
                                                                              self._services.algorithm.get_new_runner(
                                                                                  copy.deepcopy(self._get_grid()),
                                                                                  OnlineLSTM, ([],
                                                                                               {"max_it": self._max_it,
                                                                                                "load_name": kernel[
                                                                                                    1]}),
                                                                                  BasicTesting,
                                                                                  with_animations=True)),
                                                              enumerate(self.kernel_names)))

        if self._threaded:
            threaded_jobs: List[Thread] = list(
                map(lambda kernel: Thread(target=kernel[1].find_path, daemon=True), kernels))

            for j in threaded_jobs:
                j.start()

            for j in threaded_jobs:
                j.join()
        else:
            for k in kernels:
                self.__active_kernel = k[1]
                k[1].find_path()
                self.__total_path = self.__total_path.union(set(map(lambda el: el.position, k[1].map.trace)))

        self.__active_kernel = None

        # check if any found path and if they did take smallest dist
        best_kernels: List[Tuple[int, AlgorithmRunner]] = []

        for kernel in kernels:
            if kernel[1].map.is_goal_reached(kernel[1].map.agent.position):
                best_kernels.append(kernel)

        # take smallest dist kernel if any
        dist: float = float("inf")
        best_kernel: Tuple[int, AlgorithmRunner] = None
        for kernel in best_kernels:
            if dist > len(kernel[1].map.trace):
                dist = len(kernel[1].map.trace)
                best_kernel = kernel

        if best_kernel:
            best_kernel[1].map.replay_trace(self.__replay)
        else:
            # pick the one with furthest progress
            dist = -1
            best_kernel = None
            for kernel in kernels:
                if dist < len(kernel[1].map.trace):
                    dist = len(kernel[1].map.trace)
                    best_kernel = kernel

            best_kernel[1].map.replay_trace(self.__replay)
        self.kernel_call_idx = best_kernel[0]

    def __replay(self, m: Map) -> None:
        self.move_agent(m.agent.position)
        self.key_frame()
