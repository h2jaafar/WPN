from threading import Condition
from typing import Dict, Any, List, Callable, Optional

import numpy as np

from algorithms.configuration.entities.agent import Agent
from algorithms.configuration.entities.trace import Trace
from algorithms.configuration.maps.dense_map import DenseMap
from algorithms.configuration.maps.map import Map
from algorithms.configuration.maps.sparse_map import SparseMap
from simulator.services.debug import DebugLevel
from simulator.services.services import Services
from simulator.services.timer import Timer
from simulator.views.map_displays.map_display import MapDisplay
from structures import Size, Point


class BasicTesting:
    """
    Basic class used to assess the performance of an algorithms
    All tests must inherit from this class
    """

    timer: Timer
    display_info: List[MapDisplay]
    key_frame: Callable[[List, Dict], None]
    key_frame_condition: Condition
    key_frame_skip_count: int
    total_time: int

    def __init__(self, services: Services) -> None:
        self._services = services
        self.timer = None
        self.display_info = []
        self.key_frame = lambda *args, **kwargs: None
        self.key_frame_condition = None
        self.key_frame_skip_count = 0
        self.total_time = 0
        self.reset()

    def reset(self) -> None:
        """
        This method resets the testing state
        """
        self.display_info = []
        self.key_frame_condition = None
        self.key_frame_skip_count = 0
        self.total_time = 0
        if self._services.settings.simulator_key_frame_speed == 0 or not self._services.settings.simulator_graphics:
            self.key_frame = lambda *args, **kwargs: None
        else:
            self.key_frame = self.key_frame_internal

    def set_condition(self, key_frame_condition: Condition) -> None:
        """
        Setter for condition
        :param key_frame_condition:  The condition is used to synchronize key frames
        """
        self.key_frame_condition = key_frame_condition

    def algorithm_start(self) -> None:
        """
        Marks the start of the algorithms
        """
        self._services.debug.write("Algorithm " + str(self._services.algorithm.algorithm_type) + " started..",
                                   DebugLevel.BASIC)
        self.timer = Timer()

    def key_frame_internal(self, ignore_key_frame_skip: bool = False, only_render: List[int] = None, root_key_frame=None) -> None:
        """
        Internal key frame handler
        """
        self.timer.pause()

        if root_key_frame:
            root_key_frame.instance.testing.key_frame(ignore_key_frame_skip=ignore_key_frame_skip, only_render=only_render)

        if ignore_key_frame_skip or self.key_frame_skip_count >= self._services.settings.simulator_key_frame_skip:
            self.display_info = self._services.algorithm.instance.set_display_info()

            if only_render:
                self.display_info = [self.display_info[i] for i in only_render]

            if self.key_frame_condition is not None:
                self.key_frame_condition.acquire()
                self.key_frame_condition.wait()
                self.key_frame_condition.release()
            self.key_frame_skip_count = 0
        else:
            self.key_frame_skip_count += 1
        self.timer.resume()

    def algorithm_done(self) -> None:
        """
        Marks the end of the algorithms
        """
        if self.timer is None:
            self._services.debug.write_error(
                "Algorithm " + str(self._services.algorithm.algorithm_type) + "did not start!")
            return
        self.total_time = self.timer.stop()

        if self._services.settings.simulator_graphics:
            self.display_info = self._services.algorithm.instance.set_display_info()
        self.print_results()
        self._services.debug.write("", DebugLevel.BASIC, timestamp=False)

    def get_results(self) -> Dict[str, Any]:
        """
        Returns the test results
        :return: The test results
        """
        trace: List[Trace] = self._services.algorithm.map.trace

        grid: Optional[DenseMap] = self._services.algorithm.map
        if isinstance(grid, SparseMap):
            grid: DenseMap = grid.convert_to_dense_map()

        return {
            "map": self._services.algorithm.map,
            "map_obstacles_percentage": self.get_occupancy_percentage_grid(grid.grid, DenseMap.WALL_ID),
            "goal_found": self._services.algorithm.map.is_goal_reached(grid.agent.position),
            "distance_to_goal": self.distance_from_agent_to_goal(grid),
            "original_distance_to_goal": self.get_original_distance(grid),
            "trace": trace,
            "total_steps": len(trace),
            "total_distance": self.get_euclidean_distance_traveled(trace, grid.agent),
            "total_time": self.total_time,
            "algorithm_type": self._services.algorithm.algorithm_type,
        }

    def print_results(self) -> None:
        """
        Prints the test results
        """
        results: Dict[str, Any] = self.get_results()
        self._services.debug.write("Map: " + str(results["map"]), DebugLevel.MEDIUM)
        self._services.debug.write("Original distance: {0:.2f}".format(results["original_distance_to_goal"]), DebugLevel.BASIC)
        self._services.debug.write("Occupancy percentage: {0:.2f}%".format(results["map_obstacles_percentage"]), DebugLevel.BASIC)
        self._services.debug.write("Goal was " + ("FOUND" if results["goal_found"] else "NOT FOUND"), DebugLevel.BASIC)
        if not results["goal_found"]:
            self._services.debug.write("Distance to goal: " + str(results["distance_to_goal"]) + " (Original: {})".format(results["original_distance_to_goal"]))
        self._services.debug.write("Total steps: " + str(results["total_steps"]), DebugLevel.BASIC)
        self._services.debug.write("Total distance: {0:.2f}".format(results["total_distance"]), DebugLevel.BASIC)
        self._services.debug.write("Total time: " + str(results["total_time"]) + " seconds", DebugLevel.BASIC)
        self._services.debug.write("Trace: " + str(results["trace"]), DebugLevel.MEDIUM)

    @staticmethod
    def get_occupancy_percentage_grid(grid: List[List[int]], token: int) -> float:
        """
        Searches for token in grid and returns the occupancy percentage of the token
        :param grid: The grid
        :param token: The search token
        :return: The percentage
        """
        tokens: int = 0
        width: int = len(grid)
        height: int = 0 if width == 0 else len(grid[0])
        for x in range(height):
            for y in range(width):
                tokens += grid[y][x] == token
        return BasicTesting.get_occupancy_percentage_size(Size(width, height), tokens)

    @staticmethod
    def get_original_distance(mp: Map):
        agent_pos = mp.trace[0].position if mp.trace else None
        return BasicTesting.distance_from_agent_to_goal(mp, agent_pos)

    @staticmethod
    def distance_from_agent_to_goal(mp: Map, agent_pos: Point = None):
        if not agent_pos:
            agent_pos = mp.agent.position
        dist = np.linalg.norm(np.array(agent_pos) - np.array(mp.goal.position))
        return dist

    @staticmethod
    def get_occupancy_percentage_size(size: Size, items_nr: int) -> float:
        """
        Returns the percentage of present items given the size
        :param size: The size
        :param items_nr:  The number of items
        :return: The percentage
        """
        return (items_nr / (size.width * size.height)) * 100

    @staticmethod
    def get_euclidean_distance_traveled(trace: List[Trace], agent: Agent) -> float:
        trace.append(Trace(agent.position))
        dist = 0
        for i in range(1, len(trace)):
            dist += np.linalg.norm(np.array(trace[i - 1].position) - np.array(trace[i].position))
        return dist

    @property
    def map(self) -> str:
        return "map"

    @map.getter
    def map(self) -> Map:
        return self._services.algorithm.map
