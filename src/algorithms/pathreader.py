from typing import List, Tuple

import numpy as np

from algorithms.algorithm import Algorithm
from algorithms.basic_testing import BasicTesting
from algorithms.configuration.entities.agent import Agent
from algorithms.configuration.entities.goal import Goal
from algorithms.configuration.maps.map import Map
from simulator.services.services import Services
from simulator.views.map_displays.gradient_map_display import GradientMapDisplay
from simulator.views.map_displays.map_display import MapDisplay
from simulator.views.map_displays.numbers_map_display import NumbersMapDisplay
from structures import Point
import copy


class PathReader(Algorithm):

    def __init__(self, services: Services, testing: BasicTesting = None):
        super().__init__(services, testing)
        

    def set_display_info(self) -> List[MapDisplay]:
        """
        Read super description
        """
        return super().set_display_info()


    def _find_path_internal(self) -> None:
        """
        Read super description
        The internal implementation of :ref:`find_path`
        """
        grid: Map = self._get_grid()
        trace = []
        paths_list = [[1,2],[3,4]]
        '''
        Place a list of your path you want to visualize here, in the format above
        '''
        for pt in paths_list:
            trace.append(Point(pt[0],pt[1]))
        # trace =[Point(x=27, y=14), Point(x=27, y=14), Point(x=28, y=15), Point(x=28, y=15), Point(x=29, y=16), Point(x=29, y=16), Point(x=30, y=16), Point(x=30, y=16), Point(x=31, y=16), Point(x=31, y=16), Point(x=32, y=16), Point(x=32, y=16), Point(x=33, y=16), Point(x=33, y=16), Point(x=34, y=16), Point(x=34, y=16), Point(x=35, y=16), Point(x=35, y=16), Point(x=36, y=16), Point(x=36, y=16), Point(x=37, y=16), Point(x=37, y=16), Point(x=38, y=16), Point(x=38, y=16), Point(x=39, y=16), Point(x=39, y=16), Point(x=40, y=17), Point(x=40, y=17), Point(x=41, y=18), Point(x=41, y=18), Point(x=42, y=19), Point(x=42, y=19), Point(x=43, y=20), Point(x=43, y=20), Point(x=44, y=21), Point(x=44, y=21), Point(x=45, y=22), Point(x=45, y=22), Point(x=46, y=23), Point(x=46, y=23), Point(x=47, y=24), Point(x=47, y=24), Point(x=48, y=25), Point(x=48, y=25), Point(x=49, y=26), Point(x=49, y=26), Point(x=50, y=27), Point(x=50, y=27), Point(x=51, y=26), Point(x=51, y=26), Point(x=52, y=25), Point(x=52, y=25), Point(x=53, y=24), Point(x=53, y=24), Point(x=54, y=23), Point(x=54, y=23), Point(x=55, y=22), Point(x=55, y=22), Point(x=56, y=21), Point(x=56, y=21), Point(x=57, y=20), Point(x=57, y=20), Point(x=57, y=21), Point(x=57, y=21), Point(x=57, y=22), Point(x=57, y=22), Point(x=57, y=23), Point(x=57, y=23), Point(x=57, y=24), Point(x=57, y=24), Point(x=57, y=25), Point(x=57, y=25), Point(x=57, y=26), Point(x=57, y=26), Point(x=57, y=27), Point(x=57, y=27), Point(x=57, y=28), Point(x=57, y=28), Point(x=57, y=29), Point(x=57, y=29), Point(x=57, y=30), Point(x=57, y=30), Point(x=57, y=31), Point(x=57, y=31), Point(x=57, y=32), Point(x=57, y=32), Point(x=57, y=33), Point(x=57, y=33), Point(x=57, y=34), Point(x=57, y=34), Point(x=58, y=35), Point(x=58, y=35), Point(x=59, y=35), Point(x=59, y=35), Point(x=60, y=35), Point(x=60, y=35), Point(x=61, y=35), Point(x=61, y=35), Point(x=62, y=35), Point(x=62, y=35), Point(x=63, y=35), Point(x=63, y=35), Point(x=64, y=35), Point(x=64, y=35), Point(x=65, y=35), Point(x=65, y=35), Point(x=66, y=35), Point(x=66, y=35), Point(x=67, y=35), Point(x=67, y=35), Point(x=68, y=35), Point(x=68, y=35), Point(x=69, y=35), Point(x=69, y=35), Point(x=70, y=35), Point(x=70, y=35), Point(x=70, y=35), Point(x=70, y=35), Point(x=70, y=35), Point(x=70, y=35), Point(x=70, y=35), Point(x=70, y=35), Point(x=70, y=35), Point(x=70, y=35)]
        # trace = [Point(x=57, y=28), Point(x=57, y=28), Point(x=58, y=29), Point(x=58, y=29), Point(x=59, y=30), Point(x=59, y=30), Point(x=60, y=31), Point(x=60, y=31), Point(x=61, y=32), Point(x=61, y=32), Point(x=62, y=33), Point(x=62, y=33), Point(x=63, y=34), Point(x=63, y=34), Point(x=64, y=35), Point(x=64, y=35), Point(x=65, y=36), Point(x=65, y=36), Point(x=66, y=37), Point(x=66, y=37), Point(x=67, y=38), Point(x=67, y=38), Point(x=68, y=39), Point(x=68, y=39), Point(x=69, y=40), Point(x=69, y=40), Point(x=70, y=41), Point(x=70, y=41), Point(x=71, y=42), Point(x=71, y=42), Point(x=72, y=43), Point(x=72, y=43), Point(x=73, y=44), Point(x=73, y=44), Point(x=74, y=45), Point(x=74, y=45), Point(x=75, y=46), Point(x=75, y=46), Point(x=76, y=47), Point(x=76, y=47), Point(x=77, y=48), Point(x=77, y=48), Point(x=78, y=49), Point(x=78, y=49), Point(x=79, y=50), Point(x=79, y=50), Point(x=80, y=51), Point(x=80, y=51), Point(x=80, y=52), Point(x=80, y=52), Point(x=80, y=52)]


        for point in trace:
            self.move_agent(point)
            self.key_frame(ignore_key_frame_skip=True)

