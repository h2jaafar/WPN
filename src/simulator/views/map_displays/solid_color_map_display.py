import copy
import pygame
from typing import Set, Union, Tuple, List

from algorithms.configuration.entities.entity import Entity
from algorithms.configuration.maps.map import Map
from simulator.services.services import Services
from simulator.views.map_displays.map_display import MapDisplay
from structures import Point


class SolidColorMapDisplay(MapDisplay):
    radius: int
    points: Union[Set[Point], List[Entity]]
    color: pygame.Color

    def __init__(self, services: Services, points: Union[Set[Point], List[Entity]],
                 color: Union[pygame.Color, Tuple[int, int, int]], z_index=50, radius=0, custom_map: Map = None) -> None:
        super().__init__(services, z_index=z_index, custom_map=custom_map)
        self.points = points
        self.color = color
        self.radius = radius

    def render(self, screen: pygame.Surface) -> bool:
        if not super().render(screen):
            return False

        points: Union[Set[Point], List[Entity]] = copy.deepcopy(self.points)
        if points is None:
            return False

        def f(pt):
            if isinstance(pt, Entity):
                pt = pt.position
            return pt

        points = set(map(f, points))

        for point in points:
            self._root_view.render_pos(screen, Entity(point, self.radius), self.color)

        return True

    def __lt__(self, other: 'SolidColorMapDisplay') -> bool:
        return tuple(self.color) < tuple(other.color)
