from typing import List, Callable, Optional, Set

import torch

from algorithms.algorithm import Algorithm
from algorithms.basic_testing import BasicTesting
from simulator.services.services import Services
from simulator.views.map_displays.graph_map_display import GraphMapDisplay
from simulator.views.map_displays.map_display import MapDisplay
from structures import Point

from algorithms.classic.sample_based.core.vertex import Vertex

class Graph:
    root_vertex_a: Vertex
    root_vertex_b: Vertex

    def __init__(self, start_pos: Point, goal_pos: Point) -> None:
        self.root_vertex_start = Vertex(start_pos)
        self.root_vertex_goal = Vertex(goal_pos)
        self.root_vertex_a = self.root_vertex_start
        self.root_vertex_b = self.root_vertex_goal

    @staticmethod
    def add_edge(parent: Vertex, child: Optional['Vertex']):
        parent.add_child(child)
        child.set_parent(parent)

    def swap_root_vertices(self):
        buffer = self.root_vertex_a
        self.root_vertex_a = self.root_vertex_b
        self.root_vertex_b = buffer

    def walk_dfs(self, f: Callable[[Vertex], bool]):
        self.root_vertex_a.visit_children(f)
        self.root_vertex_b.visit_children(f)

    def walk_dfs_single_vertex(self, root_vertex: Vertex, f: Callable[[Vertex], bool]):
        root_vertex.visit_children(f)

    def get_nearest_vertex(self, root_vertex: Vertex, point: Point) -> Vertex:
        def get_nearest(current: Vertex, __acc) -> bool:
            dist: float = torch.norm(point.to_tensor() - current.position.to_tensor())
            if dist <= __acc[0]:
                __acc[0] = dist
                __acc[1] = current
                return True
            return False

        acc: [float, Vertex] = [float('inf'), root_vertex]
        self.walk_dfs_single_vertex(root_vertex, lambda current: get_nearest(current, acc))
        return acc[1]


class RRT_Connect(Algorithm):
    __graph: Graph
    __max_dist: float
    __iterations: int

    def __init__(self, services: Services, testing: BasicTesting = None) -> None:
        super().__init__(services, testing)
        self.__graph = Graph(self._get_grid().agent.position, self._get_grid().goal.position)
        self.__max_dist = 10
        self.__iterations = 10000

    def set_display_info(self) -> List[MapDisplay]:
        return super().set_display_info() + [GraphMapDisplay(self._services, self.__graph)]

    def __extend(self, root_vertex: Vertex, q: Point) -> str:
        self.__q_near: Vertex = self.__get_nearest_vertex(root_vertex, q)
        self.__q_new: Vertex = self.__get_new_vertex(self.__q_near, q, self.__max_dist)
        if self._get_grid().is_valid_line_sequence(self._get_grid().get_line_sequence(self.__q_near.position, self.__q_new.position)):
            self.__graph.add_edge(self.__q_near, self.__q_new)
            if self.__q_new.position == q:
                return 'reached'
            else:
                return 'advanced'
        return 'trapped'

    def __connect(self, root_vertex: Vertex, q: Vertex) -> str:
        S = 'advanced'
        while S == 'advanced':
            S = self.__extend(root_vertex, q.position)
        self.__mid_vertex = q
        return S

    def __path(self):

        # trace back
        path_mid_to_b: List[Vertex] = [self.__q_new]

        while len(path_mid_to_b[-1].parents) != 0:
            for parent in path_mid_to_b[-1].parents:
                path_mid_to_b.append(parent)
                break

        path_a_to_mid: List[Vertex] = [self.__extension_target]

        while len(path_a_to_mid[-1].parents) != 0:
            for parent in path_a_to_mid[-1].parents:
                path_a_to_mid.append(parent)
                break

        path_a_to_mid.reverse()
        path = path_a_to_mid + path_mid_to_b

        if self.__graph.root_vertex_a is self.__graph.root_vertex_goal:
            path.reverse()

        for p in path:
            self.move_agent(p.position)
            self.key_frame(ignore_key_frame_skip=True)

    def _find_path_internal(self) -> None:

        for i in range(self.__iterations):
        #while True:

            q_rand: Point = self.__get_random_sample()

            if not self.__extend(self.__graph.root_vertex_a, q_rand) == 'trapped':
                self.__extension_target = self.__q_new
                if self.__connect(self.__graph.root_vertex_b, self.__q_new) == 'reached':
                    self.__path()
                    break
            self.__graph.swap_root_vertices()

            # visualization code
            self.key_frame()

    def __get_random_sample(self) -> Point:
        while True:
            sample: Point = Point(torch.randint(0, self._get_grid().size.width, (1,)).item(),
                                  torch.randint(0, self._get_grid().size.height, (1,)).item())
            if self._get_grid().is_agent_valid_pos(sample):
                return sample

    def __get_nearest_vertex(self, graph_root_vertex: Vertex, q_sample: Point) -> Vertex:
        return self.__graph.get_nearest_vertex(graph_root_vertex, q_sample)

    @staticmethod
    def __get_new_vertex(q_near: Vertex, q_sample: Point, max_dist) -> Vertex:
        dir = q_sample.to_tensor() - q_near.position.to_tensor()
        if torch.norm(dir) <= max_dist:
            return Vertex(q_sample)

        dir_normalized = dir / torch.norm(dir)
        q_new = Point.from_tensor(q_near.position.to_tensor() + max_dist * dir_normalized)
        return Vertex(q_new)