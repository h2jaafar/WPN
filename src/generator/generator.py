import torch
from typing import Tuple, List, TYPE_CHECKING, Dict, Callable, Type, Set, Any

import numpy as np
from pygame.surface import Surface
import math
import os
import json
from algorithms.classic.graph_based.a_star import AStar
from algorithms.classic.testing.a_star_testing import AStarTesting
from algorithms.configuration.configuration import Configuration
from algorithms.configuration.entities.agent import Agent
from algorithms.configuration.entities.entity import Entity
from algorithms.configuration.maps.dense_map import DenseMap
from algorithms.configuration.maps.map import Map
from algorithms.lstm.map_processing import MapProcessing
from simulator.services.debug import DebugLevel
from simulator.services.progress import Progress
from simulator.services.resources.atlas import Atlas
from simulator.services.services import Services
from simulator.services.timer import Timer
from simulator.simulator import Simulator
from structures import Point, Size
from matplotlib import pyplot as plt
if TYPE_CHECKING:
    from main import MainRunner
from natsort import natsorted

from algorithms.lstm.LSTM_CAE_tile_by_tile import CAE


class Generator:
    """
    Used to generate maps
    """
    __services: Services
    COLOR_EPSILON: int = 230
    GOAL_COLOR: Tuple[int, int, int, int] = (0, 255, 0, 255)
    AGENT_COLOR: Tuple[int, int, int, int] = (255, 0, 0, 255)
    WALL_COLOR: Tuple[int, int, int, int] = (0, 0, 0, 255)

    AVAILABLE_GENERATION_METHODS: Set["str"] = {"uniform_random_fill", "block_map", "house"}

    def __init__(self, services: Services) -> None:
        self.__services = services
        self.generate_map_from_image = self.__services.debug.debug_func(DebugLevel.BASIC)(self.generate_map_from_image)
        self.generate_maps = self.__services.debug.debug_func(DebugLevel.BASIC)(self.generate_maps)
        self.label_maps = self.__services.debug.debug_func(DebugLevel.BASIC)(self.label_maps)

    def generate_map_from_image(self, image_name: str, rand_entities: bool = False, entity_radius: int = None, house_expo_flag: bool = False) -> Map:
        """
        Generate a map from an image
        Load the image from the default location and save the map in the default location
        :param image_name: The image name
        :return: The map
        """
        self.__services.debug.write("Started map generation from image: " + str(image_name) + " With House_expo = " + str(house_expo_flag),
        DebugLevel.BASIC)
        timer: Timer = Timer()
        if house_expo_flag:
            surface: Surface = self.__services.resources.house_expo_dir.load(image_name) #loading directory
        else:
            surface: Surface = self.__services.resources.images_dir.load(image_name) #loading directory
        self.__services.debug.write("Image loaded with Resolution:" + str(surface.get_width()) +" x "+ str(surface.get_height()),DebugLevel.HIGH)
        grid: List[List[int]] = [[0 for _ in range(surface.get_width())] for _ in range(surface.get_height())]
        agent_avg_location: np.ndarray = np.array([.0, .0])
        agent_avg_count: int = 1
        goal_avg_location: np.ndarray = np.array([.0, .0])

        if house_expo_flag: 
            '''
            We can optimize for house_expo dataset by skipping the check for the goal and agent at each pixel,
            instead, we can only identify obstacles
            '''
            self.__services.debug.write("Begin iteration through map",DebugLevel.HIGH)
            for x in range(surface.get_width()):
                for y in range(surface.get_height()):
                    if Generator.is_in_color_range(surface.get_at((x, y)), Generator.WALL_COLOR):
                        grid[y][x] = DenseMap.WALL_ID
        else: 
             for x in range(surface.get_width()):
                for y in range(surface.get_height()):
                    if Generator.is_in_color_range(surface.get_at((x, y)), Generator.AGENT_COLOR, 5):
                        agent_avg_location, agent_avg_count = \
                            Generator.increment_moving_average(agent_avg_location, agent_avg_count, np.array([x, y]))
                    elif Generator.is_in_color_range(surface.get_at((x, y)), Generator.GOAL_COLOR, 5):
                        goal_avg_location, goal_avg_count = \
                            Generator.increment_moving_average(goal_avg_location, goal_avg_count, np.array([x, y]))
                    if Generator.is_in_color_range(surface.get_at((x, y)), Generator.WALL_COLOR):
                        grid[y][x] = DenseMap.WALL_ID

        agent_avg_location = np.array(agent_avg_location, dtype=int)
        goal_avg_location = np.array(goal_avg_location, dtype=int)
        agent_radius: float = 0

        if rand_entities:
            self.__place_random_agent_and_goal(grid, Size(surface.get_width(), surface.get_height()))
            self.__services.debug.write("Placed random agent and goal ",DebugLevel.HIGH)
        else:
            grid[agent_avg_location[1]][agent_avg_location[0]] = DenseMap.AGENT_ID
            grid[goal_avg_location[1]][goal_avg_location[0]] = DenseMap.GOAL_ID
        
        if not house_expo_flag: 
            '''
            We can optimize the house_expo generation by skipping this step, 
            as we have already defined the agent radius
            '''
            self.__services.debug.write("Skipped agent_radius change checking ",DebugLevel.HIGH)

            for x in range(surface.get_width()):
                for y in range(surface.get_height()):
                    if Generator.is_in_color_range(surface.get_at((x, y)), Generator.AGENT_COLOR, 5):
                        '''
                        If color at x y is red (agent) then change the radius of the agent to the max 
                        Change the agent radius to the max between the old radius (from previous iteration )
                        and the magnitude of the agent location - the point 
                        This basically defines the agent radius as the largest red size. But we don't need to do
                        this as we are supplying our own radius
                        '''
                        agent_radius = max(agent_radius, np.linalg.norm(agent_avg_location - np.array([x, y])))

        agent_radius = int(agent_radius)

        if entity_radius:
            agent_radius = entity_radius

        res_map: DenseMap = DenseMap(grid)
        res_map.agent.radius = agent_radius
        res_map.goal.radius = agent_radius

        self.__services.debug.write("Generated initial dense map in " + str(timer.stop()) + " seconds",
                                    DebugLevel.BASIC)
        timer = Timer()
        res_map.extend_walls()
        self.__services.debug.write("Extended walls in " + str(timer.stop()) + " seconds", DebugLevel.BASIC)
        map_name: str = str(image_name.split('.')[0]) + ".pickle"
        if house_expo_flag:
            path = "resources/maps/_house_expo/"
            self.__services.resources.house_expo_dir.save(map_name, res_map, path)
        else:
            self.__services.resources.maps_dir.save(map_name, res_map)
        self.__services.debug.write("Finished generation. Map is in resources folder", DebugLevel.BASIC)
        return res_map

    def __in_bounds(self, pt: Point, dimensions: Size) -> bool:
        return 0 <= pt.x < dimensions.width and 0 <= pt.y < dimensions.height

    def __get_rand_position(self, dimensions: Size, start: Point = Point(0, 0)) -> Point:
        p_x = torch.randint(start.x, dimensions.width, (1,))
        p_y = torch.randint(start.y, dimensions.height, (1,))
        return Point(int(p_x[0]), int(p_y[0]))

    def __generate_random_map(self, dimensions: Size, obstacle_fill_rate: float = 0.3) -> Map:
        grid: List[List[int]] = [[0 for _ in range(dimensions.width)] for _ in range(dimensions.height)]
        fill: float = dimensions.width * dimensions.height * obstacle_fill_rate
        nr_of_obstacles = 0

        while nr_of_obstacles < fill:
            obst_pos: Point = self.__get_rand_position(dimensions)

            if grid[obst_pos.y][obst_pos.x] == DenseMap.CLEAR_ID:
                grid[obst_pos.y][obst_pos.x] = DenseMap.WALL_ID
                nr_of_obstacles += 1

        self.__place_random_agent_and_goal(grid, dimensions)

        return DenseMap(grid)

    def __place_random_agent_and_goal(self, grid: List[List[int]], dimensions: Size):
        while True:
            agent_pos: Point = self.__get_rand_position(dimensions)

            if grid[agent_pos.y][agent_pos.x] == DenseMap.CLEAR_ID:
                grid[agent_pos.y][agent_pos.x] = DenseMap.AGENT_ID #Changes the value of pos on grid to 2 for agent id 
                break

        while True:
            goal_pos: Point = self.__get_rand_position(dimensions)

            if grid[goal_pos.y][goal_pos.x] == DenseMap.CLEAR_ID:
                grid[goal_pos.y][goal_pos.x] = DenseMap.GOAL_ID #Changes the value of pos on grid to 3 for goal id 
                break

    def __place_entity_near_corner(self, entity: Type[Entity], corner: int, grid: List[List[int]], dimensions: Size) -> \
    List[List[int]]:
        for _ in range(corner):
            grid = self.__rotate_by_90(grid, dimensions)

        token = DenseMap.AGENT_ID if entity == Agent else DenseMap.GOAL_ID

        for l in range(min(dimensions.width, dimensions.height)):
            should_break = False
            for y in range(l + 1):
                if grid[y][l] == DenseMap.CLEAR_ID:
                    grid[y][l] = token
                    should_break = True
                    break

            if should_break:
                break

            for x in range(l + 1):
                if grid[l][x] == DenseMap.CLEAR_ID:
                    grid[x][l] = token
                    should_break = True
                    break

            if should_break:
                break

        for _ in range(4 - corner):
            grid = self.__rotate_by_90(grid, dimensions)

        return grid

    def __rotate_by_90(self, grid, dimensions):
        res: List[List[int]] = [[0 for _ in range(dimensions.height)] for _ in range(dimensions.width)]
        for i in range(len(res)):
            for j in range(len(res[i])):
                res[i][j] = grid[len(grid) - j - 1][i]
        return res

    def __generate_random_const_obstacles(self, dimensions: Size, obstacle_fill_rate: float, nr_of_obstacles: int):
        grid: List[List[int]] = [[0 for _ in range(dimensions.width)] for _ in range(dimensions.height)]

        total_fill = int(obstacle_fill_rate * dimensions.width * dimensions.height) + 1

        for i in range(nr_of_obstacles):
            next_obst_fill = torch.randint(total_fill, (1,)).item()

            if i == nr_of_obstacles - 1:
                next_obst_fill = total_fill

            if next_obst_fill == 0:
                break

            while True:
                first_side = int(torch.randint(int(math.sqrt(next_obst_fill)) + 1, (1,)).item())
                if first_side == 0:
                    continue
                second_side = int(next_obst_fill / first_side)

                size = Size(first_side, second_side)
                top_left_corner = self.__get_rand_position(dimensions)

                if self.__can_place_square(size, top_left_corner, dimensions):
                    self.__place_square(grid, size, top_left_corner)
                    break

            total_fill -= next_obst_fill

        """
        # Corner logic
        
        agent_corner = torch.randint(4, (1,)).item()
        grid = self.__place_entity_near_corner(Agent, agent_corner, grid, dimensions)
        goal_corner = (4 + agent_corner - 2) % 4
        grid = self.__place_entity_near_corner(Goal, goal_corner, grid, dimensions)
        """

        self.__place_random_agent_and_goal(grid, dimensions)
        return DenseMap(grid)

    def __place_square(self, grid: List[List[int]], size: Size, top_left_corner: Point) -> None:
        for x in range(top_left_corner.x, top_left_corner.x + size.width):
            for y in range(top_left_corner.y, top_left_corner.y + size.height):
                grid[y][x] = DenseMap.WALL_ID

    def __can_place_square(self, size: Size, top_left_corner: Point, dimensions: Size) -> bool:
        return self.__in_bounds(top_left_corner, dimensions) and \
               self.__in_bounds(Point(top_left_corner.x + size.width, top_left_corner.y), dimensions) and \
               self.__in_bounds(Point(top_left_corner.x, top_left_corner.y + size.height), dimensions) and \
               self.__in_bounds(Point(top_left_corner.x + size.width, top_left_corner.y + size.height), dimensions)

    """
    def __place_room(self, grid: List[List[int]], size: Size, top_left_corner: Point) -> None:
        def in_bounds(x, y):
            return 0 <= x < top_left_corner.x + size.width and 0 <= y < top_left_corner.y + size.height

        for x in range(top_left_corner.x, top_left_corner.x + size.width):
            for y in range(top_left_corner.y, top_left_corner.y + size.height):
                if in_bounds(x, y):
                    grid[y][x] = DenseMap.CLEAR_ID

        for x in range(top_left_corner.x, top_left_corner.x + size.width):
            y = top_left_corner.y
            if in_bounds(x, y):
                grid[y][x] = DenseMap.WALL_ID

            y = top_left_corner.y + size.height - 1
            if in_bounds(x, y):
                grid[y][x] = DenseMap.WALL_ID

        for y in range(top_left_corner.y, top_left_corner.y + size.height):
            x = top_left_corner.x
            if in_bounds(x, y):
                grid[y][x] = DenseMap.WALL_ID

            x = top_left_corner.x + size.width - 1
            if in_bounds(x, y):
                grid[y][x] = DenseMap.WALL_ID
    
    def __get_exterior_mask(self, dimensions: Size, grid: List[List[int]]) -> List[List[int]]:
        mask: List[List[int]] = [[0 for _ in range(dimensions.width)] for _ in range(dimensions.height)]
        for l in range(min(dimensions.width, dimensions.height)):
            # top to bottom - left
            for y in range(l, dimensions.height - l):
                if grid[y][l] == DenseMap.WALL_ID and (l == 0 or mask[y][l - 1] == 1):
                    mask[y][l] = 1

            # top to bottom - right
            for y in range(l, dimensions.height - l):
                if grid[y][dimensions.width - l - 1] == DenseMap.WALL_ID and (l == 0 or mask[y][dimensions.width - l] == 1):
                    mask[y][dimensions.width - l - 1] = 1

            # left to right - top
            for x in range(l, dimensions.width - l):
                if grid[l][x] == DenseMap.WALL_ID and (l == 0 or mask[l - 1][x] == 1):
                    mask[l][x] = 1

            # left to right - bottom
            for x in range(l, dimensions.width - l):
                if grid[dimensions.height - l - 1][x] == DenseMap.WALL_ID and (l == 0 or mask[dimensions.height - l][x] == 1):
                    mask[dimensions.height - l - 1][x] = 1
        return mask

    def __get_random_point_from_exterior_mask(self, exterior_mask: List[List[int]]) -> Point:
        interior_points: List[Point] = []

        for i in range(len(exterior_mask)):
            for j in range(len(exterior_mask[i])):
                if exterior_mask[i][j] == 0:
                    interior_points.append(Point(j, i))

        return interior_points[int(torch.randint(0, len(interior_points), (1,)).item())]
    """

    def __generate_random_house(self, dimensions: Size, min_room_size: Size = Size(10, 10),
                                max_room_size: Size = Size(40, 40), door_size: int = 2) -> Map:
        """
        grid: List[List[int]] = [[DenseMap.WALL_ID for _ in range(dimensions.width)] for _ in range(dimensions.height)]
        min_room_size: Size = Size(4, 4)

        # generate rooms
        for i in range(nr_of_rooms):
            if i == 0:
                rand_pos = self.__get_rand_position(Size(dimensions.width - min_room_size.width, dimensions.height - min_room_size.height))
            else:
                mask = self.__get_exterior_mask(dimensions, grid)
                rand_pos = self.__get_random_point_from_exterior_mask(mask)
                rand_pos = Point(max(min_room_size.width, min(rand_pos.x, dimensions.width - min_room_size.width - 1)), max(min_room_size.height, min(rand_pos.y, dimensions.height - min_room_size.height - 1)))

            rand_dim = self.__get_rand_position(Size(dimensions.width - rand_pos.x, dimensions.height - rand_pos.y), start=Point(min_room_size.width, min_room_size.height))
            rand_dim = Size(rand_dim.x, rand_dim.y)

            offset = self.__get_rand_position(rand_dim)
            rand_pos = Point(rand_pos.x - offset.x, rand_pos.y - offset.y)
            rand_pos = Point(max(0, rand_pos.x), max(0, rand_pos.y))

            self.__place_room(grid, rand_dim, rand_pos)

            # plt.imshow(grid, cmap="gray_r")
            # plt.show()

        # create doors

        available_spots: List[Point] = []

        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == DenseMap.CLEAR_ID:
                    available_spots.append(Point(j, i))

        agent_pos_idx = int(torch.randint(0, len(available_spots), (1,)).item())

        while True:
            goal_pos_idx = int(torch.randint(0, len(available_spots), (1,)).item())
            if goal_pos_idx != agent_pos_idx:
                break

        grid[available_spots[agent_pos_idx].y][available_spots[agent_pos_idx].x] = DenseMap.AGENT_ID
        grid[available_spots[goal_pos_idx].y][available_spots[goal_pos_idx].x] = DenseMap.GOAL_ID
        """

        grid: List[List[int]] = [[DenseMap.CLEAR_ID for _ in range(dimensions.width)] for _ in range(dimensions.height)]
        rooms: List[Tuple[Point, Size]] = []

        def in_bounds(x, y):
            return 0 <= x < len(grid[0]) and 0 <= y < len(grid)

        def __place_room(grid: List[List[int]], size: Size, top_left_corner: Point) -> None:
            for x in range(top_left_corner.x, top_left_corner.x + size.width):
                for y in range(top_left_corner.y, top_left_corner.y + size.height):
                    if in_bounds(x, y):
                        grid[y][x] = DenseMap.CLEAR_ID

            for x in range(top_left_corner.x, top_left_corner.x + size.width):
                y = top_left_corner.y
                if in_bounds(x, y):
                    grid[y][x] = DenseMap.WALL_ID

                y = top_left_corner.y + size.height - 1
                if in_bounds(x, y):
                    grid[y][x] = DenseMap.WALL_ID

            for y in range(top_left_corner.y, top_left_corner.y + size.height):
                x = top_left_corner.x
                if in_bounds(x, y):
                    grid[y][x] = DenseMap.WALL_ID

                x = top_left_corner.x + size.width - 1
                if in_bounds(x, y):
                    grid[y][x] = DenseMap.WALL_ID

            rooms.append((top_left_corner, size))

        def __get_subdivision_number(dim, vertical):
            if vertical:
                min_size = min_room_size.width
                dim_size = dim.width
            else:
                min_size = min_room_size.height
                dim_size = dim.height

            # try hallway
            # if int(torch.randint(0, 11, (1,)).item()) == 0:
            #    return hallway_size + 1

            total_split = dim_size - 2 * min_size + 1
            rand_split = torch.randint(0, total_split + 1, (1,)).item()
            return min_size + rand_split

        def __subdivide(top_left_corner, dim):
            if dim.width < 2 * min_room_size.width - 1 and dim.height < 2 * min_room_size.height - 1:
                __place_room(grid, dim, top_left_corner)
                return
            elif dim.width < 2 * min_room_size.width - 1:
                is_vertical_split = False
            elif dim.height < 2 * min_room_size.height - 1:
                is_vertical_split = True
            else:
                is_vertical_split = int(torch.randint(0, 2, (1,)).item()) == 0

            if dim.width <= max_room_size.width and dim.height <= max_room_size.height:
                if int(torch.randint(0, 2, (1,)).item()) == 0:
                    __place_room(grid, dim, top_left_corner)
                    return

            # split
            new_top_left_corner1 = top_left_corner

            if is_vertical_split:
                new_dim1 = Size(__get_subdivision_number(dim, True), dim.height)
                new_top_left_corner2 = Point(top_left_corner.x + new_dim1.width - 1, top_left_corner.y)
                new_dim2 = Size(dim.width - new_dim1.width + 1, dim.height)
            else:
                new_dim1 = Size(dim.width, __get_subdivision_number(dim, False))
                new_top_left_corner2 = Point(top_left_corner.x, top_left_corner.y + new_dim1.height - 1)
                new_dim2 = Size(dim.width, dim.height - new_dim1.height + 1)

            __subdivide(new_top_left_corner1, new_dim1)
            __subdivide(new_top_left_corner2, new_dim2)

        # place rooms
        __subdivide(Point(-1, -1), Size(dimensions.width + 2, dimensions.height + 2))

        def __get_nearby_rooms_edges(room):
            room_top_left_point, room_size = room
            edges = []
            full_edge = []
            q = 0

            for x in range(room_top_left_point.x, room_top_left_point.x + room_size.width):
                y = room_top_left_point.y
                # up
                if in_bounds(x, y):
                    full_edge.append(Point(x, y))
                    n_x, n_y = x, y - 1
                    if in_bounds(n_x, n_y) and grid[n_y][n_x] == DenseMap.WALL_ID:
                        edges.append(q)
                    q += 1

            for y in range(room_top_left_point.y, room_top_left_point.y + room_size.height):
                x = room_top_left_point.x

                # right
                x = room_top_left_point.x + room_size.width - 1
                if in_bounds(x, y):
                    full_edge.append(Point(x, y))
                    n_x, n_y = x + 1, y
                    if in_bounds(n_x, n_y) and grid[n_y][n_x] == DenseMap.WALL_ID:
                        edges.append(q)
                    q += 1

            for x in range(room_top_left_point.x, room_top_left_point.x + room_size.width):
                y = room_top_left_point.y

                # bottom
                y = room_top_left_point.y + room_size.height - 1
                if in_bounds(x, y):
                    full_edge.append(Point(x, y))
                    n_x, n_y = x, y + 1
                    if in_bounds(n_x, n_y) and grid[n_y][n_x] == DenseMap.WALL_ID:
                        edges.append(q)
                    q += 1

            for y in range(room_top_left_point.y, room_top_left_point.y + room_size.height):
                x = room_top_left_point.x

                # left
                if in_bounds(x, y):
                    full_edge.append(Point(x, y))
                    n_x, n_y = x - 1, y
                    if in_bounds(n_x, n_y) and grid[n_y][n_x] == DenseMap.WALL_ID:
                        edges.append(q)
                    q += 1

            return edges, full_edge

        doors = set()

        # place doors
        for room in rooms:
            neighbours, full_edge = __get_nearby_rooms_edges(room)

            if len(full_edge) <= door_size + 2:
                continue

            for i in range(len(neighbours)):
                nxt = (i + 1) % len(neighbours)

                if abs(neighbours[i] - neighbours[nxt]) >= door_size + 1 or len(neighbours) == 1:
                    found_door = False
                    # check if no doors are placed
                    for q in range(
                            len(full_edge) if len(neighbours) == 1 else abs(neighbours[i] - neighbours[nxt] + 1)):
                        if full_edge[(neighbours[i] + q) % len(full_edge)] in doors:
                            found_door = True
                            break

                    if found_door:
                        continue

                    # place door

                    if int(torch.randint(0, 4, (1,)).item()) == 0:
                        continue

                    diff = abs(neighbours[i] - neighbours[nxt]) - door_size + 1

                    if len(neighbours) == 1:
                        diff = len(full_edge)

                    start = int(torch.randint(1, diff, (1,)).item())
                    next_door = neighbours[i] + start
                    for q in range(door_size):
                        pt = full_edge[(next_door + q) % len(full_edge)]
                        grid[pt.y][pt.x] = DenseMap.CLEAR_ID
                        doors.add(pt)

        self.__place_random_agent_and_goal(grid, dimensions)

        return DenseMap(grid)

    def generate_maps(self, nr_of_samples: int, dimensions: Size, gen_type: str, fill_range: List[float],
                      nr_of_obstacle_range: List[int], min_map_range: List[int], max_map_range: List[int],json_save: bool = False) -> List[Map]:
        if gen_type not in Generator.AVAILABLE_GENERATION_METHODS:
            raise Exception(
                "Generation type {} does not exist in {}".format(gen_type, self.AVAILABLE_GENERATION_METHODS))

        if nr_of_samples <= 0:
            return []

        self.__services.debug.write("""Starting Generation: [
            nr_of_samples: {},
            gen_type: {},
            dimensions: {},
            fill_range: {},
            nr_of_obstacle_range: {},
            min_map_range: {},
            max_map_range: {}
        ] 
        """.format(nr_of_samples, gen_type, dimensions, fill_range, nr_of_obstacle_range, min_map_range, max_map_range), DebugLevel.BASIC)

        atlas_name = "{}_{}".format(gen_type, str(nr_of_samples))
        atlas: Atlas = self.__services.resources.maps_dir.create_atlas(atlas_name)
        progress_bar: Progress = self.__services.debug.progress_debug(nr_of_samples, DebugLevel.BASIC)
        progress_bar.start()
        maps: List[Map] = []

        for _ in range(nr_of_samples):

            fill_rate = fill_range[0] + torch.rand((1,)) * (fill_range[1] - fill_range[0])
            if gen_type == "uniform_random_fill": #random fill
                mp: Map = self.__generate_random_map(dimensions, fill_rate)
            elif gen_type == "block_map": #block
                mp: Map = self.__generate_random_const_obstacles(
                    dimensions,
                    fill_rate,
                    int(torch.randint(nr_of_obstacle_range[0], nr_of_obstacle_range[1], (1,)).item())
                )
            else: #house map
                min_map_size = int(torch.randint(min_map_range[0], min_map_range[1], (1,)).item())
                print(min_map_size)
                max_map_size = int(torch.randint(max_map_range[0], max_map_range[1], (1,)).item())
                print(max_map_size)
                mp: Map = self.__generate_random_house(
                    dimensions,
                    min_room_size=Size(min_map_size, min_map_size),
                    max_room_size=Size(max_map_size, max_map_size),
                )
            #print('grid is \n', mp.grid)
            atlas.append(mp)
            maps.append(mp)
            progress_bar.step()
          
            map_as_dict = {
                "goal" : [mp.goal.position.x,mp.goal.position.y],
                "agent" : [mp.agent.position.x, mp.agent.position.y],
                "grid" : mp.grid 
            }
            if json_save: 
                with open('output path here'+ str(_) + '.json', 'w') as outfile:
                    json.dump(map_as_dict,outfile)
                    self.__services.debug.write("Dumping JSON: " + str(_) + "\n", DebugLevel.LOW)


        #print(maps[1].grid)

        self.__services.debug.write("Finished atlas generation: " + str(atlas_name) + "\n", DebugLevel.BASIC)
        return maps

    def label_maps(self, atlases: List[str], feature_list: List[str], label_list: List[str],
                   single_feature_list: List[str], single_label_list: List[str]) -> None:
        if not atlases:
            return

        self.__services.debug.write("""Starting Labelling: [
            atlases: {},
            feature_list: {},
            label_list: {},
            single_feature_list: {},
            single_label_list: {}
        ] 
        """.format(atlases, feature_list, label_list, single_feature_list, single_label_list), DebugLevel.BASIC)

        label_atlas_name = "training_" + "_".join(atlases)

        # special case where we only have 1 atlas, overwrite is True by default
        if len(atlases) == 1:
            self.__services.debug.write("Processing single atlas (overwrite True)", DebugLevel.BASIC)
            res = self.__label_single_maps(atlases[0], feature_list, label_list, single_label_list, single_label_list,
                                        True)
            self.__save_training_data(label_atlas_name, res)
            return

        # more atlases

        t: List[Dict[str, any]] = []

        for name in atlases:
            training_atlas = "training_" + name
            self.__services.debug.write("Processing " + str(training_atlas), DebugLevel.BASIC)
            next_res: List[Dict[str, any]] = self.__label_single_maps(name, feature_list, label_list, single_label_list,
                                                                      single_label_list, False)

            # save if it does not exist
            if not self.__services.resources.training_data_dir.exists(training_atlas, ".pickle"):
                self.__save_training_data(training_atlas, next_res)
            t = t + next_res

        self.__save_training_data(label_atlas_name, t)
     
    def __save_training_data(self, training_name: str, training_data: List[Dict[str, Any]]) -> None:
        self.__services.debug.write("Saving atlas labelling: " + training_name, DebugLevel.BASIC)
        self.__services.resources.training_data_dir.save(training_name, training_data)
        self.__services.debug.write("Finished atlas labelling: " + training_name + "\n", DebugLevel.BASIC)

    def __label_single_maps(self, atlas_name, feature_list: List[str], label_list: List[str],
                            single_feature_list: List[str], single_label_list: List[str], overwrite: bool) -> List[
        Dict[str, any]]:
        """
        Passed atlas name, feature list, label list, and returns res object with the map features labelled for training
        """
        if not atlas_name:
            return []

        if not overwrite and self.__services.resources.training_data_dir.exists("training_" + atlas_name, ".pickle"):
            self.__services.debug.write("Found in training data. Loading from training data", DebugLevel.BASIC)
            return self.__services.resources.training_data_dir.load("training_" + atlas_name)

        self.__services.debug.write("Loading maps", DebugLevel.BASIC)
        maps: List[Map] = self.__services.resources.maps_dir.get_atlas(atlas_name).load_all()

        res: List[Dict[str, any]] = []

        progress_bar: Progress = self.__services.debug.progress_debug(len(maps), DebugLevel.BASIC)
        progress_bar.start()

        # process atlas
        for m in maps:
            config = Configuration()
            config.simulator_algorithm_type = AStar
            config.simulator_testing_type = AStarTesting
            config.simulator_initial_map = m
            services: Services = Services(config)
            simulator: Simulator = Simulator(services)
            testing: AStarTesting = simulator.start()

            features: Dict[str, any] = {}
            arg: str
            for arg in ["map_obstacles_percentage",
                        "goal_found",
                        "distance_to_goal",
                        "original_distance_to_goal",
                        "trace",
                        "total_steps",
                        "total_distance",
                        "total_time",
                        "algorithm_type",
                        "fringe",
                        "search_space"
                        ]:
                features[arg] = testing.get_results()[arg]

            features["features"] = MapProcessing.get_sequential_features(testing.map, feature_list)
            features["labels"] = MapProcessing.get_sequential_labels(testing.map, label_list)
            features["single_features"] = MapProcessing.get_single_features(m, single_feature_list)
            features["single_labels"] = MapProcessing.get_single_labels(m, single_label_list)
            res.append(features)
            progress_bar.step()

        return res

    def augment_label_maps(self, atlases: List[str], feature_list: List[str], label_list: List[str],
                           single_feature_list: List[str], single_label_list: List[str]) -> None:
        if not atlases:
            return

        self.__services.debug.write("""Starting Augmentation: [
            atlases: {},
            feature_list: {},
            label_list: {},
            single_feature_list: {},
            single_label_list: {}
        ] 
        """.format(atlases, feature_list, label_list, single_feature_list, single_label_list), DebugLevel.BASIC)

        label_atlas_name = "training_" + "_".join(atlases)

        self.__services.debug.write("Loading maps", DebugLevel.BASIC)
        maps: List[Map] = []
        for name in atlases:
            maps = maps + self.__services.resources.maps_dir.get_atlas(name).load_all()

        self.__services.debug.write("Loading atlas", DebugLevel.BASIC)
        t: List[Dict[str, any]] = self.__services.resources.training_data_dir.load(label_atlas_name)

        progress_bar: Progress = self.__services.debug.progress_debug(len(t), DebugLevel.BASIC)
        progress_bar.start()

        for i in range(len(t)):
            config = Configuration()
            config.simulator_algorithm_type = AStar
            config.simulator_testing_type = AStarTesting
            config.simulator_initial_map = maps[i]
            services: Services = Services(config)
            simulator: Simulator = Simulator(services)
            testing: AStarTesting = simulator.start()

            if feature_list:
                seq_features = MapProcessing.get_sequential_features(testing.map, feature_list)
                for q in range(len(t[i]["features"])):
                    t[i]["features"][q].update(seq_features[q])

            if label_list:
                seq_labels = MapProcessing.get_sequential_labels(testing.map, label_list)
                for q in range(len(t[i]["labels"])):
                    # print(q)
                    t[i]["labels"][q].update(seq_labels[q])

            if single_feature_list:
                t[i]["single_features"].update(MapProcessing.get_single_features(maps[i], single_feature_list))

            if single_label_list:
                t[i]["single_labels"].update(MapProcessing.get_single_labels(maps[i], single_label_list))
            progress_bar.step()

        self.__services.debug.write("Saving atlas augmentation: " + str(label_atlas_name), DebugLevel.BASIC)
        self.__services.resources.training_data_dir.save(label_atlas_name, t)
        self.__services.debug.write("Finished atlas augmentation: " + str(label_atlas_name) + "\n", DebugLevel.BASIC)

    def modify_map(self, map_name: str, modify_f: Callable[[Map], Map]) -> None:
        mp: Map = self.__services.resources.maps_dir.load(map_name)
        mp = modify_f(mp)
        self.__services.resources.maps_dir.save(map_name, mp)

    def convert_house_expo(self):
        path = './resources/house_expo/'
        print("Taking images from" + path) 
        #print(os.listdir(path))
        for filename in natsorted(os.listdir(path)):
            print('filename:', filename)
            self.generate_map_from_image(filename,True,2,True)

    @staticmethod
    def increment_moving_average(cur_value: np.ndarray, count: int, new_number: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Increments the average using online average update
        :param cur_value: The current value of the moving average
        :param count: The number of elements in the average
        :param new_number: The new number that needs to be added to the average
        :return: The new average along with the new number of elements
        """
        cur_value += (new_number - cur_value) / count
        return cur_value, count + 1

    @staticmethod
    def is_in_color_range(actual_color: Tuple[int, int, int, int], search_color: Tuple[int, int, int, int], eps: float = None) -> bool:
        """
        Checks if the colors are close enough to each other
        :param actual_color: The actual color
        :param search_color: The other color
        :return: The result
        """

        if not eps:
            eps = Generator.COLOR_EPSILON

        return np.linalg.norm(
            np.array(actual_color, dtype=float) - np.array(search_color, dtype=float)) < eps

    @staticmethod
    def main(m: 'MainRunner') -> None:
        """
        generator: Generator = Generator(m.main_services)
        generator.generate_map_from_image("map14", True, 2)
        return # TODO Remove this
        """
        generator: Generator = Generator(m.main_services)

        if m.main_services.settings.generator_modify:
            generator.modify_map(*m.main_services.settings.generator_modify())

        if not m.main_services.settings.generator_house_expo:
            if m.main_services.settings.generator_size == 8: #Fill rate and nr obstacle range (1,2) is for unifrom random fill (0.1,0.2)
                maps = generator.generate_maps(m.main_services.settings.generator_nr_of_examples, Size(8, 8),
                                        m.main_services.settings.generator_gen_type, [0.1, 0.2], [1, 2], [3,4], [5, 7],json_save = True)

            if m.main_services.settings.generator_size == 16:
                maps = generator.generate_maps(m.main_services.settings.generator_nr_of_examples, Size(16, 16),
                                        m.main_services.settings.generator_gen_type, [0.1, 0.2], [1, 4], [4,6], [8, 11],json_save = True)

            if m.main_services.settings.generator_size == 28:
                maps = generator.generate_maps(m.main_services.settings.generator_nr_of_examples, Size(28, 28),
                                        m.main_services.settings.generator_gen_type, [0.1, 0.3], [1, 4], [6,10], [14, 22],json_save = True)

            else:
                maps = generator.generate_maps(m.main_services.settings.generator_nr_of_examples, Size(64, 64),
                                        m.main_services.settings.generator_gen_type, [0.1, 0.3], [1, 6], [8,15], [35, 45],json_save = False)

        #This will display 5 of the maps generated
        if m.main_services.settings.generator_show_gen_sample and not m.main_services.settings.generator_house_expo:
            if m.main_services.settings.generator_nr_of_examples > 0:
                # show sample
                for i in range(5):
                    plt.imshow(maps[i].grid, cmap=CAE.MAP_COLORMAP_FULL)
                    plt.show()
       

        if m.main_services.settings.generator_aug_labelling_features or m.main_services.settings.generator_aug_labelling_labels or \
                m.main_services.settings.generator_aug_single_labelling_features or m.main_services.settings.generator_aug_single_labelling_labels:
            # augment
            generator.augment_label_maps(m.main_services.settings.generator_labelling_atlases,
                                         m.main_services.settings.generator_aug_labelling_features,
                                         m.main_services.settings.generator_aug_labelling_labels,
                                         m.main_services.settings.generator_aug_single_labelling_features,
                                         m.main_services.settings.generator_aug_single_labelling_labels)
        
        if m.main_services.settings.generator_house_expo: 
            generator.convert_house_expo()
            # generator.label_maps(m.main_services.settings.generator_labelling_atlases,
            #                      m.main_services.settings.generator_labelling_features,
            #                      m.main_services.settings.generator_labelling_labels,
            #                      m.main_services.settings.generator_single_labelling_features,
            #                      m.main_services.settings.generator_single_labelling_labels)
     
        else:
            
            generator.label_maps(m.main_services.settings.generator_labelling_atlases,
                                 m.main_services.settings.generator_labelling_features,
                                 m.main_services.settings.generator_labelling_labels,
                                 m.main_services.settings.generator_single_labelling_features,
                                 m.main_services.settings.generator_single_labelling_labels)
