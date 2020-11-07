# global #
import pickle
from tkinter import *
import faulthandler
faulthandler.enable()

# local #

# general
from main import MainRunner
from maps import Maps
from simulator.services.debug import DebugLevel

# planner testing
from algorithms.basic_testing import BasicTesting
from algorithms.classic.testing.a_star_testing import AStarTesting
from algorithms.classic.testing.combined_online_lstm_testing import CombinedOnlineLSTMTesting
from algorithms.classic.testing.dijkstra_testing import DijkstraTesting
from algorithms.classic.testing.wavefront_testing import WavefrontTesting
from algorithms.classic.testing.way_point_navigation_testing import WayPointNavigationTesting

# planner implementations
from algorithms.classic.graph_based.a_star import AStar
from algorithms.classic.graph_based.bug1 import Bug1
from algorithms.classic.graph_based.bug2 import Bug2
from algorithms.classic.graph_based.dijkstra import Dijkstra
from algorithms.classic.sample_based.sprm import SPRM
from algorithms.classic.sample_based.rt import RT
from algorithms.classic.sample_based.rrt import RRT
from algorithms.classic.sample_based.rrt_star import RRT_Star
from algorithms.classic.sample_based.rrt_connect import RRT_Connect
from algorithms.classic.graph_based.wavefront import Wavefront
from algorithms.configuration.configuration import Configuration
from algorithms.lstm.LSTM_tile_by_tile import OnlineLSTM
from algorithms.lstm.a_star_waypoint import WayPointNavigation
from algorithms.lstm.combined_online_LSTM import CombinedOnlineLSTM
#Pathreader debugging tool
from algorithms.pathreader import PathReader



class GUI:
    __maps = {
        "Uniform Random Fill": ("uniform_random_fill_10/0", True),
        "Block": ("block_map_10/6", True),
        "House": ("house_10/6", True),
        "Long Wall": (Maps.grid_map_labyrinth2, True),
        "Labyrinth": (Maps.grid_map_labyrinth, True),
        "Small Obstacle": (Maps.grid_map_one_obstacle.convert_to_dense_map(), True),
        "SLAM Map 1": ("map10", False),
        "SLAM Map 1 (compressed)": ("map11", True),
        "SLAM Map 2": ("map14", False),
        "SLAM Map 3": ("map12", False),



        }


    __algorithms = {
        "A*": (AStar, AStarTesting, ([], {})),
        "WPN1":(WayPointNavigation, WayPointNavigationTesting, ([], {"global_kernel_max_it": 20, "global_kernel": (CombinedOnlineLSTM, ([], {}))})),
        "WPN2 (LSTM) ":(WayPointNavigation, WayPointNavigationTesting, ([], {"global_kernel_max_it": 20, "global_kernel": (OnlineLSTM, ([], {"load_name": "tile_by_tile_training_uniform_random_fill_10000_block_map_10000_house_10000_model"}))})),
        "WPN3 (CAE)": (WayPointNavigation, WayPointNavigationTesting, ([], {"global_kernel_max_it": 20, "global_kernel": (OnlineLSTM, ([], {"load_name": "caelstm_section_lstm_training_block_map_10000_model"}))})),
        "LSTM Bagging": (CombinedOnlineLSTM, CombinedOnlineLSTMTesting, ([], {})),
        "Map Module (CAE) ": (OnlineLSTM, BasicTesting, ([], {"load_name": "caelstm_section_lstm_training_uniform_random_fill_10000_block_map_10000_model"})),
        "View Module (Online LSTM)": (OnlineLSTM, BasicTesting, ([], {"load_name": "tile_by_tile_training_uniform_random_fill_10000_block_map_10000_house_10000_model"})),
        "SPRM": (SPRM, BasicTesting, ([], {})),
        "RT": (RT, BasicTesting, ([], {})),
        "RRT": (RRT, BasicTesting, ([], {})),
        "RRT*": (RRT_Star, BasicTesting, ([], {})),
        "RRT-Connect": (RRT_Connect, BasicTesting, ([], {})),
        "Wave-front": (Wavefront, WavefrontTesting, ([], {})),
        "Dijkstra": (Dijkstra, DijkstraTesting, ([], {})),
        "Bug1": (Bug1, BasicTesting, ([], {})),
        "Bug2": (Bug2, BasicTesting, ([], {})),
        "Path Reader": (PathReader, BasicTesting, ([], {})),
    }

    __animations = {
        "None": (0, 0),
        "Normal": (0.00001, 0),
        "Slow": (0.5, 0),
        "Fast": (0.00001, 20)
    }

    __debug = {
        "None": DebugLevel.NONE,
        "Basic": DebugLevel.BASIC,
        "Low": DebugLevel.LOW,
        "Medium": DebugLevel.MEDIUM,
        "High": DebugLevel.HIGH,
    }

    def __init__(self):
        self.__window = Tk()
        self.__window.title("PathBench")
        self.__window.bind('<Escape>', lambda event: GUI.__on_closing_callback(self))
        self.__window.protocol("WM_DELETE_WINDOW", lambda: GUI.__on_closing_callback(self))
        self.__window.bind('<Return>', lambda event: GUI.__start_simulator_callback(self))

        self.__map_choice = StringVar(self.__window, "Small Obstacle")
        self.__algorithm_choice = StringVar(self.__window, "A*")
        self.__animations_choice = StringVar(self.__window, "None")
        self.__debug_choice = StringVar(self.__window, "Basic")

        self.__config_from_cache()

    def __cache_config(self):
        config = {
            "mp": self.__map_choice.get(),
            "algo": self.__algorithm_choice.get(),
            "ani": self.__animations_choice.get(),
            "debug": self.__debug_choice.get(),
        }

        with open("gui_config.pickle", "wb") as f:
            pickle.dump(config, f)

    def __config_from_cache(self):
        try:
            with open("gui_config.pickle", "rb") as f:
                config = pickle.load(f)
        except FileNotFoundError:
            return

        self.__map_choice.set(config["mp"])
        self.__algorithm_choice.set(config["algo"])
        self.__animations_choice.set(config["ani"])
        self.__debug_choice.set(config["debug"])

    @staticmethod
    def __on_closing_callback(gui):
        gui.__cache_config()
        gui.__window.quit()

    @staticmethod
    def __start_simulator_callback(gui): #Setting the configuration file.
        config = Configuration()

        mp = gui.__maps[gui.__map_choice.get()]
        algo = gui.__algorithms[gui.__algorithm_choice.get()]
        ani = gui.__animations[gui.__animations_choice.get()]
        debug = gui.__debug[gui.__debug_choice.get()]

        config.load_simulator = True
        config.simulator_graphics = True
        config.simulator_initial_map, config.simulator_grid_display = mp
        config.simulator_algorithm_type, config.simulator_testing_type, config.simulator_algorithm_parameters = algo
        config.simulator_key_frame_speed, config.simulator_key_frame_skip = ani
        config.simulator_write_debug_level = debug

        MainRunner(config).run()

    def __center_window(self):
        width = self.__window.winfo_reqwidth()
        height = self.__window.winfo_reqheight()
        screen_width = self.__window.winfo_screenwidth()
        screen_height = self.__window.winfo_screenheight()
        self.__window.geometry("+{}+{}".format(int(screen_width / 2 - width / 2), int(screen_height / 2 - height / 2)))

    def start(self):
        # put title
        sim_label = Label(self.__window, text="PathBench Simulator Configuration", font=("Helvetica", 16))

        # put help label
        help_label = Label(self.__window, text=
"""Please select your configuration and press Start Simulator. If you would like to run 
another session close the current simulator window and repeat the same steps.

Important runtime commands:
    * escape - exit the simulator
    * c - find the path between the agent and goal
    * mouse click - moves agent to mouse location 
    * mouse right click - moves goal to mouse location
    
For additional commands please check the simulator log (debug level >= Basic)""", justify=LEFT)

        # put map option menu
        map_frame = Frame(self.__window)
        map_label = Label(map_frame, text="Map: ")
        map_option = OptionMenu(map_frame, self.__map_choice, *self.__maps.keys())

        # put algorithm option menu
        algorithm_frame = Frame(self.__window)
        algorithm_label = Label(algorithm_frame, text="Algorithm: ")
        algorithm_option = OptionMenu(algorithm_frame, self.__algorithm_choice, *self.__algorithms.keys())

        # put animations
        animations_frame = Frame(self.__window)
        animations_label = Label(animations_frame, text="Animations: ")
        animations_option = OptionMenu(animations_frame, self.__animations_choice, *self.__animations.keys())

        # put debug level
        debug_frame = Frame(self.__window)
        debug_label = Label(debug_frame, text="Debug Level: ")
        debug_option = OptionMenu(debug_frame, self.__debug_choice, *self.__debug.keys())

        # sim start
        sim_start_button = Button(self.__window, highlightbackground='black', text='Start Simulator', width=25, command=lambda: GUI.__start_simulator_callback(self))

        sim_label.pack()
        help_label.pack()

        map_frame.pack(side=TOP)
        map_label.pack(side=LEFT, padx=5, pady=5)
        map_option.pack(side=LEFT, padx=5, pady=5)

        algorithm_frame.pack(side=TOP)
        algorithm_label.pack(side=LEFT, padx=5, pady=5)
        algorithm_option.pack(side=LEFT, padx=5, pady=5)

        animations_frame.pack(side=TOP)
        animations_label.pack(side=LEFT, padx=5, pady=5)
        animations_option.pack(side=LEFT, padx=5, pady=5)

        debug_frame.pack(side=TOP)
        debug_label.pack(side=LEFT, padx=5, pady=5)
        debug_option.pack(side=LEFT, padx=5, pady=5)

        sim_start_button.pack(pady=10)

        self.__window.update()
        self.__window.focus()
        self.__center_window()
        self.__window.mainloop()


if __name__ == '__main__':
    gui = GUI()
    gui.start()