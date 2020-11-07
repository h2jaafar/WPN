import copy
from threading import Thread
from pathos.multiprocessing import ProcessPool
#import torch.multiprocessing as multip
# import multiprocess as multip
from typing import List, Tuple, Optional, Set
from joblib import wrap_non_picklable_objects
from joblib.externals.loky import set_loky_pickler
from joblib import parallel_backend
from joblib import Parallel, delayed
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

import datetime

class CombinedOnlineLSTM(Algorithm):
    kernel_names: List[str]
    _max_it: float
    _threaded: bool

    __active_kernel: Optional[AlgorithmRunner]
    __total_path: Set[Point]
    t = datetime.datetime.now()
    def __init__(self, services: Services, testing: BasicTesting = None, kernel_names: List["str"] = None,
                 max_it: float = float('inf'), threaded: bool = False): #Changed threaded to true
        super().__init__(services, testing)
        #print(datetime.datetime.now(), ' Line 28')
        # print('\n')
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
        #print(datetime.datetime.now(), ' Line 53')
        if self.__active_kernel:
            active_kernel_displays = [
                # *self.__active_kernel.instance.set_display_info(),
                OnlineLSTMMapDisplay(self._services, custom_map=self.__active_kernel.map),
                EntitiesMapDisplay(self._services, custom_map=self.__active_kernel.map),
            ]
        #print(datetime.datetime.now(), ' Line 60')
        return super().set_display_info() + [
            *active_kernel_displays,
            SolidColorMapDisplay(self._services, self.__total_path, (0, 150, 0), z_index=80),
        ]
    
    @wrap_non_picklable_objects
    def kernels_cal(self,k):
        print('KKKK', k)
        self.__active_kernel = k[1]
        k[1].find_path()
        print('K after', k)
        print("goal reacheD:", k[1].map.is_goal_reached(k[1].map.agent.position))
        self.__total_path = self.__total_path.union(set(map(lambda el: el.position, k[1].map.trace)))
        # print('self.total_path', self.__total_path)
        return
    def __call__(self, x):   
        return self.kernels_cal(x)
    # TODO when max_it is inf take the solution where we are closer to the goal or implement special case
    def _find_path_internal(self) -> None:
        #print(datetime.datetime.now(), ' Line 68')
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
        #print('Kernels ', kernels)
        
        #TODO: HANGS here
        # print('Line 86')
        if self._threaded:
            # threaded_jobs: List[Process] = list(
            #     map(lambda kernel: multip.Process(target=kernel[1].find_path, daemon=True), kernels))
            
            # print(datetime.datetime.now(), ' Line 85')
            # print('Threaded Jobs: ', threaded_jobs)
            #multip.set_start_method('spawn')
            # print('Kernals \n \n', kernels)
            i = 0
            # lambda kernel : kernel[1].find_path #function
            #Data = kernels 
            print("Parallel")
            Parallel(n_jobs=9)(self.kernels_cal(self,_) for _ in kernels)
            p = ProcessPool(10)
            sc = p.map(self.kernels_cal, kernels)
            pool = ProcessPool(nodes = 10)
            returned_results = pool.imap(self.test_multip,num)

            # returned_results = pool.imap(self.kernels_cal,kernels)

            # for j in threaded_jobs:
            #     i+=1 
            #     print('\n Started # ',i, j)
            #     j.start()
            #     # j.join()
            # i = 0
            # for j in threaded_jobs:
            #     i+=1
            #     print('\n Joined # ',i, j)
            #     j.join()
        else: #It goes here #TODO: Figure out why it hangs
            #print('Kernels: ',kernels) #iterates through 10 kernels (max it = 10)
            for k in kernels:
                #print('kernel is ', k) #Kernel is tuple with a number (0-10) and the algorithm ()?
                self.__active_kernel = k[1] #Problem is next three lines
                # self.t1 = datetime.datetime.now()
                k[1].find_path() #HANGS HERE!!! This takes 0.2 seconds
                # self.t2 = datetime.datetime.now()
                # print('Time: ', (self.t2-self.t1))
                self.__total_path = self.__total_path.union(set(map(lambda el: el.position, k[1].map.trace)))
        # print(datetime.datetime.now(), ' Line 96')
        # print('Gets to line 114')
        self.__active_kernel = None

        # check if any found path and if they did take smallest dist
        best_kernels: List[Tuple[int, AlgorithmRunner]] = []

        for kernel in kernels:
            if kernel[1].map.is_goal_reached(kernel[1].map.agent.position):
                best_kernels.append(kernel)
        #print(datetime.datetime.now(), ' Line 106')

        # take smallest dist kernel if any
        dist: float = float("inf")
        best_kernel: Tuple[int, AlgorithmRunner] = None
        for kernel in best_kernels:
            if dist > len(kernel[1].map.trace):
                dist = len(kernel[1].map.trace)
                best_kernel = kernel
        # print(datetime.datetime.now(), ' Line 115')
        if best_kernel:
            best_kernel[1].map.replay_trace(self.__replay)
        else:
            # pick the one with furthest progress
            dist = -1
            best_kernel = None
            #print('Kernels', kernels)
            for kernel in kernels:
                #print('Kernel', kernel)
                if dist < len(kernel[1].map.trace):
                    dist = len(kernel[1].map.trace)
                    best_kernel = kernel
            # print(datetime.datetime.now(), ' Line 126')
            best_kernel[1].map.replay_trace(self.__replay)
        self.kernel_call_idx = best_kernel[0]
        # print(datetime.datetime.now(), ' Line 133')

    def __replay(self, m: Map) -> None:
        self.move_agent(m.agent.position)
        self.key_frame()