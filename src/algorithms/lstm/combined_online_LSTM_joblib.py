import copy
from threading import Thread
from typing import List, Tuple, Optional, Set
#Multiprocessing:
from pathos.multiprocessing import ProcessPool
from joblib import wrap_non_picklable_objects
from joblib.externals.loky import set_loky_pickler
from joblib import parallel_backend
from joblib import Parallel, delayed
import time


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

    def kernels_cal(self,k):
        print('In Kernels cal')
        # self.__active_kernel = k[1]
        # self.__active_kernel.find_path()
        # print("Active kernel:", self.__active_kernel)
        # k[1].find_path()
        # self.__total_path = self.__total_path.union(set(map(lambda el: el.position, k[1].map.trace)))
        return

    def kernels_cal_fake(self,k):
        print('active kernel:', k[1])
        return

    def non_pickle_kernels_cal(self,k,*args):
        '''
        This function is iterated through synchronously using joblib processpool
        It is not optimzed too well, given cloudpickle's slow serialization techniques
        
        Pass it a tuple with the kernels you which to find paths on in the structure
        (1, kernel), where 1 is the number for that kernel
        '''
        self.__active_kernel = k[1]
        # print("Active kernel:", self.__active_kernel)
        self.__active_kernel.find_path()
        self.__total_path = self.__total_path.union(set(map(lambda el: el.position, k[1].map.trace)))
        return 


    # TODO when max_it is inf take the solution where we are closer to the goal or implement special case
    def _find_path_internal(self) -> None:
        
        agent_pos = Point(self._get_grid().agent.position.x,self._get_grid().agent.position.y)



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

        #print ('kernelskernels kkkkkkkk', kernels[0][1])
        
        if self._threaded: #Multiprocessing
            
            # parallel_backend('multiprocessing')
            t1 = time.time()
            Parallel(n_jobs=4, prefer='threads')(delayed(self.non_pickle_kernels_cal)(kernel_tuple) for kernel_tuple in kernels)
            t2 = time.time()
            # print('Elapsed time', t2-t1)
            kernels_fake = [(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8) ]
            # Parallel(n_jobs=4)(delayed(print(kernel_tuple)) for kernel_tuple in kernels)
            # p = ProcessPool(10)
            # sc = p.map(self.kernels_cal, kernels)
            # pool = ProcessPool(nodes = 10)
            # returned_results = pool.imap(self.test_multip,num)


        else:
            for k in kernels:
                self.__active_kernel = k[1]
                k[1].find_path()
                # print('***************', k[1].algorithm_parameters)
                #print('33333333333333333',k[1].map.trace[1].position.x,k[1].map.trace[1].position.y)
                self.__total_path = self.__total_path.union(set(map(lambda el: el.position, k[1].map.trace)))


        self.__active_kernel = None

        # check if any found path and if they did take smallest dist
        best_kernels: List[Tuple[int, AlgorithmRunner]] = []

        for kernel in kernels:
            #   print('hihihiihihihihihihihihihihihihihihihihihihi')
            if kernel[1].map.is_goal_reached(kernel[1].map.agent.position):
                # print('kernel---------',kernel)
                best_kernels.append(kernel)

        # take smallest dist kernel if any
        dist: float = float("inf")
        best_kernel: Tuple[int, AlgorithmRunner] = None
        for kernel in best_kernels:
            if dist > len(kernel[1].map.trace):
                dist = len(kernel[1].map.trace)
                best_kernel = kernel


        if best_kernel:
            #best_kernel[1].reset_algorithm()
            global lst
            lst = []
            best_kernel[1].map.replay_trace(self.__replay)

            # print ('************ This is the list of points you want ',lst)
        else:
            # pick the one with furthest progress
            #best_kernel[1].reset_algorithm()
            lst = []
            dist = -1
            best_kernel = None
            for kernel in kernels:
                if dist < len(kernel[1].map.trace):
                    dist = len(kernel[1].map.trace)
                    best_kernel = kernel

            best_kernel[1].map.replay_trace(self.__replay)
            # print ('************ This is the list of points you want ',lst)

        self.kernel_call_idx = best_kernel[0]

    def __replay(self, m: Map) -> None:
        self.move_agent(m.agent.position)
        lst.append(m.agent.position)
        self.key_frame()

        