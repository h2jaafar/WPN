from simulator.services.resources.directory import Directory 
from simulator.services import services
import pickle
from io import StringIO
import os
import shutil

import dill as pickle  # used to pickle lambdas
from typing import BinaryIO, Callable, Any

from simulator.services.debug import DebugLevel
from simulator.services.service import Service

"""
This program will read a pickle file to analyze the conents. Useful for debugging
"""

trainingdata = open(
    "resources/maps/house_expo_52.pickle",
    "rb")
emp = pickle.load(trainingdata)
print(emp)

#The training files are full now. Before they were empty
