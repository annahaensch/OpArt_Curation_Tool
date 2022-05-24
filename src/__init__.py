""" Tools for optimal art assignment
"""
__version__ = "0.0.1"

import os
import sys

import numpy as np
import pandas as pd

from .learning import *
from .locations import *
from .visualizations import *
from .preprocessing import *


ROOT = os.popen("git rev-parse --show-toplevel").read().split("\n")[0]
sys.path.append(ROOT)
