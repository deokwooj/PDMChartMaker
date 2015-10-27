# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 23:40:30 2015

@author: deokwooj
"""

"""
* Description 
- This file defines constant values shared among python modules.
- Should be included all python modules first. 
"""
# Loading common python modules to be used. 
import os
import sys, traceback
import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm
import uuid
import pylab as pl
from scipy import signal
from scipy import stats
from scipy import sparse
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from multiprocessing import Pool
#from datetime import datetime
import datetime as dt
from dateutil import tz
import shlex, subprocess
import time
import itertools
import calendar
import random
from matplotlib.collections import LineCollection
import pprint
import warnings
import pdm_tools as pmt

FIG_OUT_DIR='./fig_out/'

# dictionary file path
