import os, natsort, re
from tqdm import tqdm
import time, random


from module_aladin.config import roles, parens, custom_hanja


from itertools import repeat, chain

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.metrics import mean_squared_log_error as msle

import torch
from torch.utils.data import DataLoader
import math
import time
from torch import nn, optim
from torch.optim import Adam
import locale
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from torcheval.metrics import functional as F_metric

from module_aladin.load_data_cls import idx_to_val
from collections import defaultdict

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["TORCH_USE_CUDA_DSA"] = '1'

# GPU device setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
torch.set_default_device(device)


locale.getpreferredencoding = lambda: "UTF-8"

from module_aladin.torch_train_cls import evaluate_w_score
