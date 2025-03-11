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

import torch

# GPU device setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
torch.set_default_device(device)

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["TORCH_USE_CUDA_DSA"] = '1'

from torch.utils.data import DataLoader

class DataLoaderDict :
  def __init__(self,dataset):
    self.dataset = dataset
  def make_iter(self,batch_size):
    return {
        mode : DataLoader(data,batch_size)#,num_workers = 4)
        for mode, data in self.dataset.items()
    }

import torch
from torch.utils.data import TensorDataset
from collections import defaultdict

def polish_idx(length,crop_idx):
    crop_idx2 = list(map(lambda x : length + x if x < 0 else x, crop_idx))
    return sorted(crop_idx2,reverse=True)

def make_cropped_data(crop_idx, X):
    crop_idx = polish_idx(X.shape[1],crop_idx)
    for i in crop_idx:
        X = np.hstack([X[:,:i],X[:,i+1:]])
    return X

def generate_dataset(data_dict,data_key,info_key='info'):
  dataset = defaultdict(dict)
  for mode, data in data_dict.items():
    X, y = data['X'], data['y'][data_key]
    X_torch, y_torch = torch.tensor(X),torch.tensor(y)
    dataset[mode] = TensorDataset(X_torch.to(torch.float32),y_torch.to(torch.float32))
  return dataset,data_dict[info_key]

def load_dataloader_iters(data_dict,batch_size,data_key='coded',info_key='info'):
  dataset,info = generate_dataset(data_dict,data_key,info_key)
  loader = DataLoaderDict(dataset)
  iter_dict = loader.make_iter(batch_size)
  rslt = {'iters' : iter_dict, 'info' : info}
  return 

def idx_to_val(data,decode_map,pad_idx=0):
  for i,ele in enumerate(data):
    if ele == pad_idx : break
  trimmed = data[:i][::-1]
  val = list(map(lambda x : str(decode_map[x]),trimmed))
#  print(data,val)
#  return int(''.join(val))
  try : return int(''.join(val))
  except : return 0