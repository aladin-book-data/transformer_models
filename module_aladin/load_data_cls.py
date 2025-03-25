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
    if mode == info_key : continue
    X, y = data['X'], data['y'][data_key]
    X_torch, y_torch = torch.tensor(X),torch.tensor(y)
    dataset[mode] = TensorDataset(X_torch.to(torch.float32),y_torch.to(torch.float32))
  return dataset,data_dict[info_key]

def load_dataloader_iters(data_dict,batch_size,data_key='coded',info_key='info'):
  dataset,info = generate_dataset(data_dict,data_key,info_key)
  loader = DataLoaderDict(dataset)
  iter_dict = loader.make_iter(batch_size)
  return {'iters' : iter_dict, 'info' : info}

def idx_to_val(data,decode_map,sos_idx,eos_idx,max_len,pad_idx=0,pad_pos='post',reverse=False):
  data = list(data)
  s = data.index(sos_idx) if sos_idx in data else -1
  e = data.index(eos_idx) if eos_idx in data else len(data) 
  trimmed = data[s+1:min(e,s+max_len-1)]
  
  pads = np.where(np.array(trimmed)==pad_idx)
  if pad_pos =='post' : trimmed= trimmed[:pads[0]]
  else : trimmed=trimmed[pads[-1]+1:]
  
  if reverse : trimmed = trimmed[::-1]
  val = list(map(lambda x : str(decode_map[x]),list(trimmed)))
  try : return int(''.join(val))
  except : return 0
  
def unzip_log_val(x):
  temp = np.power(10,x/10000)
  return np.round(temp,-2)