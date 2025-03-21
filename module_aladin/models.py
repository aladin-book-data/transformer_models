import os, natsort, re
from tqdm import tqdm
import time, random


from module_aladin.config import roles, parens, custom_hanja
from module_aladin.attention_based_model import EncoderWithEmbedding, Decoder


from itertools import repeat, chain

import numpy as np
import pandas as pd
import seaborn as sns
import os


import torch
from torch.utils.data import DataLoader
import time
from torch import nn, optim
from torch.optim import Adam
import locale

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["TORCH_USE_CUDA_DSA"] = '1'

CORPUS_SIZE = 33700

# GPU device setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
torch.set_default_device(device)

locale.getpreferredencoding = lambda: "UTF-8"

class Transformer(nn.Module):
  def __init__(self,d_model,head,d_ff,dropout,n_layers,corpus_info):
    super().__init__()
    self.set_corpus_info(corpus_info)
    self.embd_encoder = EncoderWithEmbedding(d_model,head,d_ff,self.seq_len,dropout,n_layers,device,
                                             self.corpus_size_in,self.pad_idx)
    self.decoder = Decoder(d_model,head,d_ff,self.max_len,dropout,n_layers,device,
                           self.corpus_size_out,self.pad_idx)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(dropout)
    self.linear = nn.Linear(d_model,self.corpus_size_out)
  
  def set_corpus_info(self,info):
    self.corpus_size_in = info['X']['corpus_size']
    self.corpus_size_out = info['y']['corpus_size']
    self.seq_len = info['X']['max_len'] 
    self.max_len = info['y']['max_len'] 
    decode_map = info['y']['decode_map']['map']
    self.decode_map = decode_map
    self.sos_idx = info['y']['tkn']['[SOS]']
    self.eos_idx = info['y']['tkn']['[EOS]']
    self.pad_idx = info['y']['tkn']['[PAD]']    
    
  def make_pad_mask(self,data,dim):
    pad_mask = (data!=self.pad_idx).unsqueeze(1).unsqueeze(dim)
    return pad_mask

  def make_sub_mask(self,trg):
    trg_len = trg.shape[1]
    trg_sub_mask = torch.tril(torch.ones(trg_len,trg_len))
    return trg_sub_mask.type(torch.bool).to(device)
  #bool로 되어있어서 문제가 되는 것은 아닌지? 하지만 0 == False긴 하니깐 문제 없을지도
  #unsqueeze(1) 가 안되어있는게 문제 일지도

  def forward(self,src,trg): #차원 체크해봐야 함
    src_mask = self.make_pad_mask(src,2)
    trg_mask = self.make_pad_mask(trg,3) & self.make_sub_mask(trg)
    enc_src = self.embd_encoder(src.to(torch.int32),src_mask)
    out = self.decoder(trg.to(torch.int32),enc_src,trg_mask,src_mask)
    return self.relu(self.linear(out))

  def infer(self,src):
    batch_size,_ = src.size()
    src_mask = self.make_pad_mask(src,2)
    enc_src = self.embd_encoder(src.to(torch.int32),src_mask)
       
    outputs = torch.mul(torch.ones(batch_size,self.max_len).to(torch.long).to(device),34) 
#    outputs = torch.zeros(batch_size,self.max_len).to(torch.long)
    outputs[:,0] = self.sos_idx
    out_dist = torch.zeros(batch_size,self.max_len,self.corpus_size_out)
    
    for i in range(2,self.max_len):
        trg_mask = self.make_sub_mask(outputs[:,:i]) & self.make_pad_mask(outputs[:,:i],3)
        out = self.decoder(outputs[:,:i],enc_src,trg_mask,src_mask)
        out = self.relu(self.linear(out))
        val = out[:,-1,:].max(dim=-1)
        outputs[:,i],out_dist[:,i] = val[1],out[:,-1,:]
    
    return out_dist
