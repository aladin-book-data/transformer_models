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
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_squared_log_error as msle

import torch
from torch.utils.data import DataLoader
import math
import time
from torch import nn, optim
from torch.optim import Adam
import locale
from sklearn.metrics import r2_score, mean_absolute_percentage_error

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["TORCH_USE_CUDA_DSA"] = '1'

CORPUS_SIZE = 33800

# GPU device setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
torch.set_default_device(device)

locale.getpreferredencoding = lambda: "UTF-8"

class ScaleDotProductAttention(nn.Module):
  def __init__(self):
    super().__init__()
    self.softmax = nn.Softmax(dim=-1)

  def forward(self,q,k,v,mask=None):
    # q : (batch_size, head, seq_len, head_dim) array
    # head_dim = d_k
    _,_,_,head_dim = q.size()
    attention_score = torch.matmul(q,k.transpose(-1,-2))
    attention_score /= math.sqrt(head_dim)

    # mask : 단어는 1, 마스킹 된 곳은 0으로 표시되어 있음
    if mask is not None:
      attention_score = attention_score.masked_fill(mask==0,-1e10)

    attention_score = self.softmax(attention_score)
    return torch.matmul(attention_score,v), attention_score

class PositionWiseFCFeedForwardNetwork(nn.Module):
  def __init__(self,d_model,d_ff):
    super().__init__()
    self.w_1 = nn.Linear(d_model,d_ff)
    self.relu = nn.ReLU()
    self.w_2 = nn.Linear(d_ff,d_model)

  def forward(self,x):
    x = self.w_1(x)
    x = self.relu(x)
    x = self.w_2(x)
    return x

class MultiHeadAttention(nn.Module):
  def __init__(self,d_model,head):
    # d_model : dim of model input (output of embedding layer)
    # head = num of head,
    # w_q,w_k : d_model * d_k
    # w_v : d_model * d_v
    # w_o : h d_v * d_model
    # in paper, d_k = d_v, hd_v = d_model

    super().__init__()
    self.head = head
    self.d_model = d_model
    self.head_dim = d_model // head
    self.w_q = nn.Linear(d_model,d_model)
    self.w_k = nn.Linear(d_model,d_model)
    self.w_v = nn.Linear(d_model,d_model)
    self.w_o = nn.Linear(d_model,d_model)
    self.attention = ScaleDotProductAttention()

  def forward(self,q,k,v,mask=None):
    batch_size,_,_ = q.size()
    q,k,v = self.w_q(q), self.w_k(k),self.w_v(v)

    q = q.view(batch_size,-1,self.head,self.head_dim).transpose(1,2)
    k = k.view(batch_size,-1,self.head,self.head_dim).transpose(1,2)
    v = v.view(batch_size,-1,self.head,self.head_dim).transpose(1,2)
    #Tensor.view ~ reshape, Tensor.transpose(dim0,dim1)

    out, attention_score = self.attention(q,k,v,mask)

    out = out.transpose(1,2).contiguous().view(batch_size,-1,self.d_model)

    out = self.w_o(out)

    return out, attention_score

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient
        pos = torch.arange(0, max_len,device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position
        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]
        batch_size,seq_len,_ = x.size()
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :]
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]

class EncoderLayer(nn.Module):
  def __init__(self,d_model,head,d_ff,dropout):
    super().__init__()
    self.attention = MultiHeadAttention(d_model,head)
    self.layerNorm1 = nn. LayerNorm(d_model)
    self.ffn = PositionWiseFCFeedForwardNetwork(d_model,d_ff)
    self.layerNorm2 = nn.LayerNorm(d_model)
    self.dropout = nn.Dropout(p=dropout)

  def forward(self,x,padding_mask):
    residual = x
    x, attention_score = self.attention(q=x,k=x,v=x,mask=padding_mask)
    x = self.dropout(x)+residual
    x = self.layerNorm1(x)

    residual = x
    x = self.ffn(x)
    x = self.dropout(x)+residual
    x = self.layerNorm2(x)

    return x, attention_score

class DecoderLayer(nn.Module):
  def __init__(self,d_model,head,d_ff,dropout):
    super().__init__()
    self.attention1 = MultiHeadAttention(d_model,head)
    self.layerNorm1 = nn. LayerNorm(d_model)
    self.attention2 = MultiHeadAttention(d_model,head)
    self.layerNorm2 = nn. LayerNorm(d_model)
    self.ffn = PositionWiseFCFeedForwardNetwork(d_model,d_ff)
    self.layerNorm3 = nn.LayerNorm(d_model)
    self.dropout = nn.Dropout(p=dropout)

  def forward(self,x,memory,look_ahead_mask, padding_mask):
    residual = x
    x,_  = self.attention1(q=x,k=x,v=x,mask=look_ahead_mask)
    x = self.dropout(x)+residual
    x = self.layerNorm1(x)

    x,_  = self.attention2(q=x,k=memory,v=memory,mask=padding_mask)
    x = self.dropout(x)+residual
    x = self.layerNorm2(x)

    residual = x
    x = self.ffn(x)
    x = self.dropout(x)+residual
    x = self.layerNorm3(x)

    return x

class BasicEncoder(nn.Module):
  def __init__(self,d_model,head,d_ff,max_len,dropout,n_layers,device): 
    super().__init__()

    self.pos_encoding = PositionalEncoding(d_model,max_len,device)
    self.dropout = nn.Dropout(p=dropout)

    self.encoding_layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                       head = head, d_ff=d_ff,
                                                       dropout = dropout)
                                              for _ in range(n_layers)])

  def forward(self,x,src_mask=None):
    pos_encoding = self.dropout(self.pos_encoding(x))
    batch_size,_,_ = x.size()
    
    pos_encoding=pos_encoding.unsqueeze(dim=0).repeat(batch_size,1,1)
    x = x + pos_encoding
    
    for encoder in self.encoding_layers:
      x,attention_score = encoder(x,src_mask)
    
    
    return x

class EncoderWithEmbedding(BasicEncoder):
  def __init__(self,d_model,head,d_ff,max_len,dropout,n_layers,device,corpus_size=CORPUS_SIZE,pad_idx=0):
    super().__init__(d_model,head,d_ff,max_len,dropout,n_layers,device)
    self.input_emb = nn.Embedding(corpus_size,d_model,padding_idx = pad_idx)

  def forward(self,x,src_mask):
    input_emb = self.dropout(self.input_emb(x))
    return super().forward(input_emb,src_mask)

class Decoder(nn.Module):
  def __init__(self,d_model,head,d_ff,max_len,dropout,n_layers,device,corpus_size=CORPUS_SIZE,pad_idx=0): 
    super().__init__()

    self.output_emb = nn.Embedding(corpus_size,d_model,padding_idx=pad_idx)
    self.pos_encoding = PositionalEncoding(d_model,max_len,device)
    self.dropout = nn.Dropout(p=dropout)

    self.encoding_layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                       head = head, d_ff=d_ff,
                                                       dropout = dropout)
                                              for _ in range(n_layers)])
    self.linear = nn.Linear(d_model,corpus_size)

  def forward(self,x,memory,look_ahead_mask,padding_mask):
    output_emb = self.output_emb(x)
    pos_encoding = self.pos_encoding(x)
    batch_size,_,_ = x.size()
    
    pos_encoding=pos_encoding.unsqueeze(dim=0).repeat(batch_size,1,1)
    x = self.dropout(output_emb + pos_encoding)
    
    for decoder in self.encoding_layers:
      x = decoder(x, memory,look_ahead_mask,padding_mask)
    
    
    return  self.linear(x)



'''
# REFERENCE
-https://github.com/hyunwoongko/transformer/tree/master
-https://code-angie.tistory.com/7

'''
