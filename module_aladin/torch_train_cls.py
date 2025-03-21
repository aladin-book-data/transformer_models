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

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["TORCH_USE_CUDA_DSA"] = '1'

# GPU device setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
torch.set_default_device(device)


locale.getpreferredencoding = lambda: "UTF-8"

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        x,trg = batch[0], batch[1].to(torch.long)
        optimizer.zero_grad()
        output = model(x,trg[:,1:])
        y_pred = output.contiguous().view(-1,output.shape[-1])
        y_actual = trg[:,:-1].contiguous().view(-1)
        loss = criterion(y_pred,y_actual)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def evaluate(model,iterator,criterion,mode='evaluate'):
    model.eval()
    epoch_loss, epoch_loss2,epoch_loss3  = 0, 0, 0
    Y_actual, Y_pred = list(),list()
#    softmax = nn.Softmax(dim=-1)
    with torch.no_grad():
        for i,batch in enumerate(iterator):
            x,trg = batch[0], batch[1].to(torch.long)
            if mode == 'inference': out= model.infer(x) 
            else : out= model(x,trg[:,1:])
            
            y_pred = out.contiguous().view(-1,out.shape[-1])
            y_actual = trg[:,:-1].contiguous().view(-1)
            loss = criterion(y_pred,y_actual)
            epoch_loss += loss.item()
            outputs = out.max(dim=-1)[1]
            if mode == 'inference' :
                out_eval = model(x,trg[:,1:])
                y_pred2 = out_eval.contiguous().view(-1,out_eval.shape[-1])
                loss2 = criterion(y_pred2,y_actual)
                loss3 = criterion(y_pred,y_pred2)
#                loss3 = criterion(y_pred,softmax(y_pred2))
                epoch_loss2 += loss2.item()                
                epoch_loss3 += loss3.item()                
            
            for trg_j,out_j in zip(trg,outputs):
                trg_val = idx_to_val(trg_j.detach().cpu().numpy(),
                                     model.decode_map,model.sos_idx,model.eos_idx)
                out_val = idx_to_val(out_j.detach().cpu().numpy(),
                                     model.decode_map,model.sos_idx,model.eos_idx)
                Y_pred.append(out_val)
                Y_actual.append(trg_val)

    Y_actual, Y_pred = np.array(Y_actual), np.array(Y_pred)
    
    if mode == 'inference' :
        return Y_actual, Y_pred, np.array([epoch_loss,epoch_loss2,epoch_loss3])
    else : return Y_actual, Y_pred, epoch_loss


def evaluate_w_score(model,iterator,criterion,**kwargs):
    
    Y_actual, Y_pred, epoch_loss = evaluate(model,iterator,criterion,**kwargs)

    return (Y_actual, Y_pred), epoch_loss/ len(iterator), {'r2':r2_score(Y_actual,Y_pred),
                                        'rmse':rmse(Y_actual,Y_pred),
                                        'mape':mape(Y_actual,Y_pred)}, 

  
def run(model,optimizer, scheduler, criterion,iter_dict,
        total_epoch,warmup,clip,infer_cycle,
        best_loss,save_dir,expt_name,**kwargs):
    
    train_losses, valid_losses, valid_scores,lr_list = [], [], [], []
    best_epoch=0
    train_iter,valid_iter = iter_dict['iters']['trn'],iter_dict['iters']['vld']
    
    for step in range(total_epoch):

        train_loss = train(model, train_iter, optimizer, criterion, clip)
        if (infer_cycle > 0 and (step+1) % infer_cycle == 0): eval_mode = 'inference'
        else : eval_mode = 'eval'
        _,valid_loss, valid_score = evaluate_w_score(model,valid_iter,criterion,mode=eval_mode)
        if eval_mode == 'inference' :
            valid_loss, valid_loss_etc= valid_loss[0], valid_loss[1:]
        
        if step > warmup:scheduler.step(valid_loss)

        last_lr = scheduler.get_last_lr()[0]
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        valid_scores.append(valid_score)
        lr_list.append(last_lr)

        if valid_loss < best_loss:
            best_loss,best_epoch = valid_loss,step+1
            torch.save(model, os.path.join(save_dir,'best_{0}.pt'.format(expt_name)))

        print(f'Epoch: {step + 1:>5}\t\tlr: {last_lr:.8f}')
        val_score_str = ' '.join([f'{v:.5f}' for k,v in valid_score.items()])
        valid_loss_str = f'{valid_loss:.5f}'
        if eval_mode == 'inference' :
            print('*inference*')
            valid_loss_str = valid_loss_str + ', ' + ', '.join([f'{v:.5f}' for v in valid_loss_etc]) 
        print(f'\tTrain Loss: {train_loss:.5f}\tVal Loss: {valid_loss_str}\tVal Score: {val_score_str}')
#        print(f'\tTrain Loss: {train_loss:.3f}\tVal Loss: {valid_loss:.3f}\tVal Score: {val_score_str}')

    print('Best Epoch: ',best_epoch)
    return model,train_losses,valid_losses, valid_scores,best_epoch,lr_list

def test_n_plot(model, iterator, criterion,device,scatter=True,**kwargs):
    Y_rslt,Y_truth = get_test_rslt(model,iterator,criterion,device)
    fig,ax = plt.subplots()
    if scatter: sns.scatterplot(x=Y_truth,y=Y_rslt,**kwargs)
    else :
      if 'kind' not in kwargs : kwargs['kind'] = 'hist'
      g = sns.jointplot(x=Y_truth, y=Y_rslt, xlim = (0,60000), ylim = (0,60000),**kwargs)
    score = r2_score(Y_truth,Y_rslt)
    mape = mean_absolute_percentage_error(Y_truth,Y_rslt)
    print(rmse(Y_truth,Y_rslt), "\tr2 : ",score,"\tmape : ",mape)
    return fig,ax

def plot_err_dist(y_pred,y_true,percent=False,**kwargs):
  err = y_pred - y_true
  if percent : err = err/y_true
  if 'kind' not in kwargs : kwargs['kind'] = 'hist'
  sns.jointplot(x=y_true,y=err,**kwargs)
  #return fig,ax

def plot_reg_val(y_pred,y_true,**kwargs):
    y_mean = np.mean(y_true)    
    val = np.abs((y_pred-y_mean)/(y_true-y_mean+1e-20))
    print("calc done")
    if 'kind' not in kwargs : kwargs['kind'] = 'hist'
    sns.jointplot(x=y_true,y=val,**kwargs)
   # return fig,ax

def test_plot_err_dist(model, iterator, criterion,device,percent=False,**kwargs):
  y_pred,y_true = get_test_rslt(model,iterator,criterion,device)
  plot_err_dist(y_pred,y_true,percent=percent,**kwargs)
  #return fig,ax

def test_plot_reg_val(model, iterator, criterion,device,**kwargs):
  y_pred,y_true = get_test_rslt(model,iterator,criterion,device)
  plot_reg_val(y_pred,y_true,**kwargs)
  #return fig,ax

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)
    
def trainer_setting(model,init_lr,weight_decay,adam_eps,factor,patience,cls_freq=None,**kwargs):
  print(f'The model has {count_parameters(model):,} trainable parameters')
  model.apply(initialize_weights)
  optimizer = Adam(params=model.parameters(),
                   lr=init_lr,
                   weight_decay=weight_decay,
                   eps=adam_eps)

  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                   verbose=True,
                                                   factor=factor,
                                                   patience=patience)

  #criterion = nn.CrossEntropyLoss()
  if cls_freq is not None :
      normedWeights = [np.power(1 - (x / sum(cls_freq)),5)*5 for x in cls_freq]
      normedWeights = torch.FloatTensor(normedWeights).to(device)
  else : normedWeights = None
  criterion = nn.CrossEntropyLoss(normedWeights)
#  criterion = nn.NLLLoss(normedWeights)
  return {
            'model' : model,
            'optimizer' : optimizer,
            'scheduler' : scheduler,
            'criterion' : criterion
        }
  