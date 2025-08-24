#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib as mpl
mpl.use('Agg')
import gc
from kmer_code import *
from metrics import *
from model_final import DHS
from tools import pad_sequences,get_pt_files,create1DData,createFeature

device = torch.device("cuda")

def function_test(DATAPATH ,OutputDir, MODELPATH):

    X_train, y_train = create1DData(DATAPATH)  # 创建训练数据 embeding编码
    X_train1, y_train1= createKmerData(DATAPATH)  # 创建训练数据 k-mer编码
    X_train2, y_train2= createFeature(DATAPATH)


    model_name_list = get_pt_files(MODELPATH)
    for model_name in model_name_list:

        model = DHS().to(device)
        model.load_state_dict(torch.load(MODELPATH+model_name))

        testing_result = []
        testing_result.append(calculateScore((X_train,X_train1,X_train2),y_train, model, OutputDir + model_name[:-3]+'.txt',model_name[:-3]))

        analyze_test(testing_result, OutputDir, model_name[:-3])

if __name__ == '__main__':
    DATAPATH = 'tissues/trainData/neuraltube-train.fa'
    OutputDir = '/home/wangyongbo/data/DHS/output/'
    MODELPATH = '/home/wangyongbo/data/DHS/model/'
    function_test(DATAPATH, OutputDir, MODELPATH)
