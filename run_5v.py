#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys, argparse
import os
# import pandas as pd
from test_5v import *
from train_5v import *
from tools import *
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def main(args):
    if not os.path.exists(args.datapath_train):
        print("The rain data not exist! Error\n")
        sys.exit()
    if not os.path.exists(args.outpath_train_5f):
        print("The output path not exist! Create a new folder...\n")
        os.makedirs(args.outpath_train_5f)

    if not os.path.exists(args.datapath_test):
        print("The test data not exist! Error\n")
        sys.exit()
    if not os.path.exists(args.outpath_test_5f):
        print("The output path not exist! Create a new folder...\n")
        os.makedirs(args.outpath_test_5f)

    epoch = function(args.datapath_train, args.outpath_train_5f)
    # function_test(args.datapath_train,args.datapath_test, args.outpath_test_5f, Epoch=epoch[0])#最好loss
    # function_test(args.datapath_train, args.datapath_test, args.outpath_test_5f, Epoch=epoch[1])#最好auc
    function_test(args.datapath_train, args.datapath_test, args.outpath_test_5f, Epoch=30)

def run_multiple_times( para_dataname='adult',para_datadir='timePoints' ):
    dataname = para_dataname
    modelversion = 'result/'
    # parameter = 'lr0.001d0.5_5p10/'
    # parameter = '5v3分支第3分支前6个特征/'
    # parameter = '2dse+model8+all/'
    # parameter = '2dse+model9+all/'
    # parameter = '2dse+model9融合换成相加+all/'
    # parameter = '2dse+model9MASE换成相加+all/'
    # parameter = '2dse+model9MASE和融合换成相加+all/'
    # parameter = '2dse+model9切断f特征+all/'
    # parameter = '2dse+model9切断e特征+all/'
    # parameter = 'model9_baseline_ef_loss/'
    # parameter = 'model9_baseline_e/'
    # parameter = 'model9_baseline_f/'
    # parameter = 'model9_baseline_ef/'
    # parameter = 'model9_baseline_ef_cross/'
    # parameter = 'model9_baseline_ef_cross_mase/'
    # parameter = 'model9_baseline_ef_cross_mase_2dse/'
    # parameter = 'model9_baseline_ef_2dse/'
    # parameter = 'model9_baseline_ef_mase/'
    parameter = 'new_test_1/'
    datadir = para_datadir
    outpath = '/train-5Flod/'

    datapath_train = datadir+'/trainData/' + dataname + '-train.fa' #数据集路径 数据集名称
    outpath_train_5f = modelversion + parameter + dataname + outpath # 模型版本 数据集名称 超参版本 输出路径名称

    outpath = '/test-5Flod/'

    datapath_test = datadir+'/IndependentData/' + dataname + '-test.fa'#数据集路径 数据集名称
    outpath_test_5f = modelversion + parameter + dataname + outpath # 模型版本 数据集名称 超参版本 输出路径名称

    parser = argparse.ArgumentParser(description='Manual to the DHS')
    parser.add_argument('-p1', '--datapath_train', type=str, help='data', default=datapath_train)
    parser.add_argument('-o1', '--outpath_train_5f', type=str, help='output folder',
                        default=outpath_train_5f)

    parser.add_argument('-p2', '--datapath_test', type=str, help=' data_test', default=datapath_test)
    parser.add_argument('-o2', '--outpath_test_5f', type=str, help='output folder_test', default=outpath_test_5f)
    args = parser.parse_args()
    main(args)



if __name__ == "__main__":
    dir_name_list = ['timePoints','tissues']
    # arg_sets = fa_files_dict = get_all_fa_files(dir_name_list)

    # for dir_name, file_names in arg_sets.items():
    #     for file_name in file_names:
    #         run_multiple_times(file_name, dir_name)
    #
    run_multiple_times('Late', 'timePoints')
    run_multiple_times('adult', 'timePoints')
    run_multiple_times('Early', 'timePoints')
    run_multiple_times('ESC', 'timePoints')

    run_multiple_times('stomach', 'tissues')
    # run_multiple_times('thymus', 'tissues')
    run_multiple_times('neuraltube', 'tissues')
    run_multiple_times('craniofacial', 'tissues')
    run_multiple_times('Muller_Retina_Glia', 'tissues')
    run_multiple_times('retina', 'tissues')
    run_multiple_times('limb', 'tissues')
    run_multiple_times('kidney', 'tissues')
    run_multiple_times('heart', 'tissues')
    run_multiple_times('lung', 'tissues')
    run_multiple_times('liver', 'tissues')
    run_multiple_times('hindbrain', 'tissues')
    run_multiple_times('midbrain', 'tissues')
    run_multiple_times('forebrain', 'tissues')
    #






