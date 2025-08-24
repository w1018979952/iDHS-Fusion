#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import matplotlib as mpl

mpl.use('Agg')

import numpy as np
from sklearn.metrics import confusion_matrix, recall_score, matthews_corrcoef, \
    roc_curve, roc_auc_score, auc, precision_recall_curve, accuracy_score
import matplotlib.pyplot as plt
from scipy import interp
import torch
import torchmetrics
import torchmetrics.functional as F


def calculateScore(X, y, model, OutputFile, model_name=''):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    y = y.to(device)

    with torch.no_grad():
        pred_y = model.predict(X)  # 这里应该是模型的概率输出
        y_score = pred_y.clone()  # 保存概率输出
        pred_y = (pred_y >= 0.5).int()  # 将概率转换为 0/1 分类结果

    with open(OutputFile, 'w') as fOUT:
        for index in range(len(y)):
            fOUT.write(f"{int(y[index])}\t{int(pred_y[index])}\t{y_score[index].item():.6f}\n")  # 写入三列数据

    tempLabel = pred_y  # 这里已经是 0/1 结果了

    accuracy = F.accuracy(tempLabel, y.int(), task='binary')
    confusion = F.confusion_matrix(tempLabel, y.int(), task='binary')
    TN, FP, FN, TP = confusion.ravel()
    sensitivity = F.recall(tempLabel, y.int(), task='binary')
    specificity = TN / float(TN + FP)
    MCC = F.matthews_corrcoef(tempLabel, y.int(), num_classes=2, task='binary')
    F1Score = F.f1_score(tempLabel, y.int(), task='binary')
    precision = F.precision(tempLabel, y.int(), task='binary')

    ROCArea = F.auroc(y_score, y.int(), task='binary')  # 使用概率计算 AUROC
    fpr, tpr, thresholds = F.roc(y_score, y.int(), task='binary')

    precisionPR, recallPR, _ = F.precision_recall_curve(y_score, y.int(), task='binary')
    precisionPR_cpu = precisionPR.cpu().numpy()
    recallPR_cpu = recallPR.cpu().numpy()
    aupr = auc(recallPR_cpu, precisionPR_cpu)

    print("---calculateScore---"+model_name)
    # print()
    print(f'Accuracy: {accuracy:.3f}')
    print(f'ROC AUC: {ROCArea:.3f}')
    print()

    return {
        'sn': sensitivity,
        'sp': specificity,
        'acc': accuracy,
        'MCC': MCC,
        'AUC': ROCArea,
        'precision': precision,
        'F1': F1Score,
        'fpr': fpr.cpu().numpy(),
        'tpr': tpr.cpu().numpy(),
        'thresholds': thresholds.cpu().numpy(),
        'AUPR': aupr,
        'precisionPR': precisionPR_cpu,
        'recallPR': recallPR_cpu,
        'y_real': y.cpu().numpy(),
        'y_pred': pred_y.cpu().numpy(),
        'y_score':y_score.cpu().numpy()
    }


def analyze_test(temp, OutputDir, model_name=''):
    # 打开输出文件
    file_path = f'{OutputDir.rsplit("/", 2)[0]}/all_performance.txt'
    file = open(file_path, 'a')

    # 定义需要输出的指标
    metrics = ['sn', 'sp', 'acc', 'MCC', 'AUC', 'AUPR', 'precision', 'F1']

    # 遍历指标并写入文件
    for j in metrics:
        value = temp[0][j]
        # 如果是 tensor，提取值
        if torch.is_tensor(value):
            value = value.item()
        # 写入文件
        file.write(f'{j.ljust(12)} : {value:.3f}\n')
        print(f'{j.ljust(12)} : {value:.3f}')
    file.write("\n\n\n")
    # 关闭文件
    file.close()

def analyze(temp, OutputDir, model_name=''):
    """
       Metrics and plot.
       """
    print("---analyze---")

    plt.cla()
    plt.style.use("ggplot")

    trainning_result, testing_result = temp

    # The performance output file about training, validation, and test set
    file = open(OutputDir + '/'+model_name+'performance.txt', 'w')

    index = 0
    for x in [trainning_result, testing_result]:
        title = model_name + (' ' if model_name != '' else '')
        if index == 0:
            title = title + 'training_'
        if index == 1:
            title = title + 'testing_'
        index += 1
        file.write(title + 'results\n')
        print(title + 'results')
        for j in ['sn', 'sp', 'acc', 'MCC', 'AUC', 'AUPR', 'precision', 'F1']:
            total = [val[j].item() if torch.is_tensor(val[j]) else val[j] for val in x]
            file.write(f'{j.ljust(12)} mean : {np.mean(total):.3f} \tstd : {np.std(total):.3f}\n')
            print(f'{j.ljust(12)} mean : {np.mean(total):.3f} \tstd : {np.std(total):.3f}')
        file.write('\n\n______________________________\n')
    file.close()

    print()


    # Plot ROC about training, validation, and test set
    indexROC = 0
    for x in [trainning_result, testing_result]:
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        i = 0
        for val in x:
            tpr = val['tpr']
            fpr = val['fpr']
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.3f)' % (i + 1, roc_auc))
            i += 1

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        # mean_auc = auc(mean_fpr, mean_tpr)
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b',
                 lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic curve')
        plt.legend(loc="lower right")

        titleROC = ''
        if indexROC == 0:
            titleROC = 'training_'
        if indexROC == 1:
            titleROC = 'testing_'
        plt.savefig(OutputDir + '/' + titleROC + 'ROC.png')
        plt.close('all')

        indexROC += 1

    # Plot PR curve about training, validation, and test set
    indexPR = 0
    for item in [trainning_result, testing_result]:
        y_realAll = []
        y_predAll = []
        i = 0
        for val in item:
            precisionPR = val['precisionPR']
            recallPR = val['recallPR']
            aupr = val['AUPR']
            plt.plot(recallPR, precisionPR, lw=1, alpha=0.3, label='PR fold %d (AUPR = %0.3f)' % (i + 1, aupr))

            y_realAll.append(val['y_real'])
            y_predAll.append(val['y_score'])
            i += 1

        y_realAll = np.concatenate(y_realAll)
        y_predAll = np.concatenate(y_predAll)

        precisionPRAll, recallPRAll, _ = precision_recall_curve(y_realAll, y_predAll)
        auprAll = auc(recallPRAll, precisionPRAll)

        plt.plot(recallPRAll, precisionPRAll, color='b', label=r'Precision-Recall (AUPR = %0.3f)' % (auprAll),
                 lw=1, alpha=.8)

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")

        titlePR = ''
        if indexPR == 0:
            titlePR = 'training_'
        if indexPR == 1:
            titlePR = 'testing_'
        plt.savefig(OutputDir + '/' + titlePR + 'PR.png')
        plt.close('all')

        indexPR += 1
