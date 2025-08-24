#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from torch.utils.data import TensorDataset
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm  # 导入 tqdm 来显示训练进度
import gc
from kmer_code import *
from metrics import calculateScore, analyze
from model_final import DHS
from tools import *
from datetime import datetime
from tensorboardX import SummaryWriter
import torchmetrics

# 检查是否有可用的 GPU
device = torch.device("cuda")


def function(DATAPATH, OutputDir):
    maxlen = 300  # 设置最大序列长度为300

    X_train, y_train = create1DData(DATAPATH)  # 创建训练数据 embeding编码
    X_train1, y_train1 = createKmerData(DATAPATH)  # 创建训练数据 k-mer编码
    X_train2, y_train2 = createFeature(DATAPATH)  # 创建多维特征编码

    # 确保两个数据集的标签是相同的
    if not np.array_equal(y_train, y_train1) and np.array_equal(y_train, y_train2):
        raise "y_train and y_train1 must be the same"
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)  # 使用分层交叉验证，将数据分为2份，打乱顺序，设置随机种子为7

    trainning_result = []  # 初始化训练结果列表
    testing_result = []  # 初始化测试结果列表

    for i, (train_idx, test_idx) in enumerate(kfold.split(X_train, y_train)):
        # # 打印类别分布
        train_class_distribution = np.bincount(y_train[train_idx])
        test_class_distribution = np.bincount(y_train[test_idx])

        # alpha = train_class_distribution[1] / (train_class_distribution[1] + train_class_distribution[0])
        print('\n%d Fold:' % (i + 1))  # 打印当前折数
        # print('alpha = %f' % alpha)

        best_model_name = ''  # 打印 模型名字 传参->calculateScore,analyze

        # 创建唯一的日志目录
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(OutputDir, 'logs', f'Fold_{i + 1}_{current_time}')
        writer = SummaryWriter(log_dir=OutputDir + '/logs/Fold_' + str(i + 1))  # 创建 TensorBoard writer

        # 初始化 指标计算函数
        accuracy_metric = torchmetrics.Accuracy(task='binary').to(device)
        auc_metric = torchmetrics.AUROC(task='binary').to(device)

        model = DHS().to(device)
        criterion = nn.BCEWithLogitsLoss()
        # 划分数据集
        X_train_fold, X_train1_fold, X_train2_fold, y_train_fold = X_train[train_idx], X_train1[train_idx], X_train2[
            train_idx], y_train[train_idx]
        X_val_fold, X_val1_fold, X_val2_fold, y_val_fold = X_train[test_idx], X_train1[test_idx], X_train2[test_idx], \
            y_train[test_idx]

        train_dataset = TensorDataset(X_train_fold, X_train1_fold, X_train2_fold, y_train_fold)
        val_dataset = TensorDataset(X_val_fold, X_val1_fold, X_val2_fold, y_val_fold)

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=12)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=12)

        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, )
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20,eta_min=1e-5)
        early_stopping_patience = 10  # 提前停止的耐心值
        best_val_loss = float('inf')

        best_auroc = 0.0
        epochs_no_improve = 0

        # 创建 SummaryWriter 对象，只需要一次
        w = SummaryWriter(log_dir=OutputDir + '/logs', comment='Net')
        # 添加模型图，只在第一次迭代时调用
        first_iteration = True

        for epoch in range(200):  # maxepoch=200

            model.train()  # 进入训练模式
            train_loss = 0.0
            all_train_labels = []  # 记录所有训练标签
            all_train_outputs = []  # 记录所有训练输出

            for X_batch, X1_batch, X2_bacth, y_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{200}",
                                                             leave=False):  # tqdm 显示训练进度条
                X_batch, X1_batch, X2_bacth, y_batch = X_batch.to(device), X1_batch.to(device), X2_bacth.to(
                    device), y_batch.to(device)  # 将数据移动到 GPU
                optimizer.zero_grad()  # 清零梯度

                outputs, outputs2 = model((X_batch, X1_batch, X2_bacth))  # 前向传播
                loss = criterion(outputs, y_batch.unsqueeze(1)) * 1 + criterion(outputs2[0], y_batch.unsqueeze(1))*0.2 + criterion(outputs2[1],y_batch.unsqueeze(1))*0.2
                loss.backward()  # 反向传播
                optimizer.step()  # 更新参数
                train_loss += loss.item()

                all_train_labels.append(y_batch)
                all_train_outputs.append(outputs)

                # 计算训练集指标
            all_train_labels = torch.cat(all_train_labels)
            all_train_outputs = torch.cat(all_train_outputs)
            train_accuracy = accuracy_metric(all_train_outputs, all_train_labels.unsqueeze(1).int())
            train_auc = auc_metric(all_train_outputs, all_train_labels.unsqueeze(1).int())

            model.eval()  # 进入评估模式
            val_loss = 0.0
            all_labels = []  # 记录所有标签
            all_outputs = []  # 记录所有输出

            with torch.no_grad():
                for X_batch, X1_batch, X2_bacth, y_batch in tqdm(val_loader, leave=False):
                    X_batch, X1_batch, X2_bacth, y_batch = X_batch.to(device), X1_batch.to(device), X2_bacth.to(
                        device), y_batch.to(device)
                    outputs, outputs2 = model((X_batch, X1_batch, X2_bacth))
                    loss = criterion(outputs, y_batch.unsqueeze(1)) * 1  # + criterion(outputs2[0],y_batch.unsqueeze(1)) +criterion(outputs2[1],y_batch.unsqueeze(1))*0
                    val_loss += loss.item()
                    all_labels.append(y_batch)
                    all_outputs.append(outputs)

            # 计算验证集指标
            all_labels = torch.cat(all_labels)
            all_outputs = torch.cat(all_outputs)
            val_accuracy = accuracy_metric(all_outputs, all_labels.unsqueeze(1).int())
            val_auc = auc_metric(all_outputs, all_labels.unsqueeze(1).int())

            # 写入训练集和验证集指标
            writer.add_scalar(f'Fold_{i}/Train_Loss', train_loss / len(train_loader), epoch)
            writer.add_scalar(f'Fold_{i}/Train_Accuracy', train_accuracy, epoch)
            writer.add_scalar(f'Fold_{i}/Train_AUC', train_auc, epoch)
            writer.add_scalar(f'Fold_{i}/Val_Loss', val_loss / len(val_loader), epoch)
            writer.add_scalar(f'Fold_{i}/Val_Accuracy', val_accuracy, epoch)
            writer.add_scalar(f'Fold_{i}/Val_AUC', val_auc, epoch)
            # 获取当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar(f'Fold_{i}/Learning_Rate', current_lr, epoch)  # 记录当前学习率
            print("-------------------------------------------------------")
            print(
                f'Epoch {epoch + 1} \tTrain Loss: {train_loss / len(train_loader):.3f} \tVal Loss: {val_loss / len(val_loader):.3f} ')
            print(
                f'        \ttrain acc : {train_accuracy:.3f} \tVal acc : {val_accuracy:.3f} \t')
            print(
                f'        \ttrain auc : {train_auc:.3f} \tVal auc : {val_auc:.3f} \tlr :{current_lr:.5f}')

            # 调用调度器更新学习率
            scheduler.step(val_loss / len(val_loader))
            # 检查是否提前停止

            # if val_auc > best_auroc:
            #     best_auroc = val_auc
            #     best_model_name = "model_" + str(i + 1) + "Fold" + ".pt"
            #     torch.save(model.state_dict(), OutputDir + best_model_name)  # 保存最好的模型

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_name = "model_" + str(i + 1) + "Fold" + ".pt"
                torch.save(model.state_dict(), OutputDir + best_model_name)  # 保存最好的模型
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience:
                    print("early stopping\n")
                    break

            # if val_auc > best_auroc:
            #     best_auroc = val_auc
            #     best_model_name = "model_" + str(i + 1) + "Fold" + ".pt"
            #     torch.save(model.state_dict(), OutputDir + best_model_name)  # 保存最好的模型
            #     epochs_no_improve = 0
            # else:
            #     epochs_no_improve += 1
            #     if epochs_no_improve >= early_stopping_patience:
            #         print("early stopping\n")
            #         break

        writer.close()  # 关闭TensorBoard writer

        model.load_state_dict(torch.load(OutputDir + best_model_name))

        # 计算训练结果并保存，将结果添加到训练结果列表中
        trainning_result.append(calculateScore((X_train_fold, X_train1_fold, X_train2_fold), y_train_fold, model,
                                               OutputDir + "/trainy_predy_" + str(i + 1) + ".txt", best_model_name))

        # 计算测试结果并保存，将结果添加到测试结果列表中
        testing_result.append(calculateScore((X_val_fold, X_val1_fold, X_val2_fold), y_val_fold, model,
                                             OutputDir + "/testy_predy_" + str(i + 1) + ".txt", best_model_name))

        del model  # 删除模型
        gc.collect()  # 回收内存

    temp_dict = (trainning_result, testing_result)  # 将训练结果和测试结果保存到字典中
    analyze(temp_dict, OutputDir)  # 分析结果并保存

    del trainning_result, testing_result  # 删除结果列表
    gc.collect()  # 回收内存
