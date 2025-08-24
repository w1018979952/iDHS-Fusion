#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import glob
import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader, TensorDataset
from model9 import DHS
from tools import create1DData, createFeature
from kmer_code import createKmerData
from tensorboardX import SummaryWriter
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def function_test(testpath, OutputDir, Epoch=30):
    """测试 DHS 模型的函数"""
    # 读取测试数据
    X_test, y_test = create1DData(testpath)
    X_test1, y_test1 = createKmerData(testpath)
    X_test2, y_test2 = createFeature(testpath)

    # 加载模型
    model = DHS().to(device)
    model.load_state_dict(torch.load(os.path.join(OutputDir, f"{Epoch}model.pt")))
    model.eval()  # 进入评估模式

    # 创建 TensorBoard 日志
    writer = SummaryWriter(log_dir=os.path.join(OutputDir, "logs"))

    # 初始化度量指标
    accuracy_metric = torchmetrics.Accuracy(task='binary').to(device)
    auc_metric = torchmetrics.AUROC(task='binary').to(device)

    criterion = nn.BCEWithLogitsLoss()

    # 创建数据加载器
    val_dataset = TensorDataset(X_test, X_test1, X_test2, y_test)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=12)

    all_labels = []
    all_outputs = []

    # 进行测试
    with torch.no_grad():
        for X_batch, X1_batch, X2_batch, y_batch in tqdm(val_loader, leave=False):
            X_batch, X1_batch, X2_batch, y_batch = (
                X_batch.to(device),
                X1_batch.to(device),
                X2_batch.to(device),
                y_batch.to(device),
            )
            outputs, _ = model((X_batch, X1_batch, X2_batch))
            all_labels.append(y_batch)
            all_outputs.append(outputs)

    # 计算验证集指标
    all_labels = torch.cat(all_labels)
    all_outputs = torch.cat(all_outputs)
    val_accuracy = accuracy_metric(all_outputs, all_labels.unsqueeze(1).int())
    val_auc = auc_metric(all_outputs, all_labels.unsqueeze(1).int())

    print(f"Validation Accuracy: {val_accuracy:.4f}, AUC: {val_auc:.4f}")

    return val_auc.item()  # 返回 AUROC 结果


# 结果存储文件
result_file = "results1.txt"

# 遍历 `result/2dse+model9+all/` 目录下的所有实验目录（例如 `adult`, `lung`）
experiment_dirs = glob.glob("result/2dse+model9+all/*/")

total_tests = 0  # 统计测试总次数

with open(result_file, "w") as f:
    for experiment_dir in experiment_dirs:
        experiment_dir = experiment_dir.rstrip("/")  # 清理路径
        experiment_name = os.path.basename(experiment_dir)  # 提取实验名称，如 `adult`

        # 获取模型权重路径
        weight_file = os.path.join(experiment_dir, "test-5Flod", "30model.pt")

        if not os.path.exists(weight_file):
            print(f"Warning: Model not found in {experiment_dir}. Skipping...")
            continue

        # **测试数据路径**
        # test_files = glob.glob("timePoints/IndependentData/*.fa")
        test_files = glob.glob("tissues/IndependentData/*.fa")

        if len(test_files) == 0:
            print("Warning: No test datasets found in timePoints/IndependentData/. Skipping...")
            continue

        # 获取模型的输出目录（即 `test-5Flod` 目录）
        output_dir = os.path.dirname(weight_file)  # `test-5Flod` 目录
        epoch = 30  # 从 `30model.pt` 推测

        # 遍历测试数据集并进行测试
        for test_file in test_files:
            test_name = os.path.basename(test_file)

            print(f"Testing model {weight_file} on dataset {test_file}")

            # 运行测试
            auc_result = function_test(test_file, output_dir, epoch)
            total_tests += 1

            # 记录结果
            f.write(f"{experiment_name} {test_name} {auc_result:.4f}\n")

print(f"All tests completed! Total tests run: {total_tests}")
print(f"Results saved to {result_file}")
