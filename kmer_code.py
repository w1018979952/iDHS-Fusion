import random
import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import collections
from tools import pad_sequences

def dataProcessing(seq):
    # 化学碱基的映射字典，将A, C, G, T映射为3维向量
    chem_bases = {'A': [1, 1, 1], 'C': [0, 1, 0], 'G': [1, 0, 0], 'T': [0, 0, 1]}
    # 初始化一个零矩阵，形状为 (len(seq), 3)
    Z = np.zeros((len(seq), 3))
    # 遍历每个序列
    for l, s in enumerate(seq):
        # 遍历序列中的每个字符
        for i, char in enumerate(s):
            # 如果字符在化学碱基字典中，则进行映射
            if char in chem_bases:
                Z[l] = chem_bases[char]
    return Z # 返回PyTorch张量

def get_1_trids(): # 生成字典
    # 定义四个碱基字符
    chars = ['A', 'C', 'G', 'T']
    # 初始化一个列表用于存储核苷酸组合
    nucle_com = []
    base = len(chars)  # 计算字符的基数
    end = len(chars) ** 1  # 计算所有可能的组合数（1个字符）
    for i in range(0, end):
        n = i
        ch0 = chars[n % base]  # 获取对应的字符
        nucle_com.append(ch0)  # 添加到核苷酸组合列表中
    word_index = {w: i for i, w in enumerate(nucle_com)}  # 创建字典，键是核苷酸组合，值是索引
    return word_index

def get_2_trids():
    # 定义四个碱基字符
    chars = ['A', 'C', 'G', 'T']
    # 初始化一个列表用于存储核苷酸组合
    nucle_com = []
    base = len(chars)  # 计算字符的基数
    end = len(chars) ** 2  # 计算所有可能的组合数（2个字符）
    for i in range(0, end):
        n = i
        ch0 = chars[n % base]  # 获取第一个字符
        n = n // base
        ch1 = chars[n % base]  # 获取第二个字符
        nucle_com.append(ch1 + ch0)  # 添加到核苷酸组合列表中
    word_index = {w: i for i, w in enumerate(nucle_com)}  # 创建字典，键是核苷酸组合，值是索引
    return word_index

def get_3_trids():
    # 定义四个碱基字符
    chars = ['A', 'C', 'G', 'T']
    # 初始化一个列表用于存储核苷酸组合
    nucle_com = []
    base = len(chars)  # 计算字符的基数
    end = len(chars) ** 3  # 计算所有可能的组合数（3个字符）
    for i in range(0, end):
        n = i
        ch0 = chars[n % base]  # 获取第一个字符
        n = n // base
        ch1 = chars[n % base]  # 获取第二个字符
        n = n // base
        ch2 = chars[n % base]  # 获取第三个字符
        nucle_com.append(ch2 + ch1 + ch0)  # 添加到核苷酸组合列表中
    word_index = {w: i for i, w in enumerate(nucle_com)}  # 创建字典，键是核苷酸组合，值是索引
    return word_index

def get_4_trids():
    # 初始化一个列表用于存储四核苷酸组合
    nucle_com = []
    # 定义碱基字符集
    chars = ['A', 'C', 'G', 'T']
    base = len(chars)  # 确定基数
    end = len(chars) ** 4  # 计算所有可能的组合数（4个字符）
    for i in range(0, end):
        n = i
        ch0 = chars[n % base]  # 获取第一个字符
        n = n // base
        ch1 = chars[n % base]  # 获取第二个字符
        n = n // base
        ch2 = chars[n % base]  # 获取第三个字符
        n = n // base
        ch3 = chars[n % base]  # 获取第四个字符
        nucle_com.append(ch0 + ch1 + ch2 + ch3)  # 添加到核苷酸组合列表中
    word_index = {w: i for i, w in enumerate(nucle_com)}  # 创建字典，键是核苷酸组合，值是索引
    return word_index

def frequency(seq, kmer, coden_dict):
    # 初始化一个列表用于存储k-mer值
    Value = []
    k = kmer  # 设置k-mer长度
    for i in range(len(seq) - int(k) + 1):
        kmer = seq[i:i + k]  # 提取k-mer子串
        kmer_value = coden_dict[kmer]  # 查找k-mer子串的值
        Value.append(kmer_value)  # 添加值到列表中
    freq_dict = dict(collections.Counter(Value))  # 统计k-mer值的频率
    return freq_dict

def coden(seq, kmer, tris):
    # 获取k-mer对应的字典   A[1,0,0,0] C[0,1,0,0] G[0,0,1,0] T[0,0,0,1]
    coden_dict = tris
    # 获取频率字典
    freq_dict = frequency(seq, kmer, coden_dict)
    # 初始化零矩阵
    vectors = np.zeros((len(seq), len(coden_dict.keys())))
    for i in range(len(seq) - int(kmer) + 1):
        value = freq_dict[coden_dict[seq[i:i + kmer]]]  # 获取k-mer的频率值
        vectors[i][coden_dict[seq[i:i + kmer]]] = 1  # 将频率值存储在矩阵中
    return vectors  # 返回PyTorch张量

def get_RNA_seq_concolutional_array(seq, motif_len=4):
    print(seq)
    alpha = 'ACGT'
    row = (len(seq) + 2 * motif_len - 2)
    new_array = np.zeros((row, 4))

    # 填充首部和尾部
    for i in range(motif_len - 1):
        new_array[i] = np.array([0.25] * 4)
    for i in range(row - 3, row):
        new_array[i] = np.array([0.25] * 4)

    for i, val in enumerate(seq):
        i = i + motif_len - 1
        if val not in 'ACGT':
            new_array[i] = np.array([0.25] * 4)
            continue
        try:
            index = alpha.index(val)
            new_array[i][index] = 1
        except:
            import pdb;
            pdb.set_trace()  # 调试用
    print(new_array)
    return new_array # 返回PyTorch张量

def dealwithdata(sequence):
    tris1 = get_1_trids()  # 获取1-mer组合字典
    tris2 = get_2_trids()  # 获取2-mer组合字典
    tris3 = get_3_trids()  # 获取3-mer组合字典
    dataX = []
    Z = dataProcessing(sequence)  # 处理序列，将其转换为特征矩阵
    kmer1 = coden(sequence, 1, tris1)  # 将序列编码为1-mer特征向量
    kmer2 = coden(sequence, 2, tris2)  # 将序列编码为2-mer特征向量
    kmer3 = coden(sequence, 3, tris3)  # 将序列编码为3-mer特征向量
    Kmer = np.hstack((kmer1,Z,kmer2))  # 将1-mer特征、化学基序列和2-mer特征组合在一起
    # Kmer = torch.cat((kmer1,Z,kmer2),1)  # 将1-mer特征、化学基序列和2-mer特征组合在一起
    # dataX.append(Kmer.tolist())  # 将Kmer转换为列表并添加到dataX中（已注释）
    # dataX = np.array(dataX)  # 将dataX转换为NumPy数组（已注释）
    # print(dataX.shape)  # 打印dataX的形状（已注释）
    return Kmer  # 返回组合后的特征矩阵

def createKmerData(str1):
    sequence_num = []  # 初始化序列特征列表
    label_num = []  # 初始化标签列表
    f = open(str1).readlines()  # 读取文件内容
    for i in range(0, len(f)-1, 2):  # 每次读取两行，遍历文件
        label = f[i].strip('\n').replace('>', '')  # 提取标签并去除换行符和'>'符号
        label_num.append(int(label))  # 将标签转换为整数并添加到标签列表中
        sequence = f[i + 1].strip('\n')  # 提取序列并去除换行符
        sequence_num.append(dealwithdata(sequence))  # 处理序列并添加到序列特征列表中
        # sequence_num.append((sequence))  # 处理序列并添加到序列特征列表中

    X_train = sequence_num
    labels = label_num
    seed = 113
    np.random.seed(seed)
    np.random.shuffle(X_train)
    np.random.seed(seed)
    np.random.shuffle(labels)
    X_train = np.array(X_train,dtype=object)
    X_train = pad_sequences(X_train,300)
    y_train = np.array(labels)
    y_train = torch.tensor(y_train,dtype=torch.float32)


    return X_train, y_train  # 返回训练数据和标签

