import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch.nn.functional as F
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from tools import *

class DynamicExpectationPooling(nn.Module):
    def __init__(self,dim):
        super(DynamicExpectationPooling, self).__init__()
        # 初始化 m 的参数，m 是可训练的标量，初始值为 1
        self.m = nn.Parameter(torch.ones(1))
        self.dim=dim
    def forward(self, x):
        """
        输入: x 的形状为 (batch_size, seq_len, input_dim)
        """
        # Step 1: 计算每个特征的最大值，保持维度为 (batch_size, 1, input_dim)
        X_max = torch.max(x, dim=self.dim, keepdim=True)[0]

        # Step 2: 计算 w_i 的权重，使用 softmax
        # (X - X_max) 的形状仍然为 (batch_size, seq_len, input_dim)
        diff = x - X_max
        weights = F.softmax(self.m * diff, dim=self.dim)

        # Step 3: 计算加权求和，获得输出
        out = torch.sum(weights * x, dim=self.dim)  # (batch_size, input_dim)

        return out

class cnn1(nn.Module):
    def __init__(self):
        super(cnn1, self).__init__()
        self.emb = nn.Embedding(21,128)
        self.conv1 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=5, padding='same',bias=False)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.drop2 = nn.Dropout(0.2)
    def forward(self, x):

        x = self.emb(x.int())
        x = x.permute(0, 2, 1)  # 调整形状以适应 Conv1d
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.drop2(x)

        return x

class cnn2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(cnn2, self).__init__()
        self.branch1 = nn.Conv1d(in_channels, out_channels, kernel_size=1,bias=False)  # 1x1卷积

        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1,bias=False),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1,bias=False)
        )

        self.branch5 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1,bias=False),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2,bias=False)
        )

        self.branch_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1,bias=False)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch3 = self.branch3(x)
        branch5 = self.branch5(x)
        branch_pool = self.branch_pool(x)

        outputs = torch.cat([branch1, branch3, branch5, branch_pool], dim=1)  # 在通道维度上拼接
        return outputs


class TransformerEncoder(nn.Module):
    def __init__(self, d_model=64, nhead=8, num_layers=2):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,dim_feedforward=d_model*4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):

        out = self.transformer(x)
        return out



class DHS(nn.Module):
    def __init__(self):
        super(DHS, self).__init__()

        self.cnn1 = cnn1().apply(weights_init_uniform)
        self.transformer_branch = TransformerEncoder(d_model=128)
        self.inception_branch = cnn2(in_channels=64, out_channels=32).apply(weights_init_uniform)
        self.classifier = nn.Sequential(nn.Linear(128 * 75, 100),
                                        nn.ReLU(),
                                        nn.Linear(100, 1))
        self.classifier64 = nn.Linear(64,1)
        self.coordination_layer = nn.Sequential(
            nn.Conv1d(320-64, 128, kernel_size=1,bias=False),  # 协调两个分支的特征
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(75)
        )
        self.block = CMA_Block_1D(64, 32, 64).apply(weights_init_uniform)
        self.aap = nn.AdaptiveAvgPool1d(75)
        self.amp = nn.AdaptiveMaxPool1d(75)
        self.fl = nn.Flatten()
        self.pooling_layer1 = DynamicExpectationPooling(1)

        self.bilstm = nn.LSTM(input_size=64, hidden_size=64, batch_first=True, bidirectional=True)
        self.qkv_lin = nn.Linear(64,64*3)

        self.lin128_64 = nn.Linear(128,64,bias=False)

        self.weight1 = nn.Parameter(torch.ones(64))
        self.weight2 = nn.Parameter(torch.ones(64))
    def forward(self, input):
        x = input[0]
        x = self.cnn1(x)

        inception_features = self.inception_branch(x)
        inception_features = self.amp(inception_features).permute(0,2,1)

        blstm = self.bilstm(x.permute(0,2,1))[0]
        x = blstm[:,:,:64]+blstm[:,:,-64:]
        q, k, v = torch.chunk(self.qkv_lin(self.amp(x.permute(0,2,1)).permute(0,2,1)), 3, dim=-1)

        addattn = AdditiveAttention(q,k,v,True,0.2)
        transformer_features = self.transformer_branch(inception_features)

        # combine = self.block(addattn,self.lin128_64(transformer_features))
        # combine = self.block(self.lin128_64(transformer_features),addattn,)
        combine = self.lin128_64(transformer_features)*self.weight1 + addattn*self.weight2
        # print(self.weight1.data,self.weight2.data)
        output = self.classifier64(self.pooling_layer1(combine))

        return output,output

    def predict(self, X, batch_size=128):
        # 设置模型为评估模式
        self.eval()
        predictions = []
        X1 = X[0].to("cuda")
        X2 = X[1].to("cuda")
        X3 = X[2].to("cuda")
        with torch.no_grad():
            for i in range(0, len(X1), batch_size):
                batch1 = X1[i:i + batch_size]
                batch2 = X2[i:i + batch_size]
                batch3 = X3[i:i + batch_size]
                output, loss1 = self.forward((batch1, batch2, batch3))
                # output = self.forward(batch1)
                output = torch.sigmoid(output)
                predictions.append(output)

        # 将所有小批量的预测结果拼接成一个整体
        predictions = torch.cat(predictions, dim=0)
        return predictions.flatten()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps



class CMA_Block_1D(nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel):
        super(CMA_Block_1D, self).__init__()

        self.conv1 = nn.Conv1d(in_channel, hidden_channel, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv1d(in_channel, hidden_channel, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv1d(in_channel, hidden_channel, kernel_size=1, stride=1, padding=0)

        self.scale = hidden_channel ** -0.5

        self.conv4 = nn.Sequential(
            nn.Conv1d(hidden_channel, out_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv5 = nn.Conv1d(in_channel*2, in_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, seq1, seq2):
        batch_size, seq_len, _ = seq1.size()

        # Conv1D expects input shape: (batch_size, channels, seq_len)
        # So we need to transpose the input to fit this format
        seq1 = seq1.transpose(1, 2)  # (batch_size, in_channel, seq_len)
        seq2 = seq2.transpose(1, 2)

        q = self.conv1(seq1)  # (batch_size, hidden_channel, seq_len)
        k = self.conv2(seq2)  # (batch_size, hidden_channel, seq_len)
        v = self.conv3(seq2)  # (batch_size, hidden_channel, seq_len)

        # Transpose q and k for attention calculation
        q = q.transpose(1, 2)  # (batch_size, seq_len, hidden_channel)
        k = k.transpose(1, 2)  # (batch_size, seq_len, hidden_channel)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # Attention scores
        m = attn.softmax(dim=-1)  # Attention weights

        # Apply attention weights to value
        z = torch.matmul(m, v.transpose(1, 2))  # (batch_size, seq_len, hidden_channel)
        z = z.transpose(1, 2)  # (batch_size, hidden_channel, seq_len)

        # Apply final convolution
        z = self.conv4(z)  # (batch_size, out_channel, seq_len)

        # Add the original sequence to the output
        output = seq1 + z  # Residual connection

        # Transpose back to (batch_size, seq_len, out_channel) for further layers
        output = output.transpose(1, 2)
        return output

def AdditiveAttention(Q, K, V, use_DropKey, mask_ratio):
    q_k_sum = Q.unsqueeze(-2) + K.unsqueeze(-3)

    attn = torch.tanh(q_k_sum)

    if use_DropKey:
        m_r = torch.ones_like(attn) * mask_ratio
        attn = attn + torch.bernoulli(m_r) * -1e-12

    attn = attn.mean(dim=-1)
    attn = F.softmax(attn, dim=-1)

    x = attn @ V
    return x