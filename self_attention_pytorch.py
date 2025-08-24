import torch
import torch.nn as nn
import torch.nn.functional as F
from model1 import SeqSelfAttention

class rAttention(nn.Module):
    def __init__(self, d_model, num_heads,w=21,type = 'additive'):
        super(rAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_lin = nn.Linear(d_model, 3 * d_model)
        nn.init.xavier_normal_(self.qkv_lin.weight)
        self.w = w
        self.w_half = self.w // 2

        self.out_lin = nn.Linear(d_model, d_model)

        self.bias_r_w = self._init_bias(nn.Parameter(torch.Tensor(self.num_heads, self.head_dim)))
        self.bias_r_r = self._init_bias(nn.Parameter(torch.Tensor(self.num_heads, self.head_dim)))

        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=-1)

        self.type = type
        if self.type == "additive":
            # 加性注意力的可学习权重
            self.W1 = nn.Linear(d_model, d_model)
            self.W2 = nn.Linear(d_model, d_model)
            self.V = nn.Linear(d_model, 1)
            self.bh = nn.Parameter(torch.zeros(d_model))
            self.ba = nn.Parameter(torch.zeros(1))
            self.addattn = SeqSelfAttention(64,attention_type='additive')
    def _init_bias(self, bias):
        bound = 1 / bias.size(1) ** 0.5
        return nn.init.uniform_(bias, -bound, bound)

    def forward(self, x):
        shp = x.shape
        b,c, l, d = shp
        x = x.reshape(b*c,l,d)
        q, k, v = torch.split(self.qkv_lin(x), self.num_heads * self.head_dim, dim=-1)
        if self.type != "additive":
            q = q.view(-1, l, self.num_heads, self.head_dim) * (self.head_dim ** -0.5)
            k = k.view(-1, l, self.num_heads, self.head_dim)
            v = v.view(-1, l, self.num_heads, self.head_dim)

            k = F.pad(k, (0,) * (4) + (self.w_half,) * 2).unfold(1, l, 1)
            v = F.pad(v, (0,) * (4) + (self.w_half,) * 2).unfold(1, l, 1)

            q_k = q + self.bias_r_w
            q_r = q + self.bias_r_r

            A = torch.einsum('bl...nh,bwnhl->bln...w', q_k, k)

            mask = torch.zeros(l, device=k.device).bool()
            mask = F.pad(mask, (self.w_half,) * 2, value=True).unfold(0, l, 1).T
            mask = mask.unsqueeze(1)

            mask_value = -torch.finfo(A.dtype).max
            A.masked_fill_(mask, mask_value)

            A = self.softmax(A)
            A = self.dropout(A)

            z = torch.einsum('bln...w,bwnhl->bl...nh', A, v)
        else:
            # 加性注意力计算
            q = q.unsqueeze(2)
            k = k.unsqueeze(1)
            score = self.V(torch.tanh(q + k +self.bh)).squeeze(-1)+self.ba
            # score = F.sigmoid(score)
            score = torch.exp(score - torch.max(score, dim=-1, keepdim=True)[0])
            A = score / torch.sum(score, dim=-1, keepdim=True)

            # A = self.softmax(score)
            A = self.dropout(A)
            z = torch.einsum('b c c, b c d -> b c d', A, v)
            z = self.addattn(x)
        z = z.view(b,c, l, -1)
        z = self.out_lin(z)
        return z


class cAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.20,type="additive"):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_lin = nn.Linear(d_model, 3 * d_model)
        nn.init.xavier_normal_(self.qkv_lin.weight)
        self.outlin = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.type = type
        if self.type == "additive":
            # 加性注意力的可学习权重
            self.W1 = nn.Linear(d_model, d_model)
            self.W2 = nn.Linear(d_model, d_model)
            self.V = nn.Linear(d_model, 1)
            self.bh = nn.Parameter(torch.zeros(d_model))
            self.ba = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        shp = x.shape
        b, c, l, d = shp
        x = x.permute(0,2,1,3).reshape(b*l,c,d)#b l c d
        q, k, v = torch.split(self.qkv_lin(x), self.d_model, dim=-1)
        if self.type != "additive":
            q = q.view(-1, c, self.num_heads, self.head_dim) * (self.head_dim ** -0.5)
            k = k.view(-1, c, self.num_heads, self.head_dim)
            v = v.view(-1, c, self.num_heads, self.head_dim)

            # 计算注意力分数
            A = torch.einsum('b q n h, b k n h -> b q k n', q, k)
            # 注意力分数进行 softmax 归一化
            A = self.softmax(A)
            # 应用 dropout
            A = self.dropout(A)

            # 计算注意力加权后的值
            z = torch.einsum('b q k n, b v n h -> b q n h', A, v)
        else:
            # 加性注意力计算
            q = q.unsqueeze(2)
            k = k.unsqueeze(1)
            score = self.V(torch.tanh(q + k +self.bh)).squeeze(-1)+self.ba
            score = torch.exp(score - torch.max(score, dim=-1, keepdim=True)[0])
            A = score / torch.sum(score, dim=-1, keepdim=True)
            # A = self.softmax(score)
            A = self.dropout(A)
            z = torch.einsum('b c c, b c d -> b c d', A, v)

        z = z.reshape(b, l, c, d).permute(0, 2, 1, 3)
        # 应用输出线性变换层
        z = self.outlin(z)
        return z


