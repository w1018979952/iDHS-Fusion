import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch.nn.functional as F
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from tools import *
from transformers.models.bert.configuration_bert import BertConfig
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
#
# config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
# # model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True).to(device)
#
# model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, config=config).to(device)
#
# dna = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
# inputs = tokenizer(dna, return_tensors = 'pt')["input_ids"].to(device)
# hidden_states = model(inputs)[0]  # [1, sequence_length, 768]

# # 平均池化嵌入
# embedding_mean = torch.mean(hidden_states[0], dim=0)
# print(embedding_mean.shape)  # 应为 768
#
# # 最大池化嵌入
# embedding_max = torch.max(hidden_states[0], dim=0)[0]
# print(embedding_max.shape)  # 应为 768

class SeqSelfAttention(nn.Module):
    ATTENTION_TYPE_ADD = 'additive'
    ATTENTION_TYPE_MUL = 'multiplicative'
    ATTENTION_TYPE_SDP = 'scaled_dot_product'

    def __init__(self, units=32, attention_width=None, attention_type=ATTENTION_TYPE_ADD,
                 return_attention=False, history_only=False,
                 kernel_initializer='glorot_normal', bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None, kernel_constraint=None,
                 bias_constraint=None, use_additive_bias=True, use_attention_bias=True,
                 attention_activation=None, attention_regularizer_weight=0.0,
                 max_len=75):
        super(SeqSelfAttention, self).__init__()

        self.units = units
        self.attention_width = attention_width
        self.attention_type = attention_type
        self.return_attention = return_attention
        self.history_only = history_only
        self.use_additive_bias = use_additive_bias
        self.use_attention_bias = use_attention_bias
        self.attention_activation = attention_activation
        self.attention_regularizer_weight = attention_regularizer_weight
        self.max_len = max_len
        self.kernel_initializer = 'glorot_normal'

        if attention_type == SeqSelfAttention.ATTENTION_TYPE_ADD:
            self.Wx = nn.Linear(units, units, bias=False)
            self.Wt = nn.Linear(units, units, bias=False)
            if use_additive_bias:
                self.bh = nn.Parameter(torch.zeros(units))
            else:
                self.bh = None
            self.Wa = nn.Linear(units, 1, bias=False)
            if use_attention_bias:
                self.ba = nn.Parameter(torch.zeros(1))
            else:
                self.ba = None
        elif attention_type in [SeqSelfAttention.ATTENTION_TYPE_MUL, SeqSelfAttention.ATTENTION_TYPE_SDP]:
            self.Wa = nn.Linear(units, units, bias=False)
            self.Wq = nn.Linear(units, units, bias=False)
            self.Wk = nn.Linear(units, units, bias=False)
            if use_attention_bias:
                self.ba = nn.Parameter(torch.zeros(1))
            else:
                self.ba = None

        # if bias_initializer == 'zeros':
        #     if attention_type == SeqSelfAttention.ATTENTION_TYPE_ADD:
        #         nn.init.zeros_(self.Wx.bias)
        #         nn.init.zeros_(self.Wt.bias)
        #     nn.init.zeros_(self.Wa.bias)

        self.attention_regularizer_loss = None  # 注意力正则化损失
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.2)
        # self.outlin = nn.Linear(units,units)

        self._kernel_initializer()

    def _kernel_initializer(self):
        if self.kernel_initializer == 'glorot_normal':
            if self.attention_type == SeqSelfAttention.ATTENTION_TYPE_ADD:
                nn.init.xavier_normal_(self.Wx.weight)
                nn.init.xavier_normal_(self.Wt.weight)
            nn.init.xavier_normal_(self.Wa.weight)
            # print("调用了glorot_normal")

    def forward(self, inputs, mask=None, pos=None):
        input_len = inputs.size(1)
        if pos is not None:
            relative_positions_embeddings = self.relative_position_encoding(seq_len=75, d_k=64).to(inputs.device)
        else:
            relative_positions_embeddings = None

        self.attention_regularizer_loss = 0.0  # 每个batch开始时清零正则化损失

        if self.attention_type == SeqSelfAttention.ATTENTION_TYPE_ADD:
            e = self._call_additive_emission(inputs, relative_positions_embeddings)
        elif self.attention_type == SeqSelfAttention.ATTENTION_TYPE_MUL:
            e = self._call_multiplicative_emission(inputs, relative_positions_embeddings)
        elif self.attention_type == SeqSelfAttention.ATTENTION_TYPE_SDP:
            e = self._call_scaled_dot_product_emission(inputs, relative_positions_embeddings)

        if self.attention_activation is not None:
            e = self.attention_activation(e)
        if self.attention_width is not None:
            if self.history_only:
                lower = torch.arange(0, input_len) - (self.attention_width - 1)
            else:
                lower = torch.arange(0, input_len) - self.attention_width // 2
            lower = lower.unsqueeze(-1)
            upper = lower + self.attention_width
            indices = torch.arange(0, input_len).unsqueeze(0)
            e -= 10000.0 * (1.0 - (lower <= indices).float() * (indices < upper).float())
        if mask is not None:
            mask = mask.unsqueeze(-1).float()
            e -= 10000.0 * ((1.0 - mask) * (1.0 - mask.permute(0, 2, 1)))

        e = torch.exp(e - torch.max(e, dim=-1, keepdim=True)[0])
        a = e / torch.sum(e, dim=-1, keepdim=True)

        # a = self.softmax(e)
        # a = self.dropout(a)

        v = torch.bmm(a, inputs)
        # 计算注意力正则化损失并添加到总损失中
        if self.attention_regularizer_weight > 0.0:
            self.attention_regularizer_loss += self._attention_regularizer(a)  # 更新正则化损失

        if self.return_attention:
            return v, a
        # v = self.outlin(v)
        return v

    def _call_additive_emission(self, inputs, relative_positions_embeddings=None):
        q = self.Wt(inputs).unsqueeze(2)
        k = self.Wx(inputs).unsqueeze(1)
        if relative_positions_embeddings is not None:
            relative_positions_embeddings = relative_positions_embeddings.unsqueeze(0)
            if self.use_additive_bias:
                h = torch.tanh(q + k + relative_positions_embeddings + self.bh)
            else:
                h = torch.tanh(q + k + relative_positions_embeddings)
        else:
            if self.use_additive_bias:
                h = torch.tanh(q + k + self.bh)
            else:
                h = torch.tanh(q + k)
        if self.use_attention_bias:
            e = self.Wa(h).squeeze(-1) + self.ba
        else:
            e = self.Wa(h).squeeze(-1)
        return e

    def _call_multiplicative_emission(self, inputs, relative_positions_embeddings=None):

        e = torch.bmm(self.Wa(inputs), inputs.permute(0, 2, 1))

        if relative_positions_embeddings is not None:
            e += torch.bmm(inputs, relative_positions_embeddings.permute(0, 2, 1))
        if self.use_attention_bias:
            e += self.ba
        return e

    def _call_scaled_dot_product_emission(self, inputs, relative_positions_embeddings=None):
        batch_size, seq_len, d_k = inputs.size()  # 获取输入的维度
        queries = self.Wq(inputs)  # (128, 75, 64)
        keys = self.Wk(inputs).permute(0, 2, 1)  # (128, 64, 75)
        e = torch.bmm(queries, keys)  # (128, 75, 75)
        if relative_positions_embeddings is not None:
            e += torch.einsum('bik,ijk->bij', inputs, relative_positions_embeddings)  # (128, 75, 75)
        e = e / torch.sqrt(torch.tensor(d_k, dtype=torch.float32)).to(inputs.device)
        if self.use_attention_bias:
            e += self.ba

        return e

    def _attention_regularizer(self, attention):
        batch_size = attention.size(0)
        input_len = attention.size(-1)
        indices = torch.arange(0, input_len).unsqueeze(0).to(attention.device)
        diagonal = torch.arange(0, input_len).unsqueeze(-1).to(attention.device)
        eye = (indices == diagonal).float()
        return self.attention_regularizer_weight * torch.sum(
            torch.pow(torch.bmm(attention, attention.permute(0, 2, 1)) - eye, 2)) / batch_size

    def relative_position_encoding(self, seq_len, d_k):
        range_vec = torch.arange(seq_len)
        range_mat = range_vec.unsqueeze(-1) - range_vec.unsqueeze(0)
        relative_positions_matrix = range_mat + (seq_len - 1)  # Shift to make non-negative
        R = torch.nn.Embedding(2 * seq_len - 1, d_k)
        return R(relative_positions_matrix)

class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=21, out_channels=128, kernel_size=3, padding='same')
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding='same')
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        self.bn1 = nn.BatchNorm1d(128)
        # self.ln1 = nn.LayerNorm(128)
        self.bn2 = nn.BatchNorm1d(64)
        # self.ln2 = nn.LayerNorm(64)
        self.drop=nn.Dropout(0.2)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.pool1(x)

        x = self.conv2(x)
        # x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.drop(x)
        # x = self.attn1(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x.permute(0, 2, 1)

class kmerModel(nn.Module):
    def __init__(self, in_channels=64):
        super(kmerModel, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=128, kernel_size=5, padding='same')
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding='same')
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.flt = nn.Flatten()
        # 用于VAE重构的均值和对数方差层
        self.fc_mean = nn.Linear(in_features=4800, out_features=1000)
        self.fc_logvar = nn.Linear(in_features=4800, out_features=1000)
        self.drop2 = nn.Dropout(0.2)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # PyTorch expects channels first
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.permute(0, 2, 1)
        x = self.drop2(x)# Reorder dimensions for LSTM
        kmer = x
        x = self.flt(x)
        # 输出均值和对数方差
        mu = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return kmer, mu, logvar
class kmerDecoder(nn.Module):
    def __init__(self):
        super(kmerDecoder, self).__init__()
        self.fc1 = nn.Linear(1000, 4800)
        self.deconv1 = nn.ConvTranspose1d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose1d(in_channels=128, out_channels=23, kernel_size=5, stride=2, padding=2, output_padding=1)

    def forward(self, z):
        x = self.fc1(z)
        x = x.view(z.size(0), 64, -1)  # [batch_size, channels, length]
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = x.permute(0, 2, 1)  # [batch_size, length, channels]
        return x

class embedModel(nn.Module):
    def __init__(self):
        super(embedModel, self).__init__()
        self.emb = nn.Embedding(27,128)
        self.conv1 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=10, padding='same')
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, padding='same')
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.flt = nn.Flatten()
        # 用于VAE重构的均值和对数方差层
        self.fc_mean = nn.Linear(in_features=4800, out_features=1000)
        self.fc_logvar = nn.Linear(in_features=4800, out_features=1000)
        self.drop2 = nn.Dropout(0.2)
    def forward(self, x):

        x = self.emb(x)
        x = x.permute(0, 2, 1)  # 调整形状以适应 Conv1d
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        # # 在前向传播时动态设置dropout概率
        # dropout_prob = torch.rand(1).item() * 0.3 + 0.2  # 生成0.1到0.5之间的随机数
        # x = F.dropout(x, p=dropout_prob, training=self.training)
        x = self.drop2(x)
        embed = x.permute(0, 2, 1)

        x = self.flt(x)
        # 输出均值和对数方差
        mu = self.fc_mean(x)
        logvar = self.fc_logvar(x)

        return embed,mu,logvar
class embedDecoder(nn.Module):
    def __init__(self):
        super(embedDecoder, self).__init__()
        # 假设原始输入序列的最大长度是301，特征维度是128
        self.deconv1 = nn.ConvTranspose1d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=2,
                                          output_padding=1)
        self.deconv2 = nn.ConvTranspose1d(in_channels=64, out_channels=128, kernel_size=10, stride=2, padding=4,
                                          output_padding=1)
        # 将潜在向量映射到一个中间特征表示
        self.fc1 = nn.Linear(1000, 4800)
        self.fc2 = nn.Linear(128,27)
    def forward(self, x):
        # x 是潜在空间的向量，形状为 [batch_size, latent_dim]
        # 将潜在向量映射到一个中间特征表示
        x = self.fc1(x)
        x = x.view(x.size(0), 64, -1)  # [batch_size, 64, sequence_length]
        # 第一个转置卷积层
        x = F.relu(self.deconv1(x))
        # 第二个转置卷积层
        x = F.relu(self.deconv2(x))
        # 确保输出形状正确
        x = x.permute(0, 2, 1)  # [batch_size, feature_size, sequence_length]
        x = self.fc2(x)
        # x = torch.argmax(x, dim=-1).float()  # 获取最可能的索引
        return x
# 定义 Chomp1d，用于修剪卷积后的输出
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

# 定义 TemporalBlock，即 TCN 的基本构建块
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

# 定义 TCN 网络
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=5, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


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

class DHS(nn.Module):
    def __init__(self):
        super(DHS, self).__init__()
        self.embed = embedModel().apply(weights_init_uniform)
        self.embeddecoder = embedDecoder().apply(weights_init_uniform)

        self.kmer = kmerModel(in_channels=20).apply(weights_init_uniform)
        self.kmerdecoder = kmerDecoder()#.apply(weights_init_uniform)

        self.convfeature=cnn().apply(weights_init_uniform)

        self.drop2 = nn.Dropout(p=0.2)

        self.attn = SeqSelfAttention(units=64, attention_type='multiplicative', attention_activation=F.sigmoid,)
                                    # attention_regularizer_weight=1e-4)
        self.attn1 = SeqSelfAttention(units=64, attention_type='additive', attention_activation=F.sigmoid,)
                                    #  attention_regularizer_weight=1e-4)
        self.attn2 = SeqSelfAttention(units=64, attention_type='additive', attention_activation=F.sigmoid,)
                                    #  attention_regularizer_weight=1e-4)
        # self.trans = nn.TransformerEncoder(nn.TransformerEncoderLayer(4800,8),4)
        self.flatten = nn.Flatten()

        self.gru = nn.GRU(input_size=61,hidden_size=64,bidirectional=True,batch_first=True)

        self.bilstm = nn.LSTM(input_size=64,hidden_size=64,batch_first=True,bidirectional=True)
        self.bilstm1 = nn.LSTM(input_size=64, hidden_size=64, batch_first=True, bidirectional=True)
        self.bilstm2 = nn.LSTM(input_size=64, hidden_size=64, batch_first=True, bidirectional=True)

        self.reconstruction_layer = nn.Linear(76 * 64, 304)  # 重构层，从76*64重构到304
        self.moe = MixtureOfExperts().apply(weights_init_uniform)

        self.mlp=nn.Sequential(nn.Linear(4800,256),
                               nn.ReLU(),
                               nn.Linear(256,1)
                               ).apply(weights_init_uniform)
        self.mlp2 = nn.Sequential(nn.Linear(4800, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 1)
                                 ).apply(weights_init_uniform)
        self.mlp3 = nn.Linear(64, 1)
        self.mlp4 = nn.Linear(75, 1)
        self.mlp5 = nn.Linear(64+75,1)
        self.tcn = TemporalConvNet(64, [64, 64, 64 , 64, 64 ,64,64] )
        self.pooling_layer1 = DynamicExpectationPooling(1) #压缩序列
        self.pooling_layer2 = DynamicExpectationPooling(2) #压缩特征
    def forward(self, input):

        dna, length = input[0][:, :-1], input[0][:, -1]
        kmer = input[1]
        kmer, mu1, logvar1 = self.kmer(kmer)  # 5 3的核
        kmer = kmer.permute(0, 2, 1)
        kmer = self.tcn(kmer).permute(0, 2, 1)
        kmer = self.attn1(kmer)


        dnaEmbed,mu,logvar = self.embed(dna.int())#10 5的核
        dnaEmbed = dnaEmbed.permute(0, 2, 1)
        dnaEmbed = self.tcn(dnaEmbed)
        dnaEmbed = dnaEmbed.permute(0,2,1)

        # dnaEmbed = self.attn(dnaEmbed)
        pool1 = self.pooling_layer1(dnaEmbed)
        # dnaEmbed = self.flatten(dnaEmbed)
        out1 = self.mlp3(pool1)

        return out1,0

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
                output,loss1 = self.forward((batch1, batch2,batch3))
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


class Expert(nn.Module):
    def __init__(self, input_dim, expert_dim):
        super(Expert, self).__init__()
        # self.fc = nn.Sequential(
        #     nn.Linear(input_dim, 2 * expert_dim),
        #     nn.ReLU(),
        #     nn.Linear(2 * expert_dim, expert_dim),
        #     nn.Dropout(p=0.2)
        # )
        self.fc =  nn.Linear(input_dim,expert_dim)

    def forward(self, x):
        return self.fc(x)
# class MixtureOfExperts(nn.Module):
#     def __init__(self, input_dim1=4800, input_dim2=4800, num_experts=4, expert_dim=256, output_dim=1, k=2):
#         super(MixtureOfExperts, self).__init__()
#         self.num_experts = num_experts
#         self.k = k
#         self.experts1 = nn.ModuleList([Expert(input_dim1, expert_dim) for _ in range(num_experts)])
#         self.experts2 = nn.ModuleList([Expert(input_dim2, expert_dim) for _ in range(num_experts)])
#         self.gating1 = nn.Linear(input_dim1, num_experts)
#         self.gating2 = nn.Linear(input_dim2, num_experts)
#         self.classifier = nn.Linear(expert_dim * 2, output_dim)
#
#     def forward(self, x1, x2):
#         # 专家层处理第一个模态的输入向量
#         expert_outputs1 = [expert(x1) for expert in self.experts1]
#         expert_outputs2 = [expert(x2) for expert in self.experts2]
#
#         # 将专家输出在专家维度上合并
#         expert_outputs1 = torch.stack(expert_outputs1, dim=1)  # 形状变为(batch_size, num_experts, expert_dim)
#         expert_outputs2 = torch.stack(expert_outputs2, dim=1)  # 形状变为(batch_size, num_experts, expert_dim)
#
#         # 门控层输出，分别对每个模态的输入计算
#         logits1 = self.gating1(x1)
#         logits2 = self.gating2(x2)
#
#         # 添加标准正态噪声
#         noise1 = torch.randn_like(logits1)
#         noise2 = torch.randn_like(logits2)
#         logits1 += noise1
#         logits2 += noise2
#
#         # 计算softmax得到门控权重
#         gates1 = F.softmax(logits1, dim=1)
#         gates2 = F.softmax(logits2, dim=1)
#
#         # 选择Top-K的门控权重和对应的专家输出
#         topk_gates1, topk_indices1 = torch.topk(gates1, self.k, dim=1)
#         topk_experts1 = torch.gather(expert_outputs1, 1, topk_indices1.unsqueeze(-1).expand(-1, -1, expert_outputs1.size(-1)))
#
#         topk_gates2, topk_indices2 = torch.topk(gates2, self.k, dim=1)
#         topk_experts2 = torch.gather(expert_outputs2, 1, topk_indices2.unsqueeze(-1).expand(-1, -1, expert_outputs2.size(-1)))
#
#         # 应用Top-K门控权重
#         mixed_output1 = torch.sum(topk_gates1.unsqueeze(-1) * topk_experts1, dim=1)  # 形状变为(batch_size, expert_dim)
#         mixed_output2 = torch.sum(topk_gates2.unsqueeze(-1) * topk_experts2, dim=1)  # 形状变为(batch_size, expert_dim)
#
#         # 合并两个模态的混合输出
#         combined_output = torch.cat((mixed_output1, mixed_output2), dim=1)  # 形状变为(batch_size, expert_dim * 2)
#
#         # 分类层输出
#         classification_output = self.classifier(combined_output)  # 形状变为(batch_size, output_dim)
#         return classification_output

class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim=4800, num_experts=1, expert_dim=256, output_dim=1):
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([Expert(input_dim, expert_dim) for _ in range(num_experts)])
        self.gating = nn.Linear(input_dim, num_experts)
        self.classifier = nn.Sequential(
            nn.Linear(expert_dim*3, expert_dim),
            nn.ReLU(),
            nn.Linear(expert_dim,output_dim)
        )# 修改这里，直接将每个专家的输出送入分类器

    def forward(self, x1, x2,x3=None):

        # 专家层处理每个输入向量
        expert_outputs1 = [expert(x1) for expert in self.experts]
        expert_outputs2 = [expert(x2) for expert in self.experts]
        if x3 is not None:
            expert_outputs3 = [expert(x3) for expert in self.experts]
            expert_outputs3 = torch.stack(expert_outputs3, dim=1)
            gates3 = F.softmax(self.gating(x3), dim=1)
            mixed_output3 = torch.sum(gates3.unsqueeze(-1) * expert_outputs3, dim=1)
        # 将专家输出在专家维度上合并
        expert_outputs1 = torch.stack(expert_outputs1, dim=1)  # 形状变为(batch_size, num_experts, expert_dim)
        expert_outputs2 = torch.stack(expert_outputs2, dim=1)  # 形状变为(batch_size, num_experts, expert_dim)

        # 门控层输出，分别对每个输入计算
        gates1 = F.softmax(self.gating(x1), dim=1)
        gates2 = F.softmax(self.gating(x2), dim=1)

        # 应用门控权重
        mixed_output1 = torch.sum(gates1.unsqueeze(-1) * expert_outputs1, dim=1)  # 形状变为(batch_size, expert_dim)
        mixed_output2 = torch.sum(gates2.unsqueeze(-1) * expert_outputs2, dim=1)  # 形状变为(batch_size, expert_dim)

        # 合并两个输入的混合输出
        combined_output = torch.cat((mixed_output1, mixed_output2), dim=1)  # 形状变为(batch_size, expert_dim * 2)
        if x3 is not None:
            combined_output = torch.cat((combined_output, mixed_output3), dim=1)
        # 分类层输出
        classification_output = self.classifier(combined_output)  # 形状变为(batch_size, output_dim)
        return classification_output


