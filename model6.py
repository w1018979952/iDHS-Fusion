from mamba_ssm import Mamba2, Mamba
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
# from self_attention_pytorch import MultiHeadSelfAttention
# from self_attention import SeqSelfAttention
from tools import *
from tool.res2net import Res2Net
import sys
import os
from tool.bmamba import VisionEncoderMambaBlock
# # 将mamba_simple.py所在的目录添加到sys.path
# sys.path.append(os.path.abspath('path/to/mamba1p1p1/mambassm/modules'))
# from tool.mamba1p1p1.mambassm.modules.mamba_simple import Mamba as vmamba
import torch.nn as nn


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.embedding = nn.Embedding(23, 128)
        self.dr = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1, padding=0).apply(weights_init_uniform)
        self.conv1 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, padding=0).apply(weights_init_uniform)

        self.conv11 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, padding=0).apply(weights_init_uniform)
        self.pool11 = nn.MaxPool1d(kernel_size=2)
        self.conv12 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, padding=0).apply(weights_init_uniform)

        self.conv21 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, padding=0).apply(weights_init_uniform)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv22 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, padding=0).apply(weights_init_uniform)

        self.conv31 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, padding=0).apply(weights_init_uniform)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.conv32 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, padding=0).apply(weights_init_uniform)

        self.pool4 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv41 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1).apply(weights_init_uniform)
        self.dropout = nn.Dropout(0.2)

        self.final_conv = nn.Conv1d(in_channels=64 * 5, out_channels=64, kernel_size=1).apply(weights_init_uniform)

        self.convpool1 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=4, stride=4).apply(weights_init_uniform)

        self.convpool2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, stride=5).apply(weights_init_uniform)
        self.fl = nn.Flatten()
        self.mlp = nn.Sequential(nn.Linear(960, 100), nn.ReLU(), nn.Linear(100, 1)).apply(weights_init_uniform)
        self.attn1 = SeqSelfAttention(units=64, attention_type='additive',
                                      attention_regularizer_weight=1e-4)
        self.attn2 = SeqSelfAttention(units=64, attention_type='additive', attention_activation=F.sigmoid,
                                      attention_regularizer_weight=1e-4)

    def forward(self, x):
        # x 的尺寸为 (batch_size, sequence_length, in_channels)
        x = self.embedding(x).permute(0, 2, 1)  # 调整为 (batch_size, in_channels, sequence_length)

        residuals = self.dr(x)  #降维
        residuals = self.conv1(residuals)

        x2 = self.conv12(self.conv11(residuals))

        x3 = self.conv22(self.conv21(residuals))

        x4 = self.conv32(self.conv31(residuals))

        x5 = self.conv41(self.pool4(residuals))

        x = torch.cat([residuals, x2, x3, x4, x5], dim=1)
        x = self.final_conv(x)

        x = self.convpool1(x).permute(0, 2, 1)
        x = self.attn1(x).permute(0, 2, 1)
        x = self.convpool2(F.relu(x))

        # x = self.dropout(x)
        x = self.fl(x)
        x = self.mlp(x)
        return x

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
        self.conv1 = nn.Conv1d(in_channels=77, out_channels=128, kernel_size=3, padding='same')
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding='same')
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        self.bn1 = nn.BatchNorm1d(128)
        # self.ln1 = nn.LayerNorm(128)
        self.bn2 = nn.BatchNorm1d(64)
        # self.ln2 = nn.LayerNorm(64)
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        # x = self.bn1(x)
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
        x = self.drop2(x)  # Reorder dimensions for LSTM
        kmer = x
        x = self.flt(x)
        # 输出均值和对数方差
        mu = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return kmer, mu, logvar

class embedModel(nn.Module):
    def __init__(self):
        super(embedModel, self).__init__()
        self.emb = nn.Embedding(27, 128)
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

        return embed, mu, logvar

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.avg_pool(x).view(b, c)
        fc_out = self.fc(avg_out).view(b, c, 1, 1)
        return x * fc_out.expand_as(x)

class ResidualSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, reduction=16):
        super(ResidualSEBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels, reduction=reduction)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += residual
        return F.relu(out)

class ResNetClassifier(nn.Module):
    def __init__(self, block, layers, num_classes=1, reduction=16):
        super(ResNetClassifier, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(2, self.in_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, reduction=16))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, (out.size(2), out.size(3)))  # 假设输入图像大小为32x32，这里将特征图尺寸减小到1x1
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class CustomAttention(nn.Module):
    def __init__(self):
        super(CustomAttention, self).__init__()
        self.q = nn.Linear(64, 64, bias=False)
        self.k = nn.Linear(64, 64, bias=False)
        self.v = nn.Linear(64, 64, bias=False)

    def forward(self, input1, input2):
        q,  k , v = self.q(input1),self.k(input2), self.v(input2)
        return Attention(q, k, v, True, 0.2)

class DHS(nn.Module):
    def __init__(self):
        super(DHS, self).__init__()
        self.embed = embedModel().apply(weights_init_uniform)

        self.kmer = kmerModel(in_channels=20).apply(weights_init_uniform)

        self.convfeature = cnn().apply(weights_init_uniform)

        self.drop2 = nn.Dropout(p=0.2)

        self.attn = SeqSelfAttention(units=64, attention_type='additive', attention_activation=F.sigmoid, )
        self.attn1 = SeqSelfAttention(units=64, attention_type='additive', attention_activation=F.sigmoid, )
        self.attn2 = SeqSelfAttention(units=64, attention_type='additive', attention_activation=F.sigmoid, )

        self.flatten = nn.Flatten()

        self.bilstm = nn.GRU(input_size=64, hidden_size=64, batch_first=True, bidirectional=True)
        self.bilstm1 = nn.GRU(input_size=64, hidden_size=64, batch_first=True, bidirectional=True)
        self.bilstm2 = nn.GRU(input_size=64, hidden_size=64, batch_first=True, bidirectional=True)

        self.reconstruction_layer = nn.Linear(76 * 64, 304)  # 重构层，从76*64重构到304

        self.mlp1 = nn.Sequential(nn.Linear(4800, 100),
                                  nn.ReLU(),
                                  nn.Linear(100, 1)
                                  ).apply(weights_init_uniform)
        self.mlp2 = nn.Sequential(nn.Linear(4800, 100),
                                  nn.ReLU(),
                                  nn.Linear(100, 1)
                                  ).apply(weights_init_uniform)
        self.mlp3 = nn.Sequential(nn.Linear(4800, 100),
                                  nn.ReLU(),
                                  nn.Linear(100, 1)
                                  ).apply(weights_init_uniform)
        self.mlp4 = nn.Sequential(nn.Linear(4800 * 1, 256),
                                  nn.ReLU(),
                                  nn.Linear(256, 1)
                                  ).apply(weights_init_uniform)

        layers = [1, 1, 1, 1]
        self.resSeNet = ResNetClassifier(ResidualSEBlock, layers, num_classes=1).apply(weights_init_uniform)
        self.attn_df = CustomAttention()
        self.attn_fd = CustomAttention()
        self.attn_test = CustomAttention()

        self.mam1 = MAM(dim=64)
        self.mam2 = MAM(dim=64)


        self.pooling_layer1 = DynamicExpectationPooling(1) #压缩序列
        self.classifier1 = nn.Linear(64,1,bias=False)
        self.classifier2 = nn.Linear(128, 1, bias=False)
        self.classifier3 = nn.Linear(128, 1, bias=False)

        self.block = CMA_Block_1D(64, 32, 64)
    # def forward(self, input):
    #
    #     dna, length = input[0][:, :-1], input[0][:, -1]
    #     kmer=dnakmer = input[1]
    #
    #     kmer,mu1,logvar1 = self.kmer(kmer)#5 3的核
    #     z_kmer = self.reparameterize(mu1, logvar1)
    #
    #     dnaEmbed,mu,logvar = self.embed(dna)#10 5的核
    #     z_emb = self.reparameterize(mu, logvar)
    #     # 使用解码器重构输入
    #     reconstructed_kmer = self.kmerdecoder(z_kmer)
    #     reconstructed_emb = self.embeddecoder(z_emb)
    #
    #     # 创建掩码
    #     max_length_emb = dna.size(1)
    #     mask = torch.arange(max_length_emb).expand(len(length), max_length_emb).to('cuda') < length.unsqueeze(1)
    #     mask = mask.float()
    #     mask = torch.flip(mask, dims=[1])  # 反转掩码
    #
    #     # 计算重构损失
    #     reconstruction_loss_emb = F.cross_entropy((reconstructed_emb * mask.unsqueeze(-1)).permute(0,2,1), dna ,reduction='sum') / mask.sum()
    #
    #     mask=mask[:,1:].unsqueeze(-1)
    #     reconstruction_loss_kmer = F.binary_cross_entropy_with_logits(reconstructed_kmer * mask, dnakmer ,reduction='sum') / mask.sum()
    #     # kmer = self.bilstm1(kmer)[0]
    #     # kmer = kmer[:,:,:64]+kmer[:,:,64:]
    #     kmer = self.attn1(kmer)
    #
    #     dnaEmbed = self.bilstm(dnaEmbed)[0]
    #     dnaEmbed = dnaEmbed[:,:,:64]#+dnaEmbed[:,:,64:]
    #
    #     dnaEmbed = self.attn(dnaEmbed)
    #
    #     dnaEmbed = self.flatten(dnaEmbed)
    #     kmer = self.flatten(kmer)
    #
    #     # 计算余弦相似度
    #     cosine_sim = F.cosine_similarity(dnaEmbed, kmer, dim=1)
    #     # 计算损失
    #     loss_cos = cosine_similarity_loss(cosine_sim)
    #
    #     # 创建对比损失实例
    #     contrastive_loss = ContrastiveLoss(margin=1.0)
    #
    #     # dnaEmbed = torch.cat((dnaEmbed,z_emb),dim=-1)
    #     # kmer = torch.cat((kmer,z_kmer),dim=-1)
    #
    #     out = self.moe(dnaEmbed,kmer)
    #     # out = F.sigmoid(out)
    #     # 计算KL散度损失
    #     kl_loss_emb = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1).mean()
    #     kl_loss_kmer = -0.5 * torch.sum(1 + logvar1 - mu1 ** 2 - logvar1.exp(), dim=1).mean()
    #
    #     return out,0#loss_cos#self.attn.attention_regularizer_loss+self.attn1.attention_regularizer_loss+#(reconstruction_loss_emb + reconstruction_loss_kmer)*1e-3 + (kl_loss_emb + kl_loss_kmer)*1e-4

    def forward(self, input):
        dna = input[0]
        dnaEmbed, mu, logvar = self.embed(dna.int())

        feature = input[2]

        # 计算每个特征维度 d 的最小值和最大值，沿着 batch 和序列维度
        # X_min = feature.min(dim=1, keepdim=True)[0]  # 按照序列长度维度求最小值
        # X_max = feature.max(dim=1, keepdim=True)[0]  # 按照序列长度维度求最大值
        # feature = (feature - X_min) / (X_max - X_min + 1e-5)  # 加上1e-5防止除以0
        feature = self.convfeature(feature)

        dh, dq = self.bilstm1(dnaEmbed)

        dhide = dh[:, :, :64] + dh[:, :, 64:]
        dseq = torch.cat((dq[0, :, :], dq[1, :, :]), dim=1)

        # kmer = input[1]
        # kmer, mu1, logvar1 = self.kmer(kmer)  # 5 3的核
        # kmer = self.bilstm1(kmer)[0]
        # kmer = kmer[:, :, :64] + kmer[:, :, 64:]


        fh,fq = self.bilstm2(feature)

        fhide = fh[:, :, :64]+fh[:, :, 64:]
        fseq = torch.cat((fq[0, :, :], fq[1, :, :]), dim=1)

        # feature = self.attn2(feature)

        #交叉注意力

        dfv = self.attn_df(dhide, fhide)
        fdv = self.attn_fd(fhide, dhide)

        # test = self.attn_test(dhide,dhide)


        dfv = self.mam1(dfv)
        fdv = self.mam1(fdv)

        dfv = dhide.permute(0,2,1)
        fdv = fhide.permute(0,2,1)
        df = torch.stack((dfv, fdv), dim=1)

        dkse = self.resSeNet(df)

        # dk = dfv+fdv
        # dk = self.pooling_layer1(dk)

        out_feature = self.mlp3(self.flatten(fhide))
        out_embed = self.mlp3(self.flatten(dhide))


        return dkse, (out_embed, out_feature)

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
                output, output2 = self.forward((batch1, batch2, batch3))
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


def Attention(Q, K, V, use_DropKey, mask_ratio):
    attn = (Q * (Q.shape[1] ** -0.5)) @ K.transpose(-2, -1)

    # use DropKey as regularizer
    if use_DropKey == True:
        m_r = torch.ones_like(attn) * mask_ratio
        attn = attn + torch.bernoulli(m_r) * -1e-12

    attn = attn.softmax(dim=-1)
    x = attn @ V
    return x


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


class MAM(nn.Module):
    def __init__(self, dim, r=4):
        super(MAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.Conv1d(dim, dim // r, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim // r, dim, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.IN = nn.InstanceNorm1d(dim, track_running_stats=False)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        Avg_pooled = F.avg_pool1d(x, 75)
        Max_pooled = F.max_pool1d(x, 75)

        Avg_mask = self.channel_attention(Avg_pooled)
        Max_mask = self.channel_attention(Max_pooled)

        mask = Avg_mask + Max_mask

        output = x * mask + self.IN(x) * (1 - mask)
        return output.permute(0, 2, 1)

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