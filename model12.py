# from self_attention_pytorch import MultiHeadSelfAttention
# from self_attention import SeqSelfAttention
# # 将mamba_simple.py所在的目录添加到sys.path
# sys.path.append(os.path.abspath('path/to/mamba1p1p1/mambassm/modules'))
# from tool.mamba1p1p1.mambassm.modules.mamba_simple import Mamba as vmamba

from tool.bmamba import VisionEncoderMambaBlock
from tools import *


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
        self.conv1 = nn.Conv1d(in_channels=13, out_channels=128, kernel_size=5, padding='same')
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
        x = self.bn1(x)
        x = F.relu(x)

        x = self.pool1(x)

        x = self.conv2(x)
        # x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # x = self.drop(x)
        # x = self.attn1(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x.permute(0, 2, 1)


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
        # x = self.drop2(x)
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





class DHS(nn.Module):
    def __init__(self):
        super(DHS, self).__init__()
        self.embed = embedModel().apply(weights_init_uniform)

        self.convfeature = cnn().apply(weights_init_uniform)

        self.drop2 = nn.Dropout(p=0.2)

        self.attn = SeqSelfAttention(units=64, attention_type='additive', attention_activation=F.sigmoid, )
        self.attn1 = SeqSelfAttention(units=64, attention_type='additive', attention_activation=F.sigmoid, )
        self.attn2 = SeqSelfAttention(units=64, attention_type='additive', attention_activation=F.sigmoid, )
        self.attn3 = ResAttention()
        self.attn4 = ResAttention()

        self.flatten = nn.Flatten()

        self.gru = nn.GRU(input_size=61, hidden_size=64, bidirectional=True, batch_first=True)
        self.vmamba = VisionEncoderMambaBlock(dim=64, dt_rank=4, d_state=16, dim_inner=64).to("cuda")

        self.ms_cam = MS_CAM(channels=64, r=4)
        self.aff = AFF(channels=64, r=4)
        self.iaff = iAFF(channels=64, r=4)

        self.bilstm = nn.LSTM(input_size=64, hidden_size=64, batch_first=True, bidirectional=True)
        self.rnn1 = nn.GRU(input_size=64, hidden_size=64, batch_first=True, bidirectional=True)
        self.rnn2 = nn.GRU(input_size=64, hidden_size=64, batch_first=True, bidirectional=True)

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

        layers = [1, 2, 2, 2]
        self.resSeNet = ResNetClassifier(ResidualSEBlock, layers, num_classes=1).apply(weights_init_uniform)

        self.attn_fe = CrossAttention()
        self.attn_ef = CrossAttention()

        # self.mase_cs = MASE(dim=64)#.apply(weights_init_uniform)
        self.mase_cs = MASE(dim=64)  # .apply(weights_init_uniform)
        # self.mase_ls = MASE(dim=64)#.apply(weights_init_uniform)
        self.mase_ls = MASE(dim=64)  # .apply(weights_init_uniform)

        # self.mse_ef = MSE(dim=64)#.apply(weights_init_uniform)
        self.mse_ef = MoE(input_dim=64)  # .apply(weights_init_uniform)

        self.pooling_layer1 = DynamicExpectationPooling(1)  # 压缩序列
        self.pooling_layer2 = DynamicExpectationPooling(1)  # 压缩序列
        self.pooling_layer3 = DynamicExpectationPooling(1)  # 压缩序列
        self.classifier = nn.Linear(64 * 75, 1, bias=False)
        self.classifier1 = nn.Sequential(nn.Linear(4800, 100),
                                         nn.ReLU(),
                                         nn.Linear(100, 1)
                                         )
        self.classifier2 = nn.Sequential(nn.Linear(4800, 100),
                                         nn.ReLU(),
                                         nn.Linear(100, 1)
                                         )
        self.classifier3 = nn.Sequential(nn.Linear(4800, 100),
                                         nn.ReLU(),
                                         nn.Linear(100, 1)
                                         )

    def forward(self, input):
        dna = input[0]
        e, mu, logvar = self.embed(dna.int())
        feature = input[2]
        f = self.convfeature(feature)

        e = F.dropout(e, p=0.2, training=self.training)
        f = F.dropout(f, p=0.2, training=self.training)

        eh, eq = self.rnn1(e)
        ehide = eh[:, :, :64] + eh[:, :, 64:]

        fh, fq = self.rnn2(f)
        fhide = fh[:, :, :64] + fh[:, :, 64:]

        # 加性自注意力
        res_e = self.attn2(ehide)
        res_f = self.attn4(fhide)

        # 交叉注意力
        # ef = self.attn_ef(res_e, res_f)
        # fe = self.attn_fe(res_f, res_e)

        ef = self.attn_ef(ehide, fhide)
        fe = self.attn_fe(fhide, ehide)

        xe = self.mase_cs(ef, res_e)
        xf = self.mase_cs(fe, res_f)

        # xe = F.dropout(xe, p=0.6, training=self.training)
        # xf = F.dropout(xf, p=0.3, training=self.training)
        # out_xef = self.resSeNet(torch.stack([xe,xf],dim=1))

        out_xef = self.classifier1(self.flatten(self.attn1(xe + xf)))

        part_xe = self.flatten(self.mase_ls(ehide, res_e))
        part_xf = self.flatten(self.mase_ls(fhide, res_f))

        out_xe, out_xf = self.classifier2(part_xe), self.classifier3(part_xf),

        return out_xef, (out_xe, out_xf)

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

class CustomAttention(nn.Module):
    def __init__(self):
        super(CustomAttention, self).__init__()
        self.q = nn.Linear(64, 64, bias=False)
        self.k = nn.Linear(64, 64, bias=False)
        self.v = nn.Linear(64, 64, bias=False)

    def forward(self, input1, input2):
        q, k, v = self.q(input1), self.k(input2), self.v(input2)
        # q, k, v = self.q(input1), self.k(input2), self.v(input1)
        return Attention(q, k, v, True, 0.2)

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


class MASE(nn.Module):
    def __init__(self, dim, r=1):
        super(MASE, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.Linear(dim, dim // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // r, dim, bias=False),
            nn.Tanh()
        )

        self.IN = nn.InstanceNorm1d(dim, track_running_stats=False)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, xc, x):

        xc_tmp = xc.permute(0, 2, 1)
        x_tmp  =  x.permute(0, 2, 1)
        avg_pooled = self.avg_pool(xc_tmp).permute(0, 2, 1)
        max_pooled = self.max_pool(x_tmp).permute(0, 2, 1)

        avg_mask = self.channel_attention(avg_pooled)
        max_mask = self.channel_attention(max_pooled)

        mask = (avg_mask + max_mask) / 2

        # output = x * mask + self.IN(xc) * (1 - mask)  # x和xc换位
        # output = x * mask + xc * (1 - mask)  # x和xc换位
        output = xc * mask + x * (1 - mask)  # x和xc换位

        # output = xc * mask + self.IN(x) * (1 - mask)
        # xo = x * wei + residual * (1 - wei)
        # output = x * (1 - mask) + self.IN(xc) * mask
        return output


class MoE(nn.Module):
    def __init__(self, input_dim, num_experts=4, top_k=2):
        """
        Mixture of Experts (MoE) 模块 (支持双输入)
        参数：
            input_dim (int): 输入特征的维度
            num_experts (int): 专家网络的数量
            hidden_dim (int): 专家网络的隐藏层大小
            top_k (int): 每次选择的专家数量
        """
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # 定义专家网络 (Experts)，每个专家使用 MASE 模块
        self.experts = nn.ModuleList([
            MASE(input_dim) for _ in range(num_experts)
        ])

        # 定义门控网络 (Gating Network)
        self.gate = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=1)  # 每个专家的权重
        )

    def forward(self, xc, x):
        """
        前向传播
        输入:
            xc (Tensor): 输入张量 1，形状为 (batch_size, seq_len, input_dim)
            x (Tensor): 输入张量 2，形状为 (batch_size, seq_len, input_dim)
        输出:
            output (Tensor): 融合后的输出，形状为 (batch_size, seq_len, input_dim)
        """
        # 将 xc 和 x 的第一个时间步的平均值作为门控网络的输入
        gate_input = (xc.mean(dim=1) + x.mean(dim=1)) / 2  # (batch_size, input_dim)

        # 获取门控权重
        gate_weights = self.gate(gate_input)  # (batch_size, num_experts)

        # 选择 top-k 专家
        top_k_weights, top_k_indices = torch.topk(gate_weights, self.top_k, dim=1)  # (batch_size, top_k)

        # 初始化输出
        output = torch.zeros_like(x).to(x.device)

        # 对每个专家输出进行加权求和
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]  # 第 i 个选中专家的索引
            weight = top_k_weights[:, i].unsqueeze(-1).unsqueeze(-1)  # 对应的权重 (batch_size, 1, 1)

            # 按 batch 动态选择专家
            expert_output = torch.stack([self.experts[idx](xc[b].unsqueeze(0), x[b].unsqueeze(0))
                                         for b, idx in enumerate(expert_idx)])
            output += weight * expert_output.squeeze(1)  # 加权求和

        return output


class MSE(nn.Module):
    def __init__(self, dim, r=8):
        super(MSE, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.Linear(dim, dim // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // r, dim, bias=False),
            nn.Sigmoid()
        )
        self.rule = nn.ReLU(inplace=True)
        # self.IN = nn.InstanceNorm1d(dim, track_running_stats=False)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x1, x2):
        xc_tmp = (x1 * x2).permute(0, 2, 1)
        xc_tmp = self.rule(xc_tmp)

        # avg_pooled = self.avg_pool(xc_tmp).permute(0, 2, 1)
        max_pooled = self.max_pool(xc_tmp).permute(0, 2, 1)

        # avg_mask = self.channel_attention(avg_pooled)
        max_mask = self.channel_attention(max_pooled)

        # mask = (avg_mask + max_mask) / 2

        output = (x1 * x2) * max_mask + (x1 + x2) * (1 - max_mask)  # x和xc换位
        # output = xc * mask + self.IN(x) * (1 - mask)
        # xo = x * wei + residual * (1 - wei)
        # output = x * (1 - mask) + self.IN(xc) * mask
        return output


class DynamicExpectationPooling(nn.Module):
    def __init__(self, dim):
        super(DynamicExpectationPooling, self).__init__()
        # 初始化 m 的参数，m 是可训练的标量，初始值为 1
        self.m = nn.Parameter(torch.ones(1))
        self.dim = dim

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

from einops import rearrange

class CrossAttention(nn.Module):
    def __init__(self, dim=64, num_heads=4, bias=False):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads

        # 温度参数，初始化为全 1
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 线性层，将输入投影为 Q, K, V
        self.to_q = nn.Linear(dim, dim, bias=bias)  # 用于 Query
        self.to_kv = nn.Linear(dim, dim * 2, bias=bias)  # 用于 Key 和 Value

        # 1D 卷积，进一步处理 Q, K, V
        self.q_dwconv = nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv_dwconv = nn.Conv1d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)

        # 输出投影层
        self.project_out = nn.Linear(dim, dim, bias=bias)
        # Dropout 层（未使用）
        self.attn_drop = nn.Dropout(0.)

        # 可学习的加权参数，用于组合不同稀疏掩码的输出
        self.attn1 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def forward(self, x, context):
        b, l_q, d = x.shape  # Query 的形状: (batch_size, sequence_length_q, feature_dim)
        _, l_kv, _ = context.shape  # Key 和 Value 的形状: (batch_size, sequence_length_kv, feature_dim)

        # 生成 Query
        q = self.to_q(x)  # 形状: (b, l_q, d)
        q = q.transpose(1, 2)  # 形状: (b, d, l_q)
        q = self.q_dwconv(q)  # 形状: (b, d, l_q)
        q = q.transpose(1, 2)  # 形状: (b, l_q, d)

        # 生成 Key 和 Value
        kv = self.to_kv(context)  # 形状: (b, l_kv, 2 * d)
        kv = kv.transpose(1, 2)  # 形状: (b, 2 * d, l_kv)
        kv = self.kv_dwconv(kv)  # 形状: (b, 2 * d, l_kv)
        kv = kv.transpose(1, 2)  # 形状: (b, l_kv, 2 * d)
        k, v = kv.chunk(2, dim=-1)  # 每个形状: (b, l_kv, d)

        # 将 Q, K, V 重新排列为多头注意力的形状
        q = rearrange(q, 'b l (head d) -> b head l d', head=self.num_heads)
        k = rearrange(k, 'b l (head d) -> b head l d', head=self.num_heads)
        v = rearrange(v, 'b l (head d) -> b head l d', head=self.num_heads)

        # 计算注意力得分，并缩放温度参数
        attn = (q @ k.transpose(-2, -1)) * self.temperature  # 形状: (b, head, l_q, l_kv)

        # 生成四种不同稀疏程度的掩码
        mask1 = torch.zeros(b, self.num_heads, l_q, l_kv, device=x.device, requires_grad=False)
        mask2 = torch.zeros(b, self.num_heads, l_q, l_kv, device=x.device, requires_grad=False)
        mask3 = torch.zeros(b, self.num_heads, l_q, l_kv, device=x.device, requires_grad=False)
        mask4 = torch.zeros(b, self.num_heads, l_q, l_kv, device=x.device, requires_grad=False)

        # 基于排名生成掩码
        index = torch.topk(attn, k=int(l_kv / 2), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(l_kv * 2 / 3), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(l_kv * 3 / 4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(l_kv * 4 / 5), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))

        # 计算 Softmax
        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)
        attn4 = attn4.softmax(dim=-1)

        # 计算加权输出
        out1 = (attn1 @ v)
        out2 = (attn2 @ v)
        out3 = (attn3 @ v)
        out4 = (attn4 @ v)

        # 组合不同稀疏掩码的输出
        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4

        # 重新排列为原始形状
        out = rearrange(out, 'b head l d -> b l (head d)', head=self.num_heads)

        # 输出投影
        out = self.project_out(out)
        return out

class ResAttention(nn.Module):
    def __init__(self, dim=64, num_heads=4, bias=False):
        super(ResAttention, self).__init__()
        self.num_heads = num_heads

        # 温度参数，初始化为全 1
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 线性层，将输入投影为 Q, K, V
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        # 1D 卷积，进一步处理 Q, K, V
        self.qkv_dwconv = nn.Conv1d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        # 输出投影层
        self.project_out = nn.Linear(dim, dim, bias=bias)
        # Dropout 层（未使用）
        self.attn_drop = nn.Dropout(0.)

        # 可学习的加权参数，用于组合不同稀疏掩码的输出
        self.attn1 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def forward(self, x):
        b, l, d = x.shape  # 输入形状: (batch_size, sequence_length, feature_dim)

        # 生成 Q, K, V
        qkv = self.qkv(x)  # 形状: (b, l, 3 * d)
        qkv = qkv.transpose(1, 2)  # 形状: (b, 3 * d, l)
        qkv = self.qkv_dwconv(qkv)  # 形状: (b, 3 * d, l)
        qkv = qkv.transpose(1, 2)  # 形状: (b, l, 3 * d)
        q, k, v = qkv.chunk(3, dim=-1)  # 每个形状: (b, l, d)

        # 将 Q, K, V 重新排列为多头注意力的形状
        q = rearrange(q, 'b l (head d) -> b head l d', head=self.num_heads)
        k = rearrange(k, 'b l (head d) -> b head l d', head=self.num_heads)
        v = rearrange(v, 'b l (head d) -> b head l d', head=self.num_heads)

        # 计算注意力得分，并缩放温度参数
        attn = (q @ k.transpose(-2, -1)) * self.temperature  # 形状: (b, head, l, l)

        # 生成四种不同稀疏程度的掩码
        mask1 = torch.zeros(b, self.num_heads, l, l, device=x.device, requires_grad=False)
        mask2 = torch.zeros(b, self.num_heads, l, l, device=x.device, requires_grad=False)
        mask3 = torch.zeros(b, self.num_heads, l, l, device=x.device, requires_grad=False)
        mask4 = torch.zeros(b, self.num_heads, l, l, device=x.device, requires_grad=False)

        # 基于排名生成掩码
        index = torch.topk(attn, k=int(l / 2), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(l * 2 / 3), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(l * 3 / 4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(l * 4 / 5), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))

        # 计算 Softmax
        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)
        attn4 = attn4.softmax(dim=-1)

        # 计算加权输出
        out1 = (attn1 @ v)
        out2 = (attn2 @ v)
        out3 = (attn3 @ v)
        out4 = (attn4 @ v)

        # 组合不同稀疏掩码的输出
        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4

        # 重新排列为原始形状
        out = rearrange(out, 'b head l d -> b l (head d)', head=self.num_heads)

        # 输出投影
        out = self.project_out(out)
        return out