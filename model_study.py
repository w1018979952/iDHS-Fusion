# from self_attention_pytorch import MultiHeadSelfAttention
# from self_attention import SeqSelfAttention
# # 将mamba_simple.py所在的目录添加到sys.path
# sys.path.append(os.path.abspath('path/to/mamba1p1p1/mambassm/modules'))
# from tool.mamba1p1p1.mambassm.modules.mamba_simple import Mamba as vmamba

from tool.moh import MoHAttention,CrossAttention
from tool.attention import CombinedAttention,MoETopKAttention
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


class cnn_complex(nn.Module):
    def __init__(self):
        super(cnn_complex, self).__init__()
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


class cnn_simple(nn.Module):
    def __init__(self):
        super(cnn_simple, self).__init__()
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
        embed = x.permute(0, 2, 1)
        return embed


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


class DHS(nn.Module):
    def __init__(self):
        super(DHS, self).__init__()
        self.cnn_simple = cnn_simple().apply(weights_init_uniform)

        self.cnn_complex = cnn_complex().apply(weights_init_uniform)

        self.drop2 = nn.Dropout(p=0.2)

        self.attn1 = SeqSelfAttention(units=64, attention_type='additive', attention_activation=F.sigmoid, )
        self.attn2 = SeqSelfAttention(units=64, attention_type='additive', attention_activation=F.sigmoid, )

        self.flatten = nn.Flatten()

        self.rnn1 = nn.GRU(input_size=64, hidden_size=64, batch_first=True, bidirectional=True)
        self.rnn2 = nn.GRU(input_size=64, hidden_size=64, batch_first=True, bidirectional=True)

        self.attn_fe = CrossAttention(64,8,True)
        self.attn_ef = CrossAttention(64,8,True)

        self.mase_cs = MASE(dim=64)  # .apply(weights_init_uniform)
        self.mase_ls = MAE(dim=64)  # .apply(weights_init_uniform)

        self.moe = MoE(input_dim=64)  # .apply(weights_init_uniform)
        # self.moet = MoETopKAttention(dim=64, num_heads=8, num_experts=4)
        self.moet = MoHAttention(dim=64, num_heads=4, qkv_bias=True, qk_norm=True, attn_drop=0.1, proj_drop=0.1,
                                shared_head=2, routed_head=2, head_dim=16
                                )
        # self.moet = CombinedAttention(dim=64,num_heads=8,qkv_bias=False,qk_norm=False,attn_drop=0.05,proj_drop=0.05,shared_head=4,routed_head=4,head_dim=8)
        self.pooling_layer1 = DynamicExpectationPooling(1)  # 压缩序列
        self.pooling_layer2 = DynamicExpectationPooling(1)  # 压缩序列
        self.pooling_layer3 = DynamicExpectationPooling(1)  # 压缩序列

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
        dna_embedding = input[0]
        simple = self.cnn_simple(dna_embedding.int())
        dna_multicode = input[2]
        complex = self.cnn_complex(dna_multicode)

        # eh, eq = self.rnn1(simple)
        # ehide = eh[:, :, :64] + eh[:, :, 64:]

        fh, fq = self.rnn2(complex)
        fhide = fh[:, :, :64] + fh[:, :, 64:]

        # # 交叉注意力
        # ef = self.attn_ef(fhide, ehide)
        # fe = self.attn_fe(ehide, fhide)
        # 残差自注意力
        # res_e = self.moet(ehide)
        res_f = self.moet(fhide)

        # xe = self.mase_cs(ef, res_e)
        # xf = self.mase_cs(fe, res_f)

        out_xef = self.classifier1(self.flatten(res_f))

        # part_xe = self.flatten(self.mase_ls(ehide, res_e))
        # part_xf = self.flatten(self.mase_ls(fhide, res_f))
        #
        # out_xe, out_xf = self.classifier2(part_xe), self.classifier3(part_xf),

        return out_xef, (out_xef, out_xef)

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


class MASE(nn.Module):
    def __init__(self, dim, r=8):
        super(MASE, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.Linear(dim, dim // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // r, dim, bias=False),
            nn.Sigmoid()
        )

        self.IN = nn.InstanceNorm1d(dim, track_running_stats=False)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, xc, x):
        xc_tmp = xc.permute(0, 2, 1)

        avg_pooled = self.avg_pool(xc_tmp).permute(0, 2, 1)
        max_pooled = self.max_pool(xc_tmp).permute(0, 2, 1)

        avg_mask = self.channel_attention(avg_pooled)
        max_mask = self.channel_attention(max_pooled)

        mask = (avg_mask + max_mask) / 2

        output = x * mask + self.IN(xc) * (1 - mask)  # x和xc换位

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


class MAE(nn.Module):
    def __init__(self, dim, r=8):
        super(MAE, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.Linear(dim, dim // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // r, dim, bias=False),
            nn.Sigmoid()
        )

        self.IN = nn.InstanceNorm1d(dim, track_running_stats=False)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, xc, x):
        xc_tmp = xc.permute(0, 2, 1)
        max_pooled = self.max_pool(xc_tmp).permute(0, 2, 1)
        max_mask = self.channel_attention(max_pooled)

        mask = max_mask
        output = x * mask + self.IN(xc) * (1 - mask)  # x和xc换位
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
