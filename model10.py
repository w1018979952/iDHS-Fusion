# from self_attention_pytorch import MultiHeadSelfAttention
# from self_attention import SeqSelfAttention
# # 将mamba_simple.py所在的目录添加到sys.path
# sys.path.append(os.path.abspath('path/to/mamba1p1p1/mambassm/modules'))
# from tool.mamba1p1p1.mambassm.modules.mamba_simple import Mamba as vmamba

from tool.adv import *
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

        self.conv1 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, padding='same')
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding='same')
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.drop = nn.Dropout(0.2)

    def forward(self, x,is_bn=False):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        if is_bn:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.drop(x)
        return x.permute(0, 2, 1)


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
        q, k, v = self.q(input1), self.k(input2), self.v(input2)
        # q, k, v = self.q(input1), self.k(input2), self.v(input1)
        return AdditiveAttention(q, k, v, True, 0.5)


class DHS(nn.Module):
    def __init__(self):
        super(DHS, self).__init__()
        self.emb = nn.Embedding(27, 64)
        self.mapping = nn.Linear(13, 64)
        self.convfeature = cnn().apply(weights_init_uniform)
        self.bilstm = nn.LSTM(input_size=64, hidden_size=64, batch_first=True, bidirectional=True)

        self.attn = SeqSelfAttention(units=64, attention_type='additive', attention_activation=F.sigmoid, )

        self.flatten = nn.Flatten()




        self.mlp1 = nn.Sequential(nn.Linear(4800, 100),
                                  nn.ReLU(),
                                  nn.Linear(100, 1)
                                  ).apply(weights_init_uniform)

        layers = [1, 2, 2, 2]
        self.resSeNet = ResNetClassifier(ResidualSEBlock, layers, num_classes=1).apply(weights_init_uniform)

        self.attn_fe = CustomAttention().apply(weights_init_uniform)

        self.mase = MASE(dim=64).apply(weights_init_uniform)

        self.maf = MAFusion(dim=64)

        self.pooling_layer = DynamicExpectationPooling(1)  # 压缩序列

        self.classifier = nn.Linear(64, 1, bias=False)
        self.adv = AdversarialLoss()
        self.decrease = nn.Linear(4800, 1000, bias=False)


    def forward(self, input):
        dna_embed = self.emb(input[0].int())
        feature = self.mapping(input[2])

        dna_embed = self.convfeature(dna_embed)
        feature = self.convfeature(feature,is_bn=True)

        eh, (eh_n, ec_n) = self.bilstm(dna_embed)
        ehide = eh[:, :, :64] + eh[:, :, 64:]
        ehn = torch.cat((eh_n[0], eh_n[1]), dim=-1)

        fh, (fh_n, fc_n) = self.bilstm(feature)
        fhide = fh[:, :, :64] + fh[:, :, 64:]
        fhn = torch.cat((fh_n[0], fh_n[1]), dim=-1)
        # 交叉注意力
        ef = self.attn_fe(fhide, ehide)
        fe = self.attn_fe(ehide, fhide)
        #
        xe = self.mase(ef, self.attn(ehide))
        xf = self.mase(fe, self.attn(fhide))

        # out_xef = self.resSeNet(torch.stack([xe, xf], dim=1))
        out_xef = self.resSeNet(torch.stack([xe, xf], dim=1))

        # out_xef = self.mlp1(self.flatten(xe)+self.flatten(xf))
        # out_adv = self.adv(self.decrease(self.flatten(ehide)), self.decrease(self.flatten(fhide)))
        out_adv = self.adv(ehn,fhn)
        dp_xe = self.pooling_layer(ehide)
        dp_xf = self.pooling_layer(fhide)
        out_xe, out_xf = self.classifier(dp_xe), self.classifier(dp_xf),

        return out_xef, (out_adv, out_xe, out_xf)

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
    def __init__(self, dim, r=4):
        super(MASE, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.Conv1d(dim, dim // r, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim // r, dim, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.IN = nn.InstanceNorm1d(dim, track_running_stats=False)

    def forward(self, xc, x):
        xc, x = xc.permute(0, 2, 1), x.permute(0, 2, 1)
        avg_pooled = F.avg_pool1d(xc, 75)
        max_pooled = F.max_pool1d(xc, 75)

        avg_mask = self.channel_attention(avg_pooled)
        max_mask = self.channel_attention(max_pooled)

        mask = avg_mask + max_mask

        output = x * mask + self.IN(xc) * (1 - mask)
        # xo = x * wei + residual * (1 - wei)
        # output = x * (1 - mask) + self.IN(xc) * mask
        return output.permute(0, 2, 1)


class MAFusion(nn.Module):
    def __init__(self, dim, r=4):
        super(MAFusion, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.Conv1d(dim, dim // r, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim // r, dim, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.IN = nn.InstanceNorm1d(dim, track_running_stats=False)

    def forward(self, xe, xf):
        xe, xf = xe.permute(0, 2, 1), xf.permute(0, 2, 1)

        maxe = self.channel_attention(F.max_pool1d(xe, 75)) + self.channel_attention(F.avg_pool1d(xe, 75))
        maxf = self.channel_attention(F.max_pool1d(xe, 75)) + self.channel_attention(F.avg_pool1d(xe, 75))

        output = xe * maxe + xf * maxf

        return output.permute(0, 2, 1)


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


# class MoHAttention(nn.Module):
#     fused_attn: Final[bool]
#     LOAD_BALANCING_LOSSES = []
#
#     def __init__(
#             self,
#             dim,
#             num_heads=8,
#             qkv_bias=False,
#             qk_norm=False,
#             attn_drop=0.,
#             proj_drop=0.,
#             norm_layer=nn.LayerNorm,
#             shared_head=0,
#             routed_head=0,
#             head_dim=None,
#     ):
#         super().__init__()
#         # assert dim % num_heads == 0, 'dim should be divisible by num_heads'
#         self.num_heads = num_heads
#
#         if head_dim is None:
#             self.head_dim = dim // num_heads
#         else:
#             self.head_dim = head_dim
#
#         self.scale = self.head_dim ** -0.5
#         self.fused_attn = use_fused_attn()
#
#         self.qkv = nn.Linear(dim, (self.head_dim * self.num_heads) * 3, bias=qkv_bias)
#
#         self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
#         self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(self.head_dim * self.num_heads, dim)
#
#         self.proj_drop = nn.Dropout(proj_drop)
#
#         self.shared_head = shared_head
#         self.routed_head = routed_head
#
#         if self.routed_head > 0:
#             self.wg = torch.nn.Linear(dim, num_heads - shared_head, bias=False)
#             if self.shared_head > 0:
#                 self.wg_0 = torch.nn.Linear(dim, 2, bias=False)
#
#         if self.shared_head > 1:
#             self.wg_1 = torch.nn.Linear(dim, shared_head, bias=False)
#
#     def forward(self, x):
#         B, N, C = x.shape
#
#         _x = x.reshape(B * N, C)
#
#         if self.routed_head > 0:
#             logits = self.wg(_x)
#             gates = F.softmax(logits, dim=1)
#
#             num_tokens, num_experts = gates.shape
#             _, indices = torch.topk(gates, k=self.routed_head, dim=1)
#             mask = F.one_hot(indices, num_classes=num_experts).sum(dim=1)
#
#             if self.training:
#                 me = gates.mean(dim=0)
#                 ce = mask.float().mean(dim=0)
#                 l_aux = torch.mean(me * ce) * num_experts * num_experts
#
#                 MoHAttention.LOAD_BALANCING_LOSSES.append(l_aux)
#
#             routed_head_gates = gates * mask
#             denom_s = torch.sum(routed_head_gates, dim=1, keepdim=True)
#             denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
#             routed_head_gates /= denom_s
#             routed_head_gates = routed_head_gates.reshape(B, N, -1) * self.routed_head
#
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv.unbind(0)
#         q, k = self.q_norm(q), self.k_norm(k)
#
#         if self.fused_attn:
#             x = F.scaled_dot_product_attention(
#                 q, k, v,
#                 dropout_p=self.attn_drop.p if self.training else 0.,
#             )
#         else:
#             q = q * self.scale
#             attn = q @ k.transpose(-2, -1)
#             attn = attn.softmax(dim=-1)
#             attn = self.attn_drop(attn)
#             x = attn @ v
#
#         if self.routed_head > 0:
#             x = x.transpose(1, 2)
#
#             if self.shared_head > 0:
#                 shared_head_weight = self.wg_1(_x)
#                 shared_head_gates = F.softmax(shared_head_weight, dim=1).reshape(B, N, -1) * self.shared_head
#
#                 weight_0 = self.wg_0(_x)
#                 weight_0 = F.softmax(weight_0, dim=1).reshape(B, N, 2) * 2
#
#                 shared_head_gates = torch.einsum("bn,bne->bne", weight_0[:, :, 0], shared_head_gates)
#                 routed_head_gates = torch.einsum("bn,bne->bne", weight_0[:, :, 1], routed_head_gates)
#
#                 masked_gates = torch.cat([shared_head_gates, routed_head_gates], dim=2)
#             else:
#                 masked_gates = routed_head_gates
#
#             x = torch.einsum("bne,bned->bned", masked_gates, x)
#             x = x.reshape(B, N, self.head_dim * self.num_heads)
#         else:
#             shared_head_weight = self.wg_1(_x)
#             masked_gates = F.softmax(shared_head_weight, dim=1).reshape(B, N, -1) * self.shared_head
#             x = x.transpose(1, 2)
#
#             x = torch.einsum("bne,bned->bned", masked_gates, x)
#             x = x.reshape(B, N, self.head_dim * self.num_heads)
#
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x