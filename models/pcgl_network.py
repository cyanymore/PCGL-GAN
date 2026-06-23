import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, einops
import numbers
import functools
import math


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class GFE(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(GFE, self).__init__()

        self.num_heads = num_heads
        self.temperature1 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        m = x
        x = self.norm1(x)

        x = self.qkv(x)
        qkv = self.qkv_dwconv(x)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(2, 3)) * self.temperature1
        attn = attn.softmax(dim=-1)
        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)

        V1 = out + m

        return V1


class LFE(nn.Module):
    def __init__(self, dim, num_heads, bias, N=8):
        super(LFE, self).__init__()

        self.N = N
        self.num_heads = num_heads
        self.temperature1 = nn.Parameter(torch.ones(num_heads, 1, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        m = x
        x = self.norm1(x)
        h_pad = self.N - h % self.N if not h % self.N == 0 else 0
        w_pad = self.N - w % self.N if not w % self.N == 0 else 0
        x = F.pad(x, (0, w_pad, 0, h_pad), 'reflect')

        b, c, H, W = x.shape
        x = self.qkv(x)
        qkv = self.qkv_dwconv(x)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) (h1 N1)  (w1 N2) -> b head c (N1 N2) (h1 w1)', head=self.num_heads, N1=self.N,
                      N2=self.N)
        k = rearrange(k, 'b (head c) (h1 N1)  (w1 N2) -> b head c (N1 N2) (h1 w1)', head=self.num_heads, N1=self.N,
                      N2=self.N)
        v = rearrange(v, 'b (head c) (h1 N1)  (w1 N2) -> b head c (N1 N2) (h1 w1)', head=self.num_heads, N1=self.N,
                      N2=self.N)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(3, 4)) * self.temperature1
        attn = attn.softmax(dim=-1)
        out = (attn @ v)

        out = rearrange(out, 'b head c (N1 N2) (h1 w1) -> b (head c) (h1 N1)  (w1 N2)', head=self.num_heads, N1=self.N,
                        N2=self.N, h1=H // self.N, w1=W // self.N)

        out = self.project_out(out)

        out = out[:, :, :h, :w]

        V1 = out + m

        return V1


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class GLCE_map(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2.66, bias=True, LayerNorm_type='WithBias', N=4):
        super(GLCE_map, self).__init__()

        self.attn_gfe = GFE(dim, num_heads, bias)
        self.attn_intra = LFE(dim, num_heads, bias, N=N)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        m = self.attn_gfe(x)
        z = self.attn_intra(m)

        out = z + self.ffn(self.norm2(z))

        return out


class GLCE(nn.Module):
    def __init__(self, n_layers=2, dim=576, num_heads=4, ffn_expansion_factor=2.66, bias=True,
                 LayerNorm_type='WithBias', N=4):
        super().__init__()
        self.N = N
        self.layer_stacks = nn.ModuleList([
            GLCE_map(dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, N=N)
            for _ in range(n_layers)])

    def forward(self, x):
        for layer_stack in self.layer_stacks:
            x = layer_stack(x)
        return x


class Cross_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Cross_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        dwkernel_size = 3  # 默认为3
        paddings = dwkernel_size // 2
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=dwkernel_size, stride=1, padding=paddings,
                                   groups=dim * 2, bias=bias)

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=dwkernel_size, stride=1, padding=paddings, groups=dim,
                                  bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def forward(self, x, y):
        b, c, h, w = x.shape

        kv = self.kv_dwconv(self.kv(x))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(y))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, C, _ = q.shape

        mask1 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask2 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask3 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask4 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        index = torch.topk(attn, k=int(C / 2), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C * 2 / 3), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C * 3 / 4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C * 4 / 5), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))

        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)
        attn4 = attn4.softmax(dim=-1)

        out1 = (attn1 @ v)
        out2 = (attn2 @ v)
        out3 = (attn3 @ v)
        out4 = (attn4 @ v)

        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


def get_freq_indices(method):
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                      'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2,
                             6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0,
                             5, 3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2,
                             3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5,
                             4, 3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5,
                             3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3,
                             3, 3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y


class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction=16, freq_sel_method='top16'):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, h, w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered.
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)


class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """

    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()

        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

        # fixed random init
        # self.register_buffer('weight', torch.rand(channel, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))

        # num_freq, h, w

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight

        result = torch.sum(x, dim=[2, 3])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = self.build_filter(t_x, u_x,
                                                                                           tile_size_x) * self.build_filter(
                        t_y, v_y, tile_size_y)

        return dct_filter


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CSCC(nn.Module):
    def __init__(self, out_channels, img_size=256, reduction=16):
        super(CSCC, self).__init__()
        c2wh = dict(
            [(out_channels // 4, img_size // 4), (out_channels // 2, img_size // 8), (out_channels, img_size // 16)])
        # 确保 out_channels 在 c2wh 字典中是有效的键
        if out_channels not in c2wh:
            raise ValueError(f"out_channels value {out_channels} is not supported.")

        self.spatial = SpatialAttention()
        self.frequency_channel = MultiSpectralAttentionLayer(
            channel=out_channels,
            dct_h=c2wh[out_channels],
            dct_w=c2wh[out_channels],
            reduction=reduction,
            freq_sel_method='top16'
        )

    def forward(self, x):
        x = self.frequency_channel(x)
        x = self.spatial(x) * x
        return x


class ChannelLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChannelLinear, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(-1, c)
        x = self.linear(x)
        x = x.view(b, h, w, -1)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class IAF(nn.Module):
    def __init__(self):
        super(IAF, self).__init__()
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=8)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, de1, de2, de3):
        x1 = self.up1(de1)
        x2 = self.up2(de2)
        x3 = self.up3(de3)
        out = torch.cat([x1, x2, x3], 1)
        return out


class Encoder(nn.Module):
    def __init__(self, in_ch, ngf, num_blocks=[2, 2, 2, 2, 2], heads=[2, 2, 4, 4, 8], N_blocks=[4, 4, 8, 8, 8],
                 norm_layer=nn.InstanceNorm2d):
        super(Encoder, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.conv1 = nn.Conv2d(in_ch, ngf, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.glce1 = GLCE(dim=ngf, n_layers=num_blocks[0], num_heads=heads[0], N=N_blocks[0])
        self.down1 = nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1, bias=use_bias)

        self.glce2 = GLCE(dim=ngf * 2, n_layers=num_blocks[1], num_heads=heads[1], N=N_blocks[1])
        self.down2 = nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1, bias=use_bias)

        self.glce3 = GLCE(dim=ngf * 4, n_layers=num_blocks[2], num_heads=heads[2], N=N_blocks[2])
        self.down3 = nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias)

        self.glce4 = GLCE(dim=ngf * 8, n_layers=num_blocks[3], num_heads=heads[3], N=N_blocks[3])
        self.down4 = nn.Conv2d(ngf * 8, ngf * 16, kernel_size=4, stride=2, padding=1, bias=use_bias)

        self.glce5 = GLCE(dim=ngf * 16, n_layers=num_blocks[4], num_heads=heads[4], N=N_blocks[4])

    def forward(self, x):
        x1 = self.glce1(self.conv1(x))
        x2 = self.glce2(self.down1(x1))
        x3 = self.glce3(self.down2(x2))
        x4 = self.glce4(self.down3(x3))
        out = self.glce5(self.down4(x4))

        return out, x4, x3, x2, x1


class Decoder(nn.Module):
    def __init__(self, out_ch, ngf, norm_layer=nn.InstanceNorm2d):
        super(Decoder, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1 = nn.Conv2d(ngf * 16, ngf * 8, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.cscc1 = CSCC(ngf * 16)
        self.cl1 = ChannelLinear(ngf * 16, ngf * 8)

        self.conv2 = nn.Conv2d(ngf * 8, ngf * 4, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.cscc2 = CSCC(ngf * 8)
        self.cl2 = ChannelLinear(ngf * 8, ngf * 4)

        self.conv3 = nn.Conv2d(ngf * 4, ngf * 2, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.cscc3 = CSCC(ngf * 4)
        self.cl3 = ChannelLinear(ngf * 4, ngf * 2)

        self.conv4 = nn.Conv2d(ngf * 2, ngf, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.cscc4 = CSCC(ngf * 2)
        self.cl4 = ChannelLinear(ngf * 2, ngf)

        self.fuse = IAF()
        self.conv5 = nn.Conv2d(ngf * (8 + 4 + 2 + 1), ngf, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.conv6 = nn.Conv2d(ngf, out_ch, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.tanh = nn.Tanh()

    def forward(self, x, en1, en2, en3, en4):
        x1 = torch.cat([self.conv1(self.up(x)), en1], 1)
        x2 = self.cl1(self.cscc1(x1))

        x3 = torch.cat([self.conv2(self.up(x2)), en2], 1)
        x4 = self.cl2(self.cscc2(x3))

        x5 = torch.cat([self.conv3(self.up(x4)), en3], 1)
        x6 = self.cl3(self.cscc3(x5))

        x7 = torch.cat([self.conv4(self.up(x6)), en4], 1)
        x8 = self.cl4(self.cscc4(x7))

        x_fuse = self.fuse(x2, x4, x6)
        x9 = self.conv5(torch.cat([x8, x_fuse], 1))
        out = self.tanh(self.conv6(x9))

        return out


class PCGL_GAN(nn.Module):
    def __init__(self, in_ch, out_ch, ngf, opt=None):
        super(PCGL_GAN, self).__init__()
        self.enc = Encoder(in_ch=in_ch, ngf=ngf)
        self.dec = Decoder(out_ch=out_ch, ngf=ngf)

    def forward(self, x, layers=[], encode_only=False):
        en5, en4, en3, en2, en1 = self.enc(x)
        out = self.dec(en5, en4, en3, en2, en1)

        if -1 in layers:
            layers.append(len(out))
        if len(layers) > 0:
            feat = x
            feats = []
            for layer_id, layer in enumerate(
                    [self.enc.glce1, self.enc.down1, self.enc.glce2, self.enc.down2, self.enc.glce3, self.enc.down3,
                     self.enc.glce4, self.enc.down4, self.enc.glce5]):
                feat = layer(feat)
                if layer_id in layers:
                    feats.append(feat)
                else:
                    pass
                if layer_id == layers[-1] and encode_only:
                    return None, feats
            return feat, feats
        else:
            """Standard forward"""
            return out, None


from torchsummary import summary

if __name__ == "__main__":
    input = torch.Tensor(1, 3, 256, 256).cuda()
    model = PCGL_GAN(3, 3, 32).cuda()
    model.eval()
    print(model)
    output = model(input)
    summary(model, (3, 256, 256))
    print(output.shape)
