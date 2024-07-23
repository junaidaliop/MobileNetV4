import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional, Tuple, Dict
import math

# Complete PyTorch implementation of MobileNetV4 essentials

class SwishAutoFn(torch.autograd.Function):
    """Memory Efficient Swish"""
    @staticmethod
    def forward(ctx, x):
        result = x.mul(torch.sigmoid(x))
        ctx.save_for_backward(x)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        sigmoid_x = torch.sigmoid(x)
        return grad_output * (sigmoid_x * (1 + x * (1 - sigmoid_x)))

def swish(x, inplace=False):
    return SwishAutoFn.apply(x)

def hard_swish(x, inplace=False):
    if inplace:
        return x.mul_(F.relu6(x + 3.) / 6.)
    else:
        return x * F.relu6(x + 3.) / 6.

def hard_sigmoid(x, inplace=False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.

def get_activation(activation: str):
    """Returns the activation function."""
    if activation == 'relu':
        return nn.ReLU(inplace=True)
    elif activation == 'relu6':
        return nn.ReLU6(inplace=True)
    elif activation == 'gelu':
        return nn.GELU()
    elif activation == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation == 'swish':
        return Swish()
    elif activation == 'hardswish_custom':
        return HardSwish()
    elif activation == 'swish_custom':
        return Swish()
    else:
        raise ValueError(f'Unknown activation function: {activation}')

class Swish(nn.Module):
    def forward(self, x):
        return swish(x)

class HardSwish(nn.Module):
    def forward(self, x):
        return hard_swish(x)

class HardSigmoid(nn.Module):
    def forward(self, x):
        return hard_sigmoid(x)

class Conv2DBNBlock(nn.Module):
    """A convolution block with batch normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        use_bias: bool = False,
        use_explicit_padding: bool = False,
        activation: str = 'relu6',
        norm_momentum: float = 0.1,
        norm_epsilon: float = 1e-5,
        use_normalization: bool = True,
    ):
        super(Conv2DBNBlock, self).__init__()
        self.use_normalization = use_normalization
        self.use_explicit_padding = use_explicit_padding

        if use_explicit_padding and kernel_size > 1:
            padding = (kernel_size - 1) // 2
        else:
            padding = 0

        self.pad = nn.ZeroPad2d(padding) if use_explicit_padding else None
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding=(0 if use_explicit_padding else padding),
            bias=use_bias
        )

        self.bn = nn.BatchNorm2d(out_channels, momentum=norm_momentum, eps=norm_epsilon) if use_normalization else None
        self.activation_layer = get_activation(activation)

    def forward(self, x):
        if self.use_explicit_padding and self.pad:
            x = self.pad(x)
        x = self.conv(x)
        if self.use_normalization:
            x = self.bn(x)
        x = self.activation_layer(x)
        return x

# Utility functions
def make_divisible(value: float, divisor: int, min_value: Optional[float] = None, round_down_protect: bool = True) -> int:
    """Ensure that all layers have channels that are divisible by the given divisor."""
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if round_down_protect and new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)

def round_filters(filters: int, multiplier: float, divisor: int = 8, min_depth: Optional[int] = None, round_down_protect: bool = True, skip: bool = False) -> int:
    """Rounds number of filters based on width multiplier."""
    orig_f = filters
    if skip or not multiplier:
        return filters

    new_filters = make_divisible(value=filters * multiplier,
                                 divisor=divisor,
                                 min_value=min_depth,
                                 round_down_protect=round_down_protect)
    return int(new_filters)

def get_padding_for_kernel_size(kernel_size):
    """Compute padding size given kernel size."""
    if kernel_size == 7:
        return (3, 3)
    elif kernel_size == 3:
        return (1, 1)
    else:
        raise ValueError(f'Padding for kernel size {kernel_size} not known.')

def get_stochastic_depth_rate(init_rate, i, n):
    """Get drop connect rate for the ith block."""
    if init_rate is not None:
        if init_rate < 0 or init_rate > 1:
            raise ValueError('Initial drop rate must be within 0 and 1.')
        rate = init_rate * float(i) / n
    else:
        rate = None
    return rate

class StochasticDepth(nn.Module):
    """Creates a stochastic depth layer."""
    def __init__(self, stochastic_depth_drop_rate):
        super(StochasticDepth, self).__init__()
        self._drop_rate = stochastic_depth_drop_rate

    def forward(self, x):
        if not self.training or self._drop_rate is None or self._drop_rate == 0:
            return x

        keep_prob = 1.0 - self._drop_rate
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

class OptimizedMultiQueryAttentionLayerWithDownSampling(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_heads: int,
        key_dim: int,
        value_dim: int,
        query_h_strides: int = 1,
        query_w_strides: int = 1,
        kv_strides: int = 1,
        dropout: float = 0,
        dw_kernel_size: int = 3,
        use_sync_bn: bool = False,
        norm_momentum: float = 0.99,
        norm_epsilon: float = 0.001,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.query_h_strides = query_h_strides
        self.query_w_strides = query_w_strides
        self.kv_strides = kv_strides
        self.dw_kernel_size = dw_kernel_size
        self.dropout = dropout

        self.norm = nn.SyncBatchNorm if use_sync_bn else nn.BatchNorm2d
        self.bn_axis = 1  # PyTorch uses NCHW format by default

        if query_h_strides > 1 or query_w_strides > 1:
            self.query_downsampling = nn.AvgPool2d(
                kernel_size=(query_h_strides, query_w_strides),
                stride=(query_h_strides, query_w_strides),
                padding=(query_h_strides // 2, query_w_strides // 2),
            )
            self.query_downsampling_norm = self.norm(input_dim, momentum=norm_momentum, eps=norm_epsilon)

        self.query_proj = nn.Conv2d(input_dim, num_heads * key_dim, kernel_size=1, bias=False)

        if kv_strides > 1:
            self.key_dw_conv = nn.Conv2d(input_dim, input_dim, kernel_size=dw_kernel_size, 
                                         stride=kv_strides, padding=dw_kernel_size//2, groups=input_dim, bias=False)
            self.key_dw_norm = self.norm(input_dim, momentum=norm_momentum, eps=norm_epsilon)
        self.key_proj = nn.Conv2d(input_dim, key_dim, kernel_size=1, bias=False)

        if kv_strides > 1:
            self.value_dw_conv = nn.Conv2d(input_dim, input_dim, kernel_size=dw_kernel_size, 
                                           stride=kv_strides, padding=dw_kernel_size//2, groups=input_dim, bias=False)
            self.value_dw_norm = self.norm(input_dim, momentum=norm_momentum, eps=norm_epsilon)
        self.value_proj = nn.Conv2d(input_dim, value_dim, kernel_size=1, bias=False)

        self.output_proj = nn.Conv2d(num_heads * value_dim, output_dim, kernel_size=1, bias=False)

        if query_h_strides > 1 or query_w_strides > 1:
            self.upsampling = nn.Upsample(scale_factor=(query_h_strides, query_w_strides), mode='bilinear', align_corners=False)

        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        b, c, h, w = x.shape
        assert c == self.input_dim, f"Input channel dimension {c} doesn't match expected input_dim {self.input_dim}"

        if self.query_h_strides > 1 or self.query_w_strides > 1:
            q = self.query_downsampling(x)
            q = self.query_downsampling_norm(q)
            q = self.query_proj(q)
        else:
            q = self.query_proj(x)

        q = self._reshape_projected_query(q, self.num_heads, h // self.query_h_strides, w // self.query_w_strides, self.key_dim)

        if self.kv_strides > 1:
            k = self.key_dw_conv(x)
            k = self.key_dw_norm(k)
            k = self.key_proj(k)
        else:
            k = self.key_proj(x)

        k = self._reshape_input(k)

        logits = torch.einsum('blhk,bpk->blhp', q, k)
        logits = logits / (self.key_dim ** 0.5)

        attention_scores = self.dropout_layer(F.softmax(logits, dim=-1))

        if self.kv_strides > 1:
            v = self.value_dw_conv(x)
            v = self.value_dw_norm(v)
            v = self.value_proj(v)
        else:
            v = self.value_proj(x)

        v = self._reshape_input(v)
        o = torch.einsum('blhp,bpk->blhk', attention_scores, v)

        o = self._reshape_output(o, self.num_heads, h // self.query_h_strides, w // self.query_w_strides)

        if self.query_h_strides > 1 or self.query_w_strides > 1:
            o = self.upsampling(o)

        result = self.output_proj(o)
        assert result.shape[1] == self.output_dim, f"Output channel dimension {result.shape[1]} doesn't match expected output_dim {self.output_dim}"
        return result

    def _reshape_input(self, t):
        b, c, h, w = t.shape
        return t.view(b, c, -1).transpose(1, 2)

    def _reshape_projected_query(self, t, num_heads, h_px, w_px, key_dim):
        b, c, h, w = t.shape
        t = t.view(b, num_heads, h_px, w_px, key_dim)
        return t.permute(0, 2, 3, 1, 4).contiguous().view(b, -1, num_heads, key_dim)

    def _reshape_output(self, t, num_heads, h_px, w_px):
        b, l, h, k = t.shape
        return t.view(b, h_px, w_px, h * k).permute(0, 3, 1, 2).contiguous()
    
class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_heads: int = 8,
        key_dim: int = 64,
        value_dim: int = 64,
        use_multi_query: bool = False,
        query_h_strides: int = 1,
        query_w_strides: int = 1,
        kv_strides: int = 1,
        downsampling_dw_kernel_size: int = 3,
        dropout: float = 0.0,
        use_bias: bool = False,
        use_cpe: bool = False,
        cpe_dw_kernel_size: int = 7,
        stochastic_depth_drop_rate: float = None,
        use_residual: bool = True,
        use_sync_bn: bool = False,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
        norm_momentum: float = 0.99,
        norm_epsilon: float = 0.001,
        output_intermediate_endpoints: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads or input_dim // key_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.use_multi_query = use_multi_query
        self.query_h_strides = query_h_strides
        self.query_w_strides = query_w_strides
        self.kv_strides = kv_strides
        self.downsampling_dw_kernel_size = downsampling_dw_kernel_size
        self.dropout = dropout
        self.use_bias = use_bias
        self.use_cpe = use_cpe
        self.cpe_dw_kernel_size = cpe_dw_kernel_size
        self.stochastic_depth_drop_rate = stochastic_depth_drop_rate
        self.use_residual = use_residual
        self.use_sync_bn = use_sync_bn
        self.use_layer_scale = use_layer_scale
        self.layer_scale_init_value = layer_scale_init_value
        self.norm_momentum = norm_momentum
        self.norm_epsilon = norm_epsilon
        self.output_intermediate_endpoints = output_intermediate_endpoints

        self.norm = nn.SyncBatchNorm if use_sync_bn else nn.BatchNorm2d
        self.input_norm = self.norm(input_dim, momentum=norm_momentum, eps=norm_epsilon)

        if use_cpe:
            self.cpe_dw_conv = nn.Conv2d(input_dim, input_dim, kernel_size=cpe_dw_kernel_size, 
                                         padding=cpe_dw_kernel_size//2, groups=input_dim, bias=True)

        if use_multi_query:
            if query_h_strides > 1 or query_w_strides > 1 or kv_strides > 1:
                self.multi_query_attention = OptimizedMultiQueryAttentionLayerWithDownSampling(
                    input_dim=input_dim, 
                    output_dim=output_dim,
                    num_heads=self.num_heads,
                    key_dim=key_dim,
                    value_dim=value_dim,
                    query_h_strides=query_h_strides,
                    query_w_strides=query_w_strides,
                    kv_strides=kv_strides,
                    dw_kernel_size=downsampling_dw_kernel_size,
                    dropout=dropout,
                    use_sync_bn=use_sync_bn,
                    norm_momentum=norm_momentum,
                    norm_epsilon=norm_epsilon,
                )
            else:
                # Implement MultiQueryAttentionLayerV2
                pass
        else:
            self.multi_head_attention = nn.MultiheadAttention(
                embed_dim=input_dim,
                num_heads=self.num_heads,
                dropout=dropout,
                bias=use_bias,
                batch_first=True,
            )
            if input_dim != output_dim:
                self.output_proj = nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False)
            else:
                self.output_proj = nn.Identity()

        if use_layer_scale:
            self.layer_scale = MNV4LayerScale(layer_scale_init_value)

        if stochastic_depth_drop_rate:
            self.stochastic_depth = StochasticDepth(stochastic_depth_drop_rate)
        else:
            self.stochastic_depth = None

    def forward(self, inputs):
        x = inputs
        if self.use_cpe:
            x = self.cpe_dw_conv(x) + inputs

        shortcut = x
        x = self.input_norm(x)

        if self.use_multi_query:
            if self.query_h_strides > 1 or self.query_w_strides > 1 or self.kv_strides > 1:
                x = self.multi_query_attention(x)
            else:
                # Implement MultiQueryAttentionLayerV2 if needed
                pass
        else:
            b, c, h, w = x.shape
            x = x.view(b, c, -1).transpose(1, 2)
            x, _ = self.multi_head_attention(x, x, x)
            x = x.transpose(1, 2).view(b, c, h, w)
            x = self.output_proj(x)

        if self.use_layer_scale:
            x = self.layer_scale(x)

        if self.use_residual:
            if self.stochastic_depth:
                x = self.stochastic_depth(x)
            if self.input_dim == self.output_dim:
                x = x + shortcut
            else:
                residual_proj = nn.Conv2d(self.input_dim, self.output_dim, kernel_size=1, bias=False).to(x.device)
                x = x + residual_proj(shortcut)

        if self.output_intermediate_endpoints:
            return x, {}
        return x

class MNV4LayerScale(nn.Module):
    """LayerScale as introduced in CaiT: https://arxiv.org/abs/2103.17239."""

    def __init__(self, init_value: float):
        super(MNV4LayerScale, self).__init__()
        self.init_value = init_value
        self.gamma = None

    def build(self, input_shape):
        embedding_dim = input_shape[-1]
        self.gamma = nn.Parameter(self.init_value * torch.ones(embedding_dim))

    def forward(self, x):
        if self.gamma is None:
            self.build(x.shape)
        return x * self.gamma

class UniversalInvertedBottleneckBlock(nn.Module):
    """An inverted bottleneck block with optional depthwises."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float,
        stride: int,
        middle_dw_downsample: bool = True,
        start_dw_kernel_size: int = 0,
        middle_dw_kernel_size: int = 3,
        end_dw_kernel_size: int = 0,
        stochastic_depth_drop_rate: Optional[float] = None,
        activation: str = 'relu',
        depthwise_activation: Optional[str] = None,
        dilation_rate: int = 1,
        divisible_by: int = 1,
        use_residual: bool = True,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
        norm_momentum: float = 0.99,
        norm_epsilon: float = 0.001,
    ):
        super(UniversalInvertedBottleneckBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expand_ratio = expand_ratio
        self.stride = stride
        self.middle_dw_downsample = middle_dw_downsample
        self.start_dw_kernel_size = start_dw_kernel_size
        self.middle_dw_kernel_size = middle_dw_kernel_size
        self.end_dw_kernel_size = end_dw_kernel_size
        self.stochastic_depth_drop_rate = stochastic_depth_drop_rate
        self.dilation_rate = dilation_rate
        self.use_residual = use_residual
        self.use_layer_scale = use_layer_scale
        self.layer_scale_init_value = layer_scale_init_value

        if stride > 1:
            if middle_dw_downsample and not middle_dw_kernel_size:
                raise ValueError(
                    'Requested downsampling at a non-existing middle depthwise.'
                )
            if not middle_dw_downsample and not start_dw_kernel_size:
                raise ValueError(
                    'Requested downsampling at a non-existing starting depthwise.'
                )

        expand_channels = make_divisible(in_channels * expand_ratio, divisible_by)

        layers = []

        # Starting depthwise conv
        if start_dw_kernel_size > 0:
            layers.extend([
                nn.Conv2d(in_channels, in_channels, kernel_size=start_dw_kernel_size,
                          stride=(stride if not middle_dw_downsample else 1), padding=start_dw_kernel_size // 2, groups=in_channels, bias=False, dilation=dilation_rate),
                nn.BatchNorm2d(in_channels, momentum=norm_momentum, eps=norm_epsilon),
            ])

        # Expansion conv
        layers.extend([
            nn.Conv2d(in_channels, expand_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(expand_channels, momentum=norm_momentum, eps=norm_epsilon),
            get_activation(activation)
        ])

        # Middle depthwise conv
        if middle_dw_kernel_size > 0:
            layers.extend([
                nn.Conv2d(expand_channels, expand_channels, kernel_size=middle_dw_kernel_size,
                          stride=(stride if middle_dw_downsample else 1), padding=middle_dw_kernel_size // 2, groups=expand_channels, bias=False, dilation=dilation_rate),
                nn.BatchNorm2d(expand_channels, momentum=norm_momentum, eps=norm_epsilon),
                get_activation(depthwise_activation or activation)
            ])

        # Projection conv
        layers.extend([
            nn.Conv2d(expand_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels, momentum=norm_momentum, eps=norm_epsilon)
        ])

        # Ending depthwise conv
        if end_dw_kernel_size > 0:
            layers.extend([
                nn.Conv2d(out_channels, out_channels, kernel_size=end_dw_kernel_size,
                          stride=1, padding=end_dw_kernel_size // 2, groups=out_channels, bias=False, dilation=dilation_rate),
                nn.BatchNorm2d(out_channels, momentum=norm_momentum, eps=norm_epsilon)
            ])

        self.layers = nn.Sequential(*layers)

        if use_layer_scale:
            self.layer_scale = MNV4LayerScale(layer_scale_init_value)
        else:
            self.layer_scale = None

        if stochastic_depth_drop_rate:
            self.stochastic_depth = StochasticDepth(drop_rate=stochastic_depth_drop_rate)
        else:
            self.stochastic_depth = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.layers(x)

        if self.layer_scale is not None:
            x = self.layer_scale(x)

        if self.use_residual and self.in_channels == self.out_channels and self.stride == 1:
            if self.stochastic_depth is not None:
                x = self.stochastic_depth(x, self.training)
            x = x + shortcut

        return x

class DepthwiseSeparableConvBlock(nn.Module):
    """A depthwise separable convolution block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        activation: str = 'relu',
        dilation_rate: int = 1,
        regularize_depthwise: bool = False,
        norm_momentum: float = 0.1,
        norm_epsilon: float = 1e-5,
    ):
        super(DepthwiseSeparableConvBlock, self).__init__()

        self.depthwise_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding=kernel_size // 2, groups=in_channels,
            bias=False, dilation=dilation_rate
        )
        self.depthwise_bn = nn.BatchNorm2d(in_channels, momentum=norm_momentum, eps=norm_epsilon)
        self.depthwise_act = get_activation(activation)

        self.pointwise_conv = nn.Conv2d(
            in_channels, out_channels, 1, 1, padding=0, bias=False
        )
        self.pointwise_bn = nn.BatchNorm2d(out_channels, momentum=norm_momentum, eps=norm_epsilon)
        self.pointwise_act = get_activation(activation)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x)
        x = self.depthwise_act(x)
        x = self.pointwise_conv(x)
        x = self.pointwise_bn(x)
        x = self.pointwise_act(x)
        return x

class GlobalPoolingBlock(nn.Module):
    """A global average pooling block."""

    def __init__(self, keepdims: bool = True):
        super(GlobalPoolingBlock, self).__init__()
        self.keepdims = keepdims
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.pool(x)
        if not self.keepdims:
            x = x.view(x.size(0), x.size(1))
        return x
    

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, out_channels, se_ratio, divisible_by=1, 
                 activation='relu', gating_activation='sigmoid', round_down_protect=True):
        super(SqueezeExcitation, self).__init__()
        num_reduced_filters = make_divisible(
            max(1, int(in_channels * se_ratio)), 
            divisor=divisible_by, 
            round_down_protect=round_down_protect
        )
        self.se_reduce = nn.Conv2d(in_channels, num_reduced_filters, kernel_size=1, stride=1, padding=0)
        self.se_expand = nn.Conv2d(num_reduced_filters, out_channels, kernel_size=1, stride=1, padding=0)
        self.activation_fn = get_activation(activation)
        self.gating_activation_fn = get_activation(gating_activation)

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = self.activation_fn(self.se_reduce(x))
        x = self.gating_activation_fn(self.se_expand(x))
        return x * x


class FusedInvertedBottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, stride, kernel_size=3, 
                 se_ratio=None, stochastic_depth_drop_rate=None, activation='relu', 
                 se_inner_activation='relu', se_gating_activation='sigmoid', 
                 se_round_down_protect=True, divisible_by=1, use_residual=True, 
                 norm_momentum=0.1, norm_epsilon=1e-5):
        super(FusedInvertedBottleneckBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expand_ratio = expand_ratio
        self.stride = stride
        self.kernel_size = kernel_size
        self.se_ratio = se_ratio
        self.divisible_by = divisible_by
        self.stochastic_depth_drop_rate = stochastic_depth_drop_rate
        self.use_residual = use_residual
        self.activation = activation
        self.se_inner_activation = se_inner_activation
        self.se_gating_activation = se_gating_activation
        self.se_round_down_protect = se_round_down_protect
        self.norm_momentum = norm_momentum
        self.norm_epsilon = norm_epsilon

        self.expand_channels = make_divisible(in_channels * expand_ratio, divisible_by)

        # Fused conv (combined expansion and depthwise conv)
        self.fused_conv = nn.Conv2d(in_channels, self.expand_channels, kernel_size=kernel_size, 
                                    stride=stride, padding=kernel_size // 2, bias=False)
        self.fused_bn = nn.BatchNorm2d(self.expand_channels, momentum=norm_momentum, eps=norm_epsilon)
        self.fused_act = get_activation(activation)

        # Squeeze and excitation
        if se_ratio:
            self.squeeze_excitation = SqueezeExcitation(self.expand_channels, self.expand_channels, se_ratio, 
                                                        divisible_by, activation=se_inner_activation, 
                                                        gating_activation=se_gating_activation, 
                                                        round_down_protect=se_round_down_protect)
        else:
            self.squeeze_excitation = None

        # Projection conv
        self.project_conv = nn.Conv2d(self.expand_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels, momentum=norm_momentum, eps=norm_epsilon)

        if self.stochastic_depth_drop_rate:
            self.stochastic_depth = StochasticDepth(self.stochastic_depth_drop_rate)
        else:
            self.stochastic_depth = None

    def forward(self, x):
        shortcut = x

        x = self.fused_conv(x)
        x = self.fused_bn(x)
        x = self.fused_act(x)

        if self.squeeze_excitation:
            x = self.squeeze_excitation(x)

        x = self.project_conv(x)
        x = self.project_bn(x)

        if self.use_residual and self.in_channels == self.out_channels and self.stride == 1:
            if self.stochastic_depth:
                x = self.stochastic_depth(x)
            x = x + shortcut

        return x


class InvertedBottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, stride, kernel_size=3, 
                 se_ratio=None, stochastic_depth_drop_rate=None, activation='relu', 
                 se_inner_activation='relu', se_gating_activation='sigmoid', 
                 se_round_down_protect=True, divisible_by=1, use_residual=True, 
                 norm_momentum=0.1, norm_epsilon=1e-5):
        super(InvertedBottleneckBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expand_ratio = expand_ratio
        self.stride = stride
        self.kernel_size = kernel_size
        self.se_ratio = se_ratio
        self.divisible_by = divisible_by
        self.stochastic_depth_drop_rate = stochastic_depth_drop_rate
        self.use_residual = use_residual
        self.activation = activation
        self.se_inner_activation = se_inner_activation
        self.se_gating_activation = se_gating_activation
        self.se_round_down_protect = se_round_down_protect
        self.norm_momentum = norm_momentum
        self.norm_epsilon = norm_epsilon

        self.expand_channels = make_divisible(in_channels * expand_ratio, divisible_by)

        # Expansion conv
        self.expand_conv = nn.Conv2d(in_channels, self.expand_channels, kernel_size=1, 
                                     stride=1, padding=0, bias=False)
        self.expand_bn = nn.BatchNorm2d(self.expand_channels, momentum=norm_momentum, eps=norm_epsilon)
        self.expand_act = get_activation(activation)

        # Depthwise conv
        self.depthwise_conv = nn.Conv2d(self.expand_channels, self.expand_channels, kernel_size=kernel_size, 
                                        stride=stride, padding=kernel_size // 2, groups=self.expand_channels, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(self.expand_channels, momentum=norm_momentum, eps=norm_epsilon)
        self.depthwise_act = get_activation(activation)

        # Squeeze and excitation
        if se_ratio:
            self.squeeze_excitation = SqueezeExcitation(self.expand_channels, self.expand_channels, se_ratio, 
                                                        divisible_by, activation=se_inner_activation, 
                                                        gating_activation=se_gating_activation, 
                                                        round_down_protect=se_round_down_protect)
        else:
            self.squeeze_excitation = None

        # Projection conv
        self.project_conv = nn.Conv2d(self.expand_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels, momentum=norm_momentum, eps=norm_epsilon)

        if self.stochastic_depth_drop_rate:
            self.stochastic_depth = StochasticDepth(self.stochastic_depth_drop_rate)
        else:
            self.stochastic_depth = None

    def forward(self, x):
        shortcut = x

        x = self.expand_conv(x)
        x = self.expand_bn(x)
        x = self.expand_act(x)

        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x)
        x = self.depthwise_act(x)

        if self.squeeze_excitation:
            x = self.squeeze_excitation(x)

        x = self.project_conv(x)
        x = self.project_bn(x)

        if self.use_residual and self.in_channels == self.out_channels and self.stride == 1:
            if self.stochastic_depth:
                x = self.stochastic_depth(x)
            x = x + shortcut

        return x
