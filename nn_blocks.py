import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any, Dict, Tuple, Optional
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
def make_divisible(value: float, divisor: int = 8, min_value: Optional[float] = None, round_down_protect: bool = True) -> int:
    """Ensures all layers have channels that are divisible by the given divisor."""
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

class OptimizedMultiQueryAttentionLayerWithDownSampling(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.query_h_strides = query_h_strides
        self.query_w_strides = query_w_strides
        self.kv_strides = kv_strides
        self.dw_kernel_size = dw_kernel_size
        self.dropout = dropout

        norm_layer = nn.SyncBatchNorm if use_sync_bn else nn.BatchNorm2d

        # Query layers
        self.query_layers = nn.Sequential(
            nn.AvgPool2d(kernel_size=(query_h_strides, query_w_strides), padding=0) if query_h_strides > 1 or query_w_strides > 1 else nn.Identity(),
            norm_layer(in_channels, momentum=norm_momentum, eps=norm_epsilon),
            nn.Conv2d(in_channels, num_heads * key_dim, kernel_size=1, stride=1, bias=False)
        )

        # Key layers
        self.key_layers = nn.Sequential(
            nn.Conv2d(
                num_heads * key_dim,  # Correcting the input channels
                num_heads * key_dim,
                kernel_size=dw_kernel_size,
                stride=kv_strides,
                padding=dw_kernel_size // 2,
                groups=num_heads * key_dim,  # Grouped convolution
                bias=False
            ) if kv_strides > 1 else nn.Identity(),
            norm_layer(num_heads * key_dim, momentum=norm_momentum, eps=norm_epsilon),
            nn.Conv2d(num_heads * key_dim, num_heads * key_dim, kernel_size=1, stride=1, bias=False)
        )

        # Value layers
        self.value_layers = nn.Sequential(
            nn.Conv2d(
                num_heads * key_dim,  # Correcting the input channels
                num_heads * key_dim,
                kernel_size=dw_kernel_size,
                stride=kv_strides,
                padding=dw_kernel_size // 2,
                groups=num_heads * key_dim,  # Grouped convolution
                bias=False
            ) if kv_strides > 1 else nn.Identity(),
            norm_layer(num_heads * key_dim, momentum=norm_momentum, eps=norm_epsilon),
            nn.Conv2d(num_heads * key_dim, num_heads * value_dim, kernel_size=1, stride=1, bias=False)
        )

        # Output layers
        self.output_layers = nn.Sequential(
            nn.Upsample(
                scale_factor=(query_h_strides, query_w_strides), mode='bilinear', align_corners=False
            ) if query_h_strides > 1 or query_w_strides > 1 else nn.Identity(),
            nn.Conv2d(num_heads * value_dim, out_channels, kernel_size=1, stride=1, bias=False)
        )

        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Process query
        q = self.query_layers(x)
        q = q.reshape(B, self.num_heads, self.key_dim, -1).permute(0, 1, 3, 2)

        # Process key and value
        k = self.key_layers(q.permute(0, 1, 3, 2).reshape(B, self.num_heads * self.key_dim, H, W))
        k = k.reshape(B, self.num_heads, self.key_dim, -1).permute(0, 1, 3, 2)

        v = self.value_layers(q.permute(0, 1, 3, 2).reshape(B, self.num_heads * self.key_dim, H, W))
        v = v.reshape(B, self.num_heads, self.value_dim, -1).permute(0, 1, 3, 2)

        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.key_dim)
        attn = self.dropout_layer(F.softmax(attn, dim=-1))

        # Compute output
        o = torch.matmul(attn, v)
        o = o.permute(0, 2, 1, 3).contiguous().reshape(B, -1, H // self.query_h_strides, W // self.query_w_strides)

        o = self.output_layers(o)
        return o

class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_heads: Optional[int] = 8,
        key_dim: int = 64,
        value_dim: int = 64,
        use_multi_query: bool = True,
        query_h_strides: int = 1,
        query_w_strides: int = 1,
        kv_strides: int = 1,
        downsampling_dw_kernel_size: int = 3,
        dropout: float = 0.0,
        use_bias: bool = False,
        use_cpe: bool = False,
        cpe_dw_kernel_size: int = 7,
        stochastic_depth_drop_rate: Optional[float] = None,
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
        self.num_heads = num_heads if num_heads is not None else input_dim // key_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.use_multi_query = use_multi_query
        self.query_h_strides = query_h_strides
        self.query_w_strides = query_w_strides
        self.kv_strides = kv_strides
        self.use_residual = use_residual
        self.use_layer_scale = use_layer_scale
        self.output_intermediate_endpoints = output_intermediate_endpoints
        self.use_cpe = use_cpe

        self.norm = nn.SyncBatchNorm(input_dim, momentum=norm_momentum, eps=norm_epsilon) if use_sync_bn else nn.BatchNorm2d(input_dim, momentum=norm_momentum, eps=norm_epsilon)

        if self.use_cpe:
            self.cpe_dw_conv = nn.Conv2d(input_dim, input_dim, kernel_size=cpe_dw_kernel_size, stride=1, padding=cpe_dw_kernel_size // 2, groups=input_dim, bias=True)

        if use_multi_query:
            self.attention = OptimizedMultiQueryAttentionLayerWithDownSampling(
                in_channels=input_dim,
                out_channels=output_dim,
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
            self.attention = nn.MultiheadAttention(
                embed_dim=input_dim,
                num_heads=self.num_heads,
                dropout=dropout,
                bias=use_bias,
            )

        if use_layer_scale:
            self.layer_scale = MNV4LayerScale(layer_scale_init_value)

        if stochastic_depth_drop_rate:
            self.stochastic_depth = StochasticDepth(stochastic_depth_drop_rate)
        else:
            self.stochastic_depth = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.use_cpe:
            x = self.cpe_dw_conv(inputs)
            x = x + inputs
            cpe_outputs = x
        else:
            cpe_outputs = inputs

        shortcut = cpe_outputs
        x = self.norm(cpe_outputs)

        if self.use_multi_query:
            x = self.attention(x)
        else:
            B, C, H, W = x.shape
            x = x.reshape(B, C, H * W).permute(2, 0, 1)
            x, _ = self.attention(x, x, x)
            x = x.permute(1, 2, 0).reshape(B, C, H, W)

        if self.use_layer_scale:
            x = self.layer_scale(x)

        if self.use_residual:
            if self.stochastic_depth:
                x = self.stochastic_depth(x)
            x = x + shortcut

        if self.output_intermediate_endpoints:
            return x, {}
        return x
    
def get_stochastic_depth_rate(init_rate: Optional[float], i: int, n: int) -> Optional[float]:
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
    def __init__(self, stochastic_depth_drop_rate: float):
        super().__init__()
        self._drop_rate = stochastic_depth_drop_rate

    def forward(self, inputs: torch.Tensor, training: bool = False) -> torch.Tensor:
        if not training or self._drop_rate is None or self._drop_rate == 0:
            return inputs
        keep_prob = 1.0 - self._drop_rate
        batch_size = inputs.shape[0]
        random_tensor = keep_prob + torch.rand([batch_size] + [1] * (inputs.dim() - 1), device=inputs.device, dtype=inputs.dtype)
        binary_tensor = torch.floor(random_tensor)
        output = torch.div(inputs, keep_prob) * binary_tensor
        return output

class MNV4LayerScale(nn.Module):
    def __init__(self, init_value: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1) * init_value)

    def forward(self, x):
        return x * self.gamma

class UniversalInvertedBottleneckBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float,
        strides: int = 1,
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
        use_layer_scale: bool = False,
        layer_scale_init_value: float = 1e-5,
        norm_momentum: float = 0.99,
        norm_epsilon: float = 0.001,
        **kwargs
    ):
        super().__init__()
        
        # Accept both in_channels and in_filters for compatibility
        in_filters = kwargs.get('in_filters', in_channels)
        out_filters = kwargs.get('out_filters', out_channels)
        
        # Accept both strides and stride for compatibility
        self.strides = kwargs.get('stride', strides)
        
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.expand_ratio = expand_ratio
        self.middle_dw_downsample = middle_dw_downsample
        self.start_dw_kernel_size = start_dw_kernel_size
        self.middle_dw_kernel_size = middle_dw_kernel_size
        self.end_dw_kernel_size = end_dw_kernel_size
        self.use_residual = use_residual
        self.use_layer_scale = use_layer_scale

        if self.strides > 1:
            if middle_dw_downsample and not middle_dw_kernel_size:
                raise ValueError('Requested downsampling at a non-existing middle depthwise.')
            if not middle_dw_downsample and not start_dw_kernel_size:
                raise ValueError('Requested downsampling at a non-existing starting depthwise.')

        self.activation = getattr(torch.nn.functional, activation)
        self.depthwise_activation = getattr(torch.nn.functional, depthwise_activation) if depthwise_activation else self.activation

        expand_filters = make_divisible(in_filters * expand_ratio, divisible_by)

        layers = []

        # Starting depthwise conv
        if start_dw_kernel_size:
            layers.extend([
                nn.Conv2d(in_filters, in_filters, kernel_size=start_dw_kernel_size, 
                          stride=self.strides if not middle_dw_downsample else 1, 
                          padding=start_dw_kernel_size//2, groups=in_filters, bias=False),
                nn.BatchNorm2d(in_filters, momentum=norm_momentum, eps=norm_epsilon),
            ])

        # Expansion conv
        layers.extend([
            nn.Conv2d(in_filters, expand_filters, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(expand_filters, momentum=norm_momentum, eps=norm_epsilon),
            nn.ReLU6(inplace=True)
        ])

        # Middle depthwise conv
        if middle_dw_kernel_size:
            layers.extend([
                nn.Conv2d(expand_filters, expand_filters, kernel_size=middle_dw_kernel_size, 
                          stride=self.strides if middle_dw_downsample else 1, 
                          padding=middle_dw_kernel_size//2, groups=expand_filters, bias=False),
                nn.BatchNorm2d(expand_filters, momentum=norm_momentum, eps=norm_epsilon),
                nn.ReLU6(inplace=True)
            ])

        # Projection conv
        layers.extend([
            nn.Conv2d(expand_filters, out_filters, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_filters, momentum=norm_momentum, eps=norm_epsilon)
        ])

        # Ending depthwise conv
        if end_dw_kernel_size:
            layers.extend([
                nn.Conv2d(out_filters, out_filters, kernel_size=end_dw_kernel_size, 
                          stride=1, padding=end_dw_kernel_size//2, groups=out_filters, bias=False),
                nn.BatchNorm2d(out_filters, momentum=norm_momentum, eps=norm_epsilon)
            ])

        self.layers = nn.Sequential(*layers)

        if use_layer_scale:
            self.layer_scale = MNV4LayerScale(layer_scale_init_value)
        else:
            self.layer_scale = None

        if stochastic_depth_drop_rate:
            self.stochastic_depth = StochasticDepth(stochastic_depth_drop_rate)
        else:
            self.stochastic_depth = None

    def forward(self, inputs, training=False):
        shortcut = inputs
        x = self.layers(inputs)

        if self.layer_scale:
            x = self.layer_scale(x)

        if self.use_residual and self.in_filters == self.out_filters and self.strides == 1:
            if self.stochastic_depth:
                x = self.stochastic_depth(x, training=training)
            x += shortcut

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
