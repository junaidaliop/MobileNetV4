import torch
import torch.nn as nn
from typing import Any, List, Tuple, Optional
from dataclasses import dataclass, fields, astuple
from collections import OrderedDict

from nn_blocks import (Conv2DBNBlock, DepthwiseSeparableConvBlock, FusedInvertedBottleneckBlock,
                       UniversalInvertedBottleneckBlock, MultiHeadSelfAttentionBlock, GlobalPoolingBlock,
                       round_filters, make_divisible, StochasticDepth, get_stochastic_depth_rate)

@dataclass
class BlockSpec:
    """A container class that specifies the block configuration for MobileNet."""
    block_fn: str = 'convbn'
    kernel_size: int = 3
    strides: int = 1
    filters: int = 32
    use_bias: bool = False
    use_normalization: bool = True
    activation: str = 'relu6'
    expand_ratio: Optional[float] = 6.0
    se_ratio: Optional[float] = None
    use_depthwise: bool = True
    use_residual: bool = True
    is_output: bool = True

    middle_dw_downsample: bool = True
    start_dw_kernel_size: int = 0
    middle_dw_kernel_size: int = 0
    end_dw_kernel_size: int = 0

    use_layer_scale: bool = True
    use_multi_query: bool = True
    use_downsampling: bool = False
    downsampling_dw_kernel_size: int = 3
    num_heads: int = 8
    key_dim: int = 64
    value_dim: int = 64
    query_h_strides: int = 1
    query_w_strides: int = 1
    kv_strides: int = 1


def block_spec_field_list() -> List[str]:
    """Returns the list of field names used in BlockSpec."""
    return [field.name for field in fields(BlockSpec)]


def block_spec_values_to_list(block_specs: List[BlockSpec]) -> List[Tuple[Any, ...]]:
    """Creates a list field value tuples from a list of BlockSpecs."""
    return [astuple(bs) for bs in block_specs]


def block_spec_decoder(specs: dict[str, Any], filter_size_scale: float, divisible_by: int = 8,
                       finegrain_classification_mode: bool = True) -> List[BlockSpec]:
    """Decodes specs for a block.

    Args:
        specs: A dict specification of block specs of a mobilenet version.
        filter_size_scale: A float multiplier for the filter size for all
          convolution ops. The value must be greater than zero. Typical usage will
          be to set this value in (0, 1) to reduce the number of parameters or
          computation cost of the model.
        divisible_by: An int that ensures all inner dimensions are divisible by
          this number.
        finegrain_classification_mode: If True, the model will keep the last layer
          large even for small multipliers, following
          https://arxiv.org/abs/1801.04381.

    Returns:
        A list of BlockSpec that defines structure of the base network.
    """
    spec_name = specs['spec_name']
    block_spec_schema = specs['block_spec_schema']
    block_specs = specs['block_specs']

    if not block_specs:
        raise ValueError(f'The block spec cannot be empty for {spec_name}!')

    for block_spec in block_specs:
        if len(block_spec) != len(block_spec_schema):
            raise ValueError(f'The block spec values {block_spec} do not match with the schema {block_spec_schema}')

    decoded_specs = []
    for s in block_specs:
        spec_dict = dict(zip(block_spec_schema, s))
        # # Ensure strides is always an integer, default to 1 if not specified
        # if 'strides' in spec_dict and spec_dict['strides'] is None:
        #     spec_dict['strides'] = 1
        decoded_specs.append(BlockSpec(**spec_dict))

    if spec_name != 'MobileNetV1' and finegrain_classification_mode and filter_size_scale < 1.0:
        decoded_specs[-1].filters /= filter_size_scale

    for ds in decoded_specs:
        if ds.filters:
            ds.filters = round_filters(ds.filters, filter_size_scale, divisible_by, min_depth=8)

    return decoded_specs

"""
Architecture: https://arxiv.org/abs/2404.10518

"MobileNetV4 - Universal Models for the Mobile Ecosystem"
Danfeng Qin, Chas Leichner, Manolis Delakis, Marco Fornoni, Shixin Luo, Fan
Yang, Weijun Wang, Colby Banbury, Chengxi Ye, Berkin Akin, Vaibhav Aggarwal,
Tenghui Zhu, Daniele Moro, Andrew Howard
"""
MNV4ConvSmall_BLOCK_SPECS = {
    'spec_name': 'MobileNetV4ConvSmall',
    'block_spec_schema': [
        'block_fn',
        'activation',
        'kernel_size',
        'start_dw_kernel_size',
        'middle_dw_kernel_size',
        'middle_dw_downsample',
        'strides',
        'filters',
        'expand_ratio',
        'is_output',
    ],
    'block_specs': [
        # 112px after stride 2.
        ('convbn', 'relu', 3, None, None, False, 2, 32, None, False),
        # 56px.
        ('convbn', 'relu', 3, None, None, False, 2, 32, None, False),
        ('convbn', 'relu', 1, None, None, False, 1, 32, None, True),
        # 28px.
        ('convbn', 'relu', 3, None, None, False, 2, 96, None, False),
        ('convbn', 'relu', 1, None, None, False, 1, 64, None, True),
        # 14px.
        ('uib', 'relu', None, 5, 5, True, 2, 96, 3.0, False),  # ExtraDW
        ('uib', 'relu', None, 0, 3, True, 1, 96, 2.0, False),  # IB
        ('uib', 'relu', None, 0, 3, True, 1, 96, 2.0, False),  # IB
        ('uib', 'relu', None, 0, 3, True, 1, 96, 2.0, False),  # IB
        ('uib', 'relu', None, 0, 3, True, 1, 96, 2.0, False),  # IB
        ('uib', 'relu', None, 3, 0, True, 1, 96, 4.0, True),  # ConvNext
        # 7px
        ('uib', 'relu', None, 3, 3, True, 2, 128, 6.0, False),  # ExtraDW
        ('uib', 'relu', None, 5, 5, True, 1, 128, 4.0, False),  # ExtraDW
        ('uib', 'relu', None, 0, 5, True, 1, 128, 4.0, False),  # IB
        ('uib', 'relu', None, 0, 5, True, 1, 128, 3.0, False),  # IB
        ('uib', 'relu', None, 0, 3, True, 1, 128, 4.0, False),  # IB
        ('uib', 'relu', None, 0, 3, True, 1, 128, 4.0, True),  # IB
        ('convbn', 'relu', 1, None, None, False, 1, 960, None, False),  # Conv
        (
            'gpooling',
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            False,
        ),  # Avg
        ('convbn', 'relu', 1, None, None, False, 1, 1280, None, False),  # Conv
    ],
}


def _mnv4_conv_medium_block_specs():
    """Medium-sized MobileNetV4 using only convolutional operations."""

    def convbn(kernel_size, strides, filters):
        return BlockSpec(
            block_fn='convbn',
            activation='relu',
            kernel_size=kernel_size,
            filters=filters,
            strides=strides,
            is_output=False,
        )

    def fused_ib(kernel_size, strides, filters, output=False):
        return BlockSpec(
            block_fn='fused_ib',
            activation='relu',
            kernel_size=kernel_size,
            filters=filters,
            strides=strides,
            expand_ratio=4.0,
            is_output=output,
        )

    def uib(
        start_dw_ks, middle_dw_ks, strides, filters, expand_ratio, output=False
    ):
        return BlockSpec(
            block_fn='uib',
            activation='relu',
            start_dw_kernel_size=start_dw_ks,
            middle_dw_kernel_size=middle_dw_ks,
            filters=filters,
            strides=strides,
            expand_ratio=expand_ratio,
            use_layer_scale=False,
            is_output=output,
        )

    blocks = [
        convbn(3, 2, 32),
        fused_ib(3, 2, 48, output=True),
        # 3rd stage
        uib(3, 5, 2, 80, 4.0),
        uib(3, 3, 1, 80, 2.0, output=True),
        # 4th stage
        uib(3, 5, 2, 160, 6.0),
        uib(3, 3, 1, 160, 4.0),
        uib(3, 3, 1, 160, 4.0),
        uib(3, 5, 1, 160, 4.0),
        uib(3, 3, 1, 160, 4.0),
        uib(3, 0, 1, 160, 4.0),
        uib(0, 0, 1, 160, 2.0),
        uib(3, 0, 1, 160, 4.0, output=True),
        # 5th stage
        uib(5, 5, 2, 256, 6.0),
        uib(5, 5, 1, 256, 4.0),
        uib(3, 5, 1, 256, 4.0),
        uib(3, 5, 1, 256, 4.0),
        uib(0, 0, 1, 256, 4.0),
        uib(3, 0, 1, 256, 4.0),
        uib(3, 5, 1, 256, 2.0),
        uib(5, 5, 1, 256, 4.0),
        uib(0, 0, 1, 256, 4.0),
        uib(0, 0, 1, 256, 4.0),
        uib(5, 0, 1, 256, 2.0, output=True),
        # FC layers
        convbn(1, 1, 960),
        BlockSpec(block_fn='gpooling', is_output=False),
        convbn(1, 1, 1280),
    ]
    return {
        'spec_name': 'MobileNetV4ConvMedium',
        'block_spec_schema': block_spec_field_list(),
        'block_specs': block_spec_values_to_list(blocks),
    }


MNV4ConvLarge_BLOCK_SPECS = {
    'spec_name': 'MobileNetV4ConvLarge',
    'block_spec_schema': [
        'block_fn',
        'activation',
        'kernel_size',
        'start_dw_kernel_size',
        'middle_dw_kernel_size',
        'middle_dw_downsample',
        'strides',
        'filters',
        'expand_ratio',
        'is_output',
    ],
    'block_specs': [
        ('convbn', 'relu', 3, None, None, False, 2, 24, None, False),
        ('fused_ib', 'relu', 3, None, None, False, 2, 48, 4.0, True),
        ('uib', 'relu', None, 3, 5, True, 2, 96, 4.0, False),
        ('uib', 'relu', None, 3, 3, True, 1, 96, 4.0, True),
        ('uib', 'relu', None, 3, 5, True, 2, 192, 4.0, False),
        ('uib', 'relu', None, 3, 3, True, 1, 192, 4.0, False),
        ('uib', 'relu', None, 3, 3, True, 1, 192, 4.0, False),
        ('uib', 'relu', None, 3, 3, True, 1, 192, 4.0, False),
        ('uib', 'relu', None, 3, 5, True, 1, 192, 4.0, False),
        ('uib', 'relu', None, 5, 3, True, 1, 192, 4.0, False),
        ('uib', 'relu', None, 5, 3, True, 1, 192, 4.0, False),
        ('uib', 'relu', None, 5, 3, True, 1, 192, 4.0, False),
        ('uib', 'relu', None, 5, 3, True, 1, 192, 4.0, False),
        ('uib', 'relu', None, 5, 3, True, 1, 192, 4.0, False),
        ('uib', 'relu', None, 3, 0, True, 1, 192, 4.0, True),
        ('uib', 'relu', None, 5, 5, True, 2, 512, 4.0, False),
        ('uib', 'relu', None, 5, 5, True, 1, 512, 4.0, False),
        ('uib', 'relu', None, 5, 5, True, 1, 512, 4.0, False),
        ('uib', 'relu', None, 5, 5, True, 1, 512, 4.0, False),
        ('uib', 'relu', None, 5, 0, True, 1, 512, 4.0, False),
        ('uib', 'relu', None, 5, 3, True, 1, 512, 4.0, False),
        ('uib', 'relu', None, 5, 0, True, 1, 512, 4.0, False),
        ('uib', 'relu', None, 5, 0, True, 1, 512, 4.0, False),
        ('uib', 'relu', None, 5, 3, True, 1, 512, 4.0, False),
        ('uib', 'relu', None, 5, 5, True, 1, 512, 4.0, False),
        ('uib', 'relu', None, 5, 0, True, 1, 512, 4.0, False),
        ('uib', 'relu', None, 5, 0, True, 1, 512, 4.0, False),
        ('uib', 'relu', None, 5, 0, True, 1, 512, 4.0, True),
        ('convbn', 'relu', 1, None, None, False, 1, 960, None, False),
        ('gpooling', None, None, None, None, None, None, None, None, False),
        ('convbn', 'relu', 1, None, None, False, 1, 1280, None, False),
    ],
}


def _mnv4_hybrid_medium_block_specs():
    """Medium-sized MobileNetV4 using only attention and convolutional operations."""

    def convbn(kernel_size, strides, filters):
        return BlockSpec(
            block_fn='convbn',
            activation='relu',
            kernel_size=kernel_size,
            filters=filters,
            strides=strides,
            is_output=False,
        )

    def fused_ib(kernel_size, strides, filters, output=False):
        return BlockSpec(
            block_fn='fused_ib',
            activation='relu',
            kernel_size=kernel_size,
            filters=filters,
            strides=strides,
            expand_ratio=4.0,
            is_output=output,
        )

    def uib(
        start_dw_ks, middle_dw_ks, strides, filters, expand_ratio, output=False
    ):
        return BlockSpec(
            block_fn='uib',
            activation='relu',
            start_dw_kernel_size=start_dw_ks,
            middle_dw_kernel_size=middle_dw_ks,
            filters=filters,
            strides=strides,
            expand_ratio=expand_ratio,
            use_layer_scale=True,
            is_output=output,
        )

    def mhsa_24px():
        return BlockSpec(
            block_fn='mhsa',
            activation='relu',
            filters=160,
            key_dim=64,
            value_dim=64,
            query_h_strides=1,
            query_w_strides=1,
            kv_strides=2,
            num_heads=4,
            use_layer_scale=True,
            use_multi_query=True,
            is_output=False,
        )

    def mhsa_12px():
        return BlockSpec(
            block_fn='mhsa',
            activation='relu',
            filters=256,
            key_dim=64,
            value_dim=64,
            query_h_strides=1,
            query_w_strides=1,
            kv_strides=1,
            num_heads=4,
            use_layer_scale=True,
            use_multi_query=True,
            is_output=False,
        )

    blocks = [
        convbn(3, 2, 32),
        fused_ib(3, 2, 48, output=True),
        # 3rd stage
        uib(3, 5, 2, 80, 4.0),
        uib(3, 3, 1, 80, 2.0, output=True),
        # 4th stage
        uib(3, 5, 2, 160, 6.0),
        uib(0, 0, 1, 160, 2.0),
        uib(3, 3, 1, 160, 4.0),
        uib(3, 5, 1, 160, 4.0),
        mhsa_24px(),
        uib(3, 3, 1, 160, 4.0),
        mhsa_24px(),
        uib(3, 0, 1, 160, 4.0),
        mhsa_24px(),
        uib(3, 3, 1, 160, 4.0),
        mhsa_24px(),
        uib(3, 0, 1, 160, 4.0, output=True),
        # 5th stage
        uib(5, 5, 2, 256, 6.0),
        uib(5, 5, 1, 256, 4.0),
        uib(3, 5, 1, 256, 4.0),
        uib(3, 5, 1, 256, 4.0),
        uib(0, 0, 1, 256, 2.0),
        uib(3, 5, 1, 256, 2.0),
        uib(0, 0, 1, 256, 2.0),
        uib(0, 0, 1, 256, 4.0),
        mhsa_12px(),
        uib(3, 0, 1, 256, 4.0),
        mhsa_12px(),
        uib(5, 5, 1, 256, 4.0),
        mhsa_12px(),
        uib(5, 0, 1, 256, 4.0),
        mhsa_12px(),
        uib(5, 0, 1, 256, 4.0, output=True),
        convbn(1, 1, 960),
        BlockSpec(block_fn='gpooling', is_output=False),
        convbn(1, 1, 1280),
    ]
    return {
        'spec_name': 'MobileNetV4HybridMedium',
        'block_spec_schema': block_spec_field_list(),
        'block_specs': block_spec_values_to_list(blocks),
    }


def _mnv4_hybrid_large_block_specs():
    """Large-sized MobileNetV4 using only attention and convolutional operations."""

    def convbn(kernel_size, strides, filters):
        return BlockSpec(
            block_fn='convbn',
            kernel_size=kernel_size,
            filters=filters,
            strides=strides,
            activation='gelu',
            is_output=False,
        )

    def fused_ib(kernel_size, strides, filters, output=False):
        return BlockSpec(
            block_fn='fused_ib',
            kernel_size=kernel_size,
            filters=filters,
            strides=strides,
            expand_ratio=4.0,
            is_output=output,
            activation='gelu',
        )

    def uib(
        start_dw_ks,
        middle_dw_ks,
        strides,
        filters,
        expand_ratio=4.0,
        output=False,
    ):
        return BlockSpec(
            block_fn='uib',
            start_dw_kernel_size=start_dw_ks,
            middle_dw_kernel_size=middle_dw_ks,
            filters=filters,
            strides=strides,
            expand_ratio=expand_ratio,
            use_layer_scale=True,
            is_output=output,
            activation='gelu',
        )

    def mhsa_24px():
        return BlockSpec(
            block_fn='mhsa',
            activation='relu',
            filters=192,
            key_dim=48,
            value_dim=48,
            query_h_strides=1,
            query_w_strides=1,
            kv_strides=2,
            num_heads=8,
            use_layer_scale=True,
            use_multi_query=True,
            is_output=False,
        )

    def mhsa_12px():
        return BlockSpec(
            block_fn='mhsa',
            activation='relu',
            filters=512,
            key_dim=64,
            value_dim=64,
            query_h_strides=1,
            query_w_strides=1,
            kv_strides=1,
            num_heads=8,
            use_layer_scale=True,
            use_multi_query=True,
            is_output=False,
        )

    blocks = [
        convbn(3, 2, 24),
        fused_ib(3, 2, 48, output=True),
        uib(3, 5, 2, 96),
        uib(3, 3, 1, 96, output=True),
        uib(3, 5, 2, 192),
        uib(3, 3, 1, 192),
        uib(3, 3, 1, 192),
        uib(3, 3, 1, 192),
        uib(3, 5, 1, 192),
        uib(5, 3, 1, 192),
        uib(5, 3, 1, 192),
        # add attention blocks to 2nd last stage
        mhsa_24px(),
        uib(5, 3, 1, 192),
        mhsa_24px(),
        uib(5, 3, 1, 192),
        mhsa_24px(),
        uib(5, 3, 1, 192),
        mhsa_24px(),
        uib(3, 0, 1, 192, output=True),
        # last stage
        uib(5, 5, 2, 512),
        uib(5, 5, 1, 512),
        uib(5, 5, 1, 512),
        uib(5, 5, 1, 512),
        uib(5, 0, 1, 512),
        uib(5, 3, 1, 512),
        uib(5, 0, 1, 512),
        uib(5, 0, 1, 512),
        uib(5, 3, 1, 512),
        uib(5, 5, 1, 512),
        mhsa_12px(),
        uib(5, 0, 1, 512),
        mhsa_12px(),
        uib(5, 0, 1, 512),
        mhsa_12px(),
        uib(5, 0, 1, 512),
        mhsa_12px(),
        uib(5, 0, 1, 512, output=True),
        convbn(1, 1, 960),
        BlockSpec(block_fn='gpooling', is_output=False),
        convbn(1, 1, 1280),
    ]
    return {
        'spec_name': 'MobileNetV4HybridLarge',
        'block_spec_schema': block_spec_field_list(),
        'block_specs': block_spec_values_to_list(blocks),
    }


SUPPORTED_SPECS_MAP = {
    'MobileNetV1': None,  # Implement if needed
    'MobileNetV2': None,  # Implement if needed
    'MobileNetV3Large': None,  # Implement if needed
    'MobileNetV3Small': None,  # Implement if needed
    'MobileNetV3EdgeTPU': None,  # Implement if needed
    'MobileNetMultiMAX': None,  # Implement if needed
    'MobileNetMultiAVG': None,  # Implement if needed
    'MobileNetMultiAVGSeg': None,  # Implement if needed
    'MobileNetMultiMAXSeg': None,  # Implement if needed
    'MobileNetV3SmallReducedFilters': None,  # Implement if needed
    'MobileNetV4ConvSmall': MNV4ConvSmall_BLOCK_SPECS,
    'MobileNetV4ConvMedium': _mnv4_conv_medium_block_specs(),
    'MobileNetV4ConvLarge': MNV4ConvLarge_BLOCK_SPECS,
    'MobileNetV4HybridMedium': _mnv4_hybrid_medium_block_specs(),
    'MobileNetV4HybridLarge': _mnv4_hybrid_large_block_specs(),
}

class MobileNet(nn.Module):
    def __init__(self, model_id: str = 'MobileNetV2', filter_size_scale: float = 1.0, input_specs: Tuple[int, int, int] = (3, 224, 224),
                 norm_momentum: float = 0.99, norm_epsilon: float = 0.001, kernel_initializer: str = 'VarianceScaling',
                 kernel_regularizer: Optional[nn.Module] = None, bias_regularizer: Optional[nn.Module] = None,
                 output_stride: Optional[int] = None, min_depth: int = 8, divisible_by: int = 8, stochastic_depth_drop_rate: float = 0.0,
                 flat_stochastic_depth_drop_rate: bool = True, regularize_depthwise: bool = False, use_sync_bn: bool = False,
                 finegrain_classification_mode: bool = True, output_intermediate_endpoints: bool = False, **kwargs):
        super(MobileNet, self).__init__()
        if model_id not in SUPPORTED_SPECS_MAP:
            raise ValueError(f'The MobileNet version {model_id} is not supported')

        if filter_size_scale <= 0:
            raise ValueError('filter_size_scale is not greater than zero.')

        if output_stride is not None and (output_stride == 0 or (output_stride > 1 and output_stride % 2)):
            raise ValueError('Output stride must be None, 1 or a multiple of 2.')

        self.model_id = model_id
        self.filter_size_scale = filter_size_scale
        self.min_depth = min_depth
        self.output_stride = output_stride
        self.divisible_by = divisible_by
        self.stochastic_depth_drop_rate = stochastic_depth_drop_rate
        self.flat_stochastic_depth_drop_rate = flat_stochastic_depth_drop_rate
        self.regularize_depthwise = regularize_depthwise
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.use_sync_bn = use_sync_bn
        self.norm_momentum = norm_momentum
        self.norm_epsilon = norm_epsilon
        self.finegrain_classification_mode = finegrain_classification_mode
        self.output_intermediate_endpoints = output_intermediate_endpoints

        self.input_specs = input_specs
        block_specs = SUPPORTED_SPECS_MAP.get(model_id)
        self.decoded_specs = block_spec_decoder(
            specs=block_specs,
            filter_size_scale=self.filter_size_scale,
            divisible_by=self.divisible_by,
            finegrain_classification_mode=self.finegrain_classification_mode,
        )

        self.layers = self._mobilenet_base()

    def _mobilenet_base(self):
        layers = []
        current_stride = 1
        rate = 1
        num_blocks = len(self.decoded_specs)
        input_channels = self.input_specs[0]
        for block_idx, block_def in enumerate(self.decoded_specs):
            block_stride = 1 if block_def.strides is None else block_def.strides

            if self.output_stride is not None and current_stride == self.output_stride:
                layer_stride = 1
                layer_rate = rate
                rate *= block_stride
            else:
                layer_stride = block_stride
                layer_rate = 1
                current_stride *= block_stride

            stochastic_depth_drop_rate = self.stochastic_depth_drop_rate if self.flat_stochastic_depth_drop_rate else get_stochastic_depth_rate(self.stochastic_depth_drop_rate, block_idx + 1, num_blocks)

            if block_def.block_fn == 'convbn':
                layers.append(Conv2DBNBlock(
                    in_channels=input_channels,
                    out_channels=block_def.filters,
                    kernel_size=block_def.kernel_size,
                    stride=layer_stride,
                    use_bias=block_def.use_bias,
                    use_explicit_padding=False,
                    activation=block_def.activation,
                    norm_momentum=self.norm_momentum,
                    norm_epsilon=self.norm_epsilon
                ))
                input_channels = block_def.filters

            elif block_def.block_fn == 'depsepconv':
                layers.append(DepthwiseSeparableConvBlock(
                    in_channels=input_channels,
                    out_channels=block_def.filters,
                    kernel_size=block_def.kernel_size,
                    stride=layer_stride,
                    activation=block_def.activation,
                    dilation_rate=layer_rate,
                    regularize_depthwise=self.regularize_depthwise,
                    norm_momentum=self.norm_momentum,
                    norm_epsilon=self.norm_epsilon
                ))
                input_channels = block_def.filters


            elif block_def.block_fn == 'mhsa':
                block = MultiHeadSelfAttentionBlock(
                    input_dim=input_channels,
                    output_dim=block_def.filters,
                    num_heads=block_def.num_heads,
                    key_dim=block_def.key_dim,
                    value_dim=block_def.value_dim,
                    use_multi_query=block_def.use_multi_query,
                    query_h_strides=block_def.query_h_strides,
                    query_w_strides=block_def.query_w_strides,
                    kv_strides=block_def.kv_strides,
                    downsampling_dw_kernel_size=block_def.downsampling_dw_kernel_size,
                    use_bias=False,
                    use_cpe=True,
                    cpe_dw_kernel_size=block_def.kernel_size,
                    stochastic_depth_drop_rate=stochastic_depth_drop_rate,
                    use_residual=block_def.use_residual,
                    use_sync_bn=self.use_sync_bn,
                    use_layer_scale=block_def.use_layer_scale,
                    layer_scale_init_value=1e-5,
                    norm_momentum=self.norm_momentum,
                    norm_epsilon=self.norm_epsilon,
                    output_intermediate_endpoints=self.output_intermediate_endpoints
                )
                layers.append(block)
                input_channels = block_def.filters

            elif block_def.block_fn == 'fused_ib':
                layers.append(FusedInvertedBottleneckBlock(
                    in_channels=input_channels,
                    out_channels=block_def.filters,
                    stride=layer_stride,
                    expand_ratio=block_def.expand_ratio,
                    activation=block_def.activation,
                    norm_momentum=self.norm_momentum,
                    norm_epsilon=self.norm_epsilon
                ))
                input_channels = block_def.filters

            elif block_def.block_fn in ('invertedbottleneck', 'uib'):
                use_rate = layer_rate if layer_rate > 1 and getattr(block_def, 'kernel_size', 1) != 1 else 1
                layers.append(UniversalInvertedBottleneckBlock(
                    in_channels=input_channels,
                    out_channels=block_def.filters,
                    stride=layer_stride,
                    expand_ratio=block_def.expand_ratio,
                    activation=block_def.activation,
                    use_residual=block_def.use_residual,
                    middle_dw_downsample=block_def.middle_dw_downsample,
                    start_dw_kernel_size=block_def.start_dw_kernel_size,
                    middle_dw_kernel_size=block_def.middle_dw_kernel_size,
                    end_dw_kernel_size=block_def.end_dw_kernel_size,
                    use_layer_scale=block_def.use_layer_scale,
                    stochastic_depth_drop_rate=stochastic_depth_drop_rate,
                    norm_momentum=self.norm_momentum,
                    norm_epsilon=self.norm_epsilon,
                    divisible_by=self.divisible_by
                ))
                input_channels = block_def.filters

            elif block_def.block_fn == 'gpooling':
                layers.append(GlobalPoolingBlock())

            else:
                raise ValueError(f'Unknown block type {block_def.block_fn} for layer {block_idx}')

            if stochastic_depth_drop_rate:
                layers.append(StochasticDepth(stochastic_depth_drop_rate))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x

    @property
    def output_specs(self):
        """A dict of {level: TensorShape} pairs for the model output."""
        return {str(i): layer.weight.shape for i, layer in enumerate(self.layers) if isinstance(layer, nn.Conv2d)}

def build_mobilenet(model_id: str, filter_size_scale: float = 1.0, input_specs: Tuple[int, int, int] = (3, 224, 224), **kwargs):
    return MobileNet(model_id=model_id, filter_size_scale=filter_size_scale, input_specs=input_specs, **kwargs)

# Example usage:
# mobilenet_v4_conv_small = build_mobilenet('MobileNetV4ConvSmall', input_specs=(3, 32, 32), num_classes=100)
# mobilenet_v4_conv_medium = build_mobilenet('MobileNetV4ConvMedium', input_specs=(3, 32, 32), num_classes=100)
# mobilenet_v4_conv_large = build_mobilenet('MobileNetV4ConvLarge', input_specs=(3, 32, 32), num_classes=100)
# mobilenet_v4_hybrid_medium = build_mobilenet('MobileNetV4HybridMedium', input_specs=(3, 32, 32), num_classes=100)
# mobilenet_v4_hybrid_large = build_mobilenet('MobileNetV4HybridLarge', input_specs=(3, 32, 32), num_classes=100)
