"""MRFI quantization methods

A Quantization has two static function `quantize()` and `dequantize()`, 
both have args `x` and other args have specified in config file.
These two function should modify `x` inplace, DO NOT return something.

`quantize()` should make input `x` into a integer tensor with float32 type, 
aka. pseudo quantization, therefore pytorch can forward them correctly.

Warning:
    A integer bit flip error mode *always* need a quantization.

    The `bit_width` argument and the result integer range (e.g. -128~127) should be
    *consist with* corresponding error mode argment. 
    Since MRFI does not check value bound for performance reason, wrong arguments or 
    wrong implemention of quantization may silently lead to unexpected experiment result.

!!! info "Runtime dynamic quantization"
    If you set `dynamic_range` to "auto" in MRFI config, 
    this value will be set to max range of the input tensor automatically by MRFI.
    This feature can be used to simulate runtime dynamic quantization.
    However, it should be noted that fault injection can also cause changes in the dynamic range of the later layer.

"""

class SymmericQuantization:
    """Simple symmeric quantization.
    
    Uniformly mapping a float tensor in range `[-dynamic_range*scale_factor, +dynamic_range*scale_factor]`
    into integer range `[-2**(bit_width-1)+1, 2**(bit_width-1)-1]`.\n
    Outliers are clipped.

    **Parameters:**

    |Name|Type|Description|Default|
    |-|-|-|-|
    |bit_width|`int`|Often 8 or 16.|*required*|
    |dynamic_range|`float`|Usually maximum of values.|*required*|
    |scale_factor|`float`|Extra factor on dynamic range.|`1.0`|
    """

    @staticmethod
    def quantize(x, bit_width: int, dynamic_range: float, scale_factor: float = 1.0):
        dynamic_range_scale = scale_factor * dynamic_range
        down_limit, up_limit = -(1<<bit_width-1)+1, (1<<bit_width-1)-1
        x += dynamic_range_scale
        x *= (up_limit - down_limit) / (dynamic_range_scale*2)
        x += down_limit
        x.clamp_(down_limit, up_limit)
        x.round_()

    @staticmethod
    def dequantize(x, bit_width, dynamic_range, scale_factor = 1.0):
        dynamic_range_scale = scale_factor * dynamic_range
        down_limit, up_limit = -(1<<bit_width-1)+1, (1<<bit_width-1)-1
        x -= down_limit
        x *= (dynamic_range_scale*2)/(up_limit - down_limit)
        x -= dynamic_range_scale

class PositiveQuantization:
    """Simple positive quantization.
    
    Uniformly mapping a float tensor in range `[0, dynamic_range*scale_factor]`
    into integer range `[0, 2**(bit_width)-1]`.\n 
    Outliers are clipped.

    **Parameters:**

    |Name|Type|Description|Default|
    |-|-|-|-|
    |bit_width|`int`|Often 8 or 16.|*required*|
    |dynamic_range|`float`|Usually maximum of values.|*required*|
    |scale_factor|`float`|Extra factor on dynamic range.|`1.0`|
    """
    @staticmethod
    def quantize(x, bit_width: int, dynamic_range: float, scale_factor: float = 1.0):
        dynamic_range_scale = scale_factor * dynamic_range
        up_limit = (1<<bit_width)-1
        x *= (up_limit) / (dynamic_range_scale)
        x.clamp_(0, up_limit)
        x.round_()

    @staticmethod
    def dequantize(x, bit_width, dynamic_range, scale_factor = 1.0):
        dynamic_range_scale = scale_factor * dynamic_range
        up_limit = (1<<bit_width)-1
        x *= (dynamic_range_scale)/(up_limit)

class FixPointQuantization:
    """Fixpoint quantization.
    
    Quantize a float tensor into binary fix point representation `integer_bit.decimal_bit`.\n
    So the input dynamic range is `[-2**integer_bit, 2**integer_bit]`, outliers are clipped.

    **Parameters:**

    |Name|Type|Description|Default|
    |-|-|-|-|
    |integer_bit|`int`|Integer bits of value.|*required*|
    |decimal_bit|`int`|Decimal bits of value.|*required*|
    """
    @staticmethod
    def quantize(x, integer_bit: int, decimal_bit: int):
        dynamic_range = 2**(integer_bit)
        limit = (1<<(integer_bit + decimal_bit))-1
        x *= (limit) / (dynamic_range)
        x.clamp_(-limit, limit)
        x.round_()

    @staticmethod
    def dequantize(x, integer_bit, decimal_bit):
        dynamic_range = 2**(integer_bit)
        limit = (1<<(integer_bit + decimal_bit))-1
        x *= (dynamic_range)/(limit)
