

class SymmericQuantization:
    @staticmethod
    def quantization(x, bit_width, dynamic_range, scale_factor = 1.0):
        dynamic_range_scale = scale_factor * dynamic_range
        down_limit, up_limit = -(1<<bit_width-1)+1, (1<<bit_width-1)-1
        x += dynamic_range_scale
        x *= (up_limit - down_limit) / (dynamic_range_scale*2)
        x += down_limit
        x.clamp_(down_limit, up_limit)
        x.round_()

    @staticmethod
    def dequantization(x, bit_width, dynamic_range, scale_factor = 1.0):
        dynamic_range_scale = scale_factor * dynamic_range
        down_limit, up_limit = -(1<<bit_width-1)+1, (1<<bit_width-1)-1
        x -= down_limit
        x *= (dynamic_range_scale*2)/(up_limit - down_limit)
        x -= dynamic_range_scale

class PositiveQuantization:
    @staticmethod
    def quantization(x, bit_width, dynamic_range, scale_factor = 1.0):
        dynamic_range_scale = scale_factor * dynamic_range
        up_limit = (1<<bit_width)-1
        x *= (up_limit) / (dynamic_range_scale)
        x.clamp_(0, up_limit)
        x.round_()

    @staticmethod
    def dequantization(x, bit_width, dynamic_range, scale_factor = 1.0):
        dynamic_range_scale = scale_factor * dynamic_range
        up_limit = (1<<bit_width)-1
        x *= (dynamic_range_scale)/(up_limit)

class FixPointQuantization:
    @staticmethod
    def quantization(x, integer_bit, decimal_bit):
        dynamic_range = 2**(integer_bit)
        limit = (1<<(integer_bit + decimal_bit))-1
        x *= (limit) / (dynamic_range)
        x.clamp_(-limit, limit)
        x.round_()

    @staticmethod
    def dequantization(x, integer_bit, decimal_bit):
        dynamic_range = 2**(integer_bit)
        limit = (1<<(integer_bit + decimal_bit))-1
        x *= (dynamic_range)/(limit)