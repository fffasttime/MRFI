

class SymmericQuantization:
    @staticmethod
    def quantization(x, bit_width, dynamic_range):
        down_limit, up_limit = -(1<<bit_width-1)+1, (1<<bit_width-1)-1
        x += dynamic_range
        x *= (up_limit - down_limit) / (dynamic_range*2)
        x += down_limit
        x.clamp_(down_limit, up_limit)
        x.round_()

    @staticmethod
    def dequantization(x, bit_width, dynamic_range):
        down_limit, up_limit = -(1<<bit_width-1)+1, (1<<bit_width-1)-1
        x -= down_limit
        x *= (dynamic_range*2)/(up_limit - down_limit)
        x -= dynamic_range

class PositiveQuantization:
    @staticmethod
    def quantization(x, bit_width, dynamic_range):
        up_limit = (1<<bit_width)-1
        x *= (up_limit) / (dynamic_range)
        x.clamp_(0, up_limit)
        x.round_()

    @staticmethod
    def dequantization(x, bit_width, dynamic_range):
        up_limit = (1<<bit_width)-1
        x *= (dynamic_range)/(up_limit)
