easyconfig_presets = {
"default_fi":"""
faultinject:
  - type: activation
    quantization:
      method: SymmericQuantization
      dynamic_range: auto
      bit_width: 16
    enabled: True
    selector:
      method: RandomPositionByRate
      poisson: True
      rate: 1e-3
    error_mode:
      method: IntSignBitFlip
      bit_width: 16

    module_name: [conv, fc]
    module_type: [Conv2d, Linear]
""",
"float_fi":"""
faultinject:
  - type: activation
    enabled: True
    selector:
      method: RandomPositionByRate
      poisson: True
      rate: 1e-6
    error_mode:
      method: FloatRandomBitFlip
      floattype: float32

    module_name: [conv, fc]
    module_type: [Conv2d, Linear]
""",
"float_weight_fi":"""
faultinject:
  - type: weight
    name: [weight]
    enabled: True
    selector:
      method: RandomPositionByRate
      poisson: True
      rate: 1e-3
    error_mode:
      method: FloatRandomBitFlip
      floattype: float32

    module_name: [conv, fc]
    module_type: [Conv2d, Linear]
""",
"fxp_fi":"""
faultinject:
  - type: activation
    quantization:
      method: FixPointQuantization
      integer_bit: 3
      decimal_bit: 12
    enabled: True
    selector:
      method: RandomPositionByRate
      poisson: True
      rate: 1e-3
    error_mode:
      method: IntSignBitFlip
      bit_width: 16

    module_name: [conv, fc]
    module_type: [Conv2d, Linear]
""",
"fxp_weight_fi":"""
faultinject:
  - type: weight
    name: [weight]
    quantization:
      method: FixPointQuantization
      integer_bit: 3
      decimal_bit: 13
    enabled: True
    selector:
      method: RandomPositionByRate
      poisson: True
      rate: 1e-6
    error_mode:
      method: IntRandomBitFlip
      bit_width: 16

    module_name: [conv, fc]
    module_type: [Conv2d, Linear]
""",
"weight_fi":"""
faultinject:
  - type: weight
    name: [weight]
    quantization:
      method: SymmericQuantization
      dynamic_range: auto
      bit_width: 16
    enabled: True
    selector:
      method: RandomPositionByRate
      poisson: True
      rate: 1e-3
    error_mode:
      method: IntSignBitFlip
      bit_width: 16
    module_name: [conv, fc]
    module_type: [Conv2d, Linear]
""",
}
