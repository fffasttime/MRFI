# Custom method/arguments

## Method arguments

Arguments of method can be variety type, including number, string, list, dict. 
It depends on the definition of the method. 
*In addition, argument an be emitted in configuration if a default value is given in the definition of method.*

For example, random rate position selector [`RandomPositionByRate`](../selector/#mrfi.selector.RandomPositionByRate) receives a float parameter `rate`, 
while fixed positions selector [`FixPositions`](../selector/#mrfi.selector.FixPositions) receives a list of positions.

## Use custom fault injection method/function

Custom methods can be defined according to corresponding requirements, 
see [observer doc](observer.md), [selector doc](selector.md), [error mode doc](error_mode.md), [quantization doc](error_mode.md).
Then use [`mrfi.add_function`](../mrfi/#mrfi.mrfi.add_function) to let MRFI know it. 
The custom method can be used in both EasyConfig and fine-grained ConfigTree.

The following code adds two trivial method to MRFI.

```python
from mrfi import add_function

def MyTestErrorMode(x, custom_val = 0):
    # set all selected 
    return torch.full_like(x, val)

class MyNoQuantization:
    @staticmethod
    def quantize(x, custom_arg):
        pass

    @staticmethod
    def dequantize(x, custom_arg):
        pass
  
if __name__ == '__main__':
    add_function('NoQuantization', MyNoQuantization)
    add_function('TestErrorMode', MyTestErrorMode)

```

To use the custom method, the EasyConfig can be like this:

```yaml
faultinject:
  - type: activation
    quantization:
      method: NoQuantization
      custom_arg: xxx  # can not be emitted
    error_mode:
      method: MyTestErrorMode
      custom_val: 16 # can be emitted because defaut value gived
```

If you think your function is commonly used and not provided by MRFI, 
welcome to add it to MRFI's built-in methods through Pull Request on Github.
