import sys
from importlib import import_module
import inspect

sys.stdout = open('docs/function_table.md','w')

def parse_module(name, argparser = None):
    mod = import_module('mrfi.' + name)
    print('##', name)
    print()
    print('|Name|Description|Args|')
    print('|-|-|-|')
    for k, v in mod.__dict__.items():
        if callable(v) and v.__module__ == 'mrfi.' + name and k[0]!='_':
            doc: str = v.__doc__
            if doc is None:
                doc='\n'

            link = f'../{name}/#mrfi.{name}.{k}'
            arg_str = argparser(v) if argparser else ''
            print(f"|[`{k}`]({link})|{doc.splitlines()[0]}|{arg_str}|")
    print('\n')

print('# MRFI function table\n')

def get_args(obj):
    arg_list = list(obj.__annotations__.keys())
    arg_list = ['`' + s + '`' for s in arg_list]
    return ','.join(arg_list)

def get_quantization_args(obj):
    arg_list = list(obj.quantize.__annotations__.keys())[1:]
    arg_list = ['`' + s + '`' for s in arg_list]
    return ','.join(arg_list)

parse_module('experiment', get_args)
parse_module('observer')
parse_module('selector', get_args)
parse_module('error_mode', get_args)
parse_module('quantization', get_quantization_args)
