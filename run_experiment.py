from cmath import exp
from importlib import import_module
import sys
import logging

#logging.basicConfig(level=logging.NOTSET)

if __name__=='__main__':
    ename = sys.argv[1]
    print(ename)
    mod = import_module('configs.'+ename)
    n = int(sys.argv[2])
    for expfunc in ['exp', 'experiment']:
        if expfunc in mod.__dict__:
            mod.__dict__[expfunc](n)
            break
