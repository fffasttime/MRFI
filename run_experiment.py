import torch
import argparse
import importlib
import logging

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('expname', type = str)
    parser.add_argument('-v', '--verbose', action = 'store_true', help='set logging level = INFO')
    parser.add_argument('--debug', action = 'store_true', help='set logging level = DEBUG')
    parser.add_argument('-nw', '--no-warning', action = 'store_true', help='ignore warning, set logging level = ERROR')
    parser.add_argument('--log', type=str, default = '')
    parser.add_argument('-nt', '--num-threads', type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.num_threads)

    if args.debug: logging.basicConfig(level = logging.DEBUG)
    if args.verbose: logging.basicConfig(level = logging.INFO)
    if args.no_warning: logging.basicConfig(level = logging.ERROR)
    if args.log != '': logging.basicConfig(filename = args.log)

    importlib.import_module('experiments.' + args.expname)
