#!/bin/python
# 17-11-27 Trung-Hieu Tran@ IPVS

from __future__ import print_function

import numpy as np
import time
from dsc_v1 import DSC_V1
import pprint
import math
import utils as mutils


def load_config():
    config = {}
    config['patch_size'] = 10
    config['overlap_size'] = 3
    config['dataset'] = 'BSD68_poisson'
    config['data_dir'] = '../../output/'
    config['output_dir'] = './output'
    config['dict_size'] = 256

    config['method'] = 'v1'
    config['train'] = True
    return config

def main():
    config = load_config()
    mutils.mkdir_p(config['output_dir'])

    if config['method'] == 'v1':
        DSC = DSC_V1(config)
    else:
        #TODO: implement both method a and c
        print("... on work ...")

    if config['train']:
        DSC.train()
    else:
        DSC.test()

if __name__ == "__main__":
    main()
