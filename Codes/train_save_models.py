#!/usr/bin/python

import numpy as np
import data_tools
import ml_tools
import pickle as pk

from common import parameters

if __name__ == "__main__":
    params = parameters()
    peptides = data_tools.read_data( params.data_path )
    for n in params.ntrain : 
        print n
        benchmark = ml_tools.rt_benchmark( peptides, 'elude', 'gp', n , params.nparts, params.train_ratio )
        models = ml_tools.single_train_gp( benchmark )
        save_path = params.save_tmp % ( params.data_root, params.models_tag, n )
        with open( save_path, 'w' ) as ff :
            pk.dump( [ models ], ff )
            ff.close()
        models = None 
