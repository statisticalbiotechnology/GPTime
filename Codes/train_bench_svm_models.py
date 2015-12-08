#!/usr/bin/python

import numpy as np
import data_tools
import ml_tools
import pickle as pk

from common import parameters

if __name__ == "__main__":
    params = parameters()
    peptides = data_tools.read_data( params.data_path )

    values = [];

    for n in params.ntrain : 
        print n
        benchmark = ml_tools.rt_benchmark( peptides, 'elude', 'svr', n , params.nparts, params.train_ratio )

        dt_array = []
        rmse_array = []

        for i in range( params.nparts ):
            print i
            model = benchmark.train_model(i)
            dt, rmse = benchmark.eval_model( i,model )

            dt_array.append( dt )
            rmse_array.append( rmse )

        values.append( [ n, dt_array, rmse_array ] )

    f = open('svr_bench2.pk','w');
    pk.dump( [ values ], f )
    f.close()
