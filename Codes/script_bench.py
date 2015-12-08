#!/usr/bin/python

import pickle as pk
import numpy as np

import plotting_script

def bench_svr():
    f = open('svr_bench.pk','r')
    values = pk.load(f)[0]

    for v in values :
        n = v[0]
        drt = v[1]
        rmse = v[2]
        print n, np.mean( drt ), np.std( drt )

def save_bench_gp():
    ntrain = [ 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000 ] 

    values = [];
    for n in ntrain :
        print n
        dp = plotting_script.data_plotter( n )
        dp.load_data();
        values.append( dp.benchmark_gp() )
    f = open('gp_bench.pk','w')
    pk.dump( [ values ], f )

