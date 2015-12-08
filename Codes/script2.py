#!/usr/bin/python

import numpy as np
import data_tools
import ml_tools
import pickle
import GPy


if __name__ == "__main__":
    n = 1000
    models_path = "/Users/heydar/Stuff/tmp/gprt/models_ntrain_%d.pk" % ( n )
    with open( models_path, 'r' ) as ff :
        models = pickle.load(ff)[0]
        ff.close();

    print len( models )

    raw_input()




    peptides = data_tools.read_data()
    # duplicated_message = data_tools.checked_duplicated(peptides)
    # print duplicated_message

    bench = ml_tools.rt_benchmark(peptides, 'elude', 'gp', 100, 5)

    fmat = [];
    mmat = [];
    dmat = [];

    for i in range( bench.parts.nfolds ):
        print i
        model = bench.train_model(i)
        f,m,d = bench.test_sorted(i,model)

        fmat.append(f)
        mmat.append(m)
        dmat.append(d)

    fmat = np.matrix(fmat)
    mmat = np.matrix(mmat)
    dmat = np.matrix(dmat)
