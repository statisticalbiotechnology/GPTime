#!/usr/bin/python

import getopt, sys
import data_tools

import numpy as np
import ml_tools
import pickle as pk

def load_peptides( path, randomize ):
    peptides = data_tools.read_data( path )
    if randomize :
        np.random.shuffle( peptides )
    return peptides

def load_model( path ):
    ff = open( path, 'r' )
    m = pk.load(ff)[0]
    ff.close();

    feature = m[0]
    type = m[1]
    X = m[2]
    Y = m[3]
    pa = m[4]
    norm = m[5]
    voc = m[6]
    em = m[7]

    y_params = [];
    if len(m) > 8 :
        y_params = m[8];

    mgp = GPy.models.GPRegression(X,Y)
    mgp[:] = pa

    model = rt_model( feature, type, mgp, norm,voc,em,y_params )

def usage():
    print "This is the usage"

def main():

    operation = "NA"
    peptides_path = "NA"
    model_path = "NA"
    randomize = False
    ntrain = 100
    nval = 100

    try :
        opts, args = getopt.getopt( sys.argv[1:], 
                                   "ho:p:t:v:m:r", 
                                   ["help","operation=","peptides=","ntrain=","nval=","model=","randomize"] 
                                  )
    except getopt.GetoptError as err:
        print(err)
        usage()
        sys.exit(2)

    for o, a in opts :
        if o in ( "-h", "--help" ):
            usage()
            sys.exit(2)
        elif o in ( "-o", "--operation" ):
            operation = a
        elif o in ( "-p", "--peptides" ):
            peptides_path = a
        elif o in ( "-N", "--ntrain" ):
            ntrain = int(a)
        elif o in ( "-m", "--model" ):
            model_path = a
        elif o in ( "-r", "--randomize" ):
            randomize = True

    if operation == "train" :
        assert peptides_path != "NA", "Peptides database was not given"
        assert model_path != "NA", "Path to save the model was not given"

        peptides = load_peptides( peptides_path, randomize )
        trainer = ml_tools.rt_trainer( peptides, 'elude', 'gp', ntrain )
        model_params = trainer.train_model()

        ff = open( model_path, 'w' )
        pk.dump([ model_params ], ff )
        ff.close();
    elif operation == "predict" :
        assert peptides_path != "NA", "Peptides database was not given"
        assert model_path != "NA", "Path to save the model was not given"

        peptides = load_peptides( peptides_path, randomize )
        model = load_model( model_path )

        for p in peptides :
            m,s = model.eval(p)
            print p.sequence, m, s


if __name__=="__main__" :
    main()
        

    
