#!/usr/bin/python

import getopt, sys
import numpy as np
import pickle as pk

import GPy

from Codes import data_tools
from Codes import ml_tools

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

    model = ml_tools.rt_model( feature, type, mgp, norm,voc,em,y_params )
    return model

def usage():
    print "Usage of GPTime Package"
    print "[ -o, --operation ]\t:\t The execution operation, current supported operations [ train, predict ]"
    print "[ -p, --peptides  ]\t:\t Path to the file containing their peptides and retention time"
    print "[ -n, --ntrain    ]\t:\t Number of the peptides used for training"
    print "[ -m, --model     ]\t:\t Path to the model file"
    print "                   \t\t If operation is train, then this is output model path"
    print "                   \t\t If operation is predict, then this is the input model path"
    print "[ -r, --randomize ]\t:\t Randomizing the order of the input peptides"
    print "[ -h, --help      ]\t:\t Shows this menu"


def main():

    operation = "NA"
    peptides_path = "NA"
    model_path = "NA"
    randomize = False
    ntrain = 100

    try :
        opts, args = getopt.getopt( sys.argv[1:], 
                                   "ho:p:t:m:r", 
                                   ["help","operation=","peptides=","ntrain=","model=","randomize"] 
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

        print model

        for p in peptides :
            m,v = model.eval(p) 
            s = np.sqrt(v)
            print p.sequence, p.rt,m,v,s


if __name__=="__main__" :
    main()
        

    
