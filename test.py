import argparse
import numpy as np
import pickle
import GPTime

def parse_commandline():
    parser = argparse.ArgumentParser(description="GPTime Train")
    parser.add_argument('-p','--peptides', help="Path to the file containing their peptides and retention time", required=True)
    parser.add_argument('-m','--model', help="""Path to the model file. If operation is train, then this is output model path.
                                                If operation is predict, then this is the input model path."""
                                                , required=True)
    args = parser.parse_args()

    return args

if __name__=="__main__" :
    args = parse_commandline()

    peptides = GPTime.peptides.load(args.peptides, check_duplicates=False)
    model = GPTime.model.load( args.model )

    for p in peptides :
        m,v = model.eval(p)
        s = np.sqrt(v)
        print(p.sequence, p.rt,m,v,s)
