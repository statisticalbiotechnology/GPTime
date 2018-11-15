import argparse
import numpy as np
import pickle
import GPTime

def parse_commandline():
    parser = argparse.ArgumentParser(description="GPTime Train")
    parser.add_argument('-p','--peptides', help="Path to the file containing their peptides and retention time", required=True)
    parser.add_argument('-n','--ntrain', help="Number of the peptides used for training (default=100)", type=int, default=100)
    parser.add_argument('-m','--model', help="""Path to the model file. If operation is train, then this is output model path.
                                                If operation is predict, then this is the input model path."""
                                                , required=True)
    parser.add_argument('-r','--randomize', help="Randomizing the order of the input peptides", dest="randomize",
                        action='store_true')
    parser.set_defaults(randomize=False)

    args = parser.parse_args()

    return args

if __name__=="__main__" :
    args = parse_commandline()

    peptides = GPTime.peptides.load(args.peptides, check_duplicates=False)
    trainer = GPTime.model.train(peptides,'elude',args.ntrain)
    model = trainer.train_model()
    GPTime.model.save( model, args.model )
