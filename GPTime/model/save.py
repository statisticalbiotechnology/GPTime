import pickle

def save( model, path ):
    with open(path, 'wb') as ff :
        pickle.dump([ model ], ff )
        ff.close()
