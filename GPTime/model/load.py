import pickle

def load( path ):
    with open(path, 'rb') as ff :
        model = pickle.load(ff)[0]
    return model
