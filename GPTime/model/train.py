import numpy as np
import GPy
from .. import features
from .rt_model import rt_model

class train:
    def __init__(self, peptides, feature, ntrain ):
        self.peptides = peptides
        self.feature = feature
        self.ntrain = ntrain

    def train_model( self ):
        train_peptides = self.peptides[:self.ntrain]

        mg = features.mapping_generator(train_peptides)
        voc = mg.get_bow_voc(2)
        em = mg.get_elude_model()

        Y = [];
        X = [];
        for p in train_peptides:
            Y.append(p.rt)
            if self.feature == 'bow':
                X.append(p.bow_descriptor(voc))
            elif self.feature == 'elude':
                X.append(p.elude_descriptor(em))
        X = np.matrix(X)

        Y_params = [];
        Y = np.transpose(np.matrix(Y))
        norm = features.normalizer()

        if self.feature == "elude":
            norm.normalize_maxmin(X);
        gpy_model = GPy.models.GPRegression(X, Y)
        gpy_model.optimize_restarts(num_restarts=10, verbose=False)
        pa = list(gpy_model.param_array)
        gpy_model = None
        return rt_model(self.feature, X, Y, pa , norm,voc,em, Y_params)
