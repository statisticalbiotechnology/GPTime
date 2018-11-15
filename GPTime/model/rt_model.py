import GPy
import numpy as np

class rt_model:
    def __init__(self, feature, X, Y, pa, norm, voc, em, y_params = []):
        self.feature = feature
        self.model = GPy.models.GPRegression(X,Y)
        self.model[:] = pa
        self.norm = norm
        self.voc = voc
        self.em = em
        self.y_params = y_params

    def get_vector( self, p ):
        if self.feature == "bow":
            vec = p.bow_descriptor(self.voc)
        if self.feature == "elude":
            vec = p.elude_descriptor(self.em)
        self.norm.normalize(vec)
        return vec

    def eval(self, p):
        res = [];
        if self.feature == "bow":
            vec = p.bow_descriptor(self.voc)
        if self.feature == "elude":
            vec = p.elude_descriptor(self.em)
        self.norm.normalize(vec)
        vec = np.matrix(vec)
        vals = self.model.predict(np.array(vec))

        if len( self.y_params ) > 0 :
            vals[0] = vals[0] + self.y_params[0]

        res = ( vals[0][0][0], vals[1][0][0] ) # Mean, Variance
        
        return res
