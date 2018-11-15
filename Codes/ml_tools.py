#!/usr/bin/python

import numpy as np
import random
import feature_extraction
import GPy
import math
import matplotlib.pyplot as plt
import pickle as pk
from gp_tools import my_gp

from scipy import stats
from scipy.stats.stats import pearsonr
from sklearn import svm, grid_search
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

from joblib import Parallel, delayed
import multiprocessing

def chunk_it( seq, num ):
    avg = len( seq ) / float( num )
    out = []
    last = 0.0
    while last < len( seq ):
        out.append( seq[int(last):int(last+avg)] )
        last += avg
    return out

def train(i,bench):
    model = bench.train_model(i)
    return model

def pcv_train(i, bench):
    model = bench.train_model(i)
    scores = bench.eval_model(i, model)
    return scores

def pcv_train_multi(i,bench):
    models = bench.train_multi_model(i,10)
    scores = bench.eval_multi_model(i, models)
    return scores

def parallel_train(bench):
    num_cores = multiprocessing.cpu_count()
    models = Parallel(n_jobs=num_cores)(delayed(train)(i, bench) for i in range(bench.parts.nfolds))
    return models

def single_train_gp(bench):
    models = [];
    for i in range(bench.parts.nfolds):
        print(i)
        m = bench.train_gp_model(i)
        models.append(m)
    return models

def parallel_cross_validataion(bench):
    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(pcv_train)(i, bench) for i in range(bench.parts.nfolds))
    results = np.matrix( results )
    return results

def parallel_cross_validataion_multi(bench):
    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(pcv_train_multi)(i, bench) for i in range(bench.parts.nfolds))
    results = np.matrix( results )
    return results

def section_error_independent( bench, models ):
    fracs = []
    means = []
    errs = []

    for i,model in enumerate( models ):
        print(i)
        f,m,e = bench.eval_sections_independent(i,model)
        fracs.append(f)
        means.append(m)
        errs.append(e)

    fracs = np.matrix( fracs )
    means = np.matrix( means )
    errs = np.matrix( errs )

    f_m = np.mean( fracs, axis=0 )
    m_m = np.mean( means,axis=0 )
    e_m = np.mean( errs, axis=0 )
    e_s = np.std( errs, axis=0 )

    f_m = np.squeeze( np.asarray( f_m ) )
    m_m = np.squeeze( np.asarray( m_m ) )
    e_m = np.squeeze( np.asarray( e_m ) )
    e_s = np.squeeze( np.asarray( e_s ) )

    return f_m, m_m, e_m, e_s

def section_error_interval( bench, models ):
    fracs = []
    means = []
    errs = []

    for i,model in enumerate( models ):
        print(i)
        f,m,e = bench.eval_sections_interval(i,model)
        fracs.append(f)
        means.append(m)
        errs.append(e)

    fracs = np.matrix( fracs )
    means = np.matrix( means )
    errs = np.matrix( errs )

    f_m = np.mean( fracs, axis=0 )
    m_m = np.mean( means,axis=0 )
    e_m = np.mean( errs, axis=0 )
    e_s = np.std( errs, axis=0 )

    f_m = np.squeeze( np.asarray( f_m ) )
    m_m = np.squeeze( np.asarray( m_m ) )
    e_m = np.squeeze( np.asarray( e_m ) )
    e_s = np.squeeze( np.asarray( e_s ) )

    return f_m, m_m, e_m, e_s

def section_error_overall( bench, models ):
    fracs = []
    means = []
    errs = []

    for i,model in enumerate( models ):
        print(i)
        f,m,e = bench.eval_sections_overall(i,model)
        fracs.append(f)
        means.append(m)
        errs.append(e)

    fracs = np.matrix( fracs )
    means = np.matrix( means )
    errs = np.matrix( errs )

    f_m = np.mean( fracs, axis=0 )
    m_m = np.mean( means, axis=0 )
    e_m = np.mean( errs, axis=0 )
    e_s = np.std( errs, axis=0 )

    f_m = np.squeeze( np.asarray( f_m ) )
    m_m = np.squeeze( np.asarray( m_m ) )
    e_m = np.squeeze( np.asarray( e_m ) )
    e_s = np.squeeze( np.asarray( e_s ) )

    return f_m, m_m, e_m, e_s
def actual_vs_predictive_bounds( a, p, std, min_a, max_a ,n ):
    s = ( max_a - min_a ) / (n-1)

    base = []
    base_pol = [];

    means = [];
    std_gp = [];
    std_dist = []
    low = [];
    high = [];

    for i in range( n-2 ):
        ii = i+1
        b0 = (ii-1)*s+min_a
        b1 = (ii+1)*s+min_a
        c = (ii)*s+min_a
        inds = np.where( (a >= b0) & (a<=b1) )[0]

        p_sub = p[inds]
        s_sub = std[inds]

        m = np.mean( p_sub )
        ss = np.std( p_sub )

        pol = [];

        for j in range(2):
            pol.append( c ** j );

        base.append( c )
        base_pol.append( pol )
        means.append(m)
        std_gp.append( np.mean( s_sub ))
        std_dist.append( ss )
        low.append( c-2*ss )
        high.append( c+2*ss )

    base_pol = np.matrix( base_pol )

    base_mat = np.matrix(base).T
    LR_mean = LinearRegression()
    LR_mean.fit( base_pol,means )
    means_p = LR_mean.predict( base_pol )

    LR_low = LinearRegression()
    LR_low.fit( base_pol, low )
    low_p = LR_low.predict( base_pol )

    LR_high = LinearRegression()
    LR_high.fit( base_pol, high )
    high_p = LR_high.predict( base_pol )

    return base, base, low_p, high_p


def actual_vs_predictive_std( a, p, std, min_a, max_a ,n ):
    s = ( max_a - min_a ) / (n-1)

    base = []
    base_pol = [];

    means = [];
    std_gp = [];
    std_dist = []
    low = [];
    high = [];

    for i in range( n-2 ):
        ii = i+1
        b0 = (ii-1)*s+min_a
        b1 = (ii+1)*s+min_a
        c = (ii)*s+min_a
        inds = np.where( (a >= b0) & (a<=b1) )[0]

        p_sub = p[inds]
        s_sub = std[inds]

        m = np.mean( p_sub )
        ss = np.std( p_sub )

        pol = [];

        for j in range(4):
            pol.append( c ** j );

        base.append( c )
        base_pol.append( pol )
        means.append(m)
        std_gp.append( np.mean( s_sub ))
        std_dist.append( ss )
        low.append( m-2*ss )
        high.append( m+2*ss )

    base_pol = np.matrix( base_pol )

    base_mat = np.matrix(base).T
    LR_mean = LinearRegression()
    LR_mean.fit( base_pol,means )
    means_p = LR_mean.predict( base_pol )

    LR_low = LinearRegression()
    LR_low.fit( base_pol, low )
    low_p = LR_low.predict( base_pol )

    LR_high = LinearRegression()
    LR_high.fit( base_pol, high )
    high_p = LR_high.predict( base_pol )

    return base, (high_p - low_p)/4, std_gp

class partitions:
    def __init__(self, ndata, nfolds):
        self.ndata = ndata
        self.nfolds = nfolds
        np.random.seed(8766)
    def n_train(self):
        return len(self.train_parts[0])
    def n_test(self):
        return len(self.test_parts[0])
    def gen_cross_val(self):
        perm = np.random.permutation(self.ndata)
        self.train_parts = [];
        self.test_parts = [];
        for i in range(self.nfolds):
            train = []
            test = []
            for j in range(self.ndata):
                if j % self.nfolds == i:
                    test.append(perm[j])
                else:
                    train.append(perm[j])
            self.train_parts.append(np.array(train))
            self.test_parts.append(np.array(test))
    def gen_rand_splits(self, ratio):
        self.train_parts = []
        self.test_parts = []
        for i in range(self.nfolds):
            perm = np.random.permutation(self.ndata)
            train = []
            test = []
            for j in range(self.ndata):
                r = float(j) / self.ndata
                if r < ratio:
                    train.append(perm[j])
                else:
                    test.append(perm[j])
            self.train_parts.append(np.array(train))
            self.test_parts.append(np.array(test));
    def get_train_part(self, ind):
        return self.train_parts[ind]
    def get_test_part(self, ind):
        return self.test_parts[ind]

class eval_tools:
    def mean_square_error(self, actual, predicted):
        return np.sum(np.power(actual - predicted, 2)) / len(actual)
    def root_mean_square_error(self, actual, predicted ):
        return np.sqrt( np.sum(np.power(actual - predicted, 2)) / len(actual) )
    def mean_absolute_error( self, actual, predicted ):
        return np.sum(np.abs(actual-predicted))/len(actual)
    def delta_t(self,actual,predicted,min_value=-1,max_value=-1,ratio=0.95):
        if min_value == -1 :
            min_value = np.min( actual )
        if max_value == -1 :
            max_value = np.max( actual )
        abs_diff = np.abs( actual - predicted )
        abs_diff = np.sort( abs_diff )
        ind = np.round(len( abs_diff ) * ratio)
        return 2*abs_diff[ind] / ( max_value - min_value )
    def mini_time_window(self, hist, diff, numGroup, max_total):
        max_t = max(diff)
        min_t = min(diff)
        total = sum(hist)
        threshold = round(0.95 * total)
        count = 0
        counter = 0
        for i in range(len(hist)):
            count = hist[i] + count
            if count >= threshold:
                counter = i
                break
        time_interval = 2 * counter * (max_t - min_t) / (numGroup * max_total)
        return time_interval

    def delta_t2(self, actual, predicted):
        diff = abs(actual - predicted)
        numGroup = 10
        histo = np.histogram(diff, numGroup)
        max_total = max(actual) - min(actual)
        mtw = self.mini_time_window(histo[0], diff, numGroup, max_total)
        corrcoef = pearsonr(actual, predicted)
        return mtw

def load_rt_models( path ):
    ff = open( path, 'r' )
    models_data = pk.load( ff )[0]
    ff.close()

    models = []
    kernels = []

    for m in models_data :
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

        #mgp = my_gp( X,Y,pa )
        mod = rt_model( feature, type, mgp, norm,voc,em,y_params )
        ker = rbf_kernel(X,pa)

        models.append( mod )
        kernels.append( ker )

    return models, kernels

class rbf_kernel:
    def __init__( self, X, params ):
        self.X = X
        self.rbf_var = params[0]
        self.rbf_l = params[1]
        self.n_var = params[2]

        self.train_vecs = [];
        for i in range( self.X.shape[0] ):
            row = np.squeeze(np.asarray(self.X[i,:]))
            self.train_vecs.append(row)

    def map( self, vec ):
        v = [];
        for i in range( self.X.shape[0] ):
            row = np.squeeze( np.asarray( self.X[i,:] ) )
            diff = np.sum(np.square(vec,row))
            v.append( self.rbf_var * np.exp( -(0.5*(diff/self.rbf_l) ) ) + self.n_var )
        return np.array(v)

    def min_dist( self, v ):
        values = [];
        for t in self.train_vecs :
            v = np.sqrt(np.sum( np.square( t-v )))
            values.append(v)
        values_sorted = np.sort( values )
        return values_sorted[0]

    def mean_dist( self,v ):
        values = [];
        for t in self.train_vecs :
            v = np.sqrt(np.sum( np.square( t-v )))
            values.append(v)
        return np.mean( values )

class rt_model:
    def __init__(self, feature, model_type, model, norm, voc, em, y_params = []):
        self.feature = feature
        self.model_type = model_type
        self.model = model
        self.norm = norm
        self.voc = voc
        self.em = em
        self.y_params = y_params

    def model_params( self ):
        return [ self.feature, self.model_type, self.model, self.norm,
                 self.voc, self.em, self.y_params ]

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
            print(vals[0])
            print(self.y_params[0])

            vals[0] = vals[0] + self.y_params[0]


        if self.model_type == 'gp':
            res = ( vals[0][0][0], vals[1][0][0] ) # Mean, Variance
        elif self.model_type == 'svr':
            res = ( vals[0], 0)
        return res

class rt_trainer:
    def __init__(self, peptides, feature, model_type, ntrain ):
        self.peptides = peptides
        self.feature = feature
        self.model_type = model_type
        self.ntrain = ntrain

    def train_model( self ):
        train_peptides = self.peptides[:self.ntrain]
        mg = feature_extraction.model_generator(train_peptides)
        voc = mg.get_bow_voc(2)
        em = mg.get_elude_model()

        Y = [];
        X = [];
        for p in train_peptides[0:self.ntrain]:
            Y.append(p.rt)
            if self.feature == 'bow':
                X.append(p.bow_descriptor(voc))
            elif self.feature == 'elude':
                X.append(p.elude_descriptor(em))
        X = np.matrix(X)

        Y_params = [];
        Y = np.transpose(np.matrix(Y))
        norm = feature_extraction.normalizer()

        if self.feature == "elude":
            norm.normalize_maxmin(X);
        gpy_model = GPy.models.GPRegression(X, Y)
        gpy_model.optimize_restarts(num_restarts=10, verbose=False)
        pa = list(gpy_model.param_array)
        gpy_model = None
        return [ self.feature, self.model_type, X, Y, pa , norm,voc,em, Y_params ];

class rt_benchmark:
    def __init__(self, peptides, feature, model_type, ntrain=-1, nfolds=5,tratio=0.8):
        self.peptides = peptides
        self.feature = feature
        self.model_type = model_type
        self.ntrain = ntrain
        self.parts = partitions(len(peptides), nfolds)
        self.parts.gen_rand_splits(tratio)
        # self.parts.gen_cross_val();

        if ntrain < 0 or ntrain > self.parts.n_train():
            ntrain = self.parts.n_train()

    def train_features( self, ind ):
        train_peptides = self.peptides[self.parts.get_train_part(ind)]
        Y = []
        X = []
        for p in train_peptides[0:self.ntrain]:
            Y.append(p.rt)
            if self.feature == 'bow':
                X.append(p.bow_descriptor(voc))
            elif self.feature == 'elude':
                X.append(p.elude_descriptor(em))
        X = np.matrix(X)
        Y = np.transpose(np.matrix(Y))

        return X,Y

    def train_model(self, ind):
        train_peptides = self.peptides[self.parts.get_train_part(ind)]
        mg = feature_extraction.model_generator(train_peptides)
        voc = mg.get_bow_voc(2)
        em = mg.get_elude_model()

        Y = [];
        X = [];
        for p in train_peptides[0:self.ntrain]:
            Y.append(p.rt)
            if self.feature == 'bow':
                X.append(p.bow_descriptor(voc))
            elif self.feature == 'elude':
                X.append(p.elude_descriptor(em))
        X = np.matrix(X)

        Y_params = [];
        if self.model_type == 'gp':
            Y = np.transpose(np.matrix(Y))
            Y_mean = np.mean(Y)
            Y_params.append( Y_mean )
            Y = Y - Y_mean
        elif self.model_type == 'svr':
            Y = np.transpose(Y)

        norm = feature_extraction.normalizer()

        if self.feature == "elude":
            norm.normalize_maxmin(X);
        if self.model_type == "gp":
            gpy_model = GPy.models.GPRegression(X, Y)
            gpy_model.optimize_restarts(num_restarts=10, verbose=False)
            m = my_gp(X,Y,gpy_model.param_array)
        elif self.model_type == "svr":
            m = svm.SVR(C=600, gamma=0.1, coef0=0.0, degree=3, epsilon=0.1, kernel='rbf', max_iter=-1, shrinking=True,
                        tol=0.001, verbose=False)
            m = grid_search.GridSearchCV(m, param_grid={"C": np.linspace(100, 1000, num=10),
                                                        "gamma": np.linspace(0.01, 10, num=100)})
            m.fit(X, Y)
        return rt_model(self.feature, self.model_type, m, norm, voc, em, Y_params )

    def train_gp_model( self, ind ):
        assert self.model_type == 'gp'
        train_peptides = self.peptides[self.parts.get_train_part(ind)]
        mg = feature_extraction.model_generator(train_peptides)
        voc = mg.get_bow_voc(2)
        em = mg.get_elude_model()

        Y = [];
        X = [];
        for p in train_peptides[0:self.ntrain]:
            Y.append(p.rt)
            if self.feature == 'bow':
                X.append(p.bow_descriptor(voc))
            elif self.feature == 'elude':
                X.append(p.elude_descriptor(em))
        X = np.matrix(X)

        Y_params = [];
        Y = np.transpose(np.matrix(Y))
        #Y_mean = np.mean(Y)
        #Y_params.append( Y_mean )
        #exY = Y - Y_mean

        norm = feature_extraction.normalizer()

        if self.feature == "elude":
            norm.normalize_maxmin(X);
        gpy_model = GPy.models.GPRegression(X, Y)
        gpy_model.optimize_restarts(num_restarts=10, verbose=False)
        pa = list(gpy_model.param_array)
        gpy_model = None
        return [ self.feature, self.model_type, X, Y, pa , norm,voc,em, Y_params ];

    def train_multi_model( self, ind, nmodels ):
        assert self.model_type == 'gp'

        train_peptides = self.peptides[self.parts.get_train_part(ind)]
        mg = feature_extraction.model_generator(train_peptides)
        voc = mg.get_bow_voc(2)
        em = mg.get_elude_model()

        models = [];

        for i in range( nmodels ):
            perm = np.random.permutation( len(train_peptides ) )
            Y = [];
            X = [];
            for rind in perm[0:self.ntrain]:
                p = train_peptides[rind]
                Y.append(p.rt)
                if self.feature == 'bow':
                    X.append(p.bow_descriptor(voc))
                elif self.feature == 'elude':
                    X.append(p.elude_descriptor(em))
            X = np.matrix(X)
            Y = np.transpose(np.matrix(Y))

            print(X.shape)

            norm = feature_extraction.normalizer()

            if self.feature == "elude":
                norm.normalize_maxmin(X);
            m = GPy.models.GPRegression(X, Y)
            m.optimize_restarts(num_restarts=10, verbose=False)
            models.append( rt_model(self.feature, self.model_type, m, norm, voc, em) )
        return models

    def predict(self, ind, model):
        test_peptides = self.peptides[self.parts.get_test_part(ind)]
        actual = []
        predicted = []
        var = []
        for p in test_peptides:
            actual.append(p.rt)
            m, v = model.eval(p) # This predicts the mean and the variance
            predicted.append(m)
            var.append(v)
        actual = np.array(actual)
        predicted = np.array(predicted)
        var = np.array(var)
        return actual, predicted, var

    def predict_train( self, ind, model ):
        train_peptides = self.peptides[self.parts.get_train_part(ind)]
        train_peptides = train_peptides[0:self.ntrain]

        actual = []
        predicted = []
        std = []
        for p in train_peptides:
            actual.append(p.rt)
            v, s = model.eval(p)
            predicted.append(v)
            std.append(s)
        actual = np.array(actual)
        predicted = np.array(predicted)
        std = np.array(std)
        return actual, predicted, std

    def predict_multi_model( self, ind, models ):
        test_peptides = self.peptides[self.parts.get_test_part(ind)]
        actual = []
        predicted = []
        std = []
        for p in test_peptides :
            actual.append(p.rt)

            vs = [];
            ss = [];
            for m in models :
                v,s = m.eval(p)
                vs.append( v )
                ss.append( s )

            vs = np.array( vs )
            ss = np.array( ss )

            inds = np.argsort( ss )

            predicted.append( np.mean(vs[inds[0:3]] ) )
            std.append( np.mean(ss[inds[0:3]] ) )
        actual = np.array(actual)
        predicted = np.array(predicted)
        std = np.array(std)
        return actual,predicted,std

    def hist_eval( self, ind, model, pp ):
        train_actual, train_predicted, train_std = self.predict_train(0,model)
        test_actual, test_predicted, test_std = self.predict(0,model)

        train_f, train_m, train_d = self.test_sorted2( train_actual, train_predicted, train_std )
        test_f, test_m, test_d = self.test_sorted2( test_actual, test_predicted, test_std )

        test_abs = np.abs( test_actual - test_predicted )
        #train_abs = np.abs( train_actual - train_predicted )

        inds = np.argsort( test_std )
        #for i in range(nsec):
        #    v += q;
        #    if i < r :
        #        v += 1;
        #    pp.subplot(nsec,i,)
        #    o += v


        pp.figure()
        pp.plot( test_m, test_d,'.-' )
        pp.xlabel('Predicted Standard Deviation')
        pp.ylabel('Root Mean Square Error')
        pp.grid()
        #pp.hexbin( np.log( test_std ), test_abs, cmap=plt.cm.YlOrRd_r,bins='log' )
        #pp.plot( np.log( test_std ) , test_abs, 'b.' )
        #pp.plot( np.log( train_std ), train_abs, 'r.' )
        #pp.plot( train_m, train_d,'r' )
        #pp.plot( test_m, test_d,'b' )

    def eval_model(self, ind, model):
        actual, predicted, std = self.predict(ind, model)
        et = eval_tools()
        return et.delta_t(actual, predicted),et.root_mean_square_error(actual,predicted)

    def eval_multi_model(self,ind, model ):
        actual, predicted, std = self.predict_multi_model(ind,model)
        et = eval_tools()
        return et.delta_t(actual, predicted),et.mean_square_error(actual,predicted),et.mean_absolute_error(actual,predicted)

    def test_k(self, ind, model):
        actual, predicted, std = self.predict(ind, model)
        et = eval_tools()
        print(et.delta_t(actual, predicted))

    def save_scores( self, ind , model ):
        actual, predicted, std = self.predict(ind, model)
        with open('scores.pk','w') as ff :
            pk.dump( [ actual, predicted, std ], ff )
            ff.close()

    def eval_sections_independent( self, ind, model, nsec=10 ):
        actual, predicted, var = self.predict(ind,model)
        std = np.sqrt( var )

        inds = np.argsort(std)
        q = len(inds)/nsec
        r = len(inds)%nsec

        pos = [];
        v = 0;

        min_a = np.min( actual )
        max_a = np.max( actual )

        et = eval_tools()

        fraction = [];
        err = []
        means = [];

        o = 0

        for i in range(nsec):
            v += q;
            if i < r :
                v += 1;

            a = actual[ inds[o:v ] ]
            a_part = actual[ inds[o:v] ]
            p = predicted[ inds[o:v] ]
            s = std[ inds[o:v] ]

            o = v

            fraction.append( float(i+1)/nsec )
            means.append( np.mean(s) )
            err.append( np.sqrt( et.mean_square_error(a,p) ) )

        return np.array( fraction ), np.array( means ), np.array( err )

    def eval_sections_interval( self, ind, model, nsec=10 ):
        a, p, v = self.predict(ind,model)
        s = np.sqrt(v)
        inds = np.argsort(s)
        chunks = chunk_it( inds, nsec )

        e = a - p
        n, min_max, mean, var, skew, kurt = stats.describe(e)
        std = np.sqrt( var )
        I = stats.norm.interval(0.95,loc=mean,scale=std)
        print(I)

        fraction = []
        means = []
        percentage = []

        for i,c in enumerate(chunks) :
            ce = e[c]

            inds0 = np.where( ce <= I[0] )[0]
            inds1 = np.where( ce >= I[1] )[0]
            p = (len(inds0) + len(inds1))/float(len(c)) * 100

            fraction.append( (i+1)/float(nsec) )
            means.append( np.mean( s[c] ) )
            percentage.append( p )

        print( percentage )

        return np.array( fraction ), np.array( means ), np.array( percentage )


    def eval_sections_overall( self, ind, model, nsec=10 ):
        actual, predicted, std = self.predict(ind,model)
        inds = np.argsort(std)
        q = len(inds)/nsec
        r = len(inds)%nsec

        pos = [];
        v = 0;

        min_a = np.min( actual )
        max_a = np.max( actual )

        et = eval_tools()

        fraction = [];
        err = []
        means = [];

        o = 0

        for i in range(nsec):
            v += q;
            if i < r :
                v += 1;

            a = actual[ inds[0:v ] ]
            a_part = actual[ inds[o:v] ]
            p = predicted[ inds[0:v] ]
            s = np.sqrt( std[ inds[0:v] ] )

            fraction.append( float(i+1)/nsec )
            means.append( np.mean(s) )
            err.append( np.sqrt( et.mean_square_error(a,p) ) )

        return np.array( fraction ), np.array( means ), np.array( err )

    def get_test_vectors( self, ind, model, nvec=1000 ):
        test_peptides = self.peptides[self.parts.get_test_part(ind)]
        actual, predicted, var = self.predict(ind,model)

        if nvec > len( test_peptides ):
            nvec = len( test_peptides )

        vectors = []
        for i in range( nvec ):
            vec = model.get_vector( test_peptides[i] )
            vectors.append( [vec,actual[i],predicted[i],var[i]] )
        return vectors

    def test_sorted( self, ind, model ):
        actual, predicted, std = self.predict(ind, model)
        inds = np.argsort(std)

        nsec = 10
        q = len(inds)/nsec
        r = len(inds)%nsec

        pos = [];
        v = 0;

        min_a = np.min( actual )
        max_a = np.max( actual )

        et = eval_tools()

        fraction = [];
        delta_t = []
        means = [];
        hists = [];

        hist, bin_edges = np.histogram( actual,50 )
        o = 0

        for i in range(nsec):
            v += q;
            if i < r :
                v += 1;

            a = actual[ inds[0:v ] ]
            a_part = actual[ inds[o:v] ]
            p = predicted[ inds[0:v] ]
            s = std[ inds[0:v] ]

            o = v

            h, be = np.histogram( a_part, bin_edges )

            h = np.array(h,dtype=np.float )
            h /= np.sum(h)

            fraction.append( float(i+1)/nsec )
            means.append( np.mean(s) )
            delta_t.append( et.delta_t(a, p,min_a,max_a) )
            hists.append(h)

        return np.array( fraction ), np.array( means ), np.array( delta_t )

    def test_sorted2( self, actual, predicted, std ):
        inds = np.argsort(std)

        nsec = 10
        q = len(inds)/nsec
        r = len(inds)%nsec

        pos = [];
        v = 0;

        min_a = np.min( actual )
        max_a = np.max( actual )

        et = eval_tools()

        fraction = [];
        delta_t = []
        means = [];
        hists = [];

        hist, bin_edges = np.histogram( actual,50 )
        o = 0

        for i in range(nsec):
            v += q;
            if i < r :
                v += 1;

            a = actual[ inds[o:v ] ]
            a_part = actual[ inds[o:v] ]
            p = predicted[ inds[o:v] ]
            s = np.sqrt( std[ inds[o:v] ] )

            o = v

            h, be = np.histogram( a_part, bin_edges )

            h = np.array(h,dtype=np.float )
            h /= np.sum(h)

            fraction.append( float(i+1)/nsec )
            means.append( np.mean(s) )
            delta_t.append( np.sqrt( et.mean_square_error(a,p) ) )
            #delta_t.append( et.delta_t(a, p,min_a,max_a) )
            hists.append(h)

        return np.array( fraction ), np.array( means ), np.array( delta_t )
