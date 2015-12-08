#!/usr/bin/python

import numpy as np
from matplotlib import pyplot as pp
pp.ion()

class my_gp:
    def __init__( self, X, Y, params ):
        self.X = X;
        self.Y = Y;
        self.rbf_var = params[0]
        self.rbf_l = params[1]**2
        self.n_var = params[2]
        self.ker = self.rbf_kernel( self.X, self.X )
        n = self.ker.shape[0]
        self.k_inv = np.linalg.inv( self.ker + np.eye(n) * self.n_var )

    def rbf_kernel( self, X1, X2 ):
        ker = np.zeros( (X1.shape[0],X2.shape[0]) )
        for i in range( X1.shape[0] ):
            for j in range( X2.shape[0] ):
                diff = np.sum(np.square( X1[i,:] - X2[j,:]))
                ker[i,j] = self.rbf_var * np.exp( -(0.5*(diff/self.rbf_l) ) )
        return ker

    def predict( self, x ):
        mat_x = np.matrix(x)

        k_self = np.matrix( self.rbf_kernel( mat_x,mat_x ) )
        k_basis = np.matrix( self.rbf_kernel( self.X, mat_x ) )

        Y = np.matrix( self.Y ) 
        y_bar = k_basis.T * self.k_inv * Y;
        y_var = k_self - k_basis.T * self.k_inv * k_basis + self.n_var;

        return y_bar[0,0], y_var[0,0]

    def map_feature( self, x ):
        mat_x = np.matrix(x)
        n = self.ker.shape[0]
        k_self = np.matrix( self.rbf_kernel( mat_x,mat_x ) )
        return k_self

    def eval_components( self, x ):
        mat_x = np.matrix(x)
        n = self.ker.shape[0]

        k_self = np.matrix( self.rbf_kernel( mat_x,mat_x ) )
        k_basis = np.matrix( self.rbf_kernel( self.X, mat_x ) )

        Y = np.matrix( self.Y ) 

        m = k_basis.T * k_inv * Y
        v1 = k_self
        v2 = k_basis.T * self.k_inv * k_basis
        v3 = self.n_var
        return m[0,0], v1[0,0], v2[0,0], v3
 
    def eval_array( self, values ):
        means = [];
        stds = [];
        for x in values :
            b,v = self.eval(x)
            means.append(b)
            stds.append(v)

        return np.array(means),np.array(stds)

    def eval_components_array( self, values ):
        m = [];
        v1 = [];
        v2 = [];
        v3 = [];

        for x in values : 
            a1,a2,a3,a4 = self.eval_components(x)
            m.append(a1)
            v1.append(a2)
            v2.append(a3)
            v3.append(a4)

        return np.array(m), np.array(v1), np.array(v2), np.array(v3)

    def plot_test_data( self, x ):
        m,v = self.eval_array( x )

        min_x = np.min( x )
        max_x = np.max( x )

        pp.figure()

        pp.subplot(2,1,1)
        plt1 = pp.plot( x, m, label="GP Prediction" );
        pp.fill_between( x, m-np.sqrt(v)*2, m+np.sqrt(v)*2, alpha=0.5, color ='0.75', label="Conf. Interval")
        plt2 = pp.plot( self.X, self.Y,'r.',label="Training Points" )

        [ x1,x2,y1,y2 ] = pp.axis()
        pp.axis( [ min_x, max_x, y1,y2 ])
        pp.grid()
        pp.legend(loc=2)

        pp.ylabel('Y Axis')

        pp.subplot(2,1,2)
        pp.plot( x,v )
        [ x1,x2,y1,y2 ] = pp.axis()
        pp.axis( [ min_x, max_x, y1,y2 ])
        pp.grid()

        pp.axis( )

        pp.ylabel('GP Prediction Variance')
        pp.xlabel('X Axis')



        
