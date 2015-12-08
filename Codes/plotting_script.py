#!/usr/bin/python

import platform
if platform.system() != 'Darwin':
    import matplotlib as mpl
    mpl.use('Agg')
import numpy as np
import data_tools
import pickle as pk
import ml_tools
import gp_tools
from matplotlib import pyplot as pp
pp.ion()
from sklearn.linear_model import LinearRegression
from common import parameters

class data_plotter:
    def __init__( self,n=100 ):
        self.params = parameters()
        self.n = n
    def load_data( self ):
        path = self.params.save_tmp % ( self.params.data_root, self.params.models_tag, self.n )
        self.peptides = data_tools.read_data( self.params.data_path )
        self.benchmark = ml_tools.rt_benchmark(self.peptides, 'elude', 'gp', self.n, self.params.nparts, self.params.train_ratio )
        self.models, self.kernels = ml_tools.load_rt_models( path )
    def rt_range( self ):
        rt = [];
        for p in self.peptides : 
            rt.append( p.rt )
        return np.min( rt ), np.max( rt )
    def get_test_data( self, pind=0 ):
        a,p,v = self.benchmark.predict( pind, self.models[pind] )
        test_peptides = self.peptides[ self.benchmark.parts.get_test_part(pind) ]

        ff = open('../tmp/test_peptides.txt','w')
        for i,t in enumerate( test_peptides ) : 
            ff.write('%s,%g,%g\n' % (t.sequence, a[i], p[i] ))
        ff.close()

    def benchmark_gp( self ):
        drt_values = []
        rmse_values = []

        for pind in range( self.benchmark.parts.nfolds ):
            [ drt, rmse ] = self.benchmark.eval_model(pind,self.models[pind])
            drt_values.append( drt )
            rmse_values.append( rmse )
        
        #drt_mean = np.mean( drt_values )
        #drt_std = np.std( drt_values )
        #rmse_mean = np.mean( rmse_values )
        #rmse_std = np.std( rmse_values )

        return [ self.n, drt_values, rmse_values ];

    def svr_vs_gp( self ):
        n = [ 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000 ];
        svr_m = [0.3386, 0.2953, 0.2846, 0.2725, 0.2606, 0.2551, 0.2510, 0.2472, 0.2459, 0.2427, 0.2283, 0.2212];
        svr_s = [0.0140, 0.0082, 0.00973, 0.0085, 0.0045, 0.0040, 0.0067, 0.0064, 0.0049, 0.0044, 0.0053,0.0036 ];
        gp_m = [0.3382, 0.3030, 0.2875, 0.2755, 0.2668, 0.2605, 0.2537, 0.2509, 0.2481, 0.2443, 0.2265, 0.2188 ];

        pp.figure()

        pp.plot( n,svr_m, 'r-' )
        pp.plot( n,gp_m, 'b-' )

        pp.legend(['Support Vector Regression','Gaussian Process'])
        pp.xlabel('Size of training set')
        pp.ylabel('Delta RT 0.95')
        pp.title('GP vs. SVR')
        pp.grid()
        pp.savefig('gp_vs_svr.pdf')

    def dist_vs_std( self, pind=0 ):
        test_vectors = self.benchmark.get_test_vectors(pind,self.models[pind])
        dist = [];
        std = [];

        for t in test_vectors : 
            d = self.kernels[pind].min_dist( t[0] )
            dist.append( d )
            std.append(np.sqrt(t[3]))

        dist = np.array( dist )
        std = np.array( std )

        pp.figure()
        pp.plot( dist, std, 'r.')
        pp.grid()
        pp.xlabel('Distance From Training Data')
        pp.ylabel('Predicted Standard Deviation')
        pp.title('Distance vs. Standard Deviation')
        pp.savefig('dist_vs_std.pdf')

    def section_error_independent( self ):
        f_m, m_m, e_m, e_v = ml_tools.section_error_independent( self.benchmark, self.models )

        pp.figure()
        pp.plot( m_m, e_m,'.-' )
        pp.fill_between( m_m, e_m-e_v , e_m+e_v, alpha=0.8, facecolor='0.75' )

        [ x1,x2,y1,y2 ] = pp.axis();
        pp.axis( [ np.min(m_m), np.max(m_m), y1,y2 ])

        pp.xlabel('Predicted Standard Deviation')
        pp.ylabel('Root Mean Square Error')
        pp.title('Sections Error')
        pp.grid()
        pp.savefig('ind_section.pdf')

    def section_error_overall( self ):
        f_m, m_m, e_m, e_v = ml_tools.section_error_overall( self.benchmark, self.models )

        pp.figure()
        pp.plot( f_m, e_m,'.-' )
        pp.fill_between( f_m, e_m-e_v , e_m+e_v, alpha=0.8, facecolor='0.75' )

        [ x1,x2,y1,y2 ] = pp.axis();
        pp.axis( [ np.min(f_m), np.max(f_m), y1,y2 ])

        pp.xlabel('Data Fraction')
        pp.ylabel('Root Mean Square Error')
        pp.title('Cumulative Overall Error')
        pp.grid()
        pp.savefig('cum_overall.pdf')

    def actual_vs_predicted( self, pind=0 ):
        a,p,v = self.benchmark.predict( pind, self.models[pind] )
        s = np.sqrt(v)
        rt_min,rt_max = self.rt_range()
        base, means_p,low_p, high_p = ml_tools.actual_vs_predictive_bounds(a,p,s,rt_min,rt_max,50)

        pp.figure()
        pp.plot(a,p,'r.')
        pp.plot(base,means_p,'k-',linewidth=2.0)
        pp.plot(base,low_p,'k--',linewidth=2.0)
        pp.plot(base,high_p,'k--',linewidth=2.0)
        pp.xlabel('Actual RT')
        pp.ylabel('Predicted RT')
        pp.title('Actual vs Predicted RT')
        pp.grid()
        pp.savefig('actual_vs_predicted.pdf')

    def std_histogram( self, pind=0 ):
        a,p,v = self.benchmark.predict( pind, self.models[pind] )
        rt_min,rt_max = self.rt_range()
        s = np.sqrt(v)
        inds = np.argsort( s )
        
        nsec=50
        nbins=49

        sec = len( inds )/nsec
        chunks = zip( *[iter(inds)]*sec )

        loc_bins = [];
        rt_sec = (rt_max - rt_min)/(nbins+1)

        for i in range( nbins+1 ):
            loc_bins.append( i*rt_sec + rt_min + rt_sec/2 ) 

        mat = np.zeros((nsec,nbins+1)) 

        std_bins = [];

        for i,c in enumerate( chunks ) :
            inds = []
            for j in c : 
                inds.append(j)
            a_sec = a[ inds ]
            s_sec = s[ inds ]

            std_bins.append( np.mean( s_sec ) )

            h = np.histogram( a_sec, loc_bins );
            h = np.array( h[0],dtype=float )
            h = h / np.sum(h)

            for j,v in enumerate( h ) :
                mat[i,j] = v

        print(len(loc_bins))
        print(len(std_bins))
        print(mat.shape)

        np.savetxt('x_loc_bins.txt',loc_bins)
        np.savetxt('y_std_bins.txt',std_bins)
        np.savetxt('mat.txt',mat)

        #pp.matshow( mat )# interpolation='nearest', extent=[xmin, xmax, ymin, ymax],origin='lower' )
        #pp.ylabel('Predictive STD')
        #pp.xlabel('Retention Time')

    def actual_vs_predicitive_std( self ):
        rt_min,rt_max = self.rt_range()

        base = [];

        actual = [];
        predicitive = [];
    
        for pind in range( self.benchmark.parts.nfolds ):
            print(pind)
            a,p,v = self.benchmark.predict( pind, self.models[pind] )
            s = np.sqrt(v)
            b, a_std, p_std  = ml_tools.actual_vs_predictive_variance( a,p,s,rt_min,rt_max,50 )

            base = b;
            actual.append( a_std )
            predicitive.append( p_std )
            

        actual = np.matrix( actual )
        predicitive = np.matrix( predicitive )

        av_m = np.mean( actual, axis=0 )
        av_s = np.std( actual, axis=0 )
        pv_m = np.mean( predicitive, axis=0 )
        pv_s = np.std( predicitive, axis=0 )

        av_m = np.array( np.squeeze( np.asarray( av_m ) ) )
        av_s = np.array( np.squeeze( np.asarray( av_s ) ) )
        pv_m = np.array( np.squeeze( np.asarray( pv_m ) ) )
        pv_s = np.array( np.squeeze( np.asarray( pv_s ) ) )

        pp.figure()

        pp.plot( base, av_m, 'r' )
        pp.fill_between( base, av_m-av_s , av_m+av_s, alpha=0.8, facecolor='0.75' )

        pp.plot( base, pv_m, 'b' )
        pp.fill_between( base, pv_m-pv_s , pv_m+pv_s, alpha=0.8, facecolor='0.75' )

        [ x1,x2,y1,y2 ] = pp.axis();
        pp.axis( [ np.min(base), np.max(base), y1,y2 ])
        pp.grid()

        pp.xlabel('Actual RT')
        pp.ylabel('Variance')

        pp.legend(['Actual Variance','Predicitive Variance'],loc=4)

        pp.savefig('actual_vs_predicted_variance.pdf')
