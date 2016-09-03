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
from scipy import stats
import math
from matplotlib import pyplot as pp
pp.ion()
from sklearn.linear_model import LinearRegression
from common import parameters

def chunk_it( seq, num ):
    avg = len( seq ) / float( num )
    out = []
    last = 0.0
    while last < len( seq ):
        out.append( seq[int(last):int(last+avg)] )
        last += avg
    return out

def error_percent( a,p,s ):
    e = np.abs( a - p )
    inds = np.where( e >= s*1.96 )[0]
    return float(len(inds))/float(len(e))

class data_plotter:
    def __init__( self,n=3000 ):
        self.params = parameters()
        self.n = n
    def load_data( self ):
        path = self.params.save_tmp % ( self.params.data_root, self.params.models_tag, self.n )
        self.peptides = data_tools.read_data( self.params.data_path )
        self.benchmark = ml_tools.rt_benchmark(self.peptides, 'elude', 'gp', self.n, self.params.nparts, self.params.train_ratio )
        self.models, self.kernels = ml_tools.load_rt_models( path )

    def calculate_all_a_p_v( self ):
        self.values = []
        for i in range( 10 ):
            print(i)
            a,p,v = self.benchmark.predict(i,self.models[i])
            self.values.append([a,p,v])
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

        pp.plot( n,svr_m, 'r-', linewidth=2 )
        pp.plot( n,gp_m, 'b-', linewidth=2 )

        pp.legend(['Support Vector Regression','Gaussian Process'])
        pp.xlabel('Size of training set',fontsize=18)
        pp.ylabel(r'$w_r^{95\%}$',fontsize=18)
        #pp.title('GP vs. SVR',fontsize=20)
        pp.grid()
        pp.savefig('./plots/gp_vs_svr.pdf')

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
        pp.plot( std, dist, 'r.')
        pp.grid()
        pp.ylabel('Distance From Training Data',fontsize=18)
        pp.xlabel('Predicted Standard Deviation',fontsize=18)
        #pp.title('Distance vs. Standard Deviation',fontsize=20)
        pp.savefig('./plots/dist_vs_std.pdf')

    def section_error_independent( self, over_all_err=None ):
        def eval_sections_independent( values, nsec=10 ):
            et = ml_tools.eval_tools()
            a,p,v = values
            s = np.sqrt(v)

            inds = np.argsort( s )
            chunks = chunk_it( inds, nsec )

            fraction = []
            err = []
            means = []

            for i,c in enumerate( chunks ):
                ca = a[ c ]
                cp = p[ c ]
                cs = s[ c ]

                fraction.append( float(i+1)/nsec )
                means.append( np.mean(cs) )
                err.append( np.sqrt( et.mean_square_error(ca,cp) ) )

            return np.array( fraction ), np.array( means ), np.array( err )

        fracs = []
        means = []
        errs = []

        for i in range(len(self.values)):
            f,m,e = eval_sections_independent( self.values[i] )
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

        pp.figure()
        pp.plot( m_m, e_m,'.-',label='Section RMSE' )
        pp.fill_between( m_m, e_m-e_s , e_m+e_s, alpha=0.8, facecolor='0.75' )
        if over_all_err is not None :
            pp.plot( m_m, np.ones(m_m.shape)*over_all_err, 'r-.',linewidth=2,label='Overall RMSE')

        [ x1,x2,y1,y2 ] = pp.axis();
        pp.axis( [ np.min(m_m), np.max(m_m), y1,y2 ])

        pp.xlabel('Average Predicted Standard Deviation',fontsize=18)
        pp.ylabel('Root Mean Square Error (Minutes)',fontsize=18)
        #pp.title('Sections Error',fontsize=20)
        pp.grid()
        if over_all_err is not None :
            pp.legend(loc=2)
        pp.savefig('./plots/ind_section.pdf')

    def section_error_independent_interval( self, over_all_err=None ):
        f_m, m_m, e_m, e_v = ml_tools.section_error_interval( self.benchmark, self.models )

        pp.figure()
        pp.plot( m_m, e_m,'.-',label='Section RMSE' )
        pp.fill_between( m_m, e_m-e_v , e_m+e_v, alpha=0.8, facecolor='0.75' )
        if over_all_err is not None :
            pp.plot( m_m, np.ones(m_m.shape)*over_all_error, 'r-.',linewidth=2,label='Overall RMSE')

        [ x1,x2,y1,y2 ] = pp.axis();
        pp.axis( [ np.min(m_m), np.max(m_m), y1,y2 ])

        pp.xlabel('Average Predicted Standard Deviation',fontsize=18)
        pp.ylabel(r'$p_r^{5\%}$ (Percentage)',fontsize=18)
        #pp.title('Sections Error',fontsize=20)
        pp.grid()
        pp.savefig('./plots/ind_section_interval.pdf')

    def section_error_overall( self ):
        def eval_sections_overall( values, nsec=10 ):
            et = ml_tools.eval_tools()
            a,p,v = values
            s = np.sqrt(v)

            inds = np.argsort( s )
            chunks = chunk_it( inds, nsec )

            fraction = []
            err = []
            means = []

            for i in range( len(chunks) ):
                sub_chunks = chunks[:i+1]
                selection = np.concatenate( sub_chunks )

                sa = a[ selection ]
                sp = p[ selection ]
                ss = s[ selection ]

                fraction.append( float(i+1)/nsec )
                means.append( np.mean(ss) )
                err.append( np.sqrt( et.mean_square_error(sa,sp) ) )
            return np.array( fraction ), np.array( means ), np.array( err )

        fracs = []
        means = []
        errs = []

        for i in range(len(self.values)):
            f,m,e = eval_sections_overall( self.values[i] )
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

        pp.figure()
        pp.plot( f_m, e_m,'.-' )
        pp.fill_between( f_m, e_m-e_s , e_m+e_s, alpha=0.8, facecolor='0.75' )

        [ x1,x2,y1,y2 ] = pp.axis();
        pp.axis( [ np.min(f_m), np.max(f_m), y1,y2 ])

        pp.xlabel('Fraction of peptides',fontsize=18)
        pp.ylabel('Root Mean Square Error (Minutes)',fontsize=18)
        #pp.title('Cumulative Overall Error',fontsize=20)
        pp.grid()
        pp.savefig('./plots/cum_overall.pdf')

        return e_m[-1]

    def actual_vs_predicted( self, pind=0 ):
        a,p,v = self.values[0]
        s = np.sqrt(v)
        rt_min,rt_max = self.rt_range()
        base, means_p,low_p, high_p = ml_tools.actual_vs_predictive_bounds(a,p,s,rt_min,rt_max,50)

        pp.figure()
        pp.plot(a,p,'r.',label='Data Points')
        pp.plot(base,means_p,'k-',linewidth=2.0,label='x=y')
        #pp.plot(base,low_p,'k--',linewidth=2.0)
        #pp.plot(base,high_p,'k--',linewidth=2.0)
        pp.xlabel('Actual RT (Minutes)',fontsize=18)
        pp.ylabel('Predicted RT (Minutes)',fontsize=18)
        #pp.title('Actual vs Predicted RT',fontsize=20)
        pp.grid()
        pp.legend(loc=2)
        pp.savefig('./plots/actual_vs_predicted.pdf')

    def get_predictions( self, pind=0 ):
        a,p,v = self.benchmark.predict( pind, self.models[pind] )
        s = np.sqrt(v)
        e = np.sqrt( np.abs(a - p) )

        return s,e

    def std_histogram( self, pind=0 ):
        a,p,v = self.values[0]
        s = np.array(np.sqrt(v))

        inds = np.argsort( s )
        chunks = chunk_it( inds, 10 )

        std_aves = []
        std_bounds = []

        for c in chunks :
            std_aves.append( np.mean(s[c]) )
            std_bounds.append( [ np.min(s[c]), np.max(s[c]) ] )

        std_aves = np.array( std_aves )
        std_bounds = np.array( std_bounds )

        print( std_aves )
        print( std_bounds )

        rt_inds = np.argsort( a )
        rt_chunks = chunk_it( rt_inds, 10 )

        rt_aves = []
        rt_bounds = []
        for c in rt_chunks :
            rt_aves.append( np.mean( a[c] ) )
            rt_bounds.append( [ np.min(a[c]), np.max(a[c]) ] )

        rt_aves = np.array( rt_aves )
        rt_bounds = np.array( rt_bounds )

        print( rt_aves )
        print( rt_bounds )

        mat = np.zeros( ( len(rt_aves), len(std_aves) ) )

        for i,c in enumerate( rt_chunks ):
            chunk_stds = s[c]
            for v in chunk_stds :
                selection = -1
                for j,b in enumerate( std_bounds ):
                    if v >= b[0] and v <= b[1] :
                        selection = j
                        break
                if selection >= 0 :
                    mat[i,j] += 1
            mat[i,:] = mat[i,:] / np.sum( mat[i,:] )

        xmin = np.min( std_aves )
        xmax = np.max( std_aves )
        ymin = np.min( rt_aves )
        ymax = np.max( rt_aves )

        #pp.close('all')
        pp.figure()
        pp.set_cmap('hot_r')
        pp.matshow( mat, origin='lower', extent=[xmin, xmax, ymin, ymax ], aspect='auto' )
        #, interpolation='nearest', extent=[xmin, xmax, ymin, ymax],origin='lower' )
        pp.colorbar()
        pp.xlabel('Average Predicted Standard Deviation',fontsize=18)
        pp.ylabel('Retention Time (Minutes)',fontsize=18)
        pp.savefig('./plots/selection_histogram.pdf')

    def std_histogram_old( self ):
        a,p,v = self.values[0]
        rt_min,rt_max = self.rt_range()
        s = np.sqrt(v)
        inds = np.argsort( s )

        nsec=50
        nbins=49

        sec = len( inds )/nsec
        chunks = zip( *[iter(inds)]*sec )

        loc_bins = [];ggffdf
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

            for j,val in enumerate( h ) :
                mat[i,j] = val


        xmin = np.min(s)
        xmax = np.max(s)

        mat = mat / np.max( mat )

        mat = mat.T

        pp.close('all')
        pp.figure()
        pp.set_cmap('hot_r')
        pp.matshow( mat, origin='lower', extent=[xmin, xmax, rt_min, rt_max ], aspect='auto' )
        #, interpolation='nearest', extent=[xmin, xmax, ymin, ymax],origin='lower' )
        pp.colorbar()
        pp.xlabel('Predicted Standard Deviation',fontsize=18)
        pp.ylabel('Retention Time (Minutes)',fontsize=18)
        #pp.title('Peptide PSTD Concentration',fontsize=20)
        pp.savefig('./plots/selection_histogram.pdf')
        #np.savetxt('./data/x_loc_bins.txt',loc_bins)
        #np.savetxt('./data/y_std_bins.txt',std_bins)
        #np.savetxt('./data/mat.txt',mat)

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
        pp.legend(['Actual Variance','Predicted Variance'],loc=4)
        pp.savefig('actual_vs_predicted_variance.pdf')

    def false_negative_ratio( self ):
        all_pstd = []
        all_err0 = []
        all_err1 = []

        for model_ind in range( len(self.values) ):
            a,p,v = self.values[ model_ind ]
            s = np.sqrt( v )
            e = np.abs( a - p )
            inds = np.argsort(s)
            chunks = chunk_it( inds, 10 )

            thresh = np.mean(s) * 1.96
            num = len(np.where(e>thresh)[0])
            #overall.append( (float(num) / len(e))*100 )

            pstd = []
            err0 = []
            err1 = []

            for c in chunks:
                ce = e[c]
                inds = np.where( ce > thresh )[0]

                pstd.append( np.mean( s[c] ) )
                err0.append( error_percent( a[c], p[c], s[c] ) )
                err1.append( float(len(inds))/len(ce) )

            all_pstd.append( pstd )
            all_err0.append( err0 )
            all_err1.append( err1 )

        all_pstd = np.mean( np.array( all_pstd ), axis=0 )
        ae0 = np.mean( np.array( all_err0 ), axis=0 )
        es0 = np.std( np.array( all_err0 ), axis=0 )
        ae1 = np.mean( np.array( all_err1 ), axis=0 )
        es1 = np.std( np.array( all_err1 ), axis=0 )

        pp.close()
        pp.fill_between( all_pstd, ae0-es0 , ae0+es0, alpha=0.1, facecolor='blue' )
        pp.fill_between( all_pstd, ae1-es1 , ae1+es1, alpha=0.1, facecolor='red' )

        pp.plot( all_pstd, ae0 , 'b.-', label=r'$I_{95\%}(p_n,\sigma_n)$' )
        pp.plot( all_pstd, ae1 , 'ro-', label=r'$I_{95\%}(p_n,\sigma_{ave})$' )

        pp.plot( all_pstd, np.ones( ae0.shape )*0.05, 'g-.', label='5%',linewidth=2)
        [ x1,x2,y1,y2 ] = pp.axis();
        pp.axis( [ np.min(all_pstd), np.max(all_pstd), y1,y2 ])
        pp.grid()

        pp.xlabel('Average Predicted Standard Deviation',fontsize=18)
        pp.ylabel('Fraction of false negative peptides',fontsize=18)
        pp.legend(loc=4)
        pp.savefig('./plots/PErr_vs_PSTD.pdf')

    def false_positive_ratio( self ):
        def execute_for_part( aa, pp , vv ):
            ss = np.sqrt( vv )
            inds = np.argsort(ss)
            chunks = chunk_it( inds, 10 )

            ave_pstd = np.mean( ss )

            pstd_vec = []
            fp_loc_vec = []
            fp_glob_vec = []

            pstds = []
            for c in chunks :

                pstd_values = []
                fp_loc = []
                fp_glob = []

                for i in c :
                    check = np.abs( pp - pp[i] )

                    n0 = len( np.where( check < ss[i] * 1.96)[0] )
                    n1 = len( np.where( check < ave_pstd * 1.96)[0] )

                    pstd_values.append( ss[i] )
                    fp_loc.append( float(n0) / ( len( check ) ) )
                    fp_glob.append( float(n1) / ( len( check ) ) )

                pstd_vec.append( np.mean(pstd_values) )
                fp_loc_vec.append( np.mean(fp_loc) )
                fp_glob_vec.append( np.mean(fp_glob) )

            return pstd_vec, fp_loc_vec, fp_glob_vec
        all_pstd = []
        all_fp0 = []
        all_fp1 = []

        for model_ind in range( len(self.values) ):
            a,p,v = self.values[model_ind]
            pstd, fp_loc, fp_glob = execute_for_part(a,p,v)
            all_pstd.append( pstd )
            all_fp0.append( fp_loc )
            all_fp1.append( fp_glob )


        all_pstd = np.mean( np.array( all_pstd ), axis=0 )
        ae0 = np.mean( np.array( all_fp0 ), axis=0 )
        es0 = np.std( np.array( all_fp0 ), axis=0 )
        ae1 = np.mean( np.array( all_fp1 ), axis=0 )
        es1 = np.std( np.array( all_fp1 ), axis=0 )

        pp.close()
        pp.fill_between( all_pstd, ae0-es0 , ae0+es0, alpha=0.1, facecolor='blue' )
        pp.fill_between( all_pstd, ae1-es1 , ae1+es1, alpha=0.1, facecolor='red' )

        pp.plot( all_pstd, ae0 , 'b.-', label=r'$I_{95\%}(p_n,\sigma_n)$' )
        pp.plot( all_pstd, ae1 , 'ro-', label=r'$I_{95\%}(p_n,\sigma_{ave})$' )

        #pp.plot( all_pstd, np.ones( ae0.shape )*0.05, 'g-.', label='5%',linewidth=2)
        [ x1,x2,y1,y2 ] = pp.axis();
        pp.axis( [ np.min(all_pstd), np.max(all_pstd), y1,y2 ])
        pp.grid()

        pp.xlabel('Average Predicted Standard Deviation',fontsize=18)
        pp.ylabel('Fraction of false positive peptides',fontsize=18)
        pp.legend(loc=4)
        pp.savefig('./plots/False_positive_PErr_vs_PSTD.pdf')

    def dummy():

        all_pstd = []
        all_fp = []

        for i in range( len( self.values ) ):
            a,p,v = self.values[i]
            pstd, fp = execute_for_part(a,p,v)
            all_pstd.append( pstd )
            all_fp.append( fp )

        all_pstd = np.mean( np.array( all_pstd ), axis=0 )
        fp = np.mean( np.array( all_fp ), axis=0 )
        fps = np.std( np.array( all_fp ), axis=0 )

        pp.close()
        pp.fill_between( all_pstd, fp-fps , fp+fps, alpha=0.8, facecolor='0.75' )
        pp.plot( all_pstd, fp , 'b.-' )


        [ x1,x2,y1,y2 ] = pp.axis();
        pp.axis( [ np.min(all_pstd), np.max(all_pstd), y1,y2 ])
        pp.grid()

        pp.xlabel('Average Predicted Standard Deviation',fontsize=18)
        pp.ylabel('Fraction of false positive peptides',fontsize=18)
        pp.savefig('./plots/Overall_PErr_vs_PSTD.pdf')

if __name__=="__main__" :
    dp = data_plotter()
    dp.load_data()
    dp.calculate_all_a_p_v()

    #dp.actual_vs_predicted()
    #dp.svr_vs_gp()
    #dp.dist_vs_std()
    #over_all_err = dp.section_error_overall()
    #dp.section_error_independent( over_all_err )

    #over_all_err = dp.section_error_independent()
    #dp.section_error_overall( over_all_err )
    #dp.section_error_overall()
    #dp.std_histogram()
    dp.PErr_vs_PSTD()
