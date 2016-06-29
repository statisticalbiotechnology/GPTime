#!/usr/bin/pyhon

import platform
import os

class parameters:
    def __init__( self ):
        self.data_root = '../Models'

        #self.base = "20110922_EXQ4_NaNa_SA_YeastEasy_Labelfree_06"
        self.base = "20110922_EXQ4_NaNa_SA_YeastEasy_Labelfree_06.rtimes_q_0.001"
        #self.base = "20120126_EXQ5_KiSh_SA_LabelFree_HeLa_Phospho_Control_rep1_Fr3.rtimes"
        self.data_path = '../Data/%s.tsv' % ( self.base )
        self.data_name = '%s_no_modif' % ( self.base )

        #self.data_path = './data/20121212_S25_3ug_300min_LCMS_PM3_Tryp_GRADIENT_15sec_MZ.rtimes.tsv'
        #self.data_name = 'no_modif'

        #self.data_path = './data/20120126_EXQ5_KiSh_SA_LabelFree_HeLa_Phospho_Control_rep1_Fr3.rtimes.tsv'
        #self.data_name = 'ptm_modif'

        self.models_name = 'original'
        self.models_tag = '%s_%s' % ( self.models_name, self.data_name )
        self.models_path = '%s/%s' % ( self.data_root, self.models_tag )

        #print self.models_path
        if not os.path.isdir( self.models_path ) :
            os.mkdir( self.models_path )

        self.save_tmp = '%s/%s/models_ntrain_%d.pk'
        #self.ntrain = [ 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000 ];
        self.ntrain = [ 1000 ];
        #self.ntrain = [ 3000 ];
        self.nparts = 10
        self.train_ratio = 0.5
