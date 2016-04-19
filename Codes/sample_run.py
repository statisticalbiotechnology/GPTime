#!/usr/bin/python

import os
import data_tools
import feature_extraction as fe

if __name__=="__main__" :
    pwd = os.path.dirname(os.path.realpath(__file__))

    # Reading peptides and their retention time from file
    peptides = data_tools.read_data( pwd + '/../Data/20110922_EXQ4_NaNa_SA_YeastEasy_Labelfree_01.rtimes_q_0.001.tsv')

    # Building different models
    mgen = fe.model_generator( peptides )

    aa_list = mgen.amino_acid_list()
    elude_model = mgen.get_elude_model()
    bow_voc = mgen.get_bow_voc( 2 ) # K = number of letters in each word

    peptide = peptides[0]

    print "Elude descriptor is"
    print peptide.elude_descriptor( elude_model )
    print "Bow Descriptor is"
    print peptide.bow_descriptor( bow_voc )
