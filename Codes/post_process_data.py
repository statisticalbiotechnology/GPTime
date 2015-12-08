#!/usr/bin/python

import sys

if __name__=="__main__" :
    data_path = sys.argv[1]

    f = open( data_path, 'r' )
    lines = f.readlines();
    f.close();

    for l in lines :
        [pep,v] = l[:-1].split('\t')
        pep2 = pep.replace("[UNIMOD:4]","")
        print "%s\t%s" % (pep2,v)

