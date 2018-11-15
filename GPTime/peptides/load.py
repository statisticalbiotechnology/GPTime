import numpy as np
from .peptide import peptide

def _check_duplicates(peptides):
    candiates = []
    rt = []
    for pp in peptides:
        if pp.sequence not in candiates:
            candiates.append(pp.sequence)
            rt.append(pp.rt)

    if len(candiates) == len(peptides):
        return "No duplicated peptide in this data set"
    else:
        for i in range(len(candiates)):
            for pp in peptides:
                if candiates[i] == pp.sequence and rt[i] != pp.rt:
                    return "Error: Same peptides have different RT"

    return "All same peptide have same RT"

def load( fname, check_duplicates=True, randomize=False ):
    with open(fname, 'r') as ff :
        lines = ff.readlines()

        peptides = []
        for l in lines :
            l = l.strip()
            values = l.split('\t')
            peptides.append(peptide(values[0],float(values[1])))

        peptides = np.array(peptides)
        if randomize :
            np.random.shuffle( peptides )

        if check_duplicates :
            print( _check_duplicates(peptides))

        return peptides
