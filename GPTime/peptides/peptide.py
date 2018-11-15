from ..features import word

class peptide:
    def __init__(self, sequence, rt):
        self.pre = sequence[0]
        self.post = sequence[-1]
        self.sequence = sequence[2:-2]
        self.rt = rt
        self.amino_acids = self.get_amino_acids()
        self.aa_indices = []

    def build_amino_acid_indices(self, aa_list):
        self.aa_indices = []
        for aa in self.amino_acids:
            self.aa_indices.append(aa_list.index(aa))

    def get_amino_acids(self):
        ll = []
        seq = self.sequence + '*'
        i = 0
        while i< len(seq)-1:
            if seq[i+1] == '[':
                v= seq[i:i+5]
                i = i + 5
            else:
                v = seq[i]
                i = i+1
            ll.append(v)
        return  ll

    def get_k_mers(self, k):
        k_mers = []
        for i in range(len(self.aa_indices)-k+1):
            indices = self.aa_indices[i:i+k]
            seq = ''.join(self.amino_acids[i:i+k])
            w = word(seq, indices)
            k_mers.append(w)
        return k_mers

    def bow_descriptor(self, voc):
        desc = np.array([0.0] * (voc.nwords))
        k_mers = self.get_k_mers(voc.k)
        for w in k_mers:
            ind = voc.char_seq_index(w.char_seq)
            desc[ind] = desc[ind]+1.0

        desc = desc/(np.linalg.norm(desc) + 1e-12)
        return  desc

    def sim_score(self, str1, str2):
        score = 0.0
        for s1, s2 in zip(str1, str2):
            if s1 == s2:
                score = score + 1
        return  score/len(str1)

    def wbow_descriptor(self, voc):
        desc = np.array([0.0] * (voc.nwords))
        k_mers = self.get_k_mers(voc.k)
        for w in k_mers:
            v = voc.ind_seq_score(w.ind_seq)
            desc = desc + v
        return  desc

    def elude_descriptor(self, em):
        return em.compute_features( self.sequence )
