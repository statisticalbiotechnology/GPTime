import csv
import numpy

class PSMDescription:
  peptide = ''
  retentionTime = 0.0
  
  def __eq__(self, other):
    return (isinstance(other, self.__class__)
      and self.__dict__ == other.__dict__)
  
  def __ne__(self, other):
    return not self.__eq__(other)
  
  def __lt__(self, other):
    return self.peptide < other.peptide or (self.peptide == other.peptide and self.retentionTime < other.retentionTime)
  
def getAminoAcids(peptide):
  idx = 0
  aas = set()
  while idx < len(peptide):
    if idx < len(peptide) - 1 and peptide[idx+1] == '[':
      nextIdx = peptide[idx+2:].find(']') + idx + 3
      aas.add(peptide[idx:nextIdx])
      idx = nextIdx
    else:
      aas.add(peptide[idx])
      idx += 1
  return aas

def getAminoAcidList(peptide):
  idx = 0
  aas = list()
  while idx < len(peptide):
    if idx < len(peptide) - 1 and peptide[idx+1] == '[':
      nextIdx = peptide[idx+2:].find(']') + idx + 3
      aas.append(peptide[idx:nextIdx])
      idx = nextIdx
    else:
      aas.append(peptide[idx])
      idx += 1
  return aas
  
def getAminoAcidsUnitTest():
  print(getAminoAcids("ASDIOJGSDA"))
  print(getAminoAcids("ASDUHGIDUHGI[UNIMOD:34]ASASDOIJ"))
  
def loadPeptides(fileName):
  psmDescriptions = []
  delim = '\t'
  with open(fileName, 'rb') as f:
    line = f.readline()
    sniffer = csv.Sniffer()
    dialect = sniffer.sniff(line)
    delim = dialect.delimiter

  reader = csv.reader(open(fileName, 'rb'), delimiter = delim)
  includesRT = False
  includesFlanks = False
  aaAlphabet = set()
  
  first = True
  for row in reader:
    if first:
      if len(row) > 1: includesRT = True
      if row[0][1] == '.' and row[0][-2] == '.': includesFlanks = True
      first = False
    
    psmd = PSMDescription()
    psmd.peptide = row[0].upper()
    if includesFlanks:
      psmd.peptide = psmd.peptide[2:-2]
    if includesRT:
      psmd.retentionTime = float(row[1])
    aaAlphabet = aaAlphabet.union(getAminoAcids(psmd.peptide))
    
    psmDescriptions.append(psmd)
  
  aaAlphabet = sorted(list(aaAlphabet))
  
  return psmDescriptions, aaAlphabet

def loadPeptidesUnitTest():
  psmDescriptions, aaAlphabet = loadPeptides('../data/retention_time_peptide.csv')
  print(len(psmDescriptions), psmDescriptions[10].peptide, psmDescriptions[10].retentionTime, len(aaAlphabet))

def removeDuplicates(psmDescriptions):
  psmDescriptions = sorted(psmDescriptions)
  lastPsmd = PSMDescription()
  psmDescriptionsNew = []
  for psmd in psmDescriptions:
    if psmd != lastPsmd:
      psmDescriptionsNew.append(psmd)
      lastPsmd = psmd
  return psmDescriptionsNew

def removeDuplicatesUnitTest():
  psmDescriptions, aaAlphabet = loadPeptides('../data/retention_time_peptide.csv')
  print(len(psmDescriptions), psmDescriptions[10].peptide, psmDescriptions[10].retentionTime, len(aaAlphabet))
  psmDescriptions = removeDuplicates(psmDescriptions)
  print(len(psmDescriptions))
