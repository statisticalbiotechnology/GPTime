import numpy as np
import data_manager as dm

kyteDoolittleIndex = { "A" : 1.8, "C" : 2.5, "D" : -3.5, "E" : -3.5, "F" : 2.8,
   "G" : -0.4, "H" : -3.2, "I" : 4.5, "K" : -3.9, "L" : 3.8,
   "M" : 1.9, "N" : -3.5, "P" : -1.6, "Q" : -3.5, "R" : -4.5,
   "S" : -0.8, "T" : -0.7, "V" : 4.2, "W" : -0.9, "Y" : -1.3};

bulkinessIndex = { "A" : 11.5, "C" : 13.46, "D" : 11.68, "E" : 13.57, "F" : 19.80,
   "G" : 3.40, "H" : 13.69, "I" : 21.40, "K" : 15.71, "L" : 21.40,
   "M" : 16.25, "N" : 12.82, "P" : 17.43, "Q" : 14.45, "R" : 14.28,
   "S" : 9.47, "T" : 15.77, "V" : 21.57, "W" : 21.67, "Y" : 18.03};

defaultAlphabet = set(["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"])

percentageAa = 0.25
cos300 = np.cos(300 * np.pi / 180)
cos400 = np.cos(400 * np.pi / 180)
  
def buildRetentionIndex(aaAlphabet, psmDescriptions, 
      normalizeRetentionTimes):
  featureMatrix = computeRetentionIndexFeatureMatrix(aaAlphabet, psmDescriptions)
  temp = featureMatrix[0]
  temp1 = featureMatrix[1]
  normalizeFeatures(featureMatrix)
  
  retentionTimes = [psmd.rt for psmd in psmDescriptions]
  if normalizeRetentionTimes:
    retentionTimes -= np.mean(retentionTimes)
    retentionTimes /= np.std(retentionTimes)
  
  customIndex = np.linalg.lstsq(featureMatrix, retentionTimes)[0]
  return customIndex

def computeRetentionIndexFeatureMatrix(aaAlphabet, psmDescriptions):
  featureMatrix = np.zeros((len(psmDescriptions), len(aaAlphabet)))
  t = type(psmDescriptions)
  if type(psmDescriptions) is str:
      featureMatrix = computeRetentionIndexFeatures(aaAlphabet, psmDescriptions)
  else:
      for i, psmd in enumerate(psmDescriptions):
          featureMatrix[i] = computeRetentionIndexFeatures(aaAlphabet, psmd.sequence)
  return featureMatrix
  
def computeRetentionIndexFeatures(aaAlphabet, peptide):
  aas = dm.getAminoAcidList(peptide)
  featureVector = np.zeros((1, len(aaAlphabet)))
  for aa in aas:
    featureVector[0][aaAlphabet.index(aa)] += 1
  return featureVector

def normalizeFeatures(featureMatrix):
  rows, cols = featureMatrix.shape
  colMean = list()
  colStd = list()
  if rows != 1:
      for i in range(cols):
         #featureMatrix[:,i] -= np.mean(featureMatrix[:,i])
         #featureMatrix[:,i] /= np.std(featureMatrix[:,i])
          minFeature = np.min(featureMatrix[:,i])
          maxFeature = np.max(featureMatrix[:,i])
          featureMatrix[:,i] -= minFeature
          featureMatrix[:,i] /= maxFeature - minFeature

def hasPtms(aaAlphabet):
  return sum([1 for aa in aaAlphabet if aa not in defaultAlphabet]) > 0
  
def computeRetentionFeatureVector(aaAlphabet, peptide, customIndex):
  ptmsPresent = False
  if hasPtms(aaAlphabet):
    numFeatures = 20 + 1 + len(aaAlphabet)
    ptmsPresent = True
  else:
    numFeatures = 20 + 20 + 2 + len(aaAlphabet)

  featureVector = np.zeros((1,0))
  if not ptmsPresent:
    polarAa, hydrophobicAa = getExtremeRetentionAA(kyteDoolittleIndex)
    kyteDoolittleFeatureVector = np.zeros((1, 20))
    kyteDoolittleFeatureVector[:] = computeIndexFeatures(aaAlphabet, peptide, kyteDoolittleIndex, polarAa, hydrophobicAa)
    featureVector = np.concatenate((featureVector, kyteDoolittleFeatureVector), axis = 1)

  polarAa, hydrophobicAa = getExtremeRetentionAA(customIndex)
  customFeatureVector = np.zeros((1, 20))
  customFeatureVector[:] = computeIndexFeatures(aaAlphabet, peptide, customIndex, polarAa, hydrophobicAa)
  featureVector = np.concatenate((featureVector, customFeatureVector), axis = 1)

  if not ptmsPresent:
    bulkinessFeatureVector = np.zeros((1, 1))
    aas = dm.getAminoAcidList(peptide)
    bulkinessFeatureVector[:] = indexSum(aas, bulkinessIndex)
    featureVector = np.concatenate((featureVector, bulkinessFeatureVector), axis = 1)

  lengthFeatureVector = np.zeros((1, 1))
  aas = dm.getAminoAcidList(peptide)
  lengthFeatureVector[:] = len(aas)
  featureVector = np.concatenate((featureVector, lengthFeatureVector), axis = 1)


  aaFeatureVector = computeRetentionIndexFeatureMatrix(aaAlphabet, peptide)
  featureVector = np.concatenate((featureVector, aaFeatureVector), axis = 1)


  normalizeFeatures(featureVector)
  return np.array(featureVector[0])

def getExtremeRetentionAA(index):
  numAa = int(np.ceil(percentageAa * len(index)))
  sortedIndex = sorted(index.items(), key = lambda x : x[1])
  polarAa = [x[0] for x in sortedIndex if x[1] <= sortedIndex[numAa-1][1]]
  hydrophobicAa = [x[0] for x in sortedIndex if x[1] >= sortedIndex[-1*numAa][1]]
  return polarAa, hydrophobicAa
  
def computeIndexFeatures(aaAlphabet, peptide, index, polarAa, hydrophobicAa):
  features = []
  aas = dm.getAminoAcidList(peptide)
  features.append(indexSum(aas, index))
  features.append(indexAvg(aas, index))
  features.append(indexN(aas, index))
  features.append(indexC(aas, index))
  features.append(indexNearestNeighbour(aas, index, polarAa))
  
  maxPartSum5, minPartSum5 = indexMaxMinPartialSum(aas, index, 5)
  maxPartSum2, minPartSum2 = indexMaxMinPartialSum(aas, index, 2)
  features.append(maxPartSum5)
  features.append(maxPartSum2)
  features.append(minPartSum5)
  features.append(minPartSum2)
  
  maxHsideHelix, minHsideHelix = indexMaxMinHydrophobicSideHelix(aas, index)
  features.append(maxHsideHelix)
  features.append(minHsideHelix)
  
  maxHmoment100, minHmoment100 = indexMaxMinHydrophobicMoment(aas, index, 100, 11)
  maxHmoment180, minHmoment180 = indexMaxMinHydrophobicMoment(aas, index, 180, 11)
  features.append(maxHmoment100)
  features.append(maxHmoment180)
  features.append(minHmoment100)
  features.append(minHmoment180)
  
  features.append(indexSumSquaredDiff(aas, index))
  features.append(numberTypeAA(aas, polarAa))
  features.append(numberConsecTypeAA(aas, polarAa))
  features.append(numberTypeAA(aas, hydrophobicAa))
  features.append(numberConsecTypeAA(aas, hydrophobicAa))
  return features

# calculate the sum of hydrophobicities of all aa in the peptide
def indexSum(aas, index):
    temp = 0
    for aa in aas:
        g = index[aa]
        temp = temp + g
    return temp

# calculate the average of hydrophobicities of all aa in the peptide
def indexAvg(aas, index):
  return indexSum(aas, index) / len(aas)

# calculate the hydrophobicity of the N-terminus
def indexN(aas, index):
  return index[aas[0]]
  
# calculate the hydrophobicity of the C-terminus
def indexC(aas, index):
  return index[aas[-1]]

# calculate the sum of hydrophobicities of neighbours of polar amino acids
def indexNearestNeighbour(aas, index, polarAa):
  s = 0.0
  for i, aa in enumerate(aas):
    if aa in polarAa:
      if i > 0:
        s += max([ 0.0, index[aas[i-1]] ])
      if i < len(aas) - 1:
        s += max([ 0.0, index[aas[i+1]] ])
  return s

# the most and least hydrophobic window
def indexMaxMinPartialSum(aas, index, window):
  w = min([window, len(aas) - 1])
  maxSum = indexSum(aas[:w], index)
  minSum = maxSum
  for i in range(1,len(aas) - w + 1):
    s = indexSum(aas[i:i+w], index)
    maxSum = max([maxSum, s])
    minSum = min([minSum, s])
  return maxSum, minSum

# calculate the most and least hydrophobic sides for alpha helices
def indexMaxMinHydrophobicSideHelix(aas, index):  
  if len(aas) < 9:
    avgHindex = avgHydrophobicityIndex(index)
    hSideHelix = avgHindex * (1 + 2 * cos300 + 2 * cos400)
    return hSideHelix, hSideHelix
  else:
    hydrophobicitySide = 0.0
    maxHydrophobicitySide = calcHydrophobicitySide(aas[:9], index)
    minHydrophobicitySide = maxHydrophobicitySide
    for i in range(1, len(aas) - 9 + 1):
      hSideHelix = calcHydrophobicitySide(aas[i:i+9], index)
      maxHydrophobicitySide = max([ maxHydrophobicitySide, hSideHelix ])
      minHydrophobicitySide = min([ minHydrophobicitySide, hSideHelix ])
    return maxHydrophobicitySide, minHydrophobicitySide

# calculate the maximum and minimum value of the hydrophobic moment
def indexMaxMinHydrophobicMoment(aas, index, angle, window):
  sinSum = 0.0
  cosSum = 0.0
  angleRadians = angle * np.pi / 180
  
  if len(aas) < window:
    avgHindex = avgHydrophobicityIndex(index)
    for i in range(1, window + 1):
      cosSum += np.cos(i * angleRadians)
      sinSum += np.sin(i * angleRadians)
    cosSum *= avgHindex
    sinSum *= avgHindex
    hMoment = np.sqrt(cosSum*cosSum + sinSum*sinSum)
    return hMoment, hMoment
  else:
    windowHmoment = 0.0
    for i in range(1, window + 1):
      cosSum += index[aas[i-1]] * np.cos(i * angleRadians)
      sinSum += index[aas[i-1]] * np.sin(i * angleRadians)
    maxHmoment = cosSum*cosSum + sinSum*sinSum
    minHmoment = maxHmoment
    for i in range(window + 1, len(aas) + 1):
      cosSum += index[aas[i-1]] * np.cos(i * angleRadians)
      cosSum -= index[aas[i-window-1]] * np.cos((i-window) * angleRadians)
      sinSum += index[aas[i-1]] * np.sin(i * angleRadians)
      sinSum -= index[aas[i-window-1]] * np.sin((i-window) * angleRadians)
      hMoment = cosSum*cosSum + sinSum*sinSum
      maxHmoment = max([ maxHmoment, hMoment ])
      minHmoment = min([ minHmoment, hMoment ])
    return np.sqrt(maxHmoment), np.sqrt(minHmoment)

# calculate the sum of squared differences in hydrophobicities between neighbours
def indexSumSquaredDiff(aas, index):
  squaredDiffSum = 0.0
  for i in range(len(aas)-1):
    diff = index[aas[i]] - index[aas[i+1]]
    squaredDiffSum += diff * diff
  return squaredDiffSum

# calculate the number of a certain type of aa
def numberTypeAA(aas, aasOfType):
  return sum([1 for aa in aas if aa in aasOfType])

# calculate the number of a consecutive aa pairs of a certain type
def numberConsecTypeAA(aas, aasOfType):
  return sum([1 for i in range(len(aas)-1) if aas[i] in aasOfType and aas[i+1] in aasOfType])

# calculate the average hydrophobicity of an index
def avgHydrophobicityIndex(index):
  return np.mean(index.values())

def calcHydrophobicitySide(aas, index):
  return index[aas[4]] + cos300 * (index[aas[1]] + index[aas[7]]) + cos400 * (index[aas[0]] + index[aas[8]])
