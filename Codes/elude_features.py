import data_manager as dm
import retention_model as rm
import random
import numpy as np

removeDuplicates = True
removeCommonPeptides = False # not implemented yet
removeInSourceFragments = False # not implemented yet
removeNonEnzymatic = False # not implemented yet
normalizeRetentionTimes = True

def processTrainData(trainFile, testFile = ""):
  trainPsms, trainAaAlphabet = dm.loadPeptides(trainFile)
  if removeDuplicates:
    trainPsms = dm.removeDuplicates(trainPsms)
  
  if len(testFile) > 0:
    testPsms, testAaAlphabet = dm.loadPeptides(testFile)
    if removeDuplicates:
      testPsms = dm.removeDuplicates(testPsms)
    if removeCommonPeptides:
      print "removeCommonPeptides is not implemented yet"
  
  if removeInSourceFragments:
    print "removeInSourceFragments is not implemented yet"
  
  if removeNonEnzymatic:
    print "removeNonEnzymatic is not implemented yet"
  
  rnd = random.Random(1)
  rnd.shuffle(trainPsms)
  
  return trainPsms, trainAaAlphabet

def processTrainDataUnitTest():
  psmDescriptions, aaAlphabet = processTrainData('../data/retention_time_peptide.csv')
  print len(psmDescriptions), len(aaAlphabet), psmDescriptions[0].peptide, psmDescriptions[0].retentionTime

def trainRetentionModel(aaAlphabet, psmDescriptions):
  customIndex = rm.buildRetentionIndex(aaAlphabet, psmDescriptions, normalizeRetentionTimes)
  return dict(zip(aaAlphabet, customIndex))

def trainRetentionModelUnitTest():
  psmDescriptions, aaAlphabet = processTrainData('../data/retention_time_peptide.csv')
  print len(psmDescriptions), len(aaAlphabet), psmDescriptions[0].peptide, psmDescriptions[0].retentionTime
  
  trainedIndex = trainRetentionModel(aaAlphabet, psmDescriptions)
  
  for aa, index in trainedIndex:
    print aa,':',round(index,6)

def getFeatures(trainFile):
  psmDescriptions, aaAlphabet = processTrainData(trainFile)
  customIndex = trainRetentionModel(aaAlphabet, psmDescriptions)
  featureMatrix = rm.computeRetentionFeatureMatrix(aaAlphabet, psmDescriptions, customIndex)
  return psmDescriptions, featureMatrix

def getFeaturesUnitTest():
  psmDescriptions, featureMatrix = getFeatures('../data/retention_time_peptide.csv')
  
  for idx in [100,101,102]:
    print ""
    if idx == 100:
      # feature vector for K.LCNNQEENDAVSSAK.K
      eludeFeatureVector = [0.55743, 0.392982, 0.922222, 0.0666667, 0.0974441, 0.571038, 0.798507, 0.0337838, 0.116279, 0.397471, 0.21625, 0.38153, 0.176623, 0.0741797, 0.00524934, 0.119266, 0.296296, 0.25, 0.2, 0.111111, 0.188252, 0.263656, 1, 0.126391, 0.0666641, 0.221891, 0.496811, 0.356832, 0.456211, 0.354441, 0.289677, 0.214665, 0.274368, 0.123323, 0.0191546, 0.0831153, 0.181818, 0.0666667, 0.142857, 0, 0.193677, 0.204545, 0.105263, 0.142857, 0.0555556, 0.111111, 0, 0, 0, 0, 0.333333, 0.0909091, 0, 0.375, 0, 0.0909091, 0, 0.153846, 0, 0.125, 0, 0]
    elif idx == 101:
      # feature vector for R.LGTPALTSR.G
      eludeFeatureVector = [0.690763, 0.592593, 0.922222, 0, 0, 0.480874, 0.746269, 0.611486, 0.313953, 0.477686, 0.655993, 0, 0.0127273, 0.00215584, 0.0128609, 0.0563497, 0.037037, 0, 0.133333, 0, 0.206105, 0.402284, 1, 0.140665, 0.110162, 0.308878, 0.601676, 0.547744, 0.373248, 0.322974, 0.489505, 0, 0.0285554, 0.00837665, 0.0398314, 0.142634, 0.0909091, 0, 0.142857, 0, 0.0848419, 0.0681818, 0.0526316, 0, 0, 0, 0, 0.0526316, 0, 0, 0, 0.181818, 0, 0, 0.0909091, 0, 0.333333, 0.0769231, 0.222222, 0, 0, 0]
    else:
      # feature vector for K.TVIVAALDGTFQR.K
      eludeFeatureVector = [0.763855, 0.711201, 0.422222, 0, 0.105431, 0.852459, 0.977612, 0.412162, 4.13106e-16, 0.840601, 0.473735, 0.528567, 0.25974, 0.317542, 0.152231, 0.132353, 0.111111, 0.0416667, 0.333333, 0.222222, 0.284283, 0.438798, 0.336514, 0.140665, 0.0294914, 0.506548, 0.623036, 0.400291, 0.35311, 0.661824, 0.276678, 0.564349, 0.250851, 0.449059, 0.306076, 0.115925, 0.0909091, 0, 0.357143, 0.285714, 0.191194, 0.159091, 0.105263, 0, 0.0555556, 0, 0.2, 0.0526316, 0, 0.125, 0, 0.0909091, 0, 0, 0, 0.0909091, 0.333333, 0, 0.222222, 0.25, 0, 0]
      
    print "Peptide:", psmDescriptions[idx].peptide
    # print psmDescriptions[idx].peptide, len(featureMatrix[idx]), featureMatrix[idx][:20]
    
    i = 1
    errors = 0
    for eludeFeature, ourFeature in zip(eludeFeatureVector, featureMatrix[idx]):
      if abs(eludeFeature - ourFeature) > 1e-5 and i not in range(21,41):
        print "Wrong feature:", i, ", EludeFeature:", eludeFeature, ", OurFeature:", ourFeature
        errors += 1
      i += 1
    if errors == 0:
      print "Unit test succeeded"

def getFeaturesPtmUnitTest():
  psmDescriptions, featureMatrix = getFeatures('../data/ptms.tsv')
  
  for idx in [100,101,102]:
    print ""
    if idx == 100:
      # feature vector for SLEASAADES[UNIMOD:21]DEDEEAIR
      eludeFeatureVector = [0.717583994534, 0.581047971251, 0.294176738769, 0.224364488023, 0.296109415359, 0.441661508008, 0.475393977321, 0.624725120124, 0.727760956983, 0.668211219002, 0.654429693683, 0.247200875306, 0.347081770318, 0.0351665721024, 0.00417513692853, 0.374694220463, 0.1875, 0.0, 0.333333333333, 0.0, 0.545454545455, 0.4, 0.0, 0.272727272727, 0.625, 0.0, 0.0, 0.0, 0.2, 0.0, 0.333333333333, 0.0, 0.0, 0.0, 0.0, 0.5, 0.181818181818,1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    elif idx == 101:
      # feature vector for SES[UNIMOD:21]TEVDVDGNAIR
      eludeFeatureVector = [0.669158518626, 0.546753973963, 0.294176738769, 0.224364488023, 0.187879586609, 0.361949023726, 0.475393977321, 0.645100462159, 0.604052087077, 0.718485749344, 0.565657488542, 0.405106822661, 0.329699757631, 0.177532516617, 0.00375790449087, 0.310493122797, 0.1875, 0.0, 0.5, 0.0, 0.363636363636, 0.1, 0.0, 0.181818181818, 0.25, 0.0, 0.1, 0.0, 0.2, 0.0, 0.0, 0.0, 0.111111111111, 0.0, 0.0, 0.5, 0.0909090909091,1.0, 0.166666666667, 0.0, 0.5, 0.0, 0.0]
    else:
      # feature vector for TNS[UNIMOD:21]FDMPQLNTR
      eludeFeatureVector = [0.683288189415, 0.567575941403, 0.453289245976, 0.224364488023, 0.299951401123, 0.498386937785, 0.421187055985, 0.590816202977, 0.625560159191, 0.612188682373, 0.720816928961, 0.131852825174, 0.273510954189, 0.132048018222, 0.151468105821, 0.373391158302, 0.25, 0.0, 0.333333333333, 0.0, 0.272727272727, 0.0, 0.0, 0.0909090909091, 0.0, 0.333333333333, 0.0, 0.0, 0.0, 0.0, 0.333333333333, 0.5, 0.222222222222, 0.166666666667, 0.166666666667, 0.5, 0.0,1.0, 0.333333333333, 0.0, 0.0, 0.0, 0.0] 
      
    print "Peptide:", psmDescriptions[idx].peptide
    # print psmDescriptions[idx].peptide, len(featureMatrix[idx]), featureMatrix[idx][:20]
    
    i = 1
    errors = 0
    #print ','.join(map(str,featureMatrix[idx]))
    for eludeFeature, ourFeature in zip(eludeFeatureVector, featureMatrix[idx]):
      if abs(eludeFeature - ourFeature) > 1e-5 and i not in range(21,41):
        print "Wrong feature:", i, ", EludeFeature:", eludeFeature, ", OurFeature:", ourFeature
        errors += 1
      i += 1
    if errors == 0:
      print "Unit test succeeded"
