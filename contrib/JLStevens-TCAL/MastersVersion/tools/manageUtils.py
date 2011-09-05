import topo
from topo.pattern import Disk, SineGrating, Gabor
import sys, random

def makeIntoStepPattern(pattern, period=1.0, onsetRatio=0.33,offsetRatio=0.66, amplitude=1.0):

    def stepModulator(): 
        remainder = float(topo.sim.time()) % period 
        
        if remainder < onsetRatio*period: return 0.0
        if remainder < offsetRatio*period: return amplitude
        else: return 0.0

    pattern.scale = stepModulator
    return pattern

class Settings:

   def __init__(self, name, description):

       self.name = name
       self.description = description
       self.model = {}

       self.recordedSheets = ['LGNOn']

       self.modelKeys = ['ringParams', 'saveSnapshotTime', 'plastic','timestep']
       self.LGNKeys = ['LGN_TC', 'LGNFFStrength', 'GCStrength', 'GCDelay','LGNDistanceDelay']
       self.V1Keys=['V1_TC', 'V1FFStrength', 'ExcStrength', 'InhStrength', 'LateralDelay', 'HomeoPlastic', 'V1DistanceDelay']
       
       self.allKeys = self.modelKeys + self.LGNKeys + self.V1Keys

       # Stimulus settings
       self.stepPattern = None  # Triple
       self.patternClass = None # topo.pattern Class
       self.constParams = {}    # Fixed stimulus parameters
       self.rangeParams = {}    # Ranging stimulus parameters

       # Internal attributes
       self.stimulusDict = {}
       self.seed = 42
       random.seed(self.seed)
       self.randState = random.getstate()

       # Settings for computing distance delays
       self.fourierRingRadius = None
       self.propagationVelocity = None
       self.hypercolumnDistance = None
       self.MSPerPeriod = None
       self.V1Dimensions = None

   def setNextStimulus(self):
       completeDict = self.constParams.copy(); rangeDict = {}

       # Sort keys so randomly sampled in same order every time.
       keys = self.rangeParams.keys(); keys.sort()
       # Sampling the ranging parameters
       for key in keys:
           (lower, upper) = self.rangeParams[key]
           random.setstate(self.randState)
           sample = random.uniform(lower,upper)
           self.randState = random.getstate()
           rangeDict.update({key:sample})

       completeDict.update(rangeDict)
       self.stimulusDict = completeDict

   def __call__(self):
       'Returns a dictionary that can be passed right into gcal_temporal'

       if not self.allInitialised(): print "Model '%s' not completely initialised!" % self.name; return

       if self.stimulusDict == {}: self.setNextStimulus()

       stimulus =  self.patternClass(**self.stimulusDict)    
       # Period, onset, offset, amplitude.
       (p0,p1,p2,p3) = self.stepPattern
       makeIntoStepPattern(stimulus,p0,p1,p2,p3) 
       
       runDict = self.model.copy()
       runDict.update({'stimulus':stimulus})

       infoStr =  "\n**EXPERIMENT '%s' PARAMETERS**\n" % self.name
       print '-'*len(infoStr) + infoStr + '-'*len(infoStr) + "\n"
       print 'DESCRIPTION: %s\n' % self.description

       return runDict

   def copy(self, newName, newDescription):

       if not self.allInitialised(): print "Model '%s' not completely initialised!" % self.name; return

       newSettings = Settings(newName, newDescription)
       
       newSettings.setModel(self.model)
       newSettings.setRecordedSheets(self.recordedSheets)

       # Setting the stimulus
       (period, onsetRatio, offsetRatio, amplitude) = self.stepPattern
       newSettings.setStepParams(period, onsetRatio, offsetRatio, amplitude)

       constDict = self.constParams.copy()
       constDict.update({'pattern':self.patternClass.name})
       newSettings.setConstParams(constDict)
       newSettings.setRangeParams(self.rangeParams)
       
       return newSettings

   ###########
   # Getters #
   ###########

   def getTimestep(self):
       if 'timestep' not in self.model: 
           print "*Please define model '%s' before calling this method!*" % self.name
       else: return self.model['timestep']

   def getRecordedSheets(self): return self.recordedSheets

   def getRunPeriod(self): return self.stepPattern[0]

   def getFrameSkip(self):
       frameskipDict = {0.03:1, 0.0008:10}
       return frameskipDict[self.getTimestep()]

   ###########
   # Setters #
   ###########

   def setRecordedSheets(self, sheetList):
       if 'LGNOn' not in sheetList:
           print 'LGNOn must be recorded for calibration!'
       else:
           self.recordedSheets = sheetList

   def setStepParams(self, period, onsetRatio, offsetRatio, amplitude):
       self.stepPattern = (period, onsetRatio, offsetRatio, amplitude)

       if 'timestep' not in self.model:
           print "*Please define model '%s' before calling this method*" % self.name
           return

       timestep = self.model['timestep']
       runPeriod = self.getRunPeriod()
       print "Setting homeostasis in model %s against timestep of %f and run period of %f" % (self.name, timestep, runPeriod) 
       numberOfSteps = (runPeriod / timestep)
       hfactor = (1 / numberOfSteps) * (offsetRatio - onsetRatio)
       
       self.model.update({'HomeoFactor':hfactor})
           

   def setConstParams(self, constantParamsDict):
    
       patterns = {'Disk': Disk, 'SineGrating':SineGrating, 'Gabor':Gabor}
       if 'pattern' not in constantParamsDict:
           print "Key 'pattern' must be one of %s " % ', '.join(patterns.keys())
       else:
           patternName = constantParamsDict['pattern']
           self.patternClass = patterns[patternName]
           
           del constantParamsDict['pattern']
           self.constParams = constantParamsDict
           
   def setRangeParams(self, rangeParamsDict):
       self.rangeParams = rangeParamsDict
       
   def setModel(self, modelDict):

       allKeys = self.modelKeys + self.LGNKeys + self.V1Keys       
       diff = set(allKeys) - set(modelDict.keys())
       
       if diff != set():
           print 'Please provide *exactly* the following keys: %s' % ', '.join(list(diff))
           self.model ={}
           return False
       else:
           self.model = modelDict.copy()
           # self.model.update({'V1DistanceDelay':None})
           return True


   ############
   # Modifier #
   ############

   def modifyModel(self, attribute, value):
       if attribute not in self.allKeys:
           print('%s not a valid parameter!' % attribute)
       else:
           self.model[attribute] = value

   ######################
   # Propagation delays #
   ######################

   def enableV1PropagationDelays(self, fourierRingRadius= 4.8, propagationVelocity=0.2, hypercolumnDistance = 1.0): #mm/ms!   MILLISECONDS
       ''' fourierRingRadius is in units, found from GUI fourier plot in organised map.
           propagationVelocity is real neural propagation velocity in mm/s
           hypercolumnDistance is the real distance between hypercolumns in the cortex '''

       if self.model['ringParams'][1] != 'MAX':
           print "**V1 ringParams must be set to 'MAX' for propagation delays**";  return

       self.fourierRingRadius = fourierRingRadius
       self.propagationVelocity = propagationVelocity
       self.hypercolumnDistance = hypercolumnDistance

   def disableV1PropagationDelays(self):
       self.fourierRingRadius =  self.propagationVelocity =  self.hypercolumnDistance = None

   def setV1DynamicDelayParameters(self, timeFactor, V1Shape):
       self.MSPerPeriod = timeFactor # Because period * timefactor = time. SETTING TIMEF
       print 'CURRENT timefactor: %s TARGET: 1.0' %  timeFactor,
       self.V1Dimensions = V1Shape[0]
    
       if None not in [self.fourierRingRadius, self.propagationVelocity, self.hypercolumnDistance]:
           print "Setting distance delays "
           self.setV1PropagationDelayPeriod()

   def setV1PropagationDelayPeriod(self):
       ''' Sets the propagation delay passed to the model for V1 lateral connections '''
       ###################################################################################
       # - Remember that the delay value is from one ring to another - ie ONE unit wide. #
       # - FourierRingRadius gives *periodicity* of blobs in sheet.                      #
       # - Average distance between blobs is hypercolumn distance                        #
       ###################################################################################

       # Using hypercolumn distance to get to spatial dimension
       unitsPerHypercolumn = (self.V1Dimensions / self.fourierRingRadius)
       assert (unitsPerHypercolumn == 10)

       MMPerUnit = self.hypercolumnDistance / unitsPerHypercolumn 
       assert (MMPerUnit == 0.1)

       # Computing real time (MS) to cross distance corresponding to one unit
       delayPerUnitMS = MMPerUnit / self.propagationVelocity  # Time = distance/speed
       assert (delayPerUnitMS == 0.5)
       self.model['V1DistanceDelay'] = 0.5

   ###################
   # Pretty printing #
   ###################

   def dictToString(self, dictionary, keys, sep='\n'):
       return sep.join(['%s=%s' % (k, dictionary[k]) for k in keys])

   def printDifference(self, other):

       title = "Experiment '%s' VS Experiment '%s'" % (self.name, other.name)
       differenceStr = ['-'*len(title), title, '-'*len(title)+'\n']

       if False in [self.allInitialised(), other.allInitialised()]: 
           print '\n Models not completely initialised! '
           return

       try:
           if self.patternClass.name != other.patternClass.name:
               differenceStr += ['STIMTYPE DIFFERENCES: %s VS %s' % (self.patternClass.name, other.patternClass.name)]
       except:
           differenceStr += ['STIMTYPE DIFFERENCES: DIRECTLYSET VS STEPSTIMULUS' ]

       ownPatternKeysSet = set(self.patternParams.keys());  otherPatternKeysSet = set(other.patternParams.keys())
       if ownPatternKeysSet != otherPatternKeysSet:
           patternDiffSet = ownPatternKeysSet.symmetric_difference(otherPatternKeysSet)
           differenceStr += ['STIMKEYS DIFFERENCES: %s' % ' '.join(list(patternDiffSet))]

       patternDiffs = []
       for key in ownPatternKeysSet:
           ownVal = self.patternParams[key]; otherVal = other.patternParams[key];  
           if ownVal != otherVal:  patternDiffs.append((key, ownVal,otherVal))

       patternDiffStr = ['%s: %s VS %s' % triple for triple in patternDiffs]
       if patternDiffStr != []:
           differenceStr += ['PATTERN PARAMETER DIFFERENCES: %s' % ', '.join(patternDiffStr)]

       modelDiffs = []; allKeys = self.modelKeys + self.LGNKeys + self.V1Keys
       for key in allKeys:
           ownVal = self.model[key]; otherVal = other.model[key]
           if ownVal != otherVal:  modelDiffs.append((key, ownVal, otherVal))

       modelDiffStr = ['%s: %s VS %s' % triple for triple in modelDiffs]
       if modelDiffs != []:
           differenceStr += ['MODEL DIFFERENCES: %s' % ', '.join(modelDiffStr)]

       if len(differenceStr) == 3: differenceStr += ['NO DIFFERENCES'] 
       
       differenceStr += ['\n'+'-'*len(title)]
       print '\n'.join(differenceStr)

   def allInitialised(self):
       return  ( (self.constParams != {}) and (self.stepPattern is not None) )

   def __str__(self):
       title = 'Experiment %s' % self.name
       description = 'Description: %s' % self.description

       if not self.allInitialised(): return "Model '%s' not completely initialised!" % title

       modelSettings = 'Global settings\n-------------\n%s' % self.dictToString(self.model, self.modelKeys)
       LGNSettings =  'LGN settings\n-------------\n%s' % self.dictToString(self.model, self.LGNKeys)
       V1Settings = 'V1 settings\n-------------\n%s' %  self.dictToString(self.model, self.V1Keys)

       sortedMKeys = self.model.keys(); sortedMKeys.sort()
       modelStr = '\n'.join(['%s=%s' % (k, self.model[k]) for k in sortedMKeys])
       if modelStr == '': modelStr = 'None'
       model = 'Model: %s' % modelStr
        
       return '\n\n'.join([title, description, stimulus, modelSettings, LGNSettings, V1Settings])
