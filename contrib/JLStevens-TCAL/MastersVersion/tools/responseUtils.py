import topo
import numpy
from math import log, exp, pi

import pylab

from recordUtils import recordActivity, recordData
from calibrateUtils import getLGNOnFitTimes, getreferencePeriods
from LGNOnSamplesFile import LGNOnSamples

from models.modelUtils.VSDUtils import createVSDLayer

from distanceDelays import weightsFromGCALtoTCAL

############################
# PROFILE FOR THE STIMULUS #
############################

def makeStepStimulus(times,stepParams):
    values = []
    for time in times:
        t  = (time %  stepParams[0])
        if t < (stepParams[1]*stepParams[0]):
                values.append(0.0)
        elif t < (stepParams[2]*stepParams[0]):
            values.append(stepParams[3])
        else:
            values.append(0.0)
    return values

#######################
# PROFILE FOR THE LGN #
#######################

def LGNPSTHData(sheetActivity, LGNOnSamples, settings,
                dt = 1120.0 / 803 , onsetSlice=(260,380), offsetSlice=(783,802), delayOffset=15):
    onsetSamples = LGNOnSamples[onsetSlice[0]:onsetSlice[1]]
    offsetSamples = LGNOnSamples[offsetSlice[0]:offsetSlice[1]]

    runTime = settings.stepPattern[0]
    onStepTime = (settings.stepPattern[1] * runTime) + delayOffset
    offStepTime = (settings.stepPattern[2] * runTime) + delayOffset

    onsetDuration = len(onsetSamples)*dt; offsetDuration = len(offsetSamples)*dt
    
    onsetTimes = [ (i*dt)+ onStepTime for i in range(len(onsetSamples))]
    offsetTimes = [(i*dt)+ offStepTime for i in range(len(offsetSamples))]
    
    sampleTimes = [0.0] + onsetTimes + offsetTimes
    sampleActivity = [0.0] + onsetSamples + offsetSamples

    maxSheetActivity = max(sheetActivity); maxSampleActivity = max(sampleActivity) 
    scaleFactor = maxSheetActivity / maxSampleActivity
    matchedSamples = [act * scaleFactor for act in sampleActivity]
    
    return (sampleTimes, matchedSamples)

##################
# PROFILE FOR V1 #
##################

def latencyV1(contrast, tmax=121.0, tshift=65.3, 
            power=1.80, s50=24.6):
    cpower = contrast**power
    spower = s50**power
    frac = cpower / (cpower + spower)
    return tmax - (tshift * frac)

def NREqn(contrast, n=2.4, c50=38.7):
    return contrast**n / (contrast**n + c50**n)

def V1PSTH(t, tauC, alpha=0.27, sigmaA=19.0, sigmaB=761.0):
    deltaTSq = float((t - tauC)**2)
    sigmaASq = sigmaA**2
    sigmaBSq = sigmaB**2
    
    if t < tauC:
        return exp(-log(2)*(deltaTSq / sigmaASq))
    else:
        gauss1 = exp(-log(2)*(deltaTSq / sigmaBSq))
        gauss2 = exp(-log(2)*(deltaTSq / sigmaASq))
        return (alpha * gauss1) + ((1 - alpha) * gauss2)

def sampleV1PSTH(time, contrast,
               rmax = 81.8, r0=0):

    time = time + 45 # Offset to match the data

    tauC = latencyV1(contrast)
    unmodulated = V1PSTH(time, tauC)
    contrastFactor = NREqn(contrast)
    return rmax*contrastFactor*unmodulated+ r0

def V1PSTHModel(sheetActivity, settings, contrast=1.0, samples=250):

    runPeriod = settings.stepPattern[0]
    onset = settings.stepPattern[1] * runPeriod
    offset = settings.stepPattern[2] * runPeriod

    sampleTimes = numpy.arange(0.0, runPeriod, float(runPeriod/samples)).tolist()
    V1onsetActivity = [sampleV1PSTH(t, contrast) for t in sampleTimes] 

    onsetTimes = [t + onset for t in sampleTimes]
    sheetActivityMax = max(sheetActivity);  V1onsetActivityMax = max(V1onsetActivity)

    # Factor so that the peak heights match
    factor = sheetActivityMax / V1onsetActivityMax
    adjustedV1OnsetActivity = [el*factor for el in V1onsetActivity]
    return (onsetTimes, adjustedV1OnsetActivity)

#####################
# PROFILE UTILITIES #
#####################

def getReferencePeriods(sheetPeriods, sheetActivity, settings):
    
    runTime = settings.stepPattern[0]
    onsetRatio = settings.stepPattern[1] ; offsetRatio = settings.stepPattern[2] 
    
    onsetTime = onsetRatio*runTime; offsetTime = offsetRatio*runTime
    
    peakTime = max([t for (t,a) in zip(sheetPeriods, sheetActivity) 
                    if ((t >= onsetTime) and (t <= offsetTime)) ])
    return (onsetTime, peakTime, offsetTime)


def scheduleInfo(startPeriod, endPeriod, n=10):
    stepNos = numpy.arange(float(startPeriod),float(endPeriod),n)    
    for stepNo in stepNos:
        topo.sim.schedule_command(stepNo,'pass')


def getResponses(model, settings, count=1, presentTY=None):
    ' Gets the responses over a set of stimuli' 
    responses = []

    for trial in range(count):
        settings.setNextStimulus()
        settings.setNextStimulus()
        responses.append(getResponse(model, settings,presentTY))

    return responses

def getResponse(model, settings, presentTY=None):
    ' Gets the response for one stimulus instance'
    responseData = {} 
    sheetNames = settings.recordedSheets[:]

    # Load the model, setup recording and run for period of time
    timeOffset = topo.sim.time()
    
    if presentTY is not None:
        execfile('./contrib/JLStevens-TCAL/TCAL.ty', globals()) 
        recordData(sheetNames, ports=['Activity']*len(sheetNames))
        print "Running for period %f" % settings.getRunPeriod()
        scheduleInfo(timeOffset, timeOffset+settings.getRunPeriod(), n=10) 

        if presentTY == 'LGN': 
            presentLGNCalibrationPattern()
        elif presentTY == 'V1':
            presentV1CalibrationPattern()
        elif presentTY == 'VSD':
            presentVSDCalibrationPattern()
    else:
        model(**settings())
        # Unpickling organised GCAL weights
        print "\n**Unpickling stored GCAL connections**\n"
        weightsFromGCALtoTCAL()

        # Creating the VSD layers as necessary
        if ('VSDLayer' in sheetNames) or ('VSDSignal' in sheetNames): createVSDLayer(0.2, settings)

        recordData(sheetNames, ports=['Activity']*len(sheetNames))
        print "Running for period %f" % settings.getRunPeriod()
        scheduleInfo(timeOffset, timeOffset+settings.getRunPeriod(), n=10) 
        topo.sim.run(settings.getRunPeriod())

    for sheetName in sheetNames:
        (rawProfile, unitCoords, activities) = recordActivity(sheetName, offset=timeOffset)
        (sheetPeriods, sheetActivity) = rawProfile
        # 1 period = 1 millisecond. Following line to  make gmpy to floats.
        sheetTimes = [float(p) for p in sheetPeriods]
        # The response profile from my model.
        modelProfile = (sheetTimes, sheetActivity)

        # Generating the stimulus activity values using periods.
        retinaTimes = [0.5*i for i in range(500)]
        stimulusActivities =  makeStepStimulus(retinaTimes, settings.stepPattern)
        # The stimulus profile 
        stimulusProfile = (retinaTimes, stimulusActivities)

        experimentalProfile = None
        if sheetName in ['LGNOn','LGNOn']: 
            # Fitting a the experimentally derived PSTH profile.
            experimentalProfile = LGNPSTHData(sheetActivity,LGNOnSamples, settings)

        if sheetName == 'V1':
            # Fitting a the experimentally derived V1 descriptive model profile.
            experimentalProfile = V1PSTHModel(sheetActivity, settings, contrast=1.0)

        if sheetName == 'VSDSignal':
            experimentalProfile = None

        referencePeriods = getReferencePeriods(sheetPeriods, sheetActivity, settings)
        # Now the calibration is done
        calibratedProfiles = (stimulusProfile, experimentalProfile, modelProfile)
        
        # positionData = (coords, posmap)
        data = (activities, rawProfile, unitCoords, referencePeriods, calibratedProfiles)
        responseData.update({sheetName:data})
    return responseData




        
