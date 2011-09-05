import topo
import os, numpy
from LGNOnSamplesFile import LGNOnSamples

def getreferencePeriods(sheetPeriods, sheetActivity, stepPattern):
    # first and last periods are easy
    firstPeriod = sheetPeriods[0]; lastPeriod= sheetPeriods[-1]
    # riseStart 
    (activeIndices,)= numpy.nonzero(numpy.array(sheetActivity) != 0)
    assert ((activeIndices[-1]-activeIndices[0]) == (len(activeIndices) - 1)),'Non-continuous block of non-zero activity'
    riseStartPeriod = sheetPeriods[activeIndices[0]-1] # First activity period after which rising begins

    # peakPeriod found fairly easily
    (maxIndices,)= numpy.nonzero(numpy.array(sheetActivity) == numpy.array(sheetActivity).max())
    peakPeriod = sheetPeriods[maxIndices[0]]  # First maximum activity period
    # fallEndPeriod made to match the way the stimulus step profile is created.
    # To handle feefdorward delay, fallTransition is the period when rising starts plus the sustained duration. 
    fallTransition = riseStartPeriod + (stepPattern[2] - stepPattern[1])
    (fallIndices,) = numpy.nonzero(numpy.array(sheetPeriods) >= fallTransition)
    fallStartPeriod = sheetPeriods[fallIndices[0]] # First period falling from steady state

    fallThreshold = sheetActivity[-1]
    (threshIndices,) = numpy.nonzero(numpy.array(sheetActivity[fallIndices[0]:]) <= fallThreshold)    
    
    fallEndPeriod = sheetPeriods[fallIndices[0]+threshIndices[0]]  # Last non-zero activity period
    periods = [firstPeriod, riseStartPeriod, peakPeriod, fallStartPeriod, fallEndPeriod, lastPeriod]
    return [float(el) for el in periods] 

#LGNOnsetSlice=(260,380),  LGNOffsetSlice=(775,802), dt = 1220.0 / 803):
def getLGNOnFitTimes(LGNOnSamples, referencePeriods, settings,
                     dt = 1220.0 / 803 , onsetSlice=(260,380), offsetSlice=(775,802)):

    [firstPeriod, riseStartPeriod, peakPeriod, fallStartPeriod, fallEndPeriod, lastPeriod] = referencePeriods
    
    # Getting the time before the peak for preSpanMS
    (onsetStart,onsetEnd) = onsetSlice; (offsetStart,offsetEnd) = offsetSlice
    onsetSamples = LGNOnSamples[onsetStart:onsetEnd]
    riseSpan = numpy.argmax(numpy.array(onsetSamples))
    preSpanMS = peakPeriod -  (riseSpan * dt)

    # The time recording continues after response
    postSpanMS = (lastPeriod - fallEndPeriod)
    
    # The steady state length
    onsetSpan = onsetEnd - onsetStart
    offsetSpan = offsetEnd - offsetStart
    steadySpanMS = lastPeriod - (preSpanMS + (onsetSpan + offsetSpan)*dt + postSpanMS)

    runPeriod = settings.stepPattern[0]
    onStepTime = settings.stepPattern[1] * runPeriod

    timings = [preSpanMS, steadySpanMS, postSpanMS] 
    return  [onStepTime+15 for el in timings] + [ dt, onsetSlice, offsetSlice] 


#######################################################################################################
#######################################################################################################
#    peakPeriod 
# <----------------------->
#            (MS)
#
# THEREFORE:
#
# preSpanMS =  (peakPeriod * ) -  (prepeakSpan * dt)
# <----------------->

#                     onsetSpan [prepeakSpan + postpeakSpan]
#
#                    <-------->
#
#     [prepeakSpan * dt]  |    [postpeakSpan * dt]
#          (MS)       ____|___         (MS)
#                    |        |
#                          |
#                          |
#                         ||        offsetSpan
#                        | |            __ 
#                       |  |           |  |
#                      /    \__________
#                     /                \
# ___________________/                   \_______
#                                         <------>
#                                         postSpanMS = (endPeriod - postPeriod) 

#                            <-------->
#                            steadySpanMS =  endPeriod - (preSpamMS + (onsetSpan + offsetSpan)*dt + postPanMS)
#
#
#  ( peakPeriod, postPeriod, endPeriod ) = referencePeriods()
#
#######################################################################################################
#######################################################################################################



