import sys
import numpy


import copy

import topo
from topo.misc.trace import InMemoryRecorder


def recordData(sheetNames, ports):
    """ Add a data recorder called Data and record
    the given ports from the given sheets with the
    given names """

    streamNames = [('%sActivity' % name) for name in sheetNames]
    tupleSettings = zip(sheetNames, ports, streamNames)

    topo.sim['Data'] = InMemoryRecorder()
    for (sheetName,port,name) in tupleSettings:
        topo.sim.connect(sheetName, 'Data', src_port = port, name = name) # DOES THE DELAY AFFECT ANYTHING?

def recordActivity(sheetName, offset=0.0, data=None,coordInts=None, policy='avg'):
    """ Returns the activity of a unit and the
        associated times. By default corresponds
        to the center unit """

    streamName = '%sActivity' % sheetName
    if data is None: data = topo.sim['Data'].get_data(streamName)

    stripped = [(float(time)-offset, act) for (time, act) in zip(data[0], data[1]) if (float(time) > offset)]
    times, rawActivities = zip(*stripped)
    activities = copy.deepcopy(rawActivities) #[:]  # DEEP COPY?
    
    # Getting matrix dimension
    (xdim, ydim) =  data[1][0].shape

    activity = []
    if coordInts is None:
        # Picking the maximally responding unit
        peakInds = getPeakUnit(activities,xdim,ydim) 
        maxAvgInd = getMaxAvg(activities,xdim,ydim)

        if peakInds != maxAvgInd:
            print("WARNING: No clear most responsive neuron for '%s'."  % streamName)
            if (peakInds is None) and (maxAvgInd == (0,0)):
                print "No activity in recording!"
            else:
                print('(PEAK: %s MAXAVG: %s)' % (str(peakInds),str(maxAvgInd)))
                print('Using unit with maximum average value')

            if policy == 'peak':   (xind,yind) = peakInds
            else:                  (xind,yind) = maxAvgInd
        else:                      (xind,yind) = maxAvgInd
    else:
        print "Record coordinates set to %s" % str(coordInts)
        (xind,yind) = coordInts

    for (time,matrix) in stripped:
        activity.append(matrix[xind][yind]) 

    assert len(times) == len(activity)
    return ((times, activity),(xind,yind), activities)


def getPeakUnit(data,xdim,ydim):

    maxVal = 0; maxInds = None

    for matrix in data:
        maxind = matrix.argmax()
        currMaxVal = matrix.ravel(maxind)[maxind]
        if currMaxVal > maxVal:
            maxInds = numpy.unravel_index(maxind,
                                          (xdim,ydim))
    return maxInds

def getMaxAvg(data,xdim,ydim):

    acc = numpy.zeros((xdim,ydim))
    for matrix in data:
        acc += matrix

    maxind = acc.argmax()
    return numpy.unravel_index(maxind,(xdim,ydim))
