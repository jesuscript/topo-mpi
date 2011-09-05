from scipy.misc import toimage
import pylab, numpy
from mpl_toolkits.axes_grid.axislines import Subplot
from math import exp, log

LGNFormat  = [('LGNOn', 'cal', 'peak', None ), ('Retina', 'act', 'peak', False ),('LGNOn', 'act', 'peak', None )]
VSDFormat  = [('VSDSignal', 'act', 'peak', None ), ('VSDSignal', 'act', 'offset', None )]
V1Format  = [('V1', 'cal', 'peak', False ), ('LGNOn', 'act', 'peak', False ), ('V1', 'act', 'peak', True),  ('V1', 'act', 'offset', True) ]


def makePlotDirectories(plotBaseDirectory, sheetNames):
    global LGNOnSamples
    if not os.path.exists(plotBaseDirectory):
        os.mkdir(plotBaseDirectory)
    
    plotDirectories=[]
    for sheetName in sheetNames:
        plotDirectory = plotBaseDirectory+sheetName+'/'
        if not os.path.exists(plotDirectory):
            os.mkdir(plotDirectory)            
        plotDirectories.append(plotDirectory)
    return plotDirectories

def plotCalibrated(fig,sheetName,  subplotNum, calibratedProfiles):

    (stimulusProfile, experimentalProfile, modelProfile) = calibratedProfiles
    (one,two, three) = subplotNum
    ax = Subplot(fig,one,two,three); fig.add_subplot(ax)
    pylab.title('%s' % sheetName, fontsize=10)

    pylab.plot(modelProfile[0],modelProfile[1], label='Model')
    if experimentalProfile is not None:
        pylab.plot(experimentalProfile[0],experimentalProfile[1], '--', label='Experimental')
        pylab.plot(experimentalProfile[0][0],experimentalProfile[1][0], 'yx')

    if experimentalProfile is None:
        maxVal = max(modelProfile[1])
    else:
        maxVal = max( max(experimentalProfile[1]), max(modelProfile[1]))

    stimulusActivities =  [el * (maxVal/2.0) for el in stimulusProfile[1] ]
    pylab.plot(stimulusProfile[0],stimulusActivities, label='Stimulus')

    if subplotNum != (1,1,1):
        pylab.xticks([int(max(modelProfile[0]))], fontsize=4)
    fontsize=6
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)


def plotSheet(fig, sheetName, subplotNum, activities, rawProfile, timePoint=None, timeIndex=0, position=None):

    if timePoint is not None:
        (bestIndices,)= numpy.nonzero(numpy.array(rawProfile[0]) >= (timePoint))
        try: matrixIndex = bestIndices[0]
        except: return
            
    else: matrixIndex = timeIndex

    matrix = activities[matrixIndex]

    (one,two, three) = subplotNum
    ax = Subplot(fig,one,two,three); fig.add_subplot(ax)

    ax.axis['bottom'].set_visible(False); ax.axis['top'].set_visible(False)
    ax.axis['left'].set_visible(False); ax.axis['right'].set_visible(False)        

    pylab.xlim((0,matrix.shape[0]));  pylab.ylim((0,matrix.shape[1]))
    pylab.imshow(matrix, interpolation ='nearest')

    posStr=''
    if position is not None:
        posStr = '%s' % str(position)
        pylab.plot([position[1]],[position[0]],'r+') #  Note the swapped indices.
        centerX = matrix.shape[0]/2; centerY = matrix.shape[1]/2
        pylab.plot([centerX],[centerY],'b.') #  Note the swapped indices.

        pylab.xlim((0,matrix.shape[0]));  pylab.ylim((matrix.shape[1],0))
    pylab.title('%s\n%s' % (sheetName, posStr), fontsize=10)

def gridPlot(data, plotFormats):
    ''' Plot types: act and cal;   Time points: onset, peak, offset '''

    fig = pylab.figure()
    rowNum = len(plotFormats); colNum = len(data)  
    for (i,datum) in enumerate(data):
        for (j,plotFormat) in enumerate(plotFormats):
            plotNum = (i*rowNum + j)+1
            (sheetName, plotType, timePoint, plotCoords)  = plotFormat

            (activities, rawProfile, unitCoords, referencePeriods, calibratedProfiles)  =  data[i][sheetName]
            (stimulusProfile, experimentalProfile, modelProfile) = calibratedProfiles
            
            if timePoint not in ['onset','peak','offset']:
                timeIndex = int(timePoint)
            else:
                (onset, peak, offset) = referencePeriods 
                referencePeriodsDict = {'onset':onset, 'peak':peak, 'offset':offset}
                timePoint = referencePeriodsDict[timePoint]
                timeIndex = None

            subplotNum = (colNum, rowNum,  plotNum)    
            if plotType == 'cal':
                plotCalibrated(fig, sheetName, subplotNum, calibratedProfiles)
            elif plotType == 'act':                
                if plotCoords is True: position = unitCoords
                else:                  position = unitCoords
                plotSheet(fig, sheetName, subplotNum, activities, rawProfile, 
                          timePoint=timePoint, timeIndex=timeIndex, position=position)        
    return fig
