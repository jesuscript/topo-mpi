import pickle, sys, os
import numpy

from scipy.misc import toimage

import topo
from topo import pattern
from topo import sheet


############################################################
# FUNCTIONS TO CREATE AND LOAD ORGANISED GCAL CONNECTIVITY #
############################################################

def pickleGCALWeight():
    """ Run this function after 10,000 iterations of GCAL (default settings) to generate pickle file needed by TCAL.
        Place the pickle file in the default output folder specified by topo.param.normalize_path() """

    V1Dim=48; pickleObj = {}
    pickleObj.update({'SIZE':V1Dim})

    connections = ['LGNOnAfferent', 'LGNOffAfferent', 'LateralExcitatory', 'LateralInhibitory']
    pickleObj.update({'Connections':connections})
    for connectionName in connections:
        cfsList = []
        for i in range(V1Dim):
            for j in range(V1Dim):
                cfWeights = topo.sim['V1'].projections(connectionName).cfs[i][j].weights
                cfsList.append(cfWeights[:])
        assert len(cfsList) == 48*48
        pickleObj.update({connectionName:cfsList[:]})
                
    path = os.path.join(topo.param.normalize_path(),'GCALweights.pickle')
    pickleFile = open(path,'w')
    pickle.dump(pickleObj,pickleFile)

def weightsFromGCALtoTCAL():
    " Loads GCAL connectivity from pickle file into current V1 sheet "
    # os.path.join(os.getcwd(), 'GCALweights.pickle')

    path = os.path.join(topo.param.normalize_path(),'GCALweights.pickle')
    pickleFile = open(path,'r'); pickleObj = pickle.load(pickleFile); pickleFile.close()

    V1Dim = pickleObj['SIZE']; del pickleObj['SIZE']
    connections = pickleObj['Connections']; del pickleObj['Connections']

    afferents = ['LGNOffAfferent',  'LGNOnAfferent']
    excitatory= ['LateralExcitatory-0', 'LateralExcitatory-1', 'LateralExcitatory-2', 'LateralExcitatory-3']
    inhibitory = ['LateralInhibitory-0','LateralInhibitory-1', 'LateralInhibitory-2', 'LateralInhibitory-3',
                  'LateralInhibitory-4','LateralInhibitory-5','LateralInhibitory-6', 'LateralInhibitory-7',
                  'LateralInhibitory-8','LateralInhibitory-9', 'LateralInhibitory-10']

    for connectionName in connections:
        allWeights = pickleObj[connectionName];group = None
        assert len(allWeights) == 48*48
        if connectionName == 'LateralExcitatory': group = excitatory
        if connectionName == 'LateralInhibitory':  group = inhibitory
            
        if group is not None:
            for connectionRing in group:
                groupWeights = allWeights[:]
                
                for i in range(V1Dim):
                    for j in range(V1Dim):
                        weights = groupWeights[0]
                        originalW = topo.sim['V1'].projections(connectionRing).cfs[i][j].weights

                        mask = originalW.copy()
                        mask[mask>0.0] = 1.0
                        assert (originalW.shape == weights.shape)
                        assert (weights.shape == mask.shape)
                        topo.sim['V1'].projections(connectionRing).cfs[i][j].weights = (mask*weights)
                        groupWeights = groupWeights[1:]
                assert groupWeights == []
        else:
            assert (connectionName in afferents)
            for i in range(V1Dim):
                for j in range(V1Dim):
                    weights = allWeights[0]
                    topo.sim['V1'].projections(connectionName).cfs[i][j].weights = weights
                    allWeights = allWeights[1:]
            assert allWeights == []

############################################################
############################################################

class normaliseFn:
    def __init__(self,normaliseFactor):
        self.normaliseFactor = normaliseFactor

    def __call__(self,x):
        x *= self.normaliseFactor
        
    def __repr__(self):
        return 'normaliseFn(%f)' %  self.normaliseFactor

def originalBoundsWeights(sheetName, connName, connectionParams, wPatternClass, wPatternParams, center_row,center_col):

    sheetObj = topo.sim[sheetName]
    dummyParams = connectionParams.copy()

    # To prevent name clashes
    dummyInds = [int(el.name[5:]) for el in topo.sim.connections() if el.name[:5] == 'dummy']
    if dummyInds == []: dummyInd = 0
    else:               dummyInd = max(dummyInds)+1

    dummyName= 'dummy%d' % dummyInd
    dummyParams['strength'] = 0.0; dummyParams['name'] = dummyName

    # Making the connection and getting the bounds.
    conn = topo.sim.connect(sheetName,sheetName, **dummyParams)
    cfObj = sheetObj.projections(dummyName).cfs[center_row,center_col]
    bounds = cfObj.input_sheet_slice.compute_bounds(sheetObj)

    weights = sheetObj.projections(dummyName).cfs[center_row,center_col].weights[:]
    return (bounds, weights)

def boundsChanged(bounds, sheetName, connName, center_row, center_col):

    sheetObj = topo.sim[sheetName]
    # First connection (ring) will always exist.
    ringCf = sheetObj.projections('%s-0' % connName).cfs[center_row,center_col]
    ringBounds = ringCf.input_sheet_slice.compute_bounds(sheetObj)
    
    return (ringBounds.lbrt() != bounds.lbrt())

def readBoundsWeights(sheetName, connName, connectionParams, wPatternClass, 
                      wPatternParams, center_row,center_col):

    (bounds, weights) = originalBoundsWeights(sheetName, connName, connectionParams, 
                                              wPatternClass, wPatternParams, center_row,center_col)
    # Remove the dummy connections
    [el.remove() for el in topo.sim.connections() if (el.name[:len('dummy')] == 'dummy')]
    return (bounds, weights, False)

def rawWeightPattern(wPatternClass, wPatternParams, bbwidth, extraParams):

    rawParams = wPatternParams.copy()
    rawParams.update(extraParams)

    rawParams['output_fns'] = []
    diskParams = extraParams.copy()
    diskParams.update({'smoothing':0.0, 'aspect_ratio':1.0, 'size':bbwidth})

    diskMask = pattern.Disk(**diskParams)()
    rawParams.update({'mask':diskMask})

    return wPatternClass(**rawParams)


def squareErrorPlots(weights, sheetName, connName, ringNumber,center_row,center_col, PLOTS=True):

    sheetObj = topo.sim[sheetName]
    ringNames = [ "%s-%d" % (connName, i) for i in range(ringNumber)]
    ringWeightList = [sheetObj.projections(name).cfs[center_row,center_col].weights for name in ringNames]
    ringWeights = numpy.add.reduce(ringWeightList)

    if PLOTS:
        toimage(weights).save('%s.png' % connName) # The original
        [toimage(ringWeight).save('%s-%d.png' % (connName, i)) for (i,ringWeight) in enumerate(ringWeightList)]
        toimage(ringWeights).save('COMPOSITE-%s.png' % connName)
        toimage(weights - ringWeights).save('DIFF-%s.png' % connName)

    error = ((weights - ringWeights)**2).sum()
    return error

def makeDelayedLaterals(sheetName, connName, connectionParams, ringNumber, wPatternClass, wPatternParams):

    # Getting the center and density of the sheet
    sheetObj = topo.sim[sheetName]
    sheet_rows, sheet_cols = sheetObj.shape
    center_row, center_col = sheet_rows/2,sheet_cols/2
    xdensity = ydensity = sheetObj.xdensity
    cunitx, cunity = sheetObj.matrixidx2sheet(center_row, center_col)

    (bounds, weights, newBoundsFlag) = readBoundsWeights(sheetName, connName, 
                                                         connectionParams, 
                                                         wPatternClass, wPatternParams,
                                                         center_row,center_col)
    # Getting the bounds and bounds width
    l,b,r,t = bounds.lbrt(); bbwidth = r-l

    # Making the raw weight pattern to normalise from.
    wPatternParamsRaw = wPatternParams.copy()
    extraParams = {'x':cunitx, 'y':cunity,  'xdensity':xdensity,  'ydensity':ydensity, 'bounds':bounds}
    raw_weights_pattern = rawWeightPattern(wPatternClass, wPatternParamsRaw, bbwidth,extraParams)

    # Creating the actual weight pattern
    weight_pattern = wPatternClass(**wPatternParams)
    weight_pattern.output_fns = []

    # Getting the normalisation factor
    normalisation_sum = raw_weights_pattern().sum()
    normalisation_factor = 1.0 / normalisation_sum
    
    # Adding the output function
    weight_pattern.output_fns = [normaliseFn(normalisation_factor)]

    # Setting the appropriate ring number
    if ringNumber == 'MAX':
        (dim,dim) = raw_weights_pattern().shape
        if (dim % 2) != 1: print('*WARNING*: Cf dimensions should be odd!')
        ringNumber = (dim - 1) / 2
    if ringNumber < 1:
        print('Ring number has to be greater than one!')
        sys.exit()

    thickness = bbwidth / (ringNumber * 2)
    for i in range(ringNumber):

        if (i == 0): 
            mask = pattern.Disk(size=2*thickness, smoothing=0.0)
        else: 
            ring_size = 2*thickness*i+thickness
            mask = pattern.Ring(size=ring_size, thickness=thickness, smoothing=0.0)
        
        delayName = '%s-%d' % (connName,i)


        originalWeightOutputFns = None
        if 'weights_output_fns' in connectionParams:
            originalWeightOutputFns = connectionParams['weights_output_fns'] 

        connectionParams.update({'cf_shape':mask, 'name':delayName, 
                                 'weights_generator':weight_pattern, 'autosize_mask':False, 'weights_output_fns':[]})

        connectionParams.update({'dest_port':('Activity','JointNormalize',connName)})    
        conn = topo.sim.connect(sheetName, sheetName, **connectionParams)

        if originalWeightOutputFns is not None:
            conn.weights_output_fns = originalWeightOutputFns

    # Checking to see if the bounds have changed
    boundsChangedFlag = boundsChanged(bounds, sheetName, connName, center_row, center_col)

    error = squareErrorPlots(weights, sheetName, connName, ringNumber,center_row,center_col)
    print ' Squared error in CF weights for %d rings in %s is: %f' % (ringNumber, sheetName, error)

    return newBoundsFlag | boundsChangedFlag
