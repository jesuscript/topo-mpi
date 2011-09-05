import topo
import pickle

def pickleWeight():

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
                
    pickleFile = open('weights.pickle','w')
    pickle.dump(pickleObj,pickleFile)
    
def unpickleWeightV1():
    pickleFile = open('weights.pickle','r')
    pickleObj = pickle.load(pickleFile)
    pickleFile.close()

    V1Dim = pickleObj['SIZE']
    del pickleObj['SIZE']
    connections = pickleObj['Connections']
    del pickleObj['Connections']

    afferents = ['LGNOffAfferent',  'LGNOnAfferent']
    excitatory= ['LateralExcitatory-0', 'LateralExcitatory-1', 'LateralExcitatory-2', 'LateralExcitatory-3']
    inhibitory = ['LateralInhibitory-0','LateralInhibitory-1', 'LateralInhibitory-2', 'LateralInhibitory-3',
                  'LateralInhibitory-4','LateralInhibitory-5','LateralInhibitory-6', 'LateralInhibitory-7',
                  'LateralInhibitory-8','LateralInhibitory-9', 'LateralInhibitory-10']

    for connectionName in connections:
        allWeights = pickleObj[connectionName]
        assert len(allWeights) == 48*48

        group = None
        if connectionName == 'LateralExcitatory': group = excitatory
        if connectionName == 'LateralInhibitory':  group = inhibitory
            
        if group is not None:
            for connectionRing in group:
                groupWeights = allWeights[:]
                
                for i in range(V1Dim):
                    for j in range(V1Dim):
                        weights = groupWeights[0]
                        # mask = topo.sim['V1'].projections(connectionRing).cfs[i][j].mask
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
            
