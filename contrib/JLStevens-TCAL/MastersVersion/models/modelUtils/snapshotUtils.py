import pickle, sys

import topo
from topo.command.basic import load_snapshot, save_snapshot 


def modelIsLoaded(modelName):
    if topo.sim.name != modelName: return False

    objList = topo.sim.objects().keys()
    requiredObjs = ['Retina','LGNOn','LGNOff','V1']
    diff = set(objList).symmetric_difference(set(requiredObjs))
    return ((diff == set()) or (diff == set(['Data'])))

def scheduleInfo(steps, n=10):
    stepNos = numpy.arange(0.0,steps,n)    
    for stepNo in stepNos:
        topo.sim.schedule_command(stepNo,'pass')

def loadModelSnapshot(saveSnapshotTime, 
                      LGNRingNo=5, V1RingNo=5, basename='snapshot'): 
    # LGNRingNo and V1RingNo are paramters that change the model structure
    typPath = '../models/snapshots/'

    loadSnapshot = "%s-LGN%s-V1%s-T%s.typ" % (basename, str(LGNRingNo), str(V1RingNo), str(saveSnapshotTime))
    if saveSnapshotTime is not None:
        fullTypPath = typPath+loadSnapshot
        snapshotExists = os.path.exists(fullTypPath)
        # Load snapshot and return
        if snapshotExists:
            print "Loading snapshot %s" % loadSnapshot
            load_snapshot(typPath+loadSnapshot)
            print ("Snapshot %s loaded." % loadSnapshot)
            return False
        else:
            # saveSnapshotFlag. True indicates a snapshot is needed as it is desired but doesn't exist.
            return True 
    else: 
        return False
    
def saveSnapShot(saveSnapshotTime):

    if saveSnapshotTime is not None:
        print("Snapshot missing! Regenenerating over %d periods"
              % int(saveSnapshotTime))

        # typName ='gcal_continuous_%d.typ' % int(saveSnapshotTime)
        fullTypPath = typPath+ loadSnapshot#typName
        
        scheduleInfo(saveSnapshotTime, 10)
        topo.sim.run(int(saveSnapshotTime))
        save_snapshot(os.path.abspath(fullTypPath))
        print "Snapshot generated. Rerun script."
        sys.exit()


