import topo
from tools.manageUtils import Settings

#############################################################
# This is the file that contain the settings for the models #
#############################################################

CalibratedLGNParameters = {'ringParams':(1,'MAX'),'saveSnapshotTime':None, 'plastic':False, 'timestep':0.5,
                           'LGN_TC':0.03, 'LGNFFStrength':0.5, 'GCStrength':11.0, 'GCDelay':35, 'LGNDistanceDelay':None,
                           'V1_TC': 0.01, 'V1FFStrength':8.0, 'ExcStrength':3.0,'InhStrength':-12.0, 'LateralDelay':0.0,
                           'V1DistanceDelay':0.5, 'HomeoPlastic':False }

CALIBRATEDLGN = Settings('Fitting the LGN', 'The settings used to calibrated the LGN.')
CALIBRATEDLGN.setModel(CalibratedLGNParameters)
CALIBRATEDLGN.setRecordedSheets(['Retina', 'LGNOn', 'LGNOff'])
CALIBRATEDLGN.setConstParams({'pattern':'Disk', 'size':0.07385, 'smoothing':0.0})
CALIBRATEDLGN.setStepParams(250.0, 0.01, 0.85, 1.0)

######
# V1 #
######

CALIBRATEDV1 = CALIBRATEDLGN.copy('Fitting the V1 response','The settings used to calibrated the V1 response.')
CALIBRATEDV1.setConstParams({'pattern':'SineGrating', 'frequency':2.9, 'mask_shape':topo.pattern.Disk(size=0.7, smoothing=0.0)})
CALIBRATEDV1.setRecordedSheets(['Retina', 'LGNOn', 'LGNOff','V1']);

#######
# VSD #
#######

CALIBRATEDVSD = CALIBRATEDV1.copy('Testing the VSD layers','The settings used to calibrated the the VSD response.')
CALIBRATEDVSD.setConstParams({'pattern':'Gabor', 'size':0.1})
CALIBRATEDVSD.setRecordedSheets(['Retina', 'LGNOn', 'LGNOff','V1', 'VSDSignal']);
