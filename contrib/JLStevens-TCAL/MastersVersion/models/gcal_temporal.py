# ; -*- mode: Python;-*-
import os, copy, numpy, sys

# My own imports
import gcal_vanilla
from modelUtils.VSDUtils import createVSDLayer, VSDContinuous
from modelUtils.snapshotUtils import modelIsLoaded, loadModelSnapshot, saveSnapShot

# Topographica imports
import topo
from topo.transferfn.basic import Hysteresis as Hysteresis 
from topo.transferfn.misc import HalfRectify as HalfRectify 
from topo.transferfn.misc import HomeostaticResponse

from topo.command.basic import load_snapshot

############################################################################
############################################################################

from numpy import ones
from topo.base.arrayutil import clip_lower
class HomeostaticResponse_Continuous(HomeostaticResponse):

    def __call__(self,x):
        if self.first_call:
            self.first_call = False
            if self.randomized_init:
                self.t = ones(x.shape, x.dtype.char) * self.t_init + \
                         (topo.pattern.random.UniformRandom() \
                          (xdensity=x.shape[0],ydensity=x.shape[1]) \
                          -0.5)*self.noise_magnitude*2
            else:
                self.t = ones(x.shape, x.dtype.char) * self.t_init
            self.y_avg = ones(x.shape, x.dtype.char) * self.target_activity

        x_orig = copy.copy(x); x -= self.t
        clip_lower(x,0);       x *= self.linear_slope

        if self.plastic: # & (float(topo.sim.time()) % 1.0 >= 0.54):
            self.y_avg = (1.0-self.smoothing)*x + self.smoothing*self.y_avg 
            self.t += self.learning_rate * (self.y_avg - self.target_activity)

HomeostaticResponse = HomeostaticResponse_Continuous

###############################################################################
# HELPER FUNCTIONS FOR MODIFYING THE MODEL #
############################################

def switchSheet(name, sheetType, extraAttrs=[], extraDict={}):
    """ Copies given base class attributes of GCAL sheet
    to create a new continuous sheet of the same name."""

    '''Retina: nominal_density,input_generator, period, phase, nominal_bounds
       LGNOn/Off: nominal_density, nominal_bounds, output_fns, measure_maps
       V1: above + plastic'''

    attrs=['nominal_density', 'nominal_bounds',
           'output_fns', 'plastic']
    attrs += extraAttrs
    
    # Getting GCAL attributes
    gcal = topo.sim[name]
    vals=[getattr(gcal,k) for k in attrs]
    # Switching to continuous time sheet
    d = dict(zip(attrs, vals))
    d.update(extraDict)
    newSheet =sheetType(**d)
    topo.sim[name] = newSheet

def OutputFns(fns, sheets):
    'Adding output functions to given sheet'  
    # Beware of copied random seed state when you want differences
    for sheet in sheets:
        gcalFns = topo.sim[sheet].output_fns
        deepcopied = copy.deepcopy(fns)
        topo.sim[sheet].output_fns = deepcopied
        

def switchAllSheets(VSDSheetFlag= True):
    
    d = {'measure_maps':False} # Probably not important
    # Switching to the continuous sheet (LGNOn/Off, V1)
    jointCFCont=topo.sheet.JointNormalizingCFSheet_Continuous
    switchSheet('LGNOff', jointCFCont,extraDict=d)
    switchSheet('LGNOn', jointCFCont,extraDict=d)

    if VSDSheetFlag: 
        switchSheet('V1', VSDContinuous)        
    else:            
        switchSheet('V1', jointCFCont)

    # Some V1 specific stuff
    joint_norm = topo.sheet.optimized.compute_joint_norm_totals_opt
    topo.sim['V1'].joint_norm_fn= joint_norm
    

###############################################################################
# MODIFYING THE MODEL #
#######################

def setConnStrength(name,strength,connections):
    conn = connections[name]
    conn.strength = strength

def setConnDelay(name,delay,connections):
    conn = connections[name]
    conn.delay = delay

def setupRetina(stimulus, timestep):
    retina = topo.sim['Retina']

    stimulus.xdensity = retina.xdensity
    stimulus.ydensity = retina.ydensity
    stimulus.bounds = retina.bounds

    retina.input_generator = stimulus
    retina.period = timestep

def setSheetPlasticity(plastic):
    topo.sim['Retina'].plastic = plastic
    topo.sim['LGNOn'].plastic = plastic
    topo.sim['LGNOff'].plastic = plastic
    topo.sim['V1'].plastic = plastic

def setLateralPropagationDelays(conns, LateralConnectionList, propagationDelay, fixedDelay):
    [setConnDelay(name,  fixedDelay+propagationDelay*(ind+1),  conns) for (ind,name) in enumerate(LateralConnectionList)]

def setLateralStrength(conns, LateralConnectionList, GCStrength):
     [setConnStrength(name,GCStrength,conns) for (ind,name) in enumerate(LateralConnectionList)]
     

def modifyLateralConnections(conns, lateralName, lateralStrength, fixedDelay=None, propagationDelay=None):
    
    if fixedDelay is None: fixedDelay = 0.0
    if propagationDelay is None: propagationDelay = 0.0

    if [fixedDelay, propagationDelay] == [0.0, 0.0]: 
        print 'Must have constant OR variable delays!'; sys.exit()

    # Setting the strength
    keyList = conns.keys(); keyList.sort()
    LateralConnectionList = [k for k in keyList if k[:len(lateralName)] == lateralName ]
    LateralConnectionListPairs = [(int(el[len(lateralName)+1:]),el) for el in LateralConnectionList]
    LateralConnectionListPairs.sort(); LateralConnectionList = [el for (ind,el) in  LateralConnectionListPairs]
    setLateralStrength(conns, LateralConnectionList, lateralStrength)
    
    #  Setting the appropriate delays
    setLateralPropagationDelays(conns, LateralConnectionList, propagationDelay, fixedDelay)

###############################################################################

# Default settings matched to GCAL.
def GCALTemporal(stimulus = None,
                 timestep=0.5,
                 ringParams = ('MAX','MAX'),
                 saveSnapshotTime = None,
                 plastic=True,

                 LGN_TC = None,
                 LGNFFStrength = 2.33,
                 GCStrength=0.6,
                 GCDelay=0.01,
                 LGNDistanceDelay=None,

                 # V1 Settings
                 V1_TC = None,
                 V1FFStrength=1.5,
                 ExcStrength=1.7,
                 InhStrength=-1.4,
                 LateralDelay=0.01,
                 V1DistanceDelay=None,

                 HomeoFactor=1.0, 
                 HomeoPlastic = False ): 

    VSDSheetFlag = True;  saveSnapshotFlag = False; LOAD = None

    ###################################################
    # Modifying sheets #
    ####################
                           
    modelName = 'fit-LGN-V1.py'
    (LGNRingNo,V1RingNo) = ringParams

    if LOAD is not None:
        typPath = '../models/snapshots/'
        load_snapshot(typPath+LOAD)
    else:   
        print "Modifying model loaded in memory"          
    
    ########################
    # Sheets Modifications #
    ########################

        p = gcal_vanilla.makeParams()
        gcal_vanilla.makeSheets(p)
        # Switch out the LISSOM sheets for continuous ones
        switchAllSheets(VSDSheetFlag)
        # Setup the input generator
        setupRetina(stimulus, timestep)
        # Connecting with original GCAL connections
        gcal_vanilla.connectGCAL(p,LGNRingNo,V1RingNo)
                             

    ############################
    # Connection Modifications #
    ############################
                             
    'Alternatively, could use topo.sheet[<name>].projection(<name>)'
    conns = dict([(el.name,el) for el in topo.sim.connections()])

    ''' AfferentToLGNOn, AfferentToLGNOff, LGNOnAfferent, LGNOffAfferent,
        GCLGNOn-*,  GCLGNOff-*, LateralInhibitory-*, LateralExcitatory-*
    '''

    #######
    # LGN #
    #######

    # FEEDFORWARD DELAY IS ABOUT 15MS FOR BOTH STAGES

    # Afferent from Retina to LGNOn/Off. Delay and strength.
    setConnDelay('AfferentToLGNOn',15.0,conns)
    setConnDelay('AfferentToLGNOff',15.0,conns)
    setConnStrength('AfferentToLGNOn', LGNFFStrength,conns)
    setConnStrength('AfferentToLGNOff', LGNFFStrength,conns)

    assert GCStrength > 0
    modifyLateralConnections(conns, 'GCLGNOn', GCStrength, GCDelay, LGNDistanceDelay)
    modifyLateralConnections(conns, 'GCLGNOff', GCStrength, GCDelay, LGNDistanceDelay)

    ######
    # V1 #
    ######
                             
    setConnDelay('LGNOnAfferent',15.0,conns)
    setConnDelay('LGNOffAfferent',15.0,conns)
    setConnStrength('LGNOnAfferent', V1FFStrength,conns)
    setConnStrength('LGNOffAfferent' ,V1FFStrength,conns)

    assert ExcStrength > 0;  assert InhStrength < 0
    modifyLateralConnections(conns, 'LateralExcitatory', ExcStrength, LateralDelay, V1DistanceDelay)
    modifyLateralConnections(conns, 'LateralInhibitory', InhStrength, LateralDelay, V1DistanceDelay)

    ##########################
    # Sheet output functions #
    ##########################

    setLGNOutputFns(LGN_TC)
    setV1OutputFns(V1_TC, HomeoFactor, HomeoPlastic) 

    ###############################
    # Plasticity and snapshotting #
    ###############################
    
    if saveSnapshotFlag:
        setSheetPlasticity(plastic=True) # No point making snapshot if plasticity if off                                
        saveSnapShot(saveSnapshotTime)

    setSheetPlasticity(plastic)      # Set as desired

def setLGNOutputFns(LGN_TC):
    ''' Add Hysteresis to the LGN and HalfRectification '''

    LGNoutputList = []

    if LGN_TC is not None:  
        LGNoutputList += [Hysteresis(time_constant=LGN_TC)]

    LGNoutputList += [HalfRectify()]
    OutputFns(LGNoutputList,['LGNOn','LGNOff'])


def setV1OutputFns(V1_TC, HomeoFactor=1, HomeoPlastic=False): 
    ''' Add Hysteresis and homeostatic plasticity to V1 and HalfRectification '''
    V1outputList = []

    if V1_TC is not None:
        V1outputList += [Hysteresis(time_constant=V1_TC)]
        
    # False for off and None for default
    if HomeoFactor is not False:
        learning_rate=0.001    * HomeoFactor
        smoothing=0.999 # Thanks to Jan for this conversion.
        smoothing = 1 - (1 - smoothing) / HomeoFactor  
        homeostaticFn = HomeostaticResponse(learning_rate=learning_rate)
        
        homeostaticFn.plastic = HomeoPlastic
        V1outputList += [homeostaticFn]
    else:
        print "\n*Warning*: Using half rectification instead of hysteresis!\n"
        V1outputList += [HalfRectify()] # If homeostatis is off, this non-linearity to be used instead
    
    OutputFns(V1outputList,['V1'])
    # Calibrate with homeostasis OFF - ensure balance err towards inhibition    
    # Check that learning is off properly


if __name__ == '__main__':

    # Should reproduce GCAL with delays and continuous sheets.
    # Note that tsettle has been removed.
    p = gcal_vanilla.makeParams()
    stimulus = gcal_vanilla.GCALStimulusPattern(p,'Gaussian')
    GCALTemporal(stimulus) # Using default (GCAL) parameters
