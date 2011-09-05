import topo
from topo import param
from topo.sheet.basic import JointNormalizingCFSheet_Continuous
from topo.transferfn.basic import Hysteresis as Hysteresis

import numpy
# Source: topographica/topo/base/projection.py
class VSDContinuous(JointNormalizingCFSheet_Continuous):
    """
    CFSheet that runs continuously, with no 'resting' periods between pattern presentations.
    Extended so learning can occur with any periodicity and VSD subthreshold output available on 
    a subthreshold port which allows any downstream output functions to be avoided.
    """

    src_ports = ['Activity','VSD']
    learning_period = param.Number(default=1.0,doc=""" The periodicity with which learning should occur.""")

    def activate(self):
        """
        Collect activity from each projection, combine it to calculate
        the activity for this sheet, and send the result out. 
        
        Subclasses may override this method to whatever it means to
        calculate activity in that subclass.
        """

        self.activity *= 0.0
        tmp_dict={}

        for proj in self.in_connections:
            if (proj.activity_group != None) | (proj.dest_port[0] != 'Activity'):
                if not tmp_dict.has_key(proj.activity_group[0]):
                    tmp_dict[proj.activity_group[0]]=[]
                tmp_dict[proj.activity_group[0]].append(proj)

        keys = tmp_dict.keys()
        keys.sort()
        vsd_activity = self.activity.copy() * 0.0   # HACK ALERT! MAY NOT WORK CORRECTLY WITH PRO
        for priority in keys:
            tmp_activity = self.activity.copy() * 0.0

            for proj in tmp_dict[priority]:
                vsd_activity += numpy.abs(proj.activity.copy())
                tmp_activity += proj.activity
            self.activity=tmp_dict[priority][0].activity_group[1](self.activity,tmp_activity)


        self.send_output(src_port='VSD',data=vsd_activity)

        if self.apply_output_fns:
            for of in self.output_fns:
                of(self.activity)

        self.send_output(src_port='Activity',data=self.activity)

        # self.send_output(src_port='VSD',data=self.activity)

#####################################
# 1mm distance between hypercolumns #
#####################################


def createVSDLayer(radius, settings):

    if ['VSDLayer', 'VSDSignal'] in topo.sim.objects().keys():
        del topo.sim['VSDLayer']; del topo.sim['VSDSignal']
    
  
    topo.sim['VSDLayer'] = topo.sheet.CFSheet(
        nominal_density =topo.sim['V1'].nominal_density,
        nominal_bounds= topo.sim['V1'].nominal_bounds)

    topo.sim['VSDSignal'] = topo.sheet.CFSheet(
        nominal_density =topo.sim['V1'].nominal_density,
        nominal_bounds= topo.sim['V1'].nominal_bounds)

    topo.sim.connect('V1','VSDLayer', src_port='VSD', name='V1ToVSDLayer',
                     connection_type=topo.projection.OneToOneProjection, delay=0.001)

    scaleFactor = 0.3 # A Gaussian of size 0.3 will fit in a rectangle of size 1
    blurringPattern  = topo.pattern.Gaussian(size=radius*scaleFactor, aspect_ratio=1.0, output_fns=[topo.transferfn.DivisiveNormalizeL1()])

    topo.sim.connect('VSDLayer', 'VSDSignal', name='VSDLayerToSignal', delay=0.001, strength=1.0, connection_type=topo.projection.SharedWeightCFProjection, nominal_bounds_template=topo.sheet.BoundingBox(radius=radius), weights_generator=blurringPattern)
