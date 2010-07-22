"""
Under construction: functions to measure aftereffects.

"""
from numpy import array

from topo.command.analysis import decode_feature
from topo.command.basic import pattern_present

from topo.base.arrayutil import wrap
from topo.misc.util import frange
from topo.misc.keyedlist import KeyedList

import topo
from math import pi
import copy
from topo import pattern

import pylab

from topo.command.pylabplots import vectorplot

def measure_tae():
   print "Measuring initial perception of all orientations..."
   before=test_all_orientations(0.0,0.0)
   pylab.figure(figsize=(5,5))
   vectorplot(degrees(before.keys()),  degrees(before.keys()),style="--") # add a dashed reference line
   vectorplot(degrees(before.values()),degrees(before.keys()),\
             title="Initial perceived values for each orientation")

   print "Adapting to pi/2 gaussian at the center of retina for 90 iterations..."
   for p in ["LateralExcitatory","LateralInhibitory","LGNOnAfferent","LGNOffAfferent"]:
      # Value is just an approximate match to bednar:nc00; not calculated directly
      topo.sim["V1"].projections(p).learning_rate = 0.005

   inputs = [pattern.Gaussian(x = 0.0, y = 0.0, orientation = pi/2.0,
                     size=0.088388, aspect_ratio=4.66667, scale=1.0)]
   topo.sim['Retina'].input_generator.generators = inputs
   topo.sim.run(90)


   print "Measuring adapted perception of all orientations..."
   after=test_all_orientations(0.0,0.0)
   before_vals = array(before.values())
   after_vals  = array(after.values())
   diff_vals   = before_vals-after_vals # Sign flipped to match conventions

   pylab.figure(figsize=(5,5))
   pylab.axvline(90.0)
   pylab.axhline(0.0)
   vectorplot(wrap(-90.0,90.0,degrees(diff_vals)),degrees(before.keys()),\
             title="Difference from initial perceived value for each orientation")




def test_all_orientations(x=0,y=0):
   results=KeyedList()
   for i in frange(0.0, 1.0, 1.0/36.0, inclusive=True):
       input = pattern.Gaussian(x=x, y=y, orientation=i*pi,
                        size=0.088388, aspect_ratio=4.66667, scale=1.0)
       pattern_present(inputs={'Retina' : input}, duration=1.0,
                       plastic=False,
                       overwrite_previous=True,
                       apply_output_fn=True)
       if hasattr(topo,'guimain'):
           topo.guimain.refresh_activity_windows()
       results[i]=decode_feature(topo.sim['V1'], preference_map="OrientationPreference")
   return results









def measure_dae(scale=0.6):
   print "Measuring initial perception of all directions..."
   before=test_all_directions(0.0,0.0,scale)
   pylab.figure(figsize=(5,5))
   vectorplot(degrees(before.keys()),  degrees(before.keys()),style="--") # add a dashed reference line
   vectorplot(degrees(before.values()),degrees(before.keys()),\
             title="Initial perceived values for each direction")

   print "Adapting to pi/2 gaussian at the center of retina for 90 iterations..."
   
   for p in ["LateralExcitatory","LateralInhibitory",
             "LGNOnAfferent0","LGNOffAfferent0",
             "LGNOnAfferent1","LGNOffAfferent1",
             "LGNOnAfferent2","LGNOffAfferent2",
             "LGNOnAfferent3","LGNOffAfferent3"]:
      # Value is just an approximate match to bednar:nc00; not calculated directly
      topo.sim["V1"].projections(p).learning_rate = 0.005


##    g = pattern.Gaussian(x=0.0,y=0.0,orientation=pi/2.0,size=0.088388,
##                          aspect_ratio=4.66667,scale=1.0)

   g = pattern.SineGrating(frequency=2.4,phase=0.0,orientation=pi/2,scale=scale)
   for j in range(4):
      topo.sim['Retina%s'%j].set_input_generator(pattern.Sweeper(
         generator=copy.deepcopy(g),
         speed=2.0/24.0,
         step=j))
      
   topo.sim.run(90)

   print "Measuring adapted perception of all directions..."
   after=test_all_directions(0.0,0.0,scale)
   before_vals = array(before.values())
   after_vals  = array(after.values())
   diff_vals   = before_vals-after_vals # Sign flipped to match conventions

   pylab.figure(figsize=(5,5))
   pylab.axvline(180.0)
   pylab.axhline(0.0)
   vectorplot(wrap(-2*90.0,2*90.0,degrees(diff_vals)),degrees(before.keys()),\
             title="Difference from initial perceived value for each direction")






def test_all_directions(x=0,y=0,scale=0.6):
   results=KeyedList()
##    g=pattern.Gaussian(x=x,y=y,aspect_ratio=4.66667,
##                        size=0.088388,scale=1.0)
   g = pattern.SineGrating(frequency=2.4,phase=0.0,orientation=0,scale=scale)
   inputs = {}
   for j in range(4):
      inputs['Retina%s'%j] = pattern.Sweeper(generator=copy.deepcopy(g),
                                              speed=2.0/24.0)

   for i in frange(0.0, 2.0, 2.0/18.0, inclusive=True): 

      orientation = i*pi+pi/2

      for j in range(4):
         s = inputs['Retina%s'%j] 
         s.step = j
         s.orientation = orientation
      
      pattern_present(inputs=inputs,duration=1.0,
                      plastic=False,
                      overwrite_previous=True,
                      apply_output_fn=True)

      if hasattr(topo,'guimain'):
         topo.guimain.refresh_activity_windows()

      results[i]=decode_feature(topo.sim['V1'],
                                preference_map="DirectionPreference")

   return results
         

def degrees(x):
  "Convert scalar or array argument from the range [0.0,1.0] to [0.0,180.0]"
  return 180.0*array(x)


