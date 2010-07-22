from __future__ import with_statement
"""
Grating and contour stimuli according to Hegde and Van Essen works,
usually composite using basic patterns.

Grating stimuli subclasses:
- sinusoidal
- hyberbolic
- polar (concentric-like)
- polar (radial-like)

Contour stimuli subclasses:
- bar
- tri-star
- cross
- star/circle
- acute angle
- right angle
- obtuse angle
- semi-circle
- 3/4 arc

Stimuli from one subclass have common shape characteristics but vary in orientation,
size and/or spatial frequency.

$Id$
"""
__version__ = "$Revision$"

from math import sin, cos, pi

import numpy
from numpy.oldnumeric import maximum, minimum, sqrt, divide, greater_equal
from numpy.oldnumeric import bitwise_xor, exp, fmod, absolute, arctan2
from numpy import seterr

import param
from topo.pattern.basic import *
from topo.base.patterngenerator import PatternGenerator
from topo.misc.patternfn import float_error_ignore,arc_by_radian

## New patterns for grating stimuli

class SpiralGrating(PatternGenerator):
    """
    Archimedean spiral grating. Successive turnings of the spiral have
    a constant separation distance.

    Spiral is defined by polar equation r=size*angle plotted in gausian
    plane. Spiral starts at radian 0.0; this can be changed by orientation.
    """

    aspect_ratio = param.Number(default=1.0,bounds=(0.0,None),softbounds=(0.0,2.0),
        precedence=0.31,doc="Ratio of width to height.")
    
    thickness = param.Number(default=0.02,bounds=(0.0,None),softbounds=(0.0,0.5),
        precedence=0.60,doc="Thickness (line width) of the spiral.")
    
    smoothing = param.Number(default=0.05,bounds=(0.0,None),softbounds=(0.0,0.5),
        precedence=0.61,doc="Width of the Gaussian fall-off inside and outside the spiral.")
    
    size = param.Number(default=0.05,bounds=(0.01,None),softbounds=(0.01,2.0),
        precedence=0.62,doc="Size as density of turnings - size*angle gives the actual radius.")

    def function(self,params):
        """Archemidean spiral function."""

        aspect_ratio = params['aspect_ratio']
        x = self.pattern_x/aspect_ratio
        y = self.pattern_y
        thickness = params['thickness']
        gaussian_width = params['smoothing']
        size = params['size']

        half_thickness = thickness/2.0
        spacing = size*2*pi

        distance_from_origin = sqrt(x**2+y**2)
        distance_from_spiral_middle = fmod(spacing + distance_from_origin - size*arctan2(y,x),spacing)

        distance_from_spiral_middle = minimum(distance_from_spiral_middle,spacing - distance_from_spiral_middle)
        distance_from_spiral = distance_from_spiral_middle - half_thickness

        spiral = 1.0 - greater_equal(distance_from_spiral,0.0)

        sigmasq = gaussian_width*gaussian_width

        with float_error_ignore():
            falloff = exp(divide(-distance_from_spiral*distance_from_spiral, 2.0*sigmasq))

        return maximum(falloff, spiral)


class HyperbolicGrating(PatternGenerator):
    """
    Hyperbolic grating consists of concentric rectangular hyperbolas
    with Gaussian fall-off which share the same asymptotes:
    abs(x^2/a^2 - y^2/a^2) = 1, where a mod size = 0
    """

    aspect_ratio = param.Number(default=1.0,bounds=(0.0,None),softbounds=(0.0,2.0),
        precedence=0.31,doc="Ratio of width to height.")
    
    thickness = param.Number(default=0.05,bounds=(0.0,None),softbounds=(0.0,0.5),
        precedence=0.60,doc="Thickness of the hyperbolas.")
    
    smoothing = param.Number(default=0.1,bounds=(0.0,None),softbounds=(0.0,0.5),
        precedence=0.61,doc="Width of the Gaussian fall-off inside and outside the hyperbolas.")
    
    size = param.Number(default=0.5,bounds=(0.0,None),softbounds=(0.0,2.0),
        precedence=0.62,doc="Size as distance of inner hyperbola vertices from the centre.")

    def function(self,params):
        """Hyperbolic function."""

        aspect_ratio = params['aspect_ratio']
        x = self.pattern_x/aspect_ratio
        y = self.pattern_y
        thickness = params['thickness']
        gaussian_width = params['smoothing']
        size = params['size']

        half_thickness = thickness / 2.0

        distance_from_vertex_middle = fmod(sqrt(absolute(x**2 - y**2)),size)
        distance_from_vertex_middle = minimum(distance_from_vertex_middle,size - distance_from_vertex_middle)

        distance_from_vertex = distance_from_vertex_middle - half_thickness

        hyperbola = 1.0 - greater_equal(distance_from_vertex,0.0)

        sigmasq = gaussian_width*gaussian_width

        with float_error_ignore():
            falloff = exp(divide(-distance_from_vertex*distance_from_vertex, 2.0*sigmasq))

        return maximum(falloff, hyperbola)


class RadialGrating(PatternGenerator):
    """
    Radial grating - one sector of a circle  with Gaussian fall-off
    centered along radian 0.0 with size defined in radians. The orientation
    can be changed to choose other locations.
    """

    aspect_ratio = param.Number(default=1.0,bounds=(0.0,None),softbounds=(0.0,2.0),
        precedence=0.31,doc="Ratio of width to height.")
    
    arc_length = param.Number(default=pi/4,bounds=(0.0,None),softbounds=(0.0,2.0*pi),
        precedence=0.60,doc="Length of the sector in radians.")
    
    smoothing = param.Number(default=0.4,bounds=(0.0,None),softbounds=(0.0,0.5),
        precedence=0.61,doc="Width of the Gaussian fall-off outside the sector.")
    
    def function(self,params):
        """Radial function."""

        aspect_ratio = params['aspect_ratio']
        x = self.pattern_x/aspect_ratio
        y = self.pattern_y
        gaussian_width = params['smoothing']
        
        angle = absolute(arctan2(y,x))
        half_length = params['arc_length']/2

        radius = 1.0 - greater_equal(angle,half_length)
        distance = angle - half_length

        sigmasq = gaussian_width*gaussian_width

        with float_error_ignore():
            falloff = exp(divide(-distance*distance, 2.0*sigmasq))

        return maximum(radius, falloff)


class ConcentricRings(PatternGenerator):
    """
    Concentric rings  with the solid ring-shaped region centered at (0.0,0.0) with linearly
    increasing radius. Gaussian fall-off at the edges.
    """    

    aspect_ratio = param.Number(default=1.0,bounds=(0.0,None),softbounds=(0.0,2.0),
        precedence=0.31,doc="Ratio of width to height.")
    
    thickness = param.Number(default=0.04,bounds=(0.0,None),softbounds=(0.0,0.5),
        precedence=0.60,doc="Thickness (line width) of the ring.")
    
    smoothing = param.Number(default=0.05,bounds=(0.0,None),softbounds=(0.0,0.5),
        precedence=0.61,doc="Width of the Gaussian fall-off inside and outside the rings.")
    
    size = param.Number(default=0.4,bounds=(0.01,None),softbounds=(0.1,2.0),
        precedence=0.62,doc="Radius difference of neighbouring rings.")

    def function(self,params):
        """Concentric rings."""

        aspect_ratio = params['aspect_ratio']
        x = self.pattern_x/aspect_ratio
        y = self.pattern_y
        thickness = params['thickness']
        gaussian_width = params['smoothing']
        size = params['size']

        half_thickness = thickness / 2.0

        distance_from_origin = sqrt(x**2+y**2)

        distance_from_ring_middle = fmod(distance_from_origin,size)
        distance_from_ring_middle = minimum(distance_from_ring_middle,size - distance_from_ring_middle)

        distance_from_ring = distance_from_ring_middle - half_thickness

        ring = 1.0 - greater_equal(distance_from_ring,0.0)

        sigmasq = gaussian_width*gaussian_width

        with float_error_ignore():
            falloff = exp(divide(-distance_from_ring*distance_from_ring, 2.0*sigmasq))

        return maximum(falloff, ring)

class ArcCentered(Arc):
    """
    2D arc pattern generator (centered at the middle of the arc).
    
    Draws an arc (partial ring) of the specified size (radius*2),
    with middle at radian 0.0 and starting at arc_length/2 and ending
    at -arc_length/2. The pattern is centered at the middle of the arc.

    See the Disk class for a note about the Gaussian fall-off.
    """    

    def function(self,p):

        if p.aspect_ratio==0.0:
            return self.pattern_x*0.0
        self.pattern_x -= (1+cos(pi-p.arc_length/2))*p.size/4
       
        return arc_by_radian((self.pattern_x+p.size/2)/p.aspect_ratio, self.pattern_y, p.size,
                             (2*pi-p.arc_length/2, p.arc_length/2), p.thickness, p.smoothing)


# Number of variants for each subclass

variants = 4

## GRATINGS ##

# Sinusoidal
sin1 = [SineGrating(phase=pi/2,frequency=2.0,orientation=i*pi/variants)
        for i in range(variants)]
sin2 = [SineGrating(phase=pi/2,frequency=3.1,orientation=i*pi/variants)
        for i in range(variants)]
sin3 = [SineGrating(phase=pi/2,frequency=4.2,orientation=i*pi/variants)
        for i in range(variants)]

# Hyperbolic
hyp1 = [HyperbolicGrating(orientation=i*pi/(2*variants),size=0.36,
                          smoothing=0.05,thickness=0.04)
        for i in range(variants)]
hyp2 = [HyperbolicGrating(orientation=i*pi/(2*variants),size=0.27,
                          smoothing=0.05,thickness=0.03)
        for i in range(variants)]
hyp3 = [HyperbolicGrating(orientation=i*pi/(2*variants),size=0.18,
                          smoothing=0.05,thickness=0.015)
        for i in range(variants)]

# Polar
pol1 = [ConcentricRings(size=0.35/(1+j*0.5),thickness=0.05/(1+j*0.35),smoothing=0.05/(1+j*0.15))
        for j in range(variants)]
pol2 = [Composite(generators=[SpiralGrating(size=0.45/(1+j),thickness=0.16-j*0.012,
                  smoothing=0.05/(1+j*0.15),orientation=(i*2+1)*pi/2) for i in range(2)])
        for j in range(variants)]
pol3 = [Composite(generators=[SpiralGrating(size=0.75/(1+j*0.8),thickness=0.17-j*0.013,
                  smoothing=0.06/(1+j*0.15),orientation=i*2*pi/4) for i in range(4)])
        for j in range(variants)]
pol4 = [Composite(generators=[SpiralGrating(size=1.05/(1+j*0.7),thickness=0.18-j*0.014,
                  smoothing=0.07/(1+j*0.15),orientation=i*2*pi/6) for i in range(6)])
        for j in range(variants)]
pol5 = [Composite(generators=[SpiralGrating(size=1.35/(1+j*0.6),thickness=0.19-j*0.015,
                  smoothing=0.08/(1+j*0.15),orientation=i*2*pi/8) for i in range(8)])
        for j in range(variants)]
pol6 = [Composite(generators=[RadialGrating(arc_length=1.0/(j*2.0+1),smoothing=0.15/(1+j*0.4),
                  orientation=i*2*pi/((j+1)*2)) for i in range((j+1)*2)])
        for j in range(variants)]

## CONTOUR STIMULI ##

# Bar
bar1 = [Rectangle(orientation=j*pi/4,smoothing=0.015,aspect_ratio=0.1,size=0.5)
        for j in range(variants)]
bar2 = [Rectangle(orientation=j*pi/4,smoothing=0.015,aspect_ratio=0.2,size=0.25)
        for j in range(variants)]

# Tri-star
par = 3 # number of parts of pattern
# This function is just abbreviation for next lines to avoid redundant code. Returns orientation
# of pattern parts in composite. Parameter i identify part of pattern, j stays for variant.
# This function is redefined later in code as necessary.
def angs(i,j):
  """Calculation of angles."""
  return j*pi/2+i*2*pi/par
size = 0.5 # size of basic pattern part
star1 = [Composite(generators=[Rectangle(orientation=angs(i,j),smoothing=0.015,aspect_ratio=0.2,
                   size=size/2,x=-size/4*sin(angs(i,j)),y=size/4*cos(angs(i,j)))
                   for i in range(par)])
         for j in range(variants)]
size = 0.25
star2 = [Composite(generators=[Rectangle(orientation=angs(i,j),smoothing=0.015,aspect_ratio=0.4,
                   size=size/2,x=-size/4*sin(angs(i,j)),y=size/4*cos(angs(i,j)))
                   for i in range(par)])
         for j in range(variants)]

# Cross
par = 2
def angs(i,j):
  """Calculation of angles."""
  return j*pi/8+i*pi/par
star3 = [Composite(generators=[Rectangle(orientation=angs(i,j),smoothing=0.015,
                   aspect_ratio=0.1,size=0.5) for i in range(par)])
         for j in range(variants)]
star4 = [Composite(generators=[Rectangle(orientation=angs(i,j),smoothing=0.015,
                   aspect_ratio=0.2,size=0.25) for i in range(par)])
         for j in range(variants)]

# Star/Circle
par = 5
def angs(i,j):
  """Calculation of angles."""
  return j*pi+i*2*pi/par
size = 0.5
star5 = [Composite(generators=[Rectangle(orientation=angs(i,j),smoothing=0.015,aspect_ratio=0.2,
                   size=size/2,x=-size/4*sin(angs(i,j)),y=size/4*cos(angs(i,j))) for i in range(par)])
         for j in range(2)]
par = 3
star5.append(Composite(generators=[Rectangle(orientation=i*2*pi/par,smoothing=0.015,
                       aspect_ratio=0.1,size=size) for i in range(par)]))
star5.append(Ring(smoothing=0.015,thickness=0.05,size=0.5))

par = 5
def angs(i,j):
  """Calculation of angles."""
  return j*pi+i*2*pi/par
size = 0.25
star6 = [Composite(generators=[Rectangle(orientation=angs(i,j),smoothing=0.015,aspect_ratio=0.4,
                   size=size/2,x=-size/4*sin(angs(i,j)),y=size/4*cos(angs(i,j))) for i in range(par)])
         for j in range(2)]
par = 3
star6.append(Composite(generators=[Rectangle(orientation=i*2*pi/par,smoothing=0.015,
                       aspect_ratio=0.2,size=size) for i in range(par)]))
star6.append(Ring(smoothing=0.015,thickness=0.05,size=0.25))

# Acute angle
angs = [(i*2*pi/16) for i in [-1,1]]
size = 0.5
ang1 = [Composite(generators=[Rectangle(orientation=angs[i],smoothing=0.015,
                aspect_ratio=0.1,size=size,x=-size/2*sin(angs[i]))
                for i in range(2)],orientation=j*2*pi/variants)
        for j in range(variants)]
size = 0.25
ang2 = [Composite(generators=[Rectangle(orientation=angs[i],smoothing=0.015,
                aspect_ratio=0.2,size=size,x=-size/2*sin(angs[i]))
                for i in range(2)],orientation=j*2*pi/variants)
        for j in range(variants)]

# Right angle
angs = [(i*2*pi/8) for i in [-1,1]]
size = 0.5
ang3 = [Composite(generators=[Rectangle(orientation=angs[i],smoothing=0.015,
                aspect_ratio=0.1,size=size,x=-size/2*sin(angs[i]))
                for i in range(2)],orientation=j*2*pi/variants)
        for j in range(variants)]
size = 0.25
ang4 = [Composite(generators=[Rectangle(orientation=angs[i],smoothing=0.015,
                aspect_ratio=0.2,size=size,x=-size/2*sin(angs[i]))
                for i in range(2)],orientation=j*2*pi/variants)
        for j in range(variants)]

# Obtuse angle
angs = [(i*2*pi/6) for i in [-1,1]]
size = 0.5
ang5 = [Composite(generators=[Rectangle(orientation=angs[i],smoothing=0.015,
                aspect_ratio=0.1,size=size,x=-size/2*sin(angs[i]))
                for i in range(2)],orientation=j*2*pi/variants)
        for j in range(variants)]
size = 0.25
ang6 = [Composite(generators=[Rectangle(orientation=angs[i],smoothing=0.015,
                aspect_ratio=0.2,size=size,x=-size/2*sin(angs[i]))
                for i in range(2)],orientation=j*2*pi/variants)
        for j in range(variants)]

# Quarter arc
arc1 = [ArcCentered(arc_length=pi/2,smoothing=0.015,thickness=0.05,size=0.5,
                    orientation=i*2*pi/variants)
        for i in range(variants)]
arc2 = [ArcCentered(arc_length=pi/2,smoothing=0.015,thickness=0.05,size=0.25,
                    orientation=i*2*pi/variants)
        for i in range(variants)]

# Semi-circle
arc3 = [ArcCentered(arc_length=pi,smoothing=0.015,thickness=0.05,size=0.5,
                    orientation=i*2*pi/variants)
        for i in range(variants)]
arc4 = [ArcCentered(arc_length=pi,smoothing=0.015,thickness=0.05,size=0.25,
                    orientation=i*2*pi/variants)
        for i in range(variants)]

# 3/4 arc
arc5 = [ArcCentered(arc_length=3*pi/2,smoothing=0.015,thickness=0.05,size=0.5,
                    orientation=i*2*pi/variants)
        for i in range(variants)]
arc6 = [ArcCentered(arc_length=3*pi/2,smoothing=0.015,thickness=0.05,size=0.25,
                    orientation=i*2*pi/variants)
        for i in range(variants)]


## Stimuli subclasses
sinusoidal = sin1 + sin2 + sin3
hyperbolic = hyp1 + hyp2 + hyp3
polar = pol1 + pol2 + pol3 + pol4 + pol5 + pol6
concentric_like = pol1 + pol2[1:] + pol3[2:] + pol4[2:] + pol5[3:]
radial_like = pol2[:1] + pol3[:2] + pol4[:2] + pol5[:3] + pol6
bar = bar1 + bar2
tristar = star1 + star2
cross = star3 + star4
star = star5 + star6
acute = ang1 + ang2
right = ang3 + ang4
obtuse = ang5 + ang6
quarter = arc1 + arc2
semi = arc3 + arc4
threeqtrs = arc5 + arc6

## Stimuli subclasses
all_stimuli_subclasses = [sinusoidal,
                          hyperbolic,
                          concentric_like,
                          radial_like,
                          bar,
                          tristar,
                          cross,
                          star,
                          acute,
                          right,
                          obtuse,
                          quarter,
                          semi,
                          threeqtrs ]
grating_stimuli_subclasses = all_stimuli_subclasses[:4]
contour_stimuli_subclasses = all_stimuli_subclasses[4:]

## Stimuli subclasses labels
all_stimuli_subclasses_labels = ['sinusoidal',
                             'hyperbolic',
                             'concentric-like',
                             'radial-like',
                             'bar',
                             'tri-star',
                             'cross',
                             'star/circle',
                             'acute angle',
                             'right angle',
                             'obtuse angle',
                             'quarter arc',
                             'semi-circle',
                             '3/4 arc' ]
grating_stimuli_subclasses_labels = all_stimuli_subclasses_labels[:4]
contour_stimuli_subclasses_labels = all_stimuli_subclasses_labels[4:]

