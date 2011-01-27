# -*- coding: utf-8 -*-
from topo.pattern.basic import Ring
from topo.misc.numbergenerator import UniformRandom
import scipy
import numpy

def fit_ring_to_fft(fft,display=False):
    """
    Fits a disk to two dimensional data 'fft'.
    The returned parameters are:
    (size,aspect ratio,smoothing,orientation,scale)

    The fitted ring can be obtained by:
    (Ring(size=p[0],aspect_ratio=p[1],smoothing=p[2],thickness=0.0,orientation=p[3],xdensity=sx,ydensity=sy,scale=p[4])())

    where sx,sy is the dimensions of the fft

    Notes: it is a good idea to substract the mean from the data before fft to get rid of the DC componenet, otherwise one can get strange fits

    """
    from scipy import optimize


    fitfunc = lambda p,sx,sy: (Ring(size=p[0],aspect_ratio=p[1],smoothing=p[2],thickness=0.0,orientation=p[3],xdensity=sx,ydensity=sy,scale=p[4])()) # Target function
    errfunc = lambda p,x,sx,sy: numpy.mean(numpy.power(fitfunc(p,sx,sy) - x,2)) # Distance to the target function
    
    
    sx,sy = numpy.shape(fft)

    rand = UniformRandom(seed=513)
    minerr = 100000000000000000000
    reps=100
    for i in xrange(0,reps):
    	p0 = [0.1,1.0,0.001,1.0,600] # Initial guess for the parameters
	p1 = numpy.array(p0) * [rand()*3,rand()*2.0,rand()*3.0,rand()*numpy.pi,rand()*2]

	(p,success,c)=optimize.fmin_tnc(errfunc,p1[:],bounds=[(0,0.3),(0,2),(0,0.01),(0,numpy.pi),(0.0,1200)],args=(fft,sx,sy),approx_grad=True,messages=0)
        
        err = errfunc(p,fft,sx,sy)
        if err < minerr:
	   minerr=err
           opt = p

    if display:
      import pylab
      pylab.figure()
      pylab.imshow(fitfunc(p,sx,sy))
      pylab.colorbar()
      pylab.figure()
      pylab.imshow(fft)
      pylab.colorbar()
      #pylab.show()
    return opt

