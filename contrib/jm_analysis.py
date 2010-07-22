"""
Functions that compute circular variance and orientation bandwidth from the full_matrix data.

"""
from math import pi,sin,cos,sqrt

import numpy
from numpy import array

import topo

#
# Smoothing function
#
def smooth(x,window_len=10,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string   
    From: http://www.scipy.org/Cookbook/SignalSmooth
    """
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."
    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."
    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s=numpy.r_[2*x[0]-x[window_len:1:-1],x,2*x[-1]-x[-1:-window_len:-1]]
    if window == 'flat': #moving average
        w=ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='same')
    return y[window_len-1:-window_len+1]

#
#   Statistics
#
def circular_variance(full_matrix):
    """This function expects as an input a object of type FullMatrix which contains
       responses of all neurons in a sheet to stimuli with different varying parameter values.
       It computes Circular Variance as found in
       Orientation Selectivity in Macaque V1: Diversity and Laminar Dependece
       Ringach, Shapley, Hawken, 2002
    """ 
    # And of course it is kind of speedy Gonzales 8-)

    feature_names=[feature.name for feature in full_matrix.features]
    X,Y = full_matrix.matrix_shape
    dim_or=full_matrix.dimensions[feature_names.index('orientation')]
    dim_ph=full_matrix.dimensions[feature_names.index('phase')]
    
    # list of sin. gratings orientations
    orientations=full_matrix.features[feature_names.index('orientation')].values
    
    # count the table of sin and cos, to save many expensive float operations..
    sin2o=[sin(2*o) for o in orientations]
    cos2o=[cos(2*o) for o in orientations]
    
    # list of max orientation response among phases for each cell (x,y)
    maxs = array(
        [[[array([full_matrix.full_matrix[0][o][p][x][y] for p in xrange(dim_ph)]).max()
            for y in xrange(Y)]
                for x in xrange(X)]
                    for o in xrange(dim_or)])
   
    circular_variance = array(
        [[1 - sqrt((sum([maxs[o][x][y] * sin2o[o] for o in xrange(dim_or)]))**2 +
                   (sum([maxs[o][x][y] * cos2o[o] for o in xrange(dim_or)]))**2)
            / sum([maxs[o][x][y] for o in xrange(dim_or)]) 
         for y in xrange(X)]
             for x in xrange(Y)])
    
    return circular_variance

def mean_activity(full_matrix):
    """This function expects as an input a object of type FullMatrix which contains
       responses of all neurons in a sheet to stimuly with different varying parameter values.
       It computes mean activity for each neuron.
    """ 

    feature_names=[feature.name for feature in full_matrix.features]
    X,Y = full_matrix.matrix_shape
    dim_or=full_matrix.dimensions[feature_names.index('orientation')]
    dim_ph=full_matrix.dimensions[feature_names.index('phase')]
    
    means = array(
        [[array([full_matrix.full_matrix[0][o][p][x][y] for p in xrange(dim_ph) for o in xrange(dim_or)]).mean()
            for y in xrange(Y)]
                for x in xrange(X)])
   
    return means

def orientation_bandwidth(full_matrix, height):
    """This function expects as an input a object of type FullMatrix which contains
    responses of all neurons in a sheet to stimuly with different varying parameter values.
    It computes Bandwidth as found in
      Orientation Selectivity in Macaque V1: Diversity and Laminar Dependece
      Ringach, Shapley, Hawken, 2002
    """ 

    feature_names=[feature.name for feature in full_matrix.features]
    X,Y = full_matrix.matrix_shape
    dim_or=full_matrix.dimensions[feature_names.index('orientation')]
    dim_ph=full_matrix.dimensions[feature_names.index('phase')]
    
    orientations=full_matrix.features[feature_names.index('orientation')].values

    # hanningly smoothed list of max orientation response among phases for each cell (x,y)
    smoothed_maxs = array(
        [[smooth(array([array([full_matrix.full_matrix[0][o][p][x][y] for p in xrange(dim_ph)]).max()
                         for o in xrange(dim_or)]),
                 window_len=dim_or-1)
          for y in xrange(Y)]
              for x in xrange(X)])

    def get_bandwidth(arr,height, orientations):
        """
            Function that computes the orientation bandwidth (treshold height) for a neuron specified by the array (arr) of responses to
            (orientations).
        """
        # maximum of the responses and its index
        max=arr.max()
        maxind=None
        for i in xrange(len(arr)):
            if arr[i]==max:
                maxind=i
                break
        left,right=None,None

        # find the left point
        a=maxind-1
        while a%len(arr) != maxind:
            if arr[a%len(arr)] > height*max:
                a += -1
            else:
                left=a%len(arr)
                break
        # if the left point exists:
        if left == None:
            return pi
        else:
            # find the right point
            a=maxind+1
            while a%len(arr) != maxind:
                if arr[a%len(arr)] > height*max:
                    a += 1
                else:
                    right=a%len(arr)
                    break
            # approximate the orientation at which the activity gets just below the treshold
            lefto = orientations[left] + (orientations[(left+1)%len(arr)] - orientations[left])%pi *\
                        (height*max-arr[left])/(arr[(left+1)%len(arr)]-arr[left])
            righto = orientations[right] - (orientations[right]-orientations[(right-1)%len(arr)])%pi *\
                        (height*max-arr[right])/(arr[(right-1)%len(arr)]-arr[right])

            # return the bandwidth
            return (righto  - lefto)%pi /2 

    # compute the bandwidth for all the neurons
    bandwidth = array(
      [[ get_bandwidth(smoothed_maxs[x][y],height,orientations)
         for y in xrange(Y)]
             for x in xrange(X)])
    return bandwidth
