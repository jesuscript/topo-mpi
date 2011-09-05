from parsePPM import parsePPM, getProfileRatio
from kernel import StepDeconvolutionKernel

import numpy
import pickle
import scipy.signal
import sys

import pylab

def stimulusOnAndOff(stimData, threshold):

    enumerated = [el for el in enumerate(stimData)]

    for (ind,val) in enumerated:
        stimulusOnset = ind
        if val > threshold: break

    for (ind,val) in enumerated[stimulusOnset:]:
            stimulusOffset = ind
            if val < threshold: break

    return (stimulusOnset, stimulusOffset)
                         
if __name__ == "__main__":

    timeStep = float(sys.argv[1])
    kernelDuration = 500  # Kernel spans 500 milliseconds.  
    steadyStateOffset = 600

    # Getting normalised response profile from PPM file
    (widthR, heightR, numpyImageR) = parsePPM('./images/800wide-response.ppm')
    profileR = getProfileRatio(widthR, heightR, numpyImageR)

    # Removing baseline firing rate
    minR = min(profileR)
    profileR = [el - minR for el in profileR]
    
    # Finding multiplier such that steady state is at unity
    steadyStateVal = profileR[steadyStateOffset]
    profileR = [el / steadyStateVal for el in profileR]

    # Getting normalised stimulus profile from PPM file
    (widthS, heightS, numpyImageS) = parsePPM('./images/800wide-stimulus.ppm')
    profileS = getProfileRatio(widthS, heightS, numpyImageS)

    # Generating clean stimulus step function 
    (stimulusOnset, stimulusOffset) = stimulusOnAndOff(profileS, 0.5)
    
    dataTimeDuration = 2000 #  2 seconds from figure
    stimulusOnsetTime = (float(stimulusOnset) / len(profileS)) * dataTimeDuration # *Time* of Onset
    stimulusOffsetTime = (float(stimulusOffset) / len(profileS)) * dataTimeDuration # *Time* of Offset

    #################################
    # WORKING WITH THE KERNEL CLASS #
    #################################   

    # Creating a kernel object based on time-series 2 seconds long
    kernel = StepDeconvolutionKernel(dataTimeDuration) 
    # Define step stimulus and associated response
    kernel.set_stimulus_profile(stimulusOnsetTime, stimulusOffsetTime, amplitude=1.0)
    (tvals,spline) = kernel.fit_response_profile(profileR)
 
    # Get resampled profiles
    (tvalsFit, sampledR) = kernel.sample_response_profile(timeStep)
    (tvalsFit, sampledS) = kernel.sample_stimulus_profile(timeStep)
 
    # Get kernel and sampled reconstruction
    (kernelVals, reconstructed) = kernel.compute_kernel(timeStep, kernelDuration)

    ############
    # Plotting #
    ############

    fig = pylab.Figure()
    sampleS = pylab.plot(tvalsFit, sampledS,'b.-')
    sampleR = pylab.plot(tvalsFit, sampledR,'g+-')
    recon = pylab.plot(tvalsFit, reconstructed,'r.-')
    original = pylab.plot(tvals,profileR,'k')
    pylab.legend([sampleS, sampleR, recon, original], 
                 ["sample stimulus", "sample response", "reconstructed", "original"])
    pylab.savefig("./sampled-%d.pdf" % timeStep)
    fig.clear()

    figKernel = pylab.Figure()
    pylab.hold(False)
    pylab.plot(kernelVals,'k')
    pylab.savefig("./kernel-%d.pdf" % timeStep)
    figKernel.clear()

    ############
    # Pickling #
    ############

    pickleFile = open('./kernelLGN.pickle','w')
    pickle.dump(kernel,pickleFile)
    pickleFile.close()

    pickleProfile = open('./profileR.pickle','w')
    pickle.dump((tvals,profileR),pickleProfile)
    pickleProfile.close()





