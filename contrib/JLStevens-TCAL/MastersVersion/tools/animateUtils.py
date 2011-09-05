# ; -*- mode: Python;-*-
import os, sys, subprocess

import numpy
from scipy.misc import toimage
import topo

def generateFrames(streamNames,directoryName, 
                   streams=None, frameSkip=1):
    """ Generates frames of activity using scipy's
        toimage tool for the given stream name.
        """

    if not os.path.exists(directoryName):
        os.mkdir(directoryName)

    if streams is None:
        recorder = topo.sim['Data']
        streams = [(streamName,
                    recorder.get_data(streamName)) 
                   for streamName in streamNames ]

    frameCounts = {}
    for (streamName, stream) in streams:

        counter = 0; frameCount=0; 
        minVal = 1000; maxVal = 0
        
        for matrix in stream[1]: # Frame times on [0]
            matmax = matrix.max(); matmin = matrix.min()
            if matmax > maxVal: maxVal = matmax
            if matmin < minVal: minVal = matmin

        for matrix in stream[1]: # Frame times on [0]
            if (counter % frameSkip ==0):        
                toimage(matrix, cmin=minVal,cmax=maxVal).save('%s/%s%s.png'
                                                              % (directoryName,
                                                                 streamName, 
                                                                 str(frameCount)))
                frameCount += 1
            counter += 1

        frameCounts.update({streamName:frameCount})
    return frameCounts
    

def makeGif(path, streamName, frameCounts, removeRaw=True):
    """ Uses ImageMagick to turn frames into animated
    gifs before optionally removing raw frames """

    count = frameCounts[streamName]
    gifList=[streamName+str(ind)+'.png' for ind in range(count)]

    try: 
        gifStr = "convert -border 2x2 -delay 1 -loop 0 %s %s.gif" % (' '.join(gifList), streamName)
        gifProc = subprocess.Popen(gifStr, cwd=path, shell=True)
        gifProc.wait()

    except: print("Conversion failed"); return
    
    if removeRaw:
        try:
            rmStr = "rm %s*.png" % streamName
            gifProc = subprocess.Popen(rmStr, cwd=path, shell=True)
            gifProc.wait()
        except: print("Deletion failed"); return

