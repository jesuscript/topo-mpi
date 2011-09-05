import numpy

def parsePPM(filename):
    """ Generates RGB tuples from a GIMP ASCII PPM file """  
    inputFile = open(filename,'r')
    rawData = inputFile.readlines()
    inputFile.close()

    dimStr = rawData[2] # Line with image dimensions
    [width, height] = [int(el) for el in dimStr.split()]

    brightness = rawData[3]
    pixelData = rawData[4:]

    datalen = len(pixelData)

    if datalen / (3 * width) != height:
        print "Invalid ppm!" # Assuming made with GIMP
        return None

    RGBTuples = []
    for i in range(0, datalen, 3):
        r = int(pixelData[i])    # Might by R B G!
        g = int(pixelData[i+1])
        b = int(pixelData[i+2])
        RGBTuples.append((r,g,b))

    if len(RGBTuples) != (width * height):
        print "Invalid ppm format"
        return None

    format = (height,width,3) # Goes as matrix(row,column,rgb)
    numpyImage = numpy.zeros(format, dtype=numpy.int32) 

    for row in range(height):
        for col in range(width):
            numpyImage[row][col] = RGBTuples[row*width + col]

    return (width, height, numpyImage)

def getProfileRatio(width, height, numpyImage, colour=[255,0,0]):
    """ Returns the normalised height at which the transition from colour occurs. Maximum value is unity."""
    profile = []

    for columnInd in range(width):
        redColumn = numpyImage[:,columnInd,0]
        greenColumn = numpyImage[:,columnInd,1]
        blueColumn = numpyImage[:,columnInd,2]

        for rowInd in range(height):
            red = redColumn[rowInd] 
            green = greenColumn[rowInd] 
            blue = blueColumn[rowInd] 
            if [red, green, blue] != colour:
                profile.append(height-rowInd)
                break

    maxVal = float(max(profile))
    profile = [el / maxVal for el in profile]

    return profile

