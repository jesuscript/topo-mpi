import numpy as np
import pylab

def makeCircleBand(radius,cx,cy, dim=48, halfThickness=5): # Half of band thickness is 0.5mm
    circle = np.zeros((48, 48)).astype('uint8')
    y, x = np.ogrid[-radius: radius, -radius: radius]
    index = x**2 + y**2 <= radius**2
    circle[cy-radius:cy+radius, cx-radius:cx+radius][index] = 1

    topBand = cy - halfThickness; bottomBand = cy + halfThickness
    circle[0:topBand,:] = 0; circle[bottomBand:dim,:] = 0
    return circle

def makeMask(cx,cy, num=1, dim=48): 
    radius = 5 # 5 units = 0.5mm = one band
    if num ==1:
        return makeCircleBand(radius,cx,cy)
    else:
        smaller = makeCircleBand(radius*(num-1),cx,cy)
        bigger = makeCircleBand(radius*num,cx,cy)
        return bigger - smaller

def makeMasks(cx,cy, dim=48): # Sheet dim
    return [makeMask(cx,cy, num, dim) for num in [1,2,3,4]]


def getBandAverageActivities(activityArray, cx, cy):
    ' Averages over BOTH equidistant patches '
    masks = makeMasks(cx,cy)

    avgBandActivities = []
    for (i,mask) in enumerate(masks):
        maskedActivityArray = activityArray * mask
        maskedActivitySum = maskedActivityArray.sum()
        maskedUnitNumber = mask.sum()
        avgBandActivities.append( maskedActivitySum/maskedUnitNumber )

    return avgBandActivities


def getBandProfiles(activities, cx, cy):
    
    zipped = [getBandAverageActivities(activity, cx, cy) for activity in activities]
    return zip(*zipped)     # Profile at center is first, the first one out etc etc
    

def normaliseProfile(profile):

    minValue = min(profile);     maxValue = max(profile)
    rangeValue = maxValue - minValue
    shifted = [el - minValue for el in profile]
    return [el / rangeValue for el in profile]

def getNormalisedBandProfiles(activities, cx, cy):
    profiles = getBandProfiles(activities, cx, cy)
    return [ normaliseProfile(profile) for profile in profiles]


def makeVSDPlot(data, dataIndex, lineThickness=5): # In pixels

    VSDdata = data[dataIndex]['VSDSignal']
    activities = VSDdata[0]
    (unitX, unitY) = VSDdata[2]

    neededRadius = 20 # 4 sets of 5 units
    if unitX < neededRadius:  print "UnitX:%d. Minimum needed: %d " % (unitX, neededRadius); return
    if unitY < neededRadius:  print "UnitX:%d. Minimum needed: %d " % (unitX, neededRadius); return

    profiles = getNormalisedBandProfiles(activities, unitX, unitY)
    image = []
    for profile in profiles:
        for i in range(lineThickness):
            image.append(profile)

    pylab.imshow(image, interpolation='nearest')
    pylab.show()
