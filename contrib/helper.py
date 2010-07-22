import numpy
import pylab
from topo import numbergen

def flatten(x):
    """flatten(sequence) -> list

    Returns a single, flat list which contains all elements retrieved
    from the sequence and all recursively contained sub-sequences
    (iterables).

    Examples:
    >>> [1, 2, [3,4], (5,6)]
    [1, 2, [3, 4], (5, 6)]
    >>> flatten([[[1,2,3], (42,None)], [4,5], [6], 7, MyVector(8,9,10)])
    [1, 2, 3, 42, None, 4, 5, 6, 7, 8, 9, 10]"""

    result = []
    for el in x:
        #if isinstance(el, (list, tuple)):
        if hasattr(el, "__iter__") and not isinstance(el, basestring):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result
	
	


def normalize_images(): 
    import PIL
    import Image
    import ImageOps
    index = numpy.arange(1001,1100,1)
    	
    image = Image.open('/home/antolikjan/Desktop/a10b_10000.tif')	
    	
    flip = True
    reverse = True
    
    #image_filenames=["/home/antolikjan/topographica/topographica/Flogl/DataOct2009/20090925_image_list_used/image_%04d.tif" %(i) for i in index]
    
    #images=[Image.open(f) for f in image_filenames]
    
    normalized_images=[]
    for i in xrange(0,5000):
	image.seek(i)
	d = numpy.array(list(image.getdata()))/255.0
	x,y = image.size
	
	mi = numpy.sort(d)[int(len(d)*0.2)]
	ma = numpy.sort(d)[int(len(d)*0.8)]
	
	#clip 5% of top and bottom
	d = numpy.multiply(d, (d > mi)*1.0) + (d <= mi)*mi
	d = numpy.multiply(d, (d < ma)*1.0) + (d >= ma)*ma
	
	d = d - numpy.mean(d)
	c = numpy.std(d)
	if c==0: continue
	a = c / 0.6
	d = d / a
	
	if numpy.min(d) > -1.0 and numpy.max(d) < 1.0:
	   d = (d + 1.0)*128
	   print numpy.mean(d)
	   print numpy.std(d)
	   #im = Image.new('L',image.size)
	   #im.putdata(d)
	   normalized_images.append(d)
    
    
    dissimilar_images=[]	
    
    for i in xrange(0,len(normalized_images)):
	flag = 0		
	print i
	for j in xrange(0,len(dissimilar_images)):
	    if (numpy.sum(numpy.multiply(normalized_images[i],dissimilar_images[j]))/numpy.sqrt(numpy.sum(numpy.power(normalized_images[i],2)) * numpy.sum(numpy.power(dissimilar_images[j],2)))) > 0.85:
	        flag = 1
		break
	if not flag:       
		dissimilar_images.append(normalized_images[i])
    
    for i in xrange(0,len(dissimilar_images)):
	framefile = "./Mice/Stimuli/NG/dissimilar/frame%05d.tif" % i
        im = Image.new('L',image.size)
	im.putdata(dissimilar_images[i])
        im.save(framefile)
    
    images = dissimilar_images
    new_im=[]
    if reverse:
       for i in images:
	   new_im.append(i)
	   new_im.append((((i/256 - 0.5)*-1.0) + 0.5)*256)
    
    images = new_im
    to_export=[]
    if flip:
       for i in images:
  	   im = Image.new('L',image.size)
	   im.putdata(i)
	   to_export.append(im)
           to_export.append(im.transpose(Image.FLIP_LEFT_RIGHT))
	   to_export.append(im.transpose(Image.FLIP_TOP_BOTTOM))
    print len(to_export)

    for i in xrange(0,len(to_export)):
	framefile = "./Mice/Stimuli/NG/tifsequence/frame%05d.tif" % i
	to_export[i].save(framefile)




def package_images(packages=1,randomize=False):
    import Image
    import subprocess
    index = numpy.arange(0,180*packages,1)
    image_filenames=["./Mice/Stimuli/NG/tifsequence/frame%05d.tif" %(i) for i in index]
    #image_filenames=["./Mice/Stimuli/SparseNoise_DS=5.0_Step=2_Density=10/up_frame%05d.tif" %(i) for i in index]
    
    if randomize:
       import random
       random.shuffle(image_filenames)
    
    for i in xrange(0,packages):
	command = ["tiffcp"]    
	for j in xrange(0,180):
	    command.append(image_filenames[i*180+j])		
	command.append("./Mice/Stimuli/NG/NIFliInvStack"+str(i)+".tif")
	#command.append("./Mice/Stimuli/SparseNoise_DS=5.0_Step=2_Density=10/SparseNoiseStack"+str(i)+".tif")
	subprocess.call(command)

def de_package_images(packages=3,package_offset=0):
    import Image
    import subprocess
    #dirr = "./Mice/Stimuli/SparseNoise_DS=3.0_Step=2_Density=20/"
    #stackname = "SparseNoiseStack"
    dirr = "/home/antolikjan/topographica/topographica/Mice/Stimuli/NG/1800/"
    stackname = "NIFliInvStack"
    for i in xrange(0,packages):
	image = Image.open(dirr +stackname +str(i+package_offset)+".tif")
	for j in xrange(0,180):
	    image.seek(j)	
            im = Image.new('L',image.size)
	    im.putdata(image.getdata())
	    filename = "frame%05d.tif" % int(i*180+j)
            im.save(dirr+'depackaged_val/'+filename)
	
	
def monitor_view_angle(monitor_size,monitor_dist):	
    return 2*abs(numpy.arctan(monitor_size/2.0/monitor_dist))/numpy.pi*2*90
    	
	
def generateSparseNoiseStimuli(square_in_deg,steps,num_inputs,density):
    import pylab
    import Image
    
    #in cm
    monitor_size = 59.0
    monitor_dist = 20.0
    view_angle = monitor_view_angle(monitor_size,monitor_dist)
    sub_square_ratio=view_angle/square_in_deg*steps
    	
    print 'View angle:',view_angle	
    #how many pairs of dots will be presented in the whole image	
    rand=numbergen.UniformRandom()
    
    #generate inputs one by one
    for i in xrange(0,num_inputs):
	#create new field corresponding to pixels of grey     
	up_im = numpy.zeros((768,416))+128
	im = numpy.zeros((int(sub_square_ratio),int(sub_square_ratio/1.846153846)))+128
	sx = 768.0 / int(sub_square_ratio)
	sy = 416.0 / int(sub_square_ratio/1.846153846)
	
	#find a random number of dot pairs between 1 and density
	num_dots = 1+int(rand()*density)
	
	positions=[]
	
	for j in xrange(0,num_dots*2):
	    #find random position on the virtual grid	
	    lx = int(rand()*(sub_square_ratio-steps))
	    ly = int(rand()*(sub_square_ratio/1.846153846-steps))
	    #make sure we haven't already picked such position
	    while numpy.sum((positions == lx*sub_square_ratio+ly)*1.0) !=0: 
	    	lx = rand()*(sub_square_ratio-steps)
	        ly = rand()*(sub_square_ratio/1.846153846-steps)
	    positions.append(ly*sub_square_ratio+ly)
	    if j < num_dots:
	    	up_im[lx*sx:(lx+steps)*sx,ly*sy:(ly+steps)*sy]=0
		im[lx:(lx+steps),ly:(ly+steps)]=0
	    else:
		up_im[lx*sx:(lx+steps)*sx,ly*sy:(ly+steps)*sy]=255
		im[lx:(lx+steps),ly:(ly+steps)]=255
        
	#pylab.figure()
	#pylab.imshow(im)
	up_image = Image.new('L',(768,416))
	image = Image.new('L',(int(sub_square_ratio),int(sub_square_ratio/1.846153846)))
	image.putdata(im.T.flatten())
	up_image.putdata(up_im.T.flatten())
	image.putdata(im.T.flatten())
	
	framefile = "./Mice/Stimuli/SparseNoise_DS=5.0_Step=2_Density=10/frame%05d.tif" % i
	image.save(framefile)	
	framefile = "./Mice/Stimuli/SparseNoise_DS=5.0_Step=2_Density=10/up_frame%05d.tif" % i
	up_image.save(framefile)
	

def rename():
    import subprocess	
    n = [3178,3083,4770,2912,2912,2912,1957,4614,3931,3243]
    z = 0
    for i in xrange(0,len(n)):
	for j in xrange(0,n[i]):
	    filein = "./Mice/Stimuli/CatCam/Raw/mov%d/Catt%04d.tif" % (i+1,j+1)
	    fileout =  "./Mice/Stimuli/CatCam/Raw/mov/Catt%05d.tif" % z
	    command = ["cp", filein, fileout ]
	    subprocess.call(command) 
	    z = z+1
		