import pylab
import contrib.modelfit
import numpy
import __main__
from contrib.modelfit import * 
from contrib.JanA.ofestimation import *


def showRFS(rfs,cog=False,centers=None,joinnormalize=True,axis=False):
	print numpy.shape(rfs)
	pylab.figure()
	m = numpy.max([numpy.abs(numpy.min(rfs)),numpy.abs(numpy.max(rfs))])
	for i in xrange(0,len(rfs)):
		pylab.subplot(15,15,i+1)
		w = numpy.array(rfs[i])
		pylab.show._needmain=False
		
		if not joinnormalize:
		   m = numpy.max([numpy.abs(numpy.min(w)),numpy.abs(numpy.max(w))])
		
		pylab.imshow(w,vmin=-m,vmax=m,interpolation='nearest',cmap=pylab.cm.RdBu)
		if centers != None:
			cir = Circle( (centers[i][0],centers[i][1]), radius=1,color='r')
			pylab.gca().add_patch(cir)
		if cog:
			xx,yy = contrib.modelfit.centre_of_gravity(rfs[i])
			cir = Circle( (xx,yy), radius=1,color='b')
			pylab.gca().add_patch(cir)
			
		if not axis:
			pylab.axis('off')
		i+=1
   
   
def compareModelPerformanceWithRPI(training_set,validation_set,training_inputs,validation_inputs,pred_act,pred_val_act,raw_validation_set,sizex,sizey,modelname='Model'):
    from contrib.JanA.regression import laplaceBias	
    
    num_neurons = numpy.shape(pred_act)[1]
    
    kernel_size= numpy.shape(validation_inputs)[1]
    laplace = laplaceBias(sizex,sizey)
    X = numpy.mat(training_inputs)
    rpi = numpy.linalg.pinv(X.T*X + __main__.__dict__.get('RPILaplaceBias',0.0001)*laplace) * X.T * training_set 
    rpi_pred_act = training_inputs * rpi
    rpi_pred_val_act = validation_inputs * rpi

    showRFS(numpy.reshape(numpy.array(rpi.T),(-1,sizex,sizey)))	
    
    print numpy.shape(numpy.mat(training_set))
    print numpy.shape(numpy.mat(pred_act))
    ofs = run_nonlinearity_detection(numpy.mat(training_set),numpy.mat(pred_act),num_bins=10,display=True,name=(modelname+'_piece_wise_nonlinearity.png'))
    pred_act_t = numpy.mat(apply_output_function(numpy.mat(pred_act),ofs))
    pred_val_act_t = numpy.mat(apply_output_function(numpy.mat(pred_val_act),ofs))

    ofs = run_nonlinearity_detection(numpy.mat(training_set),numpy.mat(rpi_pred_act),num_bins=10,display=True,name='RPI_piece_wise_nonlinearity.png')
    rpi_pred_act_t = numpy.mat(apply_output_function(numpy.mat(rpi_pred_act),ofs))
    rpi_pred_val_act_t = numpy.mat(apply_output_function(numpy.mat(rpi_pred_val_act),ofs))
    
    pylab.figure()
    pylab.title('RPI')
    for i in xrange(0,num_neurons):
	pylab.subplot(11,11,i+1)    
    	pylab.plot(rpi_pred_val_act[:,i],validation_set[:,i],'o')
    release_fig('RPI_val_relationship.png')	
	
    pylab.figure()
    pylab.title(modelname)
    for i in xrange(0,num_neurons):
	pylab.subplot(11,11,i+1)    
 	pylab.plot(pred_val_act[:,i],validation_set[:,i],'o') 
    release_fig('GLM_val_relationship.png')	  
    
    
    
    pylab.figure()
    pylab.title('RPI')
    for i in xrange(0,num_neurons):
	pylab.subplot(11,11,i+1)    
    	pylab.plot(rpi_pred_val_act_t[:,i],validation_set[:,i],'o')
    release_fig('RPI_t_val_relationship.png')	
	
	
    pylab.figure()
    pylab.title(modelname)
    for i in xrange(0,num_neurons):
	pylab.subplot(11,11,i+1)    
 	pylab.plot(pred_val_act_t[:,i],validation_set[:,i],'o')   
    release_fig('GLM_t_val_relationship.png')
    
    pylab.figure()
    print numpy.shape(numpy.mean(numpy.power(validation_set - rpi_pred_val_act_t,2)[:,:num_neurons],0))
    print numpy.shape(numpy.mean(numpy.power(validation_set - pred_val_act,2)[:,:num_neurons],0))
    pylab.plot(numpy.mean(numpy.power(validation_set - rpi_pred_val_act_t,2)[:,:num_neurons],0),numpy.mean(numpy.power(validation_set - pred_val_act,2)[:,:num_neurons],0),'o')
    pylab.hold(True)
    pylab.plot([0.0,1.0],[0.0,1.0])
    pylab.xlabel('RPI')
    pylab.ylabel(modelname)
    release_fig('GLM_vs_RPI_MSE.png')
    
    print '\n \n RPI \n'
    
    print 'Without TF'
    performance_analysis(training_set,validation_set,rpi_pred_act,rpi_pred_val_act,raw_validation_set,85)
    print 'With TF'
    (signal_power,noise_power,normalized_noise_power,training_prediction_power,rpi_validation_prediction_power,signal_power_variance) = performance_analysis(training_set,validation_set,rpi_pred_act_t,rpi_pred_val_act_t,raw_validation_set,85)
	
    print '\n \n', modelname, '\n'
	
    print 'Without TF'
    (signal_power,noise_power,normalized_noise_power,training_prediction_power,validation_prediction_power,signal_power_variance) = performance_analysis(training_set,validation_set,pred_act,pred_val_act,raw_validation_set,85)
    print 'With TF'
    (signal_power_t,noise_power_t,normalized_noise_power_t,training_prediction_power_t,validation_prediction_power_t,signal_power_variance_t) =    performance_analysis(training_set,validation_set,pred_act_t,pred_val_act_t,raw_validation_set,85)
    
    
    significant = numpy.array(numpy.nonzero((numpy.array(normalized_noise_power) < 85) * 1.0))[0]
    
    print significant
    
    pylab.figure()
    pylab.plot(rpi_validation_prediction_power[significant],validation_prediction_power[significant],'o')
    pylab.hold(True)
    pylab.plot([0.0,1.0],[0.0,1.0])
    pylab.xlabel('RPI')
    pylab.ylabel(modelname)
    release_fig('GLM_vs_RPI_prediction_power.png')
    
    pylab.figure()
    pylab.plot(rpi_validation_prediction_power[significant],validation_prediction_power_t[significant ],'o')
    pylab.hold(True)
    pylab.plot([0.0,1.0],[0.0,1.0])
    pylab.xlabel('RPI')
    pylab.ylabel(modelname+'+TF')
    release_fig('GLM_vs_RPI_prediction_power.png')



def extractContours(RFs):
    a = 1
	
    tr = 140 
    size = numpy.shape(RFs)[1]    
    on_contours = []
    off_contours = []
    onoff_contours = []
    on_filled = []
    off_filled = []
    
    showRFS(RFs)
    
    for rf in RFs:
	import PIL
    	import Image
	
	#pylab.figure()
	#pylab.imshow(rf,interpolation='nearest')

	image = Image.new('F',(size,size))
	image.putdata(rf.T.flatten()/400+0.5)
	image = image.resize((int(size*6), int(size*6)), Image.ANTIALIAS)
	rf = (numpy.array(image.getdata()).reshape(int(size*6), int(size*6))-0.5)*400
	rf = rf.T
		
	#pylab.figure()
	#pylab.imshow(rf,interpolation='nearest')
	
	rf_on = rf * (rf > tr )   
	rf_off = rf * (rf < -tr )
	on = extractContour(rf_on)
	off = extractContour(-rf_off)
	
	
	on_contours.append(on)
	off_contours.append(-off)
	onoff_contours.append(on-off)
	
	on_filled.append((rf > tr))
	off_filled.append((rf < -tr))

		
    #showRFS(onoff_contours)
    #showRFS(on_contours)
    #showRFS(off_contours)
        
    return (on_contours,off_contours,onoff_contours,on_filled,off_filled)	
	
def extractContour(rf):
	size = numpy.shape(rf)[0]
	
	rf1 = numpy.copy(rf)
	for x in xrange(0,size):
	    for y in xrange(0,size):
		flag = 0
		
		if x+1 < size:
		   if rf[x+1,y] == 0:	
		      flag=1	
		
		if x-1 >= 0:
		   if rf[x-1,y] == 0:	
		      flag=1	
	
		if y-1 >= 0:
		   if rf[x,y-1] == 0:	
		      flag=1	
		
		if y+1 < size:
		   if rf[x,y+1] == 0:	
		      flag=1	

		
		if x+1 < size and y+1 < size:
		   if rf[x+1,y+1] == 0:	
		      flag=1	
		
		if x-1 >= 0 and y-1 >= 0:
		   if rf[x-1,y-1] == 0:	
		      flag=1	
	
		if x+1 < size and y-1 >= 0:
		   if rf[x+1,y-1] == 0:	
		      flag=1	
		
		if x-1 >= 0 and y+1 < size:
		   if rf[x-1,y+1] == 0:	
		      flag=1	

		rf1[x,y]*= flag
		rf1[x,y]= rf1[x,y] > 0  	 
	
    	return rf1

def OnOffCenterOfGravityPlot(rfs):
    a = 1
    tr = 100 
    
    on_center = []
    off_center = []

    for rf in rfs:
	rf_on = rf * (rf > tr )   
	rf_off = -rf * (rf < -tr )

	on_center.append(centre_of_gravity(rf_on))
	off_center.append(centre_of_gravity(rf_off))
    
    f = pylab.figure(dpi=100,facecolor='w',figsize=(3,3))
    x, y = zip(*on_center)
    pylab.plot(numpy.array(x),numpy.array(y),'bo')
    x, y = zip(*off_center)
    pylab.plot(numpy.array(x),numpy.array(y),'ro')
    pylab.xlim(0.0,numpy.shape(rfs[0])[0])
    pylab.ylim(0.0,numpy.shape(rfs[0])[1])
    pylab.hold('on')
    



def visualize2DOF(pred_act1,pred_act2,act,num_bins=10):
    bin_size1 = (numpy.max(pred_act1,axis=0) - numpy.min(pred_act1,axis=0))/num_bins 
    bin_size2 = (numpy.max(pred_act2,axis=0) - numpy.min(pred_act2,axis=0))/num_bins
    	
    of = numpy.zeros((numpy.shape(act)[1],num_bins,num_bins))
    ofn = numpy.zeros((numpy.shape(act)[1],num_bins,num_bins))
	
    for i in xrange(0,numpy.shape(act)[0]):
	idx1 = numpy.round_((pred_act1[i,:]-numpy.min(pred_act1,axis=0)) / bin_size1)    	
	idx2 = numpy.round_((pred_act2[i,:]-numpy.min(pred_act2,axis=0)) / bin_size2)
	
	idx1 = idx1 -(idx1 >= num_bins)
	idx2 = idx2 -(idx2 >= num_bins)
	
	j=0
	for (x,y) in zip(numpy.array(idx1).flatten().tolist(),numpy.array(idx2).flatten().tolist()):
            of[j,x,y] = of[j,x,y] +  act[i,j]
	    ofn[j,x,y] = ofn[j,x,y] + 1 
	    j=j+1
    
    of = of - (ofn <= 0)
    ofn = ofn + (ofn <= 0)
    of = of/ofn
    print of[0]
    print of[1]
    print ofn[0]
    print ofn[1]

    
    
    showRFS(of,joinnormalize=False)

def printCorrelationAnalysis(act,val_act,pred_act,pred_val_act):
    num_pres,num_neurons = numpy.shape(act)
    import scipy.stats
    train_c=[]
    val_c=[]
    
    for i in xrange(0,num_neurons):
	train_c.append(scipy.stats.pearsonr(numpy.array(act)[:,i].flatten(),numpy.array(pred_act)[:,i].flatten())[0])
        val_c.append(scipy.stats.pearsonr(numpy.array(val_act)[:,i].flatten(),numpy.array(pred_val_act)[:,i].flatten())[0])
    
    print 'Correlation Coefficients (training/validation): ' + str(numpy.mean(train_c)) + '/' + str(numpy.mean(val_c))