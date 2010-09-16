import pylab
import contrib.modelfit
import numpy
import __main__
from contrib.modelfit import * 


def showRFS(rfs,cog=False,centers=None):
	print numpy.shape(rfs)
	pylab.figure()
	m = numpy.max([numpy.abs(numpy.min(rfs)),numpy.abs(numpy.max(rfs))])
	for i in xrange(0,len(rfs)):
		pylab.subplot(15,15,i+1)
		w = numpy.array(rfs[i])
		pylab.show._needmain=False
		pylab.imshow(w,vmin=-m,vmax=m,interpolation='nearest',cmap=pylab.cm.RdBu)
		if centers != None:
			cir = Circle( (centers[i][0],centers[i][1]), radius=1,color='r')
			pylab.gca().add_patch(cir)
		if cog:
			xx,yy = contrib.modelfit.centre_of_gravity(rfs[i])
			cir = Circle( (xx,yy), radius=1,color='b')
			pylab.gca().add_patch(cir)
		pylab.axis('off')
		i+=1
   
   
def compareModelPerformanceWithRPI(training_set,validation_set,training_inputs,validation_inputs,pred_act,pred_val_act,raw_validation_set,modelname='Model'):
    from contrib.JanA.regression import laplaceBias	
    
    num_neurons = numpy.shape(pred_act)[1]
    
    kernel_size= numpy.shape(validation_inputs)[1]
    laplace = laplaceBias(numpy.sqrt(kernel_size),numpy.sqrt(kernel_size))
    X = numpy.mat(training_inputs)
    rpi = numpy.linalg.pinv(X.T*X + __main__.__dict__.get('RPILaplaceBias',0.0001)*laplace) * X.T * training_set 
    rpi_pred_act = training_inputs * rpi
    rpi_pred_val_act = validation_inputs * rpi

    showRFS(numpy.reshape(numpy.array(rpi),(-1,numpy.sqrt(kernel_size),numpy.sqrt(kernel_size))))	

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
    
    print numpy.shape(validation_set - rpi_pred_val_act_t)
    print numpy.shape(validation_set - pred_val_act)
    print numpy.shape(numpy.mean(numpy.power(validation_set - rpi_pred_val_act_t,2)[:,:num_neurons],0))
    print numpy.shape(numpy.mean(numpy.power(validation_set - pred_val_act,2)[:,:num_neurons],0))
    
    pylab.figure()
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
    performance_analysis(training_set,validation_set,pred_act_t,pred_val_act_t,raw_validation_set,85)
    
    pylab.figure()
    pylab.plot(rpi_validation_prediction_power[:num_neurons],validation_prediction_power[:num_neurons],'o')
    pylab.hold(True)
    pylab.plot([0.0,1.0],[0.0,1.0])
    pylab.xlabel('RPI')
    pylab.ylabel(modelname)
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