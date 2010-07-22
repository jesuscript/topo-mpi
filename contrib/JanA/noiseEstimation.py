import numpy 
import pylab

def signal_power_test(raw_validation_data_set, training_set, validation_set, pred_act, pred_val_act):
	
        signal_power=[]
	noise_power=[]
	normalized_noise_power=[]
	signal_power_variance=[]
	for i in xrange(0,len(raw_validation_data_set)):
	    (sp,np,nnp,spv) = signal_and_noise_power(raw_validation_data_set[i])
	    signal_power.append(sp)
	    noise_power.append(np)
	    normalized_noise_power.append(nnp)
	    signal_power_variance.append(spv)
	
	significant = numpy.mat(numpy.nonzero(((numpy.array(signal_power) - 0.5*numpy.array(signal_power_variance)) > 0.0)*1.0)).getA1()
	pylab.figure()
	pylab.subplot(131)
	pylab.title('distribution of estimated signal power in neurons')
	pylab.errorbar(noise_power,signal_power,fmt='ro',yerr=signal_power_variance)
	pylab.errorbar(numpy.array(noise_power)[significant],numpy.array(signal_power)[significant],fmt='bo',yerr=numpy.array(signal_power_variance)[significant])
	pylab.ylabel('signal power')
	pylab.xlabel('noise power')
	
	training_prediction_power=numpy.divide(numpy.var(training_set,axis=0) - numpy.var(pred_act - training_set,axis=0), signal_power)
	validation_prediction_power=numpy.divide(numpy.var(validation_set,axis=0) - numpy.var(pred_val_act - validation_set,axis=0), signal_power)
	pylab.subplot(132)
	pylab.title('distribution of estimated prediction power ')
	pylab.plot(numpy.array(normalized_noise_power)[significant],numpy.array(training_prediction_power)[significant],'ro',label='training')
	pylab.plot(numpy.array(normalized_noise_power)[significant],numpy.array(validation_prediction_power)[significant],'bo',label='validation')
	pylab.axis([20.0,100.0,-2.0,2.0])
	pylab.xlabel('normalized noise power')
	pylab.ylabel('prediction power')
	pylab.legend()

	pylab.subplot(133)
	pylab.title('relationship between test set prediction power \n and validation prediction power')
	pylab.plot(validation_prediction_power[significant],training_prediction_power[significant],'ro')
	pylab.axis([-2.0,2.0,0.0,2.0])
	pylab.xlabel('validation set prediction power')
	pylab.ylabel('test set prediction power')
	

	return (signal_power,noise_power,normalized_noise_power,training_prediction_power,validation_prediction_power,signal_power_variance)

def signal_and_noise_power(responses):
    (trials,n) = numpy.shape(responses)	
    sp =  (1 / (trials-1.0)) * (trials * numpy.var(numpy.mean(responses,axis=0)) - numpy.mean(numpy.var(responses,axis=1)))
    np =  numpy.mean(numpy.var(responses,axis=1)) - sp
    nnp =  (numpy.mean(numpy.var(responses,axis=1)) - sp) / numpy.mean(numpy.var(responses,axis=1)) * 100
    
    ni = numpy.mean(numpy.mat(responses),0)
    nni = numpy.mean(ni)
    noise = responses - numpy.tile(ni,(trials,1))
    Cov = numpy.mat(numpy.mat(noise)).T * numpy.mat(numpy.mat(noise))  
    s = numpy.mean(Cov,0)
    ss = numpy.mean(s)
    spv = numpy.sum((4.0/trials) * ((ni*Cov*ni.T)/(n*n) - 2*nni*ni*s.T/n + nni*nni*ss) + 2.0/(trials*(trials-1))*( numpy.trace(Cov*Cov)/(n*n) - (2.0/n)*s*s.T + ss*ss)) 
    return (sp,np,nnp,spv)

def estimateNoise(trials):
    (num_neurons,num_trials,num_resp) = numpy.shape(trials)	
	
    mean_responses = numpy.mean(trials,1)
    		
    for i in xrange(0,10):
	pylab.figure()
	pylab.subplot(3,3,2)
	pylab.hist(mean_responses[i,:])
	bins = numpy.arange(numpy.min(mean_responses[i,:]),numpy.max(mean_responses[i,:]) + (numpy.max(mean_responses[i,:])+0.00001-numpy.min(mean_responses[i,:]))/5.0,( numpy.max(mean_responses[i,:])+0.00001-numpy.min(mean_responses[i,:]))/5.0)
	print numpy.min(mean_responses[i,:])
	print numpy.max(mean_responses[i,:])
	print bins
	#membership = numpy.zeros(numpy.shape(mean_responses[i,:]))
	for j in xrange(0,5):
	    membership = numpy.nonzero(numpy.array(((mean_responses[i,:] >= bins[j]) &  (mean_responses[i,:] < bins[j+1])))) 
	    raw_responses = trials[i,:,membership].flatten()
	    pylab.subplot(3,3,3+j+1)
	    if(len(raw_responses) != 0):
	    	pylab.hist(raw_responses)
		pylab.xlabel(str(bins[j])+'-'+str(bins[j+1]))	
		
