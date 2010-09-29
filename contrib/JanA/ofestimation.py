import numpy


def run_nonlinearity_detection(activities,predicted_activities,num_bins=20,display=False ,name='piece_wise_nonlinearity.png'):
            (num_act,num_neurons) = numpy.shape(activities)
            a=pylab.rcParams['font.size']
            os = []
            if display:
	       pylab.rc('font', size=1)
               pylab.figure(dpi=100,facecolor='w',figsize=(17,12))
            for i in xrange(0,num_neurons):
                min_pact = numpy.min(predicted_activities[:,i])
                max_pact = numpy.max(predicted_activities[:,i])
                bins = numpy.arange(0,num_bins+1,1)/(num_bins*1.0)*(max_pact-min_pact) + min_pact
                bins[-1]+=0.000001
                ps = numpy.zeros(num_bins)
                pss = numpy.zeros(num_bins)
                    
                for j in xrange(0,num_act):
                    bin = numpy.nonzero(bins>=predicted_activities[j,i])[0][0]-1
                    ps[bin]+=1
                    pss[bin]+=activities[j,i] 
                
		idx = numpy.nonzero(ps==0)
                ps[idx]=1.0
                tf = pss/ps
                tf[idx]=0.0
                
                if display:
                   pylab.subplot(15,7,i+1)
                   #pylab.plot(bins[0:-1],ps)
                   #pylab.plot(bins[0:-1],pss)
                   pylab.plot(bins[0:-1],tf)
		   
                
                os.append((bins,tf))
	    pylab.rc('font', size=a)
	    if display:
		release_fig(name)    
            return os

def apply_output_function(activities,of):
    (x,y) = numpy.shape(activities)
    acts = numpy.zeros(numpy.shape(activities))
    for i in xrange(0,x):
        for j in xrange(0,y):
            (bins,tf) = of[j]
            
            if activities[i,j] >= numpy.max(bins):
                acts[i,j] = tf[-1]
            elif activities[i,j] <= numpy.min(bins):
                 acts[i,j] = tf[0]
            else:
                bin = numpy.nonzero(bins>=activities[i,j])[0][0]-1
                # do linear interpolation
                a = bins[bin]
                b = bins[bin+1]
                alpha = (activities[i,j]-a)/(b-a)
                
                if bin!=0:
                   c = (tf[bin]+tf[bin-1])/2
                else:
                   c = tf[bin]
                
                if bin!=len(tf)-1:
                   d = (tf[bin]+tf[bin+1])/2
                else:
                   d = tf[bin]
                
                acts[i,j] = c + (d-c)* alpha
    
    return acts

def fit_sigmoids_to_of(activities,predicted_activities,offset=True,display=True,name='piece_wise_nonlinearity.png'):
	
    (num_in,num_ne) = numpy.shape(activities)	
    from scipy import optimize
    rand =numbergen.UniformRandom(seed=513)
    a=pylab.rcParams['font.size']
    if display: 	
        pylab.rc('font', size=1)
    	pylab.figure(dpi=100,facecolor='w',figsize=(17,12))

    fitfunc = lambda p, x:  (offset*p[2])+p[3] / (1 + numpy.exp(-p[0]*(x-p[1]))) # Target function
    errfunc = lambda p,x, y: numpy.mean(numpy.power(fitfunc(p, x) - y,2)) # Distance to the target function
    
    params=[]
    for i in xrange(0,num_ne):
	min_err = 10e10
	best_p = 0
	for j in xrange(0,1000):
		p0 = [20*rand(),10*(rand()-0.5),20*(rand()-0.5),50*rand()] 
		(p,success,c)=optimize.fmin_tnc(errfunc,p0[:],bounds=[(0,20),(-5,5),(-10,10),(0,50)],args=(numpy.array(predicted_activities[:,i].T)[0],numpy.array(activities[:,i].T)[0]),approx_grad=True,messages=0,maxfun=1000)
		err  = errfunc(p,numpy.array(predicted_activities[:,i].T)[0],numpy.array(activities[:,i].T)[0])
		if err < min_err:
		   best_p = p 
	
        params.append(best_p)
	if display:
		pylab.subplot(15,7,i+1)
	        pylab.plot(numpy.array(predicted_activities[:,i].T)[0],numpy.array(activities[:,i].T)[0],'go')
        	pylab.plot(numpy.array(predicted_activities[:,i].T)[0],fitfunc(best_p,numpy.array(predicted_activities[:,i].T)[0]),'bo')
		
    if display:
	release_fig(name)    
		
    pylab.rc('font', size=a)		

    return params
    
def fit_exponential_to_of(activities,predicted_activities,offset=True,display=True):
	
    (num_in,num_ne) = numpy.shape(activities)	
    from scipy import optimize 	
    pylab.figure()

    fitfunc = lambda p, x: offset*p[0] + p[1] * numpy.exp(p[2]*(x-p[3])) # Target function
    errfunc = lambda p,x, y: numpy.mean(numpy.power(fitfunc(p, x) - y,2)) # Distance to the target function
    
    params=[]
    for i in xrange(0,num_ne):
    	p0 = [0.0,1.0,0.1,0.0] # Initial guess for the parameters
	(p,success,c)=optimize.fmin_tnc(errfunc,p0[:],bounds=[(-20,20),(-10,10),(0,10),(-5,5)],args=(numpy.array(predicted_activities[:,i].T)[0],numpy.array(activities[:,i].T)[0]),approx_grad=True,messages=0)
        params.append(p)
	if display:
		pylab.subplot(13,13,i+1)
	        pylab.plot(numpy.array(predicted_activities[:,i].T)[0],numpy.array(activities[:,i].T)[0],'go')
        	pylab.plot(numpy.array(predicted_activities[:,i].T)[0],fitfunc(p,numpy.array(predicted_activities[:,i].T)[0]),'bo')

    return params
    
def fit_power_to_of(activities,predicted_activities,display=True):
	
    (num_in,num_ne) = numpy.shape(activities)	
    from scipy import optimize 	
    pylab.figure()

    fitfunc = lambda p, x: p[0] + p[1] * numpy.power(x,p[2]) # Target function
    errfunc = lambda p,x, y: numpy.mean(numpy.power(fitfunc(p, x) - y,2)) # Distance to the target function
    
    params=[]
    for i in xrange(0,num_ne):
    	p0 = [0.0,1.0,-0.5] # Initial guess for the parameters
	(p,success,c)=optimize.fmin_tnc(errfunc,p0[:],bounds=[(-20,20),(-1,1),(-1,2)],args=(numpy.array(predicted_activities[:,i].T)[0],numpy.array(activities[:,i].T)[0]),approx_grad=True,messages=0)
        params.append(p)
	if display:
		pylab.subplot(15,7,i+1)
	        pylab.plot(numpy.array(predicted_activities[:,i].T)[0],numpy.array(activities[:,i].T)[0],'go')
        	pylab.plot(numpy.array(predicted_activities[:,i].T)[0],fitfunc(p,numpy.array(predicted_activities[:,i].T)[0]),'bo')

    return params
    
    
    
def apply_sigmoid_output_function(activities,of,offset=True):
    sig = lambda p, x: (offset*p[2]) + p[3] * 1 / (1 + numpy.exp(-p[0]*(x-p[1])))
    (x,y) = numpy.shape(activities)	
    new_acts = numpy.zeros((x,y))
    
    for i in xrange(0,y):
	new_acts[:,i] = sig(of[i],numpy.array(activities[:,i].T)[0]).T
    return new_acts

def apply_exponential_output_function(activities,of,offset=True):
    sig = lambda p, x: offset*p[0] + p[1] * numpy.exp(p[2]*(x-p[3])) 
    (x,y) = numpy.shape(activities)	
    new_acts = numpy.zeros((x,y))
    
    for i in xrange(0,y):
	new_acts[:,i] = sig(of[i],numpy.array(activities[:,i].T)[0]).T
    return new_acts

def apply_power_output_function(activities,of):
    sig = lambda p, x: p[0] + p[1] * numpy.power(x,p[2])
    (x,y) = numpy.shape(activities)	
    new_acts = numpy.zeros((x,y))
    
    for i in xrange(0,y):
	new_acts[:,i] = sig(of[i],numpy.array(activities[:,i].T)[0]).T
    return new_acts
    
    
    
def fit2DOF(pred_act1,pred_act2,act,num_bins=10):
    bin_size1 = (numpy.max(pred_act1,axis=0) - numpy.min(pred_act1,axis=0))/6.0 
    bin_size2 = (numpy.max(pred_act2,axis=0) - numpy.min(pred_act2,axis=0))/6.0
    	
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
    
    ofn = ofn + (ofn <= 0)
    
    return (of/ofn,bin_size1,bin_size2,numpy.min(pred_act1,axis=0),numpy.min(pred_act2,axis=0))
    

def apply2DOF(activities1,activities2,ofs):
    (of,bin_size1,bin_size2,offset1,offset2) = ofs
    
    new_activities = numpy.zeros(numpy.shape(activities1))
    
    idx = numpy.arange(0,numpy.shape(of)[0],1)
    
    for i in xrange(0,numpy.shape(activities1)[0]):
	idx1 = numpy.round_((activities1[i,:]-offset1) / bin_size1)    	
	idx2 = numpy.round_((activities2[i,:]-offset2) / bin_size2)
	
	
	idx1 = idx1 - (idx1 >= numpy.shape(of)[1])
	idx2 = idx2 - (idx2 >= numpy.shape(of)[1])
	
	new_activities[i,:] = of(zip(idx,idx1,idx2))
	
    return new_activities     	