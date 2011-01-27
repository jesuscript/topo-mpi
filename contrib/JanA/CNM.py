from scipy.optimize import fmin_ncg, fmin_tnc
import scipy
import __main__
import numpy
import pylab
import sys
sys.path.append('/home/antolikjan/topographica/Theano/')
import theano 
from theano import tensor as T
from topo.misc.filepath import normalize_path, application_path
from contrib.JanA.ofestimation import *
from contrib.modelfit import *
import contrib.dd
import contrib.JanA.dataimport
from contrib.JanA.regression import laplaceBias
from contrib.JanA.visualization import compareModelPerformanceWithRPI, showRFS, visualize2DOF


class ContrastNormalizationModel(object):
	
	def __init__(self,XX,YY,ZZ,sizex,sizey,of_aff='Exp',of_surr='Linear'):
	    (self.num_pres,self.kernel_size) = numpy.shape(XX) 	
	    
	    self.Y = theano.shared(YY)
    	    self.X = theano.shared(XX)
	    self.Z = theano.shared(ZZ)
	    
    	    self.xx = theano.shared(numpy.repeat([numpy.arange(0,sizex,1)],sizey,axis=0).T.flatten())	
	    self.yy = theano.shared(numpy.repeat([numpy.arange(0,sizey,1)],sizex,axis=0).flatten())

	    self.K = T.dvector('K')
	    self.x = self.K[0]
	    self.y = self.K[1]
	    self.surr_size = self.K[2]
	    self.surr_gain = self.K[3]
	    self.surr_c50 = self.K[4]
	    self.n1 = self.K[5]
	    self.n2 = self.K[6]
	    self.k = self.K[7:sizex*sizey+7]
	    
	    self.of_aff = of_aff
	    self.of_surr = of_surr
	    #self.a = T.reshape(self.K[5:sizex*sizey],(sizex,sizey))
	    
        
	def model_output(self):
	    #surr = self.surr_gain *T.var(self.X,axis=1)**2
	    #surr = self.surr_gain *T.mean(self.X,axis=1)**2
	    surr = self.surr_gain * T.dot(self.X,T.exp(-T.div_proxy(((self.xx - self.x)**2 + (self.yy - self.y)**2),self.surr_size)).T/T.sqrt(self.surr_size*numpy.pi))
	    aff = T.dot(self.X,self.k.T)
	    
	    #lgn_output = theano.printing.Print(message='lgn output:')(lgn_output)
	    
	    lin = self.construct_of(aff / (self.surr_c50 + self.construct_of(surr,self.of_surr)) - self.n1,self.of_aff) 
	    return lin 
	
	def log_likelyhood(self):
	    mo = self.model_output()	
	    ll = T.sum(mo) - T.sum(T.dot(self.Y.T,  T.log(mo)))
            ll = ll + T.sum(T.dot(self.k ,T.dot(__main__.__dict__.get('LaplaceBias',0.0004)*self.Z,self.k.T)))
	    
	    return ll	
	
	def func(self):
	    return theano.function(inputs=[self.K], outputs=self.log_likelyhood(),mode='FAST_RUN') 
			
	def der(self):
	    g_K = T.grad(self.log_likelyhood(), self.K)
	    return theano.function(inputs=[self.K], outputs=g_K,mode='FAST_RUN')
 
 	def hess(self):
            g_K = T.grad(self.log_likelyhood(), self.K,consider_constant=[self.Y,self.X])
	    H, updates = theano.scan(lambda i,v: T.grad(g_K[i],v), sequences= T.arange(g_K.shape[0]), non_sequences=self.K)
  	    f = theano.function(inputs=[self.K], outputs=H,mode='FAST_RUN')
	    return f
	
	def construct_of(self,inn,of):
    	    if of == 'Exp':
	       return T.exp(inn)
	    elif of == 'Sigmoid':
	       return 1 / (1 + T.exp(-inn)) 
	    elif of == 'Square':
	       return T.sqr(inn)
	    elif of == 'ExpExp':
	       return T.exp(T.exp(inn))
	    elif of == 'Linear':
	       return inn  	
	    elif of == 'ExpSquare':
	       return T.exp(T.sqr(inn))
	    elif of == 'LogisticLoss':
	       return 1*T.log(1+T.exp(1*inn))
	    elif of == 'Zero':
	       return inn*0   
	    elif of == 'Linear':
	       return inn   
	    
	def response(self,X,kernels):
	    self.X.value = X
	       
	    resp = theano.function(inputs=[self.K], outputs=self.model_output())
	    
	    (a,b) = numpy.shape(kernels)
	    (c,d) = numpy.shape(X)
	    
	    responses = numpy.zeros((c,a))
	    for i in xrange(0,a):
		responses[:,i] = resp(kernels[i,:]).T
	    
	    return responses
	
	    
def runCNM():
    res = contrib.dd.loadResults("newest_dataset.dat")
    (sizex,sizey,training_inputs,training_set,validation_inputs,validation_set,ff,db_node) = contrib.JanA.dataimport.sortOutLoading(res)
    raw_validation_set = db_node.data["raw_validation_set"]
    
    params={}
    params["SCM"]=True
    db_node = db_node.get_child(params)
    
    params={}
    params["LaplacaBias"] = __main__.__dict__.get('LaplaceBias',0.0004)
    params["OFAff"] = __main__.__dict__.get('OFAff','Exp')
    params["OFSurr"] = __main__.__dict__.get('OFSurr','Linear')
    params["num_neurons"] = __main__.__dict__.get('NumNeurons',103)
     
    significant = [0,1,6,7,8,9,12,13,15,16,17,19,20,21,22,23,24,25,26,27,28,30,31,32,33,34,38,39,40,42,43,45,46,47,48,49,50,51,52,53,56,58,59,61,62,63,64,65,66,69,71,72,75,77,78,79,80,83,85,89,91,92,93,94,96,100,101,102] 
    # creat history
    training_set = numpy.mat(training_set)[:,significant]
    validation_set = numpy.mat(validation_set)[:,significant]
    training_inputs= numpy.mat(training_inputs)
    validation_inputs= numpy.mat(validation_inputs)
    
    for i in xrange(0,len(raw_validation_set)):
        raw_validation_set[i] = numpy.mat(raw_validation_set[i])[:,significant]
	
    db_node1 = db_node
    db_node = db_node.get_child(params)
    
    num_pres,num_neurons = numpy.shape(training_set)
    num_pres,kernel_size = numpy.shape(training_inputs)
    num_neurons_to_run=params["num_neurons"]
    
    Ks = numpy.zeros((num_neurons,kernel_size+7))
    
    print 'Kernel size',kernel_size
    
    laplace = laplaceBias(sizex,sizey)
    
    
    rpi = numpy.linalg.pinv(training_inputs.T*training_inputs + __main__.__dict__.get('RPILaplaceBias',0.0004)*laplace) * training_inputs.T * training_set
    
    
    
     
    bounds  = []
     
    for i in xrange(0,kernel_size):
	bounds.append((-1000000000,10000000000))
    
    bounds = [(6,26),(6,25),(1,1000000),(0.0,500000),(0.001,100000),(-100,100),(-100,100)] + bounds
    
    for i in xrange(0,num_neurons_to_run): 
	print i
	k0 = [15,15,20,100000,1,0,0] + rpi[:,i].getA1().tolist() 
	
	print numpy.shape(k0)
	print sizex,sizey
	scm = ContrastNormalizationModel(training_inputs,numpy.mat(training_set[:,i]),laplace,sizex,sizey,of_aff=params["OFAff"],of_surr=params["OFSurr"])

	#K = fmin_ncg(scm.func(),numpy.array(k0),scm.der(),fhess = scm.hess(),avextol=0.0000001,maxiter=20)
	(K,success,c)=fmin_tnc(scm.func(),numpy.array(k0)[:],fprime=scm.der(),bounds = bounds,maxfun = 100000,messages=0)
	print scm.func()(K)
	Ks[i,:] = K

    pred_act = scm.response(training_inputs,Ks)
    pred_val_act = scm.response(validation_inputs,Ks)
    
    from contrib.JanA.sparsness_analysis import TrevesRollsSparsness
    
    showRFS(numpy.reshape(numpy.array(rpi.T),(-1,sizex,sizey)))
    
    showRFS(numpy.reshape(Ks[:,7:kernel_size+7],(-1,sizex,sizey)))
    
    print Ks[:,:7]

    pylab.figure()
    pylab.hist(TrevesRollsSparsness(numpy.mat(pred_val_act)).flatten()) 
    
    pylab.figure()
    pylab.hist(TrevesRollsSparsness(numpy.mat(pred_val_act.T)).flatten()) 

    pylab.figure()
    pylab.hist(TrevesRollsSparsness(numpy.mat(validation_set)).flatten()) 
    
    pylab.figure()
    pylab.hist(TrevesRollsSparsness(numpy.mat(validation_set.T)).flatten()) 


    compareModelPerformanceWithRPI(training_set[:,:num_neurons_to_run],validation_set[:,:num_neurons_to_run],training_inputs,validation_inputs,numpy.mat(pred_act)[:,:num_neurons_to_run],numpy.mat(pred_val_act)[:,:num_neurons_to_run],numpy.array(raw_validation_set)[:,:,:num_neurons_to_run],sizex,sizey,'SCM')	

    db_node.add_data("Kernels",Ks,force=True)
    db_node.add_data("GLM",scm,force=True)

    #contrib.dd.saveResults(res,"newest_dataset.dat")
    
    
