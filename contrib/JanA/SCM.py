from scipy.optimize import fmin_ncg, fmin_tnc
import __main__
import numpy
import pylab
import sys
sys.path.append('/home/antolikjan/topographica/Theano/')
import theano 
from theano import tensor as T
from topo.misc.filepath import normalize_path, application_path
from contrib.modelfit import *
import contrib.dd
import contrib.JanA.dataimport
from contrib.JanA.regression import laplaceBias
from contrib.JanA.visualization import compareModelPerformanceWithRPI, showRFS, visualize2DOF
from contrib.JanA.ofestimation import *

class SimpleContextualModel(object):
	
	def __init__(self,XX,YY,ZZ,of1='Exp',of2='Sqrt'):
	    (self.num_pres,self.kernel_size) = numpy.shape(XX) 	
	    
	    self.Y = theano.shared(YY)
    	    self.X = theano.shared(XX)
	    self.Z = theano.shared(ZZ)
	    self.K = T.dvector('K')
	    self.k1 = self.K[0:self.kernel_size]
	    self.k2 = self.K[self.kernel_size:2*self.kernel_size]
	    self.n1 = self.K[2*self.kernel_size]
	    self.n2 = self.K[2*self.kernel_size+1]
	    self.a1 = self.K[2*self.kernel_size+2]
	    self.a2 = self.K[2*self.kernel_size+3]
	    self.of1 = of1
	    self.of2 = of2
        
	def model_output(self):
	    lin = ( self.construct_of(T.dot(self.X,self.k1.T) - self.n1,self.of1)) / ((1+self.construct_of(T.dot(self.X,self.k2.T) - self.n2,self.of2)))
	    return lin 
	
	def log_likelyhood(self):
	    mo = self.model_output()	
	    ll = T.sum(mo) - T.sum(T.dot(self.Y.T,  T.log(mo)))
            ll = ll + T.sum(T.dot(self.k1 ,T.dot(__main__.__dict__.get('CLaplaceBias',0.0004)*self.Z,self.k1.T))) + T.sum(T.dot(self.k2 ,T.dot(__main__.__dict__.get('SLaplaceBias',0.0004)*self.Z,self.k2.T)))
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
	       return inn*inn
	    elif of == 'ExpExp':
	       return T.exp(T.exp(inn))  	
	    elif of == 'ExpSquare':
	       return T.exp(T.sqr(inn))
	    elif of == 'LogisticLoss':
	       return 1*T.log(1+T.exp(1*inn))
	    elif of == 'Zero':
	       return inn*0   
	    
	def response(self,X,kernels):
	    self.X.value = X
	       
	    resp = theano.function(inputs=[self.K], outputs=self.model_output())
	    
	    (a,b) = numpy.shape(kernels)
	    (c,d) = numpy.shape(X)
	    
	    responses = numpy.zeros((c,a))
	    for i in xrange(0,a):
		responses[:,i] = resp(kernels[i,:]).T
	    
	    return responses
	
	    
def runSCM():
    res = contrib.dd.loadResults("newest_dataset.dat")
    (sizex,sizey,training_inputs,training_set,validation_inputs,validation_set,ff,db_node) = contrib.JanA.dataimport.sortOutLoading(res)
    raw_validation_set = db_node.data["raw_validation_set"]
    
    params={}
    params["SCM"]=True
    db_node = db_node.get_child(params)
    
    params={}
    params["CLaplacaBias"] = __main__.__dict__.get('CLaplaceBias',0.0004)
    params["SLaplacaBias"] = __main__.__dict__.get('SLaplaceBias',0.0004)
    params["OF1"] = __main__.__dict__.get('OF1','Exp')
    params["OF2"] = __main__.__dict__.get('OF2','Square')
    params["num_neurons"] = __main__.__dict__.get('NumNeurons',103)
     
    # creat history
    training_set = numpy.mat(training_set)
    validation_set = numpy.mat(validation_set)
    training_inputs= numpy.mat(training_inputs)
    validation_inputs= numpy.mat(validation_inputs)
    
    db_node1 = db_node
    db_node = db_node.get_child(params)
    
    num_pres,num_neurons = numpy.shape(training_set)
    num_pres,kernel_size = numpy.shape(training_inputs)
    num_neurons_to_run=params["num_neurons"]
    
    Ks = numpy.zeros((num_neurons,kernel_size*2+4))
    
    print 'Kernel size',kernel_size
    
    laplace = laplaceBias(numpy.sqrt(kernel_size),numpy.sqrt(kernel_size))
    
    rpi = numpy.linalg.pinv(training_inputs.T*training_inputs + __main__.__dict__.get('RPILaplaceBias',0.0001)*laplace) * training_inputs.T * training_set
     
    for i in xrange(0,num_neurons_to_run): 
	print i
	#k0 = rpi[:,i].getA1().tolist() +  numpy.zeros((1,kernel_size)).flatten().tolist() + [0,0]
	k0 = numpy.zeros((1,kernel_size*2)).flatten().tolist() + [0,0,0,0]
	scm = SimpleContextualModel(numpy.mat(training_inputs),numpy.mat(training_set[:,i]),laplace,of1=params["OF1"],of2=params["OF2"])

	
	#K = fmin_ncg(scm.func(),numpy.array(k0),scm.der(),fhess = scm.hess(),avextol=0.0000001,maxiter=20)
	(K,success,c)=fmin_tnc(scm.func(),numpy.array(k0)[:],fprime=scm.der(),maxfun = 10000,messages=0)
	#print success
	#print c
	Ks[i,:] = K
    
    pred_act = scm.response(training_inputs,Ks)
    pred_val_act = scm.response(validation_inputs,Ks)
    
    print Ks[0,:]
    
    
    showRFS(numpy.reshape(Ks[:,0:kernel_size],(-1,numpy.sqrt(kernel_size),numpy.sqrt(kernel_size))))
    release_fig('K1.png')
    showRFS(numpy.reshape(Ks[:,kernel_size:2*kernel_size],(-1,numpy.sqrt(kernel_size),numpy.sqrt(kernel_size))))
    release_fig('K2.png')
        
    compareModelPerformanceWithRPI(training_set[:,:num_neurons_to_run],validation_set[:,:num_neurons_to_run],training_inputs,validation_inputs,numpy.mat(pred_act)[:,:num_neurons_to_run],numpy.mat(pred_val_act)[:,:num_neurons_to_run],numpy.array(raw_validation_set)[:,:,:num_neurons_to_run],'SCM')	

    db_node.add_data("Kernels",Ks,force=True)
    db_node.add_data("GLM",scm,force=True)

    #contrib.dd.saveResults(res,"newest_dataset.dat")
    
    
    
def sequentialFilterFinding():
    d = contrib.dd.loadResults("newest_dataset.dat")
    (sizex,sizey,training_inputs,training_set,validation_inputs,validation_set,ff,db_node) = sortOutLoading(d)
    raw_validation_set = db_node.data["raw_validation_set"]
    contrib.modelfit.save_fig_directory='/home/antolikjan/Doc/reports/Sparsness/SequentialFilterFitting/'
    
    params={}
    params["SequentialFilterFitting"]=True
    db_node = db_node.get_child(params)
    
    params={}
    params["alpha"] = __main__.__dict__.get('Alpha',0.02)
    params["num_neurons"]= __main__.__dict__.get('NumNeurons',10)
    params["OF"] = __main__.__dict__.get('OF','Square')
    db_node = db_node.get_child(params)
    
    num_neurons_to_run=params["num_neurons"]
    
    training_set = numpy.mat(training_set)[:,0:num_neurons_to_run]
    validation_set = numpy.mat(validation_set)[:,0:num_neurons_to_run]
    training_inputs= numpy.mat(training_inputs)
    validation_inputs= numpy.mat(validation_inputs)
    raw_validation_set = numpy.array(raw_validation_set)[:,:,0:num_neurons_to_run] 

    
    num_pres,kernel_size = numpy.shape(training_inputs)
    
    laplace = laplaceBias(numpy.sqrt(kernel_size),numpy.sqrt(kernel_size))
    rpi = numpy.linalg.pinv(training_inputs.T*training_inputs + __main__.__dict__.get('RPILaplaceBias',0.0001)*laplace) * training_inputs.T * training_set

    new_training_inputs = numpy.zeros(numpy.shape(training_inputs))
    new_validation_inputs = numpy.zeros(numpy.shape(validation_inputs)) 
            
    Ks = numpy.zeros((num_neurons_to_run,kernel_size*2+4))
    
    second_pred_act=[]
    second_pred_val_act=[]
    
    for i in xrange(0,num_neurons_to_run):
	print i
	# project out STA
	a = rpi[:,i]/numpy.sqrt(numpy.power(rpi[:,i],2))
	
	for j in xrange(0,numpy.shape(training_inputs)[0]):
		new_training_inputs[j,:] = training_inputs[j,:] - a.T * ((training_inputs[j,:]*a)[0,0])
	
	for j in xrange(0,numpy.shape(validation_inputs)[0]):
		new_validation_inputs[j,:] = validation_inputs[j,:] - a.T * ((validation_inputs[j,:]*a)[0,0])

	
	
	k0 = numpy.zeros((1,kernel_size*2)).flatten().tolist() + [0,0,0,0]
	scm = SimpleContextualModel(numpy.mat(new_training_inputs),numpy.mat(training_set[:,i]),laplace,of1=params["OF"],of2='Zero')
	#K = fmin_ncg(scm.func(),numpy.array(k0),scm.der(),fhess = scm.hess(),avextol=0.0000001,maxiter=20)
	(K,success,c)=fmin_tnc(scm.func(),numpy.array(k0)[:],fprime=scm.der(),maxfun = 1000,messages=0)
	Ks[i,:] = K
        
	second_pred_act.append(scm.response(new_training_inputs,numpy.array([K])))
        second_pred_val_act.append(scm.response(new_validation_inputs,numpy.array([K])))

    second_pred_act = numpy.hstack(second_pred_act)
    second_pred_val_act = numpy.hstack(second_pred_val_act)

    showRFS(numpy.reshape(Ks[:,0:kernel_size],(-1,numpy.sqrt(kernel_size),numpy.sqrt(kernel_size))))
    release_fig('K1.png')
    showRFS(numpy.reshape(Ks[:,kernel_size:2*kernel_size],(-1,numpy.sqrt(kernel_size),numpy.sqrt(kernel_size))))
    release_fig('K2.png')

    
    rpi_pred_act = training_inputs * rpi
    rpi_pred_val_act = validation_inputs * rpi
    
    #ofs = run_nonlinearity_detection(numpy.mat(training_set),numpy.mat(rpi_pred_act),num_bins=10,display=True)
    #rpi_pred_act_t = numpy.mat(apply_output_function(numpy.mat(rpi_pred_act),ofs))
    #rpi_pred_val_act_t = numpy.mat(apply_output_function(numpy.mat(rpi_pred_val_act),ofs))

    #ofs = run_nonlinearity_detection(numpy.mat(training_set),numpy.mat(second_pred_act),num_bins=10,display=True)
    #second_pred_act_t = numpy.mat(apply_output_function(numpy.mat(second_pred_act),ofs))
    #second_pred_val_act_t = numpy.mat(apply_output_function(numpy.mat(second_pred_val_act),ofs))

    visualize2DOF(rpi_pred_act,numpy.mat(second_pred_act),training_set)
    
    visualize2DOF(rpi_pred_val_act,numpy.mat(second_pred_val_act),validation_set)
    
    #pred_act = second_pred_act_t + rpi_pred_act_t #numpy.multiply(second_pred_act_t,rpi_pred_act_t)
    #pred_val_act = second_pred_val_act_t + rpi_pred_val_act_t #numpy.multiply(second_pred_val_act_t,rpi_pred_val_act_t)
    
    of = fit2DOF(rpi_pred_act,numpy.mat(second_pred_act),training_set)
    
    pred_act = apply2DOF(rpi_pred_act,numpy.mat(second_pred_act),of)
    pred_val_act = apply2DOF(rpi_pred_val_act,numpy.mat(second_pred_val_act),of)

    compareModelPerformanceWithRPI(training_set,validation_set,training_inputs,validation_inputs,numpy.mat(pred_act),numpy.mat(pred_val_act),numpy.array(raw_validation_set),'BilinearModel')
    
