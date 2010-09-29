import __main__
import numpy
import pylab
import sys
from contrib.modelfit import *
import contrib.modelfit
import contrib.dd
import contrib.JanA.dataimport
from contrib.JanA.visualization import compareModelPerformanceWithRPI, showRFS
    
    
    
def bilinearRegression(training_inputs,training_set,sizex,sizey,num_steps,alpha,num_neurons):
    laplace = laplaceBias(sizex,sizey)
    
    (num_pres,kernel_size) = numpy.shape(training_inputs)
    
    X = numpy.mat(training_inputs)
    Y = numpy.mat(training_set)
    
    print numpy.shape(alpha*laplace)
    print numpy.shape(X.T*X)
    
    K1 = []
    K2 = []
    
    kk1 = numpy.linalg.pinv(X.T*X + alpha*laplace) * X.T * Y
    
    
    M = numpy.zeros((num_pres,kernel_size,kernel_size))
    for xx in xrange(0,sizex):
        for yy in xrange(0,sizey):
		for x in xrange(0,sizex):
			for y in xrange(0,sizey):
				if( ((xx + (x-sizex/2)) < 0) or ((xx + (x-sizex/2)) >= sizex) or ((yy + (y-sizey/2)) < 0) or ((yy + (y-sizey/2)) >= sizey)):
					M[:,yy*sizex+xx, y*sizex+x] = 0  
				else:
					M[:,yy*sizex+xx, y*sizex+x] = training_inputs[:,(yy + (y-sizey/2))*sizex + (xx + (x-sizex/2))]
    
    for n in xrange(0,num_neurons):
	print 'Neuron: ', n
	k1 = kk1[:,n]	
	for i in xrange(0,num_steps):
		print 'Step: ', i
		A =  numpy.hstack([numpy.mat(M[:,:,i]) * k1 for i in xrange(0,sizex*sizey)])
		k2 = numpy.linalg.pinv(A.T*A + alpha*laplace) * A.T * Y[:,n]
		B =  numpy.hstack([numpy.mat(M[:,i,:]) * k2 for i in xrange(0,sizex*sizey)])
		k1 = numpy.linalg.pinv(B.T*B + alpha*laplace) * B.T * Y[:,n]
	K1.append(k1)
	K2.append(k2)
     
     
     
    K1 = numpy.hstack(K1).T 
    K2 = numpy.hstack(K2).T
    
    print sizex
    print sizey 
    
    print shape(K1) 
    numpy.reshape(numpy.array(K1),(-1,sizex,sizey))
    print numpy.shape(numpy.reshape(numpy.array(K1),(-1,sizex,sizey)))
    
    showRFS(numpy.reshape(numpy.array(K1),(-1,sizex,sizey)))
    release_fig('K1.png')
    
    showRFS(numpy.reshape(numpy.array(K2),(-1,sizex,sizey)))
    release_fig('K2.png')
     
    return (K1,K2)

def bilinearModelResponse(inputs,k1,k2,sizex,sizey):
    (num_pres,kernel_size) = numpy.shape(inputs)
    num_neurons = numpy.shape(k1)[0]
    print sizex,sizey,kernel_size
	
    M = numpy.zeros((num_pres,kernel_size,kernel_size))	
    
    for xx in xrange(0,sizex):
        for yy in xrange(0,sizey):
		for x in xrange(0,sizex):
			for y in xrange(0,sizey):
				if( ((xx + (x-sizex/2)) < 0) or ((xx + (x-sizex/2)) >= sizex) or ((yy + (y-sizey/2)) < 0) or ((yy + (y-sizey/2)) >= sizey)):
					M[:,yy*sizex+xx, y*sizex+x] = 0  
				else:
					M[:,yy*sizex+xx, y*sizex+x] = inputs[:,(yy + (y-sizey/2))*sizex + (xx + (x-sizex/2))]
					
    response=[]					
    for n in xrange(0,num_neurons):
	     A =  numpy.hstack([numpy.mat(M[:,:,i]) * k1.T[:,n] for i in xrange(0,sizex*sizey)])
    	     response.append(A*k2.T[:,n])
     
    return numpy.hstack(response)
	
	

def runBilinearRegression():
    d = contrib.dd.loadResults("newest_dataset.dat")
    (sizex,sizey,training_inputs,training_set,validation_inputs,validation_set,ff,db_node) = sortOutLoading(d)
    raw_validation_set = db_node.data["raw_validation_set"]
    
    contrib.modelfit.save_fig_directory='/home/antolikjan/Doc/reports/Sparsness/InputContext/'
    
    params={}
    params["Bilinear"]=True
    db_node = db_node.get_child(params)
    
    params={}
    params["alpha"] = __main__.__dict__.get('Alpha',0.02)
    params["num_steps"] = __main__.__dict__.get('NumSteps',1)
    params["num_neurons"]= __main__.__dict__.get('NumNeurons',10)
    db_node = db_node.get_child(params)
    
    kernels = bilinearRegression(training_inputs,training_set,sizex,sizey,params["num_steps"],params["alpha"],params["num_neurons"])
     
    pred_act = bilinearModelResponse(training_inputs,kernels[0],kernels[1],sizex,sizey)
    pred_val_act = bilinearModelResponse(validation_inputs,kernels[0],kernels[1],sizex,sizey)
    
    compareModelPerformanceWithRPI(training_set[:,0:params["num_neurons"]],validation_set[:,0:params["num_neurons"]],training_inputs,validation_inputs,numpy.mat(pred_act),numpy.mat(pred_val_act),numpy.array(raw_validation_set)[:,:,0:params["num_neurons"]],'BilinearModel')
    
    db_node.add_data("Kernels",kernels,force=True)
    contrib.dd.saveResults(d,"newest_dataset.dat")


def analyseBlilinearModel():
    d = contrib.dd.loadResults("newest_dataset.dat")
    	
    dataset_node = d.children[0].children[0]
	
    training_set = dataset_node.data["training_set"]
    validation_set = dataset_node.data["validation_set"]
    training_inputs= dataset_node.data["training_inputs"]
    validation_inputs= dataset_node.data["validation_inputs"]
    raw_validation_set = dataset_node.data["raw_validation_set"]
	
    kernels = dataset_node.children[1].children[4].data['Kernels']
    
    K1 = kernels[0]
    K2 = kernels[1]
       
    pred_act = numpy.multiply(training_inputs*K1.T,training_inputs*K2.T)
    pred_val_act = numpy.multiply(validation_inputs*K1.T,validation_inputs*K2.T)
    
    compareModelPerformanceWithRPI(training_set,validation_set,training_inputs,validation_inputs,numpy.mat(pred_act),numpy.mat(pred_val_act),raw_validation_set,85)	


def laplaceBias(sizex,sizey):
	S = numpy.zeros((sizex*sizey,sizex*sizey))
	for x in xrange(0,sizex):
		for y in xrange(0,sizey):
			norm = numpy.mat(numpy.zeros((sizex,sizey)))
			norm[x,y]=4
			if x > 0:
				norm[x-1,y]=-1
			if x < sizex-1:
				norm[x+1,y]=-1   
			if y > 0:
				norm[x,y-1]=-1
			if y < sizey-1:
				norm[x,y+1]=-1
			S[x*sizex+y,:] = norm.flatten()
	S=numpy.mat(S)
        return S*S.T
