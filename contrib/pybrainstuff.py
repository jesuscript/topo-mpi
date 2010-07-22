from scipy import reshape, dot, outer
import numpy

from pybrain.structure.connections.connection import Connection
from pybrain.structure.parametercontainer import ParameterContainer
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import FullConnection
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.tests.helpers import gradientCheck
from pybrain.tools.shortcuts import buildNetwork
from pybrain.utilities           import percentError


import contrib.modelfit

import pybrain.tools.functions 

class RBFConnection(Connection, ParameterContainer):
    """Connection which assumes the input layer to be of 2D sheets with dx,dy sizes of dimensions,
       and defines the weight profile to be a Gaussian, with free parameters x,y defining it's center and sigma
       defining its width"""
       	
    def __init__(self, dx,dy, *args, **kwargs):
        Connection.__init__(self, *args, **kwargs)
	self.dx = dx
	self.dy = dy
	ParameterContainer.__init__(self,3*self.outdim)
	
	for i in xrange(0,self.outdim):
	    self.params[2+i*3] = dx/5
	
        self.xx = (numpy.repeat([numpy.arange(0,self.dx,1)],self.dy,axis=0).T)	
	self.yy = (numpy.repeat([numpy.arange(0,self.dy,1)],self.dx,axis=0))

	
	assert self.indim == self.dx * self.dy, "Indim (%i) does not equal dx * dy (%i %i)" % (self.indim, self.dx, self.dy)

	

    def _forwardImplementation(self, inbuf, outbuf):
	par = reshape(self.params, (3,self.outdim))
	inn = reshape(inbuf, (self.dx,self.dy))
	self.out = numpy.zeros((self.outdim,self.dx,self.dy))
	for k in xrange(0,len(outbuf)):
	    kernel = 	((self.xx - par[0][k])**2  + (self.yy - par[1][k])**2)/(2*par[2][k]**2)
	    self.out[k] =  numpy.multiply(inn,pybrain.tools.functions.safeExp(-kernel))
	    outbuf[k]+=numpy.sum(self.out[k])

    
    def _backwardImplementation(self, outerr, inerr, inbuf):
        inerr += numpy.dot(numpy.reshape(self.out,(self.outdim,self.dx*self.dy)).T, outerr.T)
	par = reshape(self.params, (3,self.outdim))
	der = numpy.zeros(numpy.shape(par))
	#inn = reshape(inbuf, (self.dx,self.dy))
	
	for k in xrange(0,self.outdim):
	    der[0][k] = numpy.sum( numpy.multiply(self.out[k], (self.xx - par[0][k])/(par[2][k]**2))) * outerr[k]
	    der[1][k] = numpy.sum( numpy.multiply(self.out[k], (self.yy - par[1][k])/(par[2][k]**2)))* outerr[k]
	    der[2][k] = numpy.sum( numpy.multiply(self.out[k], ((self.xx - par[0][k])**2  + (self.yy - par[1][k])**2) /par[2][k]**3)) * outerr[k]
	ds = self.derivs
	ds += -der.flatten()

class LGNConnection(Connection, ParameterContainer):
    """Connection which assumes the input layer to be of 2D sheets with dx,dy sizes of dimensions,
       and defines the weight profile to be a Gaussian, with free parameters x,y defining it's center and sigma
       defining its width"""
       	
    def __init__(self, dx,dy, *args, **kwargs):
        Connection.__init__(self, *args, **kwargs)
	self.dx = dx
	self.dy = dy
	ParameterContainer.__init__(self,4*self.outdim)
	
	for i in xrange(0,self.outdim):
	    self.params[2+i*4] = dx/6.0
	    self.params[3+i*4] = dx/4.0
	
        self.xx = (numpy.repeat([numpy.arange(0,self.dx,1)],self.dy,axis=0).T)	
	self.yy = (numpy.repeat([numpy.arange(0,self.dy,1)],self.dx,axis=0))
	assert self.indim == self.dx * self.dy, "Indim (%i) does not equal dx * dy (%i %i)" % (self.indim, self.dx, self.dy)

	

    def _forwardImplementation(self, inbuf, outbuf):
	par = reshape(self.params, (4,self.outdim))
	inn = reshape(inbuf, (self.dx,self.dy))
	self.out = numpy.zeros((self.outdim,self.dx,self.dy))
	for k in xrange(0,len(outbuf)):
	    kernel = ((self.xx - par[0][k])**2  + (self.yy - par[1][k])**2)
	    self.out[k] =  numpy.multiply(inn,pybrain.tools.functions.safeExp(-kernel/2*par[2][k]**2)- pybrain.tools.functions.safeExp( - kernel/2*par[3][k]**2) )
	    outbuf[k]+=numpy.sum(self.out[k])

    
    def _backwardImplementation(self, outerr, inerr, inbuf):
        inerr += numpy.dot(numpy.reshape(self.out,(self.outdim,self.dx*self.dy)).T, outerr.T)
	par = reshape(self.params, (4,self.outdim))
	der = numpy.zeros(numpy.shape(par))
	#inn = reshape(inbuf, (self.dx,self.dy))
	
	for k in xrange(0,self.outdim):
	    der[0][k] = numpy.sum( numpy.multiply(self.out[k], (self.xx - par[0][k])/(par[2][k]**2) - (self.xx - par[0][k])/(par[3][k]**2)) ) * outerr[k]
	    der[1][k] = numpy.sum( numpy.multiply(self.out[k], (self.yy - par[1][k])/(par[2][k]**2) - (self.yy - par[1][k])/(par[3][k]**2)) ) * outerr[k]
	    
	    der[2][k] = 1.0 #numpy.sum( numpy.multiply(self.out[k], -((self.xx - par[0][k])**2  + (self.yy - par[1][k])**2) /par[2][k]**3)) * outerr[k]
	    der[3][k] = 1.0 #numpy.sum( numpy.multiply(self.out[k], -((self.xx - par[0][k])**2  + (self.yy - par[1][k])**2) /par[3][k]**3)) * outerr[k]
	ds = self.derivs
	ds += -der.flatten()



def run():
	import scipy
	from scipy import linalg
	f = open("modelfitDatabase1.dat",'rb')
	import pickle
	dd = pickle.load(f)
	node = dd.children[13]
	
	rfs  = node.children[0].data["ReversCorrelationRFs"]
	
	
	
	pred_act  = numpy.array(node.children[0].data["ReversCorrelationPredictedActivities"])
	pred_val_act  = numpy.array(node.children[0].data["ReversCorrelationPredictedValidationActivities"])
	
	training_set = node.data["training_set"]
	validation_set = node.data["validation_set"]
	training_inputs = node.data["training_inputs"]
	validation_inputs = node.data["validation_inputs"]
	
	ofs = contrib.modelfit.fit_sigmoids_to_of(numpy.mat(training_set),numpy.mat(pred_act))
	pred_act_t = contrib.modelfit.apply_sigmoid_output_function(numpy.mat(pred_act),ofs)
	pred_val_act_t= contrib.modelfit.apply_sigmoid_output_function(numpy.mat(pred_val_act),ofs)
	
	
	(sx,sy) = numpy.shape(rfs[0])
	print sx,sy
	n = FeedForwardNetwork()
	
	inLayer = LinearLayer(sx*sy)
	hiddenLayer = SigmoidLayer(4)
	outputLayer= SigmoidLayer(1)
	
	n.addInputModule(inLayer)
	n.addModule(hiddenLayer)
	n.addOutputModule(outputLayer)

	
	in_to_hidden = RBFConnection(sx,sy,inLayer, hiddenLayer)
	#in_to_hidden = FullConnection(inLayer, hiddenLayer)
	hidden_to_out = FullConnection(hiddenLayer, outputLayer)
	
	n.addConnection(in_to_hidden)
	n.addConnection(hidden_to_out)
	n.sortModules()
	gradientCheck(n)
	return
	
	from pybrain.datasets import SupervisedDataSet
	ds = SupervisedDataSet(sx*sy, 1)
	val = SupervisedDataSet(sx*sy, 1)
	
	for i in xrange(0,len(training_inputs)):
		ds.addSample(training_inputs[i],training_set[i,0])
	
	for i in xrange(0,len(validation_inputs)):
		val.addSample(validation_inputs[i],validation_set[i,0])
	
	
	tstdata, trndata = ds.splitWithProportion( 0.1 )
	
	from pybrain.supervised.trainers import BackpropTrainer
	trainer = BackpropTrainer(n, trndata, momentum=0.1, verbose=True, learningrate=0.002)
	
	training_set = numpy.array(numpy.mat(training_set)[:,0])
	validation_set = numpy.array(numpy.mat(validation_set)[:,0])
	pred_val_act_t = numpy.array(numpy.mat(pred_val_act_t)[:,0])
	
	out = n.activateOnDataset(val)
	(ranks,correct,pred) = contrib.modelfit.performIdentification(validation_set,out)
	print "Correct:", correct , "Mean rank:", numpy.mean(ranks) , "MSE", numpy.mean(numpy.power(validation_set - out,2))

	
	print 'Start training'
	for i in range(50):
	    trnresult = percentError(trainer.testOnData(),trndata)
    	    tstresult = percentError(trainer.testOnData(dataset=tstdata),tstdata)

            print "epoch: %4d" % trainer.totalepochs, "  train error: %5.2f%%" % trnresult, "  test error: %5.2f%%" % tstresult
	    trainer.trainEpochs( 1 )	
	    
	    out = n.activateOnDataset(val)
	    (ranks,correct,pred) = contrib.modelfit.performIdentification(validation_set,out)
	    print "Correct:", correct , "Mean rank:", numpy.mean(ranks) , "MSE", numpy.mean(numpy.power(validation_set - out,2))

	out = n.activateOnDataset(val)
	
	
	print numpy.shape(out)
	print numpy.shape(validation_set)
	
	(ranks,correct,pred) = contrib.modelfit.performIdentification(validation_set,out)
	print "Correct:", correct , "Mean rank:", numpy.mean(ranks) , "MSE", numpy.mean(numpy.power(validation_set - out,2))

	(ranks,correct,pred) = contrib.modelfit.performIdentification(validation_set,pred_val_act_t)
	print "Correct:", correct , "Mean rank:", numpy.mean(ranks) , "MSE", numpy.mean(numpy.power(validation_set - pred_val_act_t,2))


	return n	
	
