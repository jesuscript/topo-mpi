from scipy.optimize import fmin_ncg, anneal, fmin_cg, fmin_bfgs, fmin_tnc, fmin_l_bfgs_b
import __main__
import numpy
import pylab
import sys
sys.path.append('/home/jan/Theano/')
import theano
theano.config.floatX='float32' 
#theano.config.warn.sum_sum_bug=False
from theano import tensor as T
from theano import function, config, shared, sandbox
from param import normalize_path
from contrib.JanA.ofestimation import *
from contrib.modelfit import *
import contrib.dd
import contrib.JanA.dataimport
from contrib.JanA.regression import laplaceBias
from pyevolve import *
from contrib.JanA.visualization import printCorrelationAnalysis

pylab.show=lambda:1; 
pylab.interactive(True)

class LSCSM(object):
	def __init__(self,XX,YY,num_lgn,num_neurons):
	    (self.num_pres,self.kernel_size) = numpy.shape(XX)
	    self.num_lgn = num_lgn
	    self.num_neurons = num_neurons
	    self.size = numpy.sqrt(self.kernel_size)

	    self.xx = theano.shared(numpy.repeat([numpy.arange(0,self.size,1)],self.size,axis=0).T.flatten())	
	    self.yy = theano.shared(numpy.repeat([numpy.arange(0,self.size,1)],self.size,axis=0).flatten())
	    self.Y = theano.shared(YY)
    	    self.X = theano.shared(XX)

	    
	    self.v1of = __main__.__dict__.get('V1OF','Exp')
	    self.lgnof = __main__.__dict__.get('LGNOF','Exp')
	    
	    self.K = T.dvector('K')
	    
	    self.x = self.K[0:self.num_lgn]
	    self.y = self.K[self.num_lgn:2*self.num_lgn]
	    self.sc = self.K[2*self.num_lgn:3*self.num_lgn]
	    self.ss = self.K[3*self.num_lgn:4*self.num_lgn]
	    
	    idx = 4*self.num_lgn
	    
	    if not __main__.__dict__.get('BalancedLGN',True):
		    self.rc = self.K[idx:idx+self.num_lgn]
		    self.rs = self.K[idx+self.num_lgn:idx+2*self.num_lgn]
		    idx = idx  + 2*self.num_lgn
	    
	    if __main__.__dict__.get('LGNTreshold',False):
	    	self.ln = self.K[idx:idx + self.num_lgn]
		idx += self.num_lgn
	    
	    
	    
	    if __main__.__dict__.get('SecondLayer',False):
	       self.a = T.reshape(self.K[idx:idx+int(num_neurons*__main__.__dict__.get('HiddenLayerSize',1.0))*self.num_lgn],(self.num_lgn,int(self.num_neurons*__main__.__dict__.get('HiddenLayerSize',1.0))))
	       idx +=  int(num_neurons*__main__.__dict__.get('HiddenLayerSize',1.0))*self.num_lgn		    
	       self.a1 = T.reshape(self.K[idx:idx+num_neurons*int(self.num_neurons*__main__.__dict__.get('HiddenLayerSize',1.0))],(int(self.num_neurons*__main__.__dict__.get('HiddenLayerSize',1.0)),self.num_neurons))
	       idx = idx+num_neurons*int(num_neurons*__main__.__dict__.get('HiddenLayerSize',1.0))
	    else:
	       self.a = T.reshape(self.K[idx:idx+num_neurons*self.num_lgn],(self.num_lgn,self.num_neurons))
	       idx +=  num_neurons*self.num_lgn

	    
	    self.n = self.K[idx:idx+self.num_neurons]
	    
	    if __main__.__dict__.get('SecondLayer',False):
	       self.n1 = self.K[idx+self.num_neurons:idx+self.num_neurons+int(self.num_neurons*__main__.__dict__.get('HiddenLayerSize',1.0))]
	    
	    #if __main__.__dict__.get('BalancedLGN',True):
		#lgn_kernel = lambda i,x,y,sc,ss: T.dot(self.X,(T.exp(-((self.xx - x[i])**2 + (self.yy - y[i])**2)/2/sc[i]).T/ (2*sc[i]*numpy.pi)) - (T.exp(-((self.xx - x[i])**2 + (self.yy - y[i])**2)/2/ss[i]).T/ (2*ss[i]*numpy.pi)))
		#lgn_output,updates = theano.scan(lgn_kernel , sequences= T.arange(self.num_lgn), non_sequences=[self.x,self.y,self.sc,self.ss])
	    
	    #else:
		#lgn_kernel = lambda i,x,y,sc,ss,rc,rs: T.dot(self.X,rc[i]*(T.exp(-((self.xx - x[i])**2 + (self.yy - y[i])**2)/2/sc[i]).T/ (2*sc[i]*numpy.pi)) - rs[i]*(T.exp(-((self.xx - x[i])**2 + (self.yy - y[i])**2)/2/ss[i]).T/ (2*ss[i]*numpy.pi)))
	        #lgn_output,updates = theano.scan(lgn_kernel,sequences=T.arange(self.num_lgn),non_sequences=[self.x,self.y,self.sc,self.ss,self.rc,self.rs])
	    
	    lgn_kernel = lambda i,x,y,sc,ss: T.dot(self.X,rc[i]*(T.exp(-((self.xx - x[i])**2 + (self.yy - y[i])**2)/2/sc[i]).T/ (2*sc[i]*numpy.pi)))
	    lgn_output,updates = theano.scan(lgn_kernel , sequences= T.arange(self.num_lgn), non_sequences=[self.x,self.y,self.sc,self.ss])
	    
	    lgn_output = lgn_output.T
	    
	    if __main__.__dict__.get('LGNTreshold',False):
	       lgn_output = lgn_output - self.ln.T
 
	       
	    self.output = T.dot(self.construct_of(lgn_output,self.lgnof),self.a)
	    if __main__.__dict__.get('SecondLayer',False):
	       self.model_output = self.construct_of(self.output-self.n1,self.v1of)
	       self.model_output = self.construct_of(T.dot(self.model_output , self.a1) - self.n,self.v1of)
	    else:
	       self.model_output = self.construct_of(self.output-self.n,self.v1of)	    
	    
   	    if __main__.__dict__.get('LL',True):
	       ll = T.sum(self.model_output) - T.sum(self.Y * T.log(self.model_output+0.0000000000000000001))
	       
	       if __main__.__dict__.get('Sparse',False):
		  ll += __main__.__dict__.get('FLL1',1.0)*T.sum(abs(self.a)) + __main__.__dict__.get('FLL2',1.0)*T.sum(self.a**2) 
 		  if __main__.__dict__.get('SecondLayer',False):
			ll += __main__.__dict__.get('SLL1',1.0)*T.sum(abs(self.a1)) + __main__.__dict__.get('SLL2',1.0)**T.sum(self.a1**2)
	       
	    else:
	       ll = T.sum(T.sqr(self.model_output - self.Y)) 

	    self.loglikelyhood =  ll
	
	def func(self):
	    return theano.function(inputs=[self.K], outputs=self.loglikelyhood,mode='FAST_RUN')
	
	def der(self):
	    g_K = T.grad(self.loglikelyhood, self.K)
	    return theano.function(inputs=[self.K], outputs=g_K,mode='FAST_RUN')
	
	def response(self,X,kernels):
	    self.X.value = X
	    
	    resp = theano.function(inputs=[self.K], outputs=self.model_output,mode='FAST_RUN')
	    return resp(kernels)	
	
	def construct_of(self,inn,of):
   	    if of == 'Linear':
	       return inn
    	    if of == 'Exp':
	       return T.exp(inn)
	    elif of == 'Sigmoid':
	       return 5.0 / (1.0 + T.exp(-inn))
    	    elif of == 'SoftSign':
	       return inn / (1 + T.abs_(inn)) 
	    elif of == 'Square':
	       return T.sqr(inn)
	    elif of == 'ExpExp':
	       return T.exp(T.exp(inn))  	
	    elif of == 'ExpSquare':
	       return T.exp(T.sqr(inn))
	    elif of == 'LogisticLoss':
	       return __main__.__dict__.get('LogLossCoef',1.0)*T.log(1+T.exp(__main__.__dict__.get('LogLossCoef',1.0)*inn))

	
	def returnRFs(self,K):
	    x = K[0:self.num_lgn]
	    y = K[self.num_lgn:2*self.num_lgn]
	    sc = K[2*self.num_lgn:3*self.num_lgn]
	    ss = K[3*self.num_lgn:4*self.num_lgn]
	    idx = 4*self.num_lgn
	    
	    if not __main__.__dict__.get('BalancedLGN',True):
		    rc = K[idx:idx+self.num_lgn]
		    rs = K[idx+self.num_lgn:idx+2*self.num_lgn]
		    idx = idx  + 2*self.num_lgn
	    
    	    if __main__.__dict__.get('LGNTreshold',False):
	    	ln = K[idx:idx + self.num_lgn]
            	idx += self.num_lgn
		
	    if __main__.__dict__.get('SecondLayer',False):
	       a = numpy.reshape(K[idx:idx+int(self.num_neurons*__main__.__dict__.get('HiddenLayerSize',1.0))*self.num_lgn],(self.num_lgn,int(self.num_neurons*__main__.__dict__.get('HiddenLayerSize',1.0))))
	       idx +=  int(self.num_neurons*__main__.__dict__.get('HiddenLayerSize',1.0))*self.num_lgn		    
	       a1 = numpy.reshape(K[idx:idx+self.num_neurons*int(self.num_neurons*__main__.__dict__.get('HiddenLayerSize',1.0))],(int(self.num_neurons*__main__.__dict__.get('HiddenLayerSize',1.0)),self.num_neurons))
	       idx = idx+self.num_neurons*int(self.num_neurons*__main__.__dict__.get('HiddenLayerSize',1.0))
	    else:
	       a = numpy.reshape(K[idx:idx+self.num_neurons*self.num_lgn],(self.num_lgn,self.num_neurons))
	       idx +=  self.num_neurons*self.num_lgn
	
	    n = K[idx:idx+self.num_neurons]

	    if __main__.__dict__.get('SecondLayer',False):
	       n1 = K[idx+self.num_neurons:idx+self.num_neurons+int(self.num_neurons*__main__.__dict__.get('HiddenLayerSize',1.0))]

            rfs = numpy.zeros((self.num_neurons,self.kernel_size))
	    
	    xx = numpy.repeat([numpy.arange(0,self.size,1)],self.size,axis=0).T.flatten()	
	    yy = numpy.repeat([numpy.arange(0,self.size,1)],self.size,axis=0).flatten()
	    			    
            print 'X'				    
            print x
	    print 'Y'
	    print y
	    print 'SS'
	    print ss
	    print 'SC'
	    print sc
	    print 'A'
	    print a
	    print 'N'
	    print n
	    
	    if not __main__.__dict__.get('BalancedLGN',True):
		print 'RS'
		print rs
	    	print 'RC'
	    	print rc
	    
	    if __main__.__dict__.get('SecondLayer',False):
		print 'A1'
		print a1
	    if __main__.__dict__.get('LGNTreshold',False):
	       print 'LN'	    
	       print ln
	    
	    if __main__.__dict__.get('SecondLayer',False):
	    	num_neurons_first_layer = int(self.num_neurons*__main__.__dict__.get('HiddenLayerSize',1.0))  
	    else:
		num_neurons_first_layer = self.num_neurons
		
	    for j in xrange(num_neurons_first_layer):
	    	for i in xrange(0,self.num_lgn):
		    if  __main__.__dict__.get('BalancedLGN',True):			
		    	rfs[j,:] += a[i,j]*(numpy.exp(-((xx - x[i])**2 + (yy - y[i])**2)/2/sc[i])/(2*sc[i]*numpy.pi) - numpy.exp(-((xx - x[i])**2 + (yy - y[i])**2)/2/ss[i])/(2*ss[i]*numpy.pi)) 
		    else:
			rfs[j,:] += a[i,j]*(rc[i]*numpy.exp(-((xx - x[i])**2 + (yy - y[i])**2)/2/sc[i])/(2*sc[i]*numpy.pi) - rs[i]*numpy.exp(-((xx - x[i])**2 + (yy - y[i])**2)/2/ss[i])/(2*ss[i]*numpy.pi))
			
	        
	    return rfs
	
	def generateBounds(self):
	      bounds = []
	      for j in xrange(0,self.num_lgn):
		  bounds.append((6,(numpy.sqrt(self.kernel_size)-6)))
		  bounds.append((6,(numpy.sqrt(self.kernel_size)-6)))
	      for j in xrange(0,self.num_lgn):	
		  bounds.append((1.0,25))
		  bounds.append((1.0,25))

	      if not __main__.__dict__.get('BalancedLGN',True):	
		  for j in xrange(0,self.num_lgn):	
			  bounds.append((0.0,1.0))
			  bounds.append((0.0,1.0))

	      if __main__.__dict__.get('LGNTreshold',False):
		for j in xrange(0,self.num_lgn):
		    bounds.append((0,20))
		  

	      if __main__.__dict__.get('NegativeLgn',True):
		  minw = -__main__.__dict__.get('MaxW',5000)
	      else:
		  minw = 0
	      maxw = __main__.__dict__.get('MaxW',5000)
	      print __main__.__dict__.get('MaxW',5000)
	      
	      if __main__.__dict__.get('SecondLayer',False):
		  for j in xrange(0,self.num_lgn):		
			  for k in xrange(0,int(self.num_neurons*__main__.__dict__.get('HiddenLayerSize',1.0))):
				  bounds.append((minw,maxw))
			  
		  for j in xrange(0,int(self.num_neurons*__main__.__dict__.get('HiddenLayerSize',1.0))):		
			  for k in xrange(0,self.num_neurons):
				  bounds.append((-__main__.__dict__.get('MaxWL2',4),__main__.__dict__.get('MaxWL2',4)))
	      else:
		  for j in xrange(0,self.num_lgn):		
			  for k in xrange(0,self.num_neurons):
				  bounds.append((minw,maxw))
				  
			  
				  
	      for k in xrange(0,self.num_neurons):
		  bounds.append((0,20))
		  
	      if __main__.__dict__.get('SecondLayer',False):
		  for k in xrange(0,int(self.num_neurons*__main__.__dict__.get('HiddenLayerSize',1.0))):
			  bounds.append((0,20))
	      return bounds




class LSCSMNEW(object):
	def __init__(self,XX,YY,num_lgn,num_neurons,batch_size=100):
	    (self.num_pres,self.kernel_size) = numpy.shape(XX)
	    self.num_lgn = num_lgn
	    self.num_neurons = num_neurons
	    self.size = numpy.sqrt(self.kernel_size)
	    self.hls = __main__.__dict__.get('HiddenLayerSize',1.0)
	    self.divisive = __main__.__dict__.get('Divisive',False)
	    self.batch_size=batch_size	    

	    #self.xx = theano.shared(numpy.asarray(numpy.repeat([numpy.arange(0,self.size,1)],self.size,axis=0).T.flatten(),dtype=theano.config.floatX))	
	    #self.yy = theano.shared(numpy.asarray(numpy.repeat([numpy.arange(0,self.size,1)],self.size,axis=0).flatten(),dtype=theano.config.floatX))
	    self.xx = theano.shared(numpy.repeat([numpy.arange(0,self.size,1)],self.size,axis=0).T.flatten())	
	    self.yy = theano.shared(numpy.repeat([numpy.arange(0,self.size,1)],self.size,axis=0).flatten())
	    
	    #self.Y = theano.shared(numpy.asarray(YY,dtype=theano.config.floatX))
    	    #self.X = theano.shared(numpy.asarray(XX,dtype=theano.config.floatX))
    	    self.Y = theano.shared(YY)
    	    self.X = theano.shared(XX)

	    
	    self.v1of = __main__.__dict__.get('V1OF','Exp')
	    self.lgnof = __main__.__dict__.get('LGNOF','Exp')
	    
	    #self.K = T.fvector('K')
	    self.K = T.dvector('K')
	    #self.index = T.lscalar('I')
	    
	    #srng = RandomStreams(seed=234)
	    #self.index = srng.random_integers((1,1),high=self.num_pres-batch_size)[0]

	    
	    self.x = self.K[0:self.num_lgn]
	    self.y = self.K[self.num_lgn:2*self.num_lgn]
	    self.sc = self.K[2*self.num_lgn:3*self.num_lgn]
	    self.ss = self.K[3*self.num_lgn:4*self.num_lgn]
	    
	    idx = 4*self.num_lgn
	    
	    if not __main__.__dict__.get('BalancedLGN',True):
		    self.rc = self.K[idx:idx+self.num_lgn]
		    self.rs = self.K[idx+self.num_lgn:idx+2*self.num_lgn]
		    idx = idx  + 2*self.num_lgn
	    
	    if __main__.__dict__.get('LGNTreshold',False):
	    	self.ln = self.K[idx:idx + self.num_lgn]
		idx += self.num_lgn
	    
	    
	    
	    if __main__.__dict__.get('SecondLayer',False):
	       self.a = T.reshape(self.K[idx:idx+int(num_neurons*self.hls)*self.num_lgn],(self.num_lgn,int(self.num_neurons*self.hls)))
	       idx +=  int(num_neurons*self.hls)*self.num_lgn		    
	       self.a1 = T.reshape(self.K[idx:idx+num_neurons*int(self.num_neurons*self.hls)],(int(self.num_neurons*self.hls),self.num_neurons))
	       idx = idx+num_neurons*int(num_neurons*self.hls)
	       if self.divisive:
		       self.d = T.reshape(self.K[idx:idx+int(num_neurons*self.hls)*self.num_lgn],(self.num_lgn,int(self.num_neurons*self.hls)))
		       idx +=  int(num_neurons*self.hls)*self.num_lgn		    
	       	       self.d1 = T.reshape(self.K[idx:idx+num_neurons*int(self.num_neurons*self.hls)],(int(self.num_neurons*self.hls),self.num_neurons))
	       	       idx = idx+num_neurons*int(num_neurons*self.hls)
	    else:
	       self.a = T.reshape(self.K[idx:idx+num_neurons*self.num_lgn],(self.num_lgn,self.num_neurons))
	       idx +=  num_neurons*self.num_lgn
	       if self.divisive:	       
	               self.d = T.reshape(self.K[idx:idx+num_neurons*self.num_lgn],(self.num_lgn,self.num_neurons))
		       idx +=  num_neurons*self.num_lgn

	    
	    self.n = self.K[idx:idx+self.num_neurons]
	    idx +=  num_neurons
	    
	    if self.divisive:
		    self.nd = self.K[idx:idx+self.num_neurons]
		    idx +=  num_neurons
	    
	    if __main__.__dict__.get('SecondLayer',False):
	       self.n1 = self.K[idx:idx+int(self.num_neurons*self.hls)]
	       idx +=  int(self.num_neurons*self.hls)
	       if self.divisive:
		       self.nd1 = self.K[idx:idx+int(self.num_neurons*self.hls)]
		       idx +=  int(self.num_neurons*self.hls)
	    
	    if __main__.__dict__.get('BalancedLGN',True):
		lgn_kernel = lambda i,x,y,sc,ss: T.dot(self.X,(T.exp(-((self.xx - x[i])**2 + (self.yy - y[i])**2)/2/sc[i]).T/ (2*sc[i]*numpy.pi)) - (T.exp(-((self.xx - x[i])**2 + (self.yy - y[i])**2)/2/ss[i]).T/ (2*ss[i]*numpy.pi)))
		lgn_output,updates = theano.scan(lgn_kernel , sequences= T.arange(self.num_lgn), non_sequences=[self.x,self.y,self.sc,self.ss])
	    
	    else:
		lgn_kernel = lambda i,x,y,sc,ss,rc,rs: T.dot(self.X,rc[i]*(T.exp(-((self.xx - x[i])**2 + (self.yy - y[i])**2)/2/sc[i]).T/ (2*sc[i]*numpy.pi)) - rs[i]*(T.exp(-((self.xx - x[i])**2 + (self.yy - y[i])**2)/2/ss[i]).T/ (2*ss[i]*numpy.pi)))
	        lgn_output,updates = theano.scan(lgn_kernel,sequences=T.arange(self.num_lgn),non_sequences=[self.x,self.y,self.sc,self.ss,self.rc,self.rs])
	    
	    #lgn_output = theano.printing.Print(message='lgn output:')(lgn_output)
	    
	    lgn_output = lgn_output.T
	    
	    if __main__.__dict__.get('LGNTreshold',False):
	       lgn_output = lgn_output - self.ln.T
            lgn_output = self.construct_of(lgn_output,self.lgnof)
	       
	    self.output = T.dot(lgn_output,self.a)
	    if self.divisive:
		self.d_output = T.dot(lgn_output,self.d)
	    #self.output = theano.printing.Print(message='Output1:')(self.output)
	    
	    #self.n = theano.printing.Print(message='N:')(self.n)
	    
	    #self.output = theano.printing.Print(message='Output2:')(self.output)
	    
	    if __main__.__dict__.get('SecondLayer',False):
	       if self.divisive:
			self.model_output = self.construct_of(self.output-self.n1,self.v1of)
			self.d_model_output = self.construct_of(self.d_output-self.nd1,self.v1of)
	       		self.model_output = self.construct_of((T.dot(self.model_output , self.a1) - self.n)/(1+self.construct_of(T.dot(self.d_model_output , self.d1) - self.nd,self.v1of)),self.v1of)
	       else:
		        self.model_output = self.construct_of(self.output-self.n1,self.v1of)
	       		self.model_output = self.construct_of(T.dot(self.model_output , self.a1) - self.n,self.v1of)
	    else:
	       if self.divisive:
                  self.model_output = self.construct_of((self.output-self.n)/(1.0+T.dot(lgn_output,self.d)-self.nd),self.v1of)
	       else:
		  self.model_output = self.construct_of(self.output-self.n,self.v1of)
	    
	    
   	    if __main__.__dict__.get('LL',True):
	       #self.model_output = theano.printing.Print(message='model output:')(self.model_output)
	       ll = T.sum(self.model_output) - T.sum(self.Y * T.log(self.model_output+0.0000000000000000001))
	       
	       if __main__.__dict__.get('Sparse',False):
		  ll += __main__.__dict__.get('FLL1',1.0)*T.sum(abs(self.a)) + __main__.__dict__.get('FLL2',1.0)*T.sum(self.a**2) 
 		  if __main__.__dict__.get('SecondLayer',False):
			ll += __main__.__dict__.get('SLL1',1.0)*T.sum(abs(self.a1)) + __main__.__dict__.get('SLL2',1.0)**T.sum(self.a1**2)
	       
	    else:
	       ll = T.sum(T.sqr(self.model_output - self.Y)) 

	    #ll = theano.printing.Print(message='LL:')(ll)
	    self.loglikelyhood =  ll
	
	def func(self):
	    return theano.function(inputs=[self.K], outputs=self.loglikelyhood,mode='FAST_RUN')
	
	def der(self):
	    g_K = T.grad(self.loglikelyhood, self.K)
	    return theano.function(inputs=[self.K], outputs=g_K,mode='FAST_RUN')
	
	def response(self,X,kernels):
	    self.X.value = X
	    
	    resp = theano.function(inputs=[self.K], outputs=self.model_output,mode='FAST_RUN')
	    return resp(kernels)	
	
	def construct_of(self,inn,of):
   	    if of == 'Linear':
	       return inn
    	    if of == 'Exp':
	       return T.exp(inn)
	    elif of == 'Sigmoid':
	       return 5.0 / (1.0 + T.exp(-inn))
    	    elif of == 'SoftSign':
	       return inn / (1 + T.abs_(inn)) 
	    elif of == 'Square':
	       return T.sqr(inn)
	    elif of == 'ExpExp':
	       return T.exp(T.exp(inn))  	
	    elif of == 'ExpSquare':
	       return T.exp(T.sqr(inn))
	    elif of == 'LogisticLoss':
	       return __main__.__dict__.get('LogLossCoef',1.0)*T.log(1+T.exp(__main__.__dict__.get('LogLossCoef',1.0)*inn))

	
	def returnRFs(self,K):
	    x = K[0:self.num_lgn]
	    y = K[self.num_lgn:2*self.num_lgn]
	    sc = K[2*self.num_lgn:3*self.num_lgn]
	    ss = K[3*self.num_lgn:4*self.num_lgn]
	    idx = 4*self.num_lgn
	    
	    if not __main__.__dict__.get('BalancedLGN',True):
		    rc = K[idx:idx+self.num_lgn]
		    rs = K[idx+self.num_lgn:idx+2*self.num_lgn]
		    idx = idx  + 2*self.num_lgn
	    
    	    if __main__.__dict__.get('LGNTreshold',False):
	    	ln = K[idx:idx + self.num_lgn]
            	idx += self.num_lgn
		
	    if __main__.__dict__.get('SecondLayer',False):
	       a = numpy.reshape(K[idx:idx+int(self.num_neurons*self.hls)*self.num_lgn],(self.num_lgn,int(self.num_neurons*self.hls)))
	       idx +=  int(self.num_neurons*self.hls)*self.num_lgn		    
	       a1 = numpy.reshape(K[idx:idx+self.num_neurons*int(self.num_neurons*self.hls)],(int(self.num_neurons*self.hls),self.num_neurons))
	       idx = idx+self.num_neurons*int(self.num_neurons*self.hls)
	       if self.divisive:
		       d = numpy.reshape(K[idx:idx+int(self.num_neurons*self.hls)*self.num_lgn],(self.num_lgn,int(self.num_neurons*self.hls)))
		       idx +=  int(self.num_neurons*self.hls)*self.num_lgn		    
	       	       d1 = numpy.reshape(K[idx:idx+self.num_neurons*int(self.num_neurons*self.hls)],(int(self.num_neurons*self.hls),self.num_neurons))
	       	       idx = idx+self.num_neurons*int(self.num_neurons*self.hls)

	    else:
	       a = numpy.reshape(K[idx:idx+self.num_neurons*self.num_lgn],(self.num_lgn,self.num_neurons))
	       idx +=  self.num_neurons*self.num_lgn
       	       if self.divisive:	       
	               d = numpy.reshape(K[idx:idx+self.num_neurons*self.num_lgn],(self.num_lgn,self.num_neurons))
		       idx +=  self.num_neurons*self.num_lgn

	
	    n = K[idx:idx+self.num_neurons]

	    if __main__.__dict__.get('SecondLayer',False):
	       n1 = K[idx+self.num_neurons:idx+self.num_neurons+int(self.num_neurons*self.hls)]

	    xx = numpy.repeat([numpy.arange(0,self.size,1)],self.size,axis=0).T.flatten()	
	    yy = numpy.repeat([numpy.arange(0,self.size,1)],self.size,axis=0).flatten()
	    			    
            print 'X'				    
            print x
	    print 'Y'
	    print y
	    print 'SS'
	    print ss
	    print 'SC'
	    print sc
	    print 'A'
	    print a
	    print 'N'
	    print n
	    
	    if not __main__.__dict__.get('BalancedLGN',True):
		print 'RS'
		print rs
	    	print 'RC'
	    	print rc
	    
	    if __main__.__dict__.get('SecondLayer',False):
		print 'A1'
		print a1
		print self.hls
		pylab.figure()    
		pylab.imshow(a1)
	    
	    if __main__.__dict__.get('LGNTreshold',False):
	       print 'LN'	    
	       print ln
	    
	    if __main__.__dict__.get('SecondLayer',False):
	    	num_neurons_first_layer = int(self.num_neurons*self.hls)  
	    else:
		num_neurons_first_layer = self.num_neurons
            
	    rfs = numpy.zeros((num_neurons_first_layer,self.kernel_size))		
	    
	    for j in xrange(num_neurons_first_layer):
	    	for i in xrange(0,self.num_lgn):
		    if  __main__.__dict__.get('BalancedLGN',True):			
		    	rfs[j,:] += a[i,j]*(numpy.exp(-((xx - x[i])**2 + (yy - y[i])**2)/2/sc[i])/(2*sc[i]*numpy.pi) - numpy.exp(-((xx - x[i])**2 + (yy - y[i])**2)/2/ss[i])/(2*ss[i]*numpy.pi)) 
		    else:
			rfs[j,:] += a[i,j]*(rc[i]*numpy.exp(-((xx - x[i])**2 + (yy - y[i])**2)/2/sc[i])/(2*sc[i]*numpy.pi) - rs[i]*numpy.exp(-((xx - x[i])**2 + (yy - y[i])**2)/2/ss[i])/(2*ss[i]*numpy.pi))
			
	    return rfs

	def generateBounds(self):
	      bounds = []

	      for j in xrange(0,self.num_lgn):
		  bounds.append((6,(numpy.sqrt(self.kernel_size)-6)))
		  bounds.append((6,(numpy.sqrt(self.kernel_size)-6)))
		  
	      for j in xrange(0,self.num_lgn):	
		  bounds.append((1.0,25))
		  bounds.append((1.0,25))
	      if not __main__.__dict__.get('BalancedLGN',True):	
		  for j in xrange(0,self.num_lgn):	
			  bounds.append((0.0,1.0))
			  bounds.append((0.0,1.0))
		  
		  
	      if __main__.__dict__.get('LGNTreshold',False):
		for j in xrange(0,self.num_lgn):
		    bounds.append((0,20))
		  

	      if __main__.__dict__.get('NegativeLgn',True):
		  minw = -__main__.__dict__.get('MaxW',5000)
	      else:
		  minw = 0
	      maxw = __main__.__dict__.get('MaxW',5000)
	      print __main__.__dict__.get('MaxW',5000)
	      
	      if __main__.__dict__.get('Divisive',False):
		  d=2
	      else:
		  d=1

	      
	      if __main__.__dict__.get('SecondLayer',False):
		  for j in xrange(0,self.num_lgn):		
			  for k in xrange(0,int(self.num_neurons*__main__.__dict__.get('HiddenLayerSize',1.0))):
				  bounds.append((minw,maxw))
			  
		  for j in xrange(0,int(self.num_neurons*__main__.__dict__.get('HiddenLayerSize',1.0))):		
			  for k in xrange(0,self.num_neurons):
				  bounds.append((-__main__.__dict__.get('MaxWL2',4),__main__.__dict__.get('MaxWL2',4)))
		  if __main__.__dict__.get('Divisive',False):
			  for j in xrange(0,self.num_lgn):		
				  for k in xrange(0,int(self.num_neurons*__main__.__dict__.get('HiddenLayerSize',1.0))):
					  bounds.append((minw,maxw))
				  
			  for j in xrange(0,int(self.num_neurons*__main__.__dict__.get('HiddenLayerSize',1.0))):		
				  for k in xrange(0,self.num_neurons):
					  bounds.append((0,2))
				  
	      else:
		  for i in xrange(0,d):    
			  for j in xrange(0,self.num_lgn):		
				  for k in xrange(0,self.num_neurons):
					  bounds.append((minw,maxw))
				  
			  
				  
	      for k in xrange(0,self.num_neurons):
		  for i in xrange(0,d):
			  bounds.append((0,20))
		  
	      if __main__.__dict__.get('SecondLayer',False):
		  for i in xrange(0,d):
			  for k in xrange(0,int(self.num_neurons*__main__.__dict__.get('HiddenLayerSize',1.0))):
				  bounds.append((0,20))

	      return bounds


def fitLSCSM(training_inputs,training_set,lgn_num,num_neurons,validation_inputs,validation_set):
    num_pres,num_neurons = numpy.shape(training_set) 
    
    if __main__.__dict__.get('EarlyStopping',False):
       frac=0.1
    else:
       frac=0.01
    
    early_stopping_set = training_set[-num_pres*frac:,:]
    early_stopping_inputs = training_inputs[-num_pres*frac:,:]
    training_set = training_set[:-num_pres*frac,:]
    training_inputs = training_inputs[:-num_pres*frac,:]
    
    if __main__.__dict__.get('LSCSMOLD',True):
	lscsm = LSCSM(training_inputs,training_set,lgn_num,num_neurons)
    else:
        lscsm = LSCSMNEW(training_inputs,training_set,lgn_num,num_neurons)
    func = lscsm.func() 
    der = lscsm.der()
    bounds = lscsm.generateBounds()
    
    rand =numbergen.UniformRandom(seed=__main__.__dict__.get('Seed',513))
    
    Ks = []
    terr=[]
    eserr=[]
    verr=[]
    
    pylab.figure()
    
    pylab.show()
    pylab.hold(True)

    [Ks.append(a[0]+rand()*(a[1]-a[0])/2.0)  for a in bounds]
    
    best_Ks = list(Ks)
    best_eserr = 100000000000000000000000000000000000
    time_since_best=0
    
    for i in xrange(0,__main__.__dict__.get('NumEpochs',100)):
        print i
        
	(Ks,success,c)=fmin_tnc(func,Ks,fprime=der,bounds=bounds,maxfun = __main__.__dict__.get('EpochSize',1000),messages=0)
	
	terr.append(func(numpy.array(Ks))/numpy.shape(training_set)[0])
	lscsm.X.value = early_stopping_inputs
	lscsm.Y.value = early_stopping_set
	eserr.append(func(numpy.array(Ks))/ numpy.shape(early_stopping_set)[0])
	lscsm.X.value = validation_inputs
	lscsm.Y.value = validation_set
	verr.append(func(numpy.array(Ks))/numpy.shape(validation_set)[0])
	lscsm.X.value = training_inputs
	lscsm.Y.value = training_set
        print terr[-1],verr[-1],eserr[-1]
	pylab.plot(verr,'r')
	pylab.plot(terr,'b')
	pylab.plot(eserr,'g')	
	pylab.draw()
	
	if best_eserr > eserr[-1]:    
	   best_eserr = eserr[-1]
	   best_Ks = list(Ks)
	   time_since_best = 0
	else:
	   time_since_best+=1
	
	if __main__.__dict__.get('EarlyStopping',False):
	   if time_since_best > 50:
	      break

    if __main__.__dict__.get('EarlyStopping',False):
       Ks = best_Ks

    print 'Final training error: ', func(numpy.array(Ks))/ numpy.shape(training_set)[0] 
    lscsm.X.value = early_stopping_inputs
    lscsm.Y.value = early_stopping_set
    print 'Final testing error: ', func(numpy.array(Ks))/ numpy.shape(early_stopping_set)[0] 

    pylab.savefig(normalize_path('Error_evolution.png'))
    rfs = lscsm.returnRFs(Ks)
    kernel_size =  numpy.shape(training_inputs)[1]
    laplace = laplaceBias(numpy.sqrt(kernel_size),numpy.sqrt(kernel_size))
    rpi = numpy.linalg.pinv(training_inputs.T*training_inputs + __main__.__dict__.get('RPILaplaceBias',0.0001)*laplace) * training_inputs.T * training_set
    return [Ks,rpi,lscsm,rfs]	
  

def runLSCSM():
    import noiseEstimation
    
    import param
        
    contrib.modelfit.save_fig_directory=param.normalize_path.prefix
    
    print contrib.modelfit.save_fig_directory

    res = contrib.dd.DB(None)
    (sizex,sizey,training_inputs,training_set,validation_inputs,validation_set,ff,db_node) = contrib.JanA.dataimport.sortOutLoading(res)

    raw_validation_set = db_node.data["raw_validation_set"]
    
    num_pres,num_neurons = numpy.shape(training_set)
    num_pres,kernel_size = numpy.shape(training_inputs)
    size = numpy.sqrt(kernel_size)

    raw_validation_data_set=numpy.rollaxis(numpy.array(raw_validation_set),2)
    
    params={}
    params["LSCSM"]=True
    db_node = db_node.get_child(params)
    
    params={}
    params["LaplacaBias"] = __main__.__dict__.get('LaplaceBias',0.0004)
    params["LGN_NUM"] = __main__.__dict__.get('LgnNum',6)
    params["num_neurons"] = __main__.__dict__.get('NumNeurons',103)
    params["sequential"] = __main__.__dict__.get('Sequential',False)
    params["ll"] =  __main__.__dict__.get('LL',True)
    params["V1OF"] = __main__.__dict__.get('V1OF','Exp')
    params["LGNOF"] = __main__.__dict__.get('LGNOF','Exp')
    params["BalancedLGN"] = __main__.__dict__.get('BalancedLGN',True)
    params["LGNTreshold"] =  __main__.__dict__.get('LGNTreshold',False)
    params["SecondLayer"] = __main__.__dict__.get('SecondLayer',False)
    params["HiddenLayerSize"] = __main__.__dict__.get('HiddenLayerSize',1.0)
    params["LogLossCoef"] = __main__.__dict__.get('LogLossCoef',1.0)
    params["NegativeLgn"] = __main__.__dict__.get('NegativeLgn',True)
    params["MaxW"] = __main__.__dict__.get('MaxW',5000)
    params["GenerationSize"] = __main__.__dict__.get('GenerationSize',100)
    params["PopulationSize"] = __main__.__dict__.get('PopulationSize',100)
    params["MutationRate"] = __main__.__dict__.get('MutationRate',0.05)
    params["CrossoverRate"] = __main__.__dict__.get('CrossoverRate',0.9)
    params["FromWhichNeuron"] = __main__.__dict__.get('FromWhichNeuron',0)
    
    db_node1 = db_node
    db_node = db_node.get_child(params)
    
    num_neurons=params["num_neurons"]

    training_set = training_set[:,params["FromWhichNeuron"]:params["FromWhichNeuron"]+num_neurons]
    validation_set = validation_set[:,params["FromWhichNeuron"]:params["FromWhichNeuron"]+num_neurons]
    for i in xrange(0,len(raw_validation_set)):
	raw_validation_set[i] = raw_validation_set[i][:,params["FromWhichNeuron"]:params["FromWhichNeuron"]+num_neurons]
    	
    
    raw_validation_data_set=numpy.rollaxis(numpy.array(raw_validation_set),2)
    
    [K,rpi,glm,rfs]=  fitLSCSM(numpy.mat(training_inputs),numpy.mat(training_set),params["LGN_NUM"],num_neurons,numpy.mat(validation_inputs),numpy.mat(validation_set))
    
    rpi_pred_act = training_inputs * rpi
    rpi_pred_val_act = validation_inputs * rpi
    
    if __main__.__dict__.get('Sequential',False):
	glm_pred_act = numpy.hstack([glm[i].response(training_inputs,K[i]) for i in xrange(0,num_neurons)])
	glm_pred_val_act = numpy.hstack([glm[i].response(validation_inputs,K[i]) for i in xrange(0,num_neurons)])
    else:
    	glm_pred_act = glm.response(training_inputs,K)
    	glm_pred_val_act = glm.response(validation_inputs,K)

    runLSCSMAnalysis(rpi_pred_act,rpi_pred_val_act,glm_pred_act,glm_pred_val_act,training_set,validation_set,num_neurons,raw_validation_data_set)

    pylab.figure()
    print numpy.shape(rfs)
    print num_neurons
    print kernel_size
    m = numpy.max(numpy.abs(rfs))
    for i in xrange(0,numpy.shape(rfs)[0]):
	pylab.subplot(11,11,i+1)    
    	pylab.imshow(numpy.reshape(rfs[i,0:kernel_size],(size,size)),vmin=-m,vmax=m,cmap=pylab.cm.RdBu,interpolation='nearest')
    pylab.savefig(normalize_path('GLM_rfs.png'))
    
    pylab.figure()
    m = numpy.max(numpy.abs(rpi))
    for i in xrange(0,num_neurons):
	pylab.subplot(11,11,i+1)
    	pylab.imshow(numpy.reshape(rpi[:,i],(size,size)),vmin=-m,vmax=m,cmap=pylab.cm.RdBu,interpolation='nearest')
    pylab.savefig(normalize_path('RPI_rfs.png'))


    signal_power,noise_power,normalized_noise_power,training_prediction_power,validation_prediction_power,signal_power_variance = signal_power_test(raw_validation_data_set, numpy.array(training_set), numpy.array(validation_set), glm_pred_act, glm_pred_val_act)
    to_delete = numpy.array(numpy.nonzero((numpy.array(normalized_noise_power) > 85) * 1.0))[0]
    print 'After deleting ' , len(to_delete) , 'most noisy neurons (<15% signal to noise ratio)\n\n\n'
        
    if len(to_delete) != num_neurons:
    
	training_set = numpy.delete(training_set, to_delete, axis = 1)
	validation_set = numpy.delete(validation_set, to_delete, axis = 1)
	glm_pred_act = numpy.delete(glm_pred_act, to_delete, axis = 1)
	glm_pred_val_act = numpy.delete(glm_pred_val_act, to_delete, axis = 1)
	rpi_pred_act = numpy.delete(rpi_pred_act, to_delete, axis = 1)
	rpi_pred_val_act = numpy.delete(rpi_pred_val_act, to_delete, axis = 1)
	
	for i in xrange(0,len(raw_validation_set)):
		raw_validation_set[i] = numpy.delete(raw_validation_set[i], to_delete, axis = 1)
	
	raw_validation_data_set=numpy.rollaxis(numpy.array(raw_validation_set),2)	
	runLSCSMAnalysis(rpi_pred_act,rpi_pred_val_act,glm_pred_act,glm_pred_val_act,training_set,validation_set,num_neurons-len(to_delete),raw_validation_data_set)
	
    db_node.add_data("Kernels",K,force=True)
    db_node.add_data("LSCSM",glm,force=True)
	
    contrib.dd.saveResults(res,normalize_path(__main__.__dict__.get('save_name','BestLSCSM.dat')))
    

def loadLSCSMandAnalyse(): 
    
    res = contrib.dd.DB(None)
    (sizex,sizey,training_inputs,training_set,validation_inputs,validation_set,ff,db_node) = contrib.JanA.dataimport.sortOutLoading(res)
    raw_validation_set = db_node.data["raw_validation_set"]
    
    res = contrib.dd.loadResults("BestLSCSM.datt")
    
    dataset_node = res.children[0].children[0]
    
    training_set = dataset_node.data["training_set"]
    validation_set = dataset_node.data["validation_set"]
    training_inputs= dataset_node.data["training_inputs"]
    validation_inputs= dataset_node.data["validation_inputs"]

    if __main__.__dict__.get('LSCSMOLD',True):
	    lscsm_new = contrib.JanA.LSCSM.LSCSM(numpy.mat(training_inputs),numpy.mat(training_set),15,103)
    else:
            lscsm_new = contrib.JanA.LSCSMNEW.LSCSM(numpy.mat(training_inputs),numpy.mat(training_set),15,103)
    
    
    K = res.children[0].children[0].children[0].children[0].data["Kernels"]
    lscsm = res.children[0].children[0].children[0].children[0].data["LSCSM"]
    rfs  = lscsm_new.returnRFs(K)
    	
    pred_act = lscsm.response(training_inputs,K)
    pred_val_act = lscsm.response(validation_inputs,K)
   
    sizex=numpy.sqrt(numpy.shape(training_inputs)[1])
    from contrib.JanA.visualization import compareModelPerformanceWithRPI  
    
    compareModelPerformanceWithRPI(numpy.mat(training_set),numpy.mat(validation_set),numpy.mat(training_inputs),numpy.mat(validation_inputs),numpy.mat(pred_act),numpy.mat(pred_val_act),numpy.array(raw_validation_set),sizex,sizex,modelname='Model')
   	
	
    
def runLSCSMAnalysis(rpi_pred_act,rpi_pred_val_act,glm_pred_act,glm_pred_val_act,training_set,validation_set,num_neurons,raw_validation_data_set):
    
    ofs = run_nonlinearity_detection(numpy.mat(training_set),numpy.mat(rpi_pred_act),display=True)
    rpi_pred_act_t = apply_output_function(numpy.mat(rpi_pred_act),ofs)
    rpi_pred_val_act_t = apply_output_function(numpy.mat(rpi_pred_val_act),ofs)
    
    ofs = run_nonlinearity_detection(numpy.mat(training_set),numpy.mat(glm_pred_act),display=True)
    glm_pred_act_t = apply_output_function(numpy.mat(glm_pred_act),ofs)
    glm_pred_val_act_t = apply_output_function(numpy.mat(glm_pred_val_act),ofs)
    
    
    pylab.figure()
    
    for i in xrange(0,num_neurons):
	pylab.subplot(11,11,i+1)    
    	pylab.plot(rpi_pred_val_act[:,i],validation_set[:,i],'o')
    pylab.savefig(normalize_path('RPI_val_relationship.png'))
	
    pylab.figure()
    for i in xrange(0,num_neurons):
	pylab.subplot(11,11,i+1)    
 	pylab.plot(glm_pred_val_act[:,i],validation_set[:,i],'o')   
    pylab.savefig(normalize_path('GLM_val_relationship.png'))
    
    
    pylab.figure()
    for i in xrange(0,num_neurons):
	pylab.subplot(11,11,i+1)    
    	pylab.plot(rpi_pred_val_act_t[:,i],validation_set[:,i],'o')
    pylab.savefig(normalize_path('RPI_t_val_relationship.png'))
	
	
    pylab.figure()
    for i in xrange(0,num_neurons):
	pylab.subplot(11,11,i+1)    
 	pylab.plot(glm_pred_val_act_t[:,i],validation_set[:,i],'o')
        pylab.title('RPI')   
    pylab.savefig(normalize_path('GLM_t_val_relationship.png'))
    
    
    print numpy.shape(validation_set)
    print numpy.shape(rpi_pred_val_act_t)
    print numpy.shape(glm_pred_val_act)
    
    pylab.figure()
    pylab.plot(numpy.mean(numpy.power(validation_set - rpi_pred_val_act_t,2),0),numpy.mean(numpy.power(validation_set - glm_pred_val_act,2),0),'o')
    pylab.hold(True)
    pylab.plot([0.0,1.0],[0.0,1.0])
    pylab.xlabel('RPI')
    pylab.ylabel('GLM')
    pylab.savefig(normalize_path('GLM_vs_RPI_MSE.png'))
    
    
    print 'RPI \n'
	
    (ranks,correct,pred) = performIdentification(validation_set,rpi_pred_val_act)
    print "Natural:", correct , "Mean rank:", numpy.mean(ranks) , "MSE", numpy.mean(numpy.power(validation_set - rpi_pred_val_act,2))
	
    (ranks,correct,pred) = performIdentification(validation_set,rpi_pred_val_act_t)
    print "Natural+TF:", correct , "Mean rank:", numpy.mean(ranks) , "MSE", numpy.mean(numpy.power(validation_set - rpi_pred_val_act_t,2))
		
    signal_power,noise_power,normalized_noise_power,training_prediction_power,validation_prediction_power,signal_power_variance = signal_power_test(raw_validation_data_set, numpy.array(training_set), numpy.array(validation_set), numpy.array(rpi_pred_act), numpy.array(rpi_pred_val_act))
    signal_power,noise_power,normalized_noise_power,training_prediction_power_t,rpi_validation_prediction_power_t,signal_power_variance = signal_power_test(raw_validation_data_set, numpy.array(training_set), numpy.array(validation_set), numpy.array(rpi_pred_act_t), numpy.array(rpi_pred_val_act_t))
	
    print "Prediction power on training set / validation set: ", numpy.mean(training_prediction_power) , " / " , numpy.mean(validation_prediction_power)
    print "Prediction power after TF on training set / validation set: ", numpy.mean(training_prediction_power_t) , " / " , numpy.mean(rpi_validation_prediction_power_t)

	
    print '\n \n GLM \n'
	
    (ranks,correct,pred) = performIdentification(validation_set,glm_pred_val_act)
    print "Natural:", correct , "Mean rank:", numpy.mean(ranks) , "MSE", numpy.mean(numpy.power(validation_set - glm_pred_val_act,2))
	
    (ranks,correct,pred) = performIdentification(validation_set,glm_pred_val_act_t)
    print "Natural+TF:", correct , "Mean rank:", numpy.mean(ranks) , "MSE", numpy.mean(numpy.power(validation_set - glm_pred_val_act_t,2))
		
    glm_signal_power,glm_noise_power,glm_normalized_noise_power,glm_training_prediction_power,glm_validation_prediction_power,glm_signal_power_variance = signal_power_test(raw_validation_data_set, numpy.array(training_set), numpy.array(validation_set), numpy.array(glm_pred_act), numpy.array(glm_pred_val_act))
    glm_signal_power_t,glm_noise_power_t,glm_normalized_noise_power_t,glm_training_prediction_power_t,glm_validation_prediction_power_t,glm_signal_power_variances_t = signal_power_test(raw_validation_data_set, numpy.array(training_set), numpy.array(validation_set), numpy.array(glm_pred_act_t), numpy.array(glm_pred_val_act_t))
	
    print "Prediction power on training set / validation set: ", numpy.mean(glm_training_prediction_power) , " / " , numpy.mean(glm_validation_prediction_power)
    print "Prediction power after TF on training set / validation set: ", numpy.mean(glm_training_prediction_power_t) , " / " , numpy.mean(glm_validation_prediction_power_t)
    
    pylab.figure()
    pylab.plot(rpi_validation_prediction_power_t[:num_neurons],glm_validation_prediction_power[:num_neurons],'o')
    pylab.hold(True)
    pylab.plot([0.0,1.0],[0.0,1.0])
    pylab.xlabel('RPI_t')
    pylab.ylabel('GLM')
    pylab.savefig(normalize_path('GLM_vs_RPI_prediction_power.png'))

    
    pylab.figure()
    pylab.plot(rpi_validation_prediction_power_t[:num_neurons],glm_validation_prediction_power_t[:num_neurons],'o')
    pylab.hold(True)
    pylab.plot([0.0,1.0],[0.0,1.0])
    pylab.xlabel('RPI_t')
    pylab.ylabel('GLM_t')
    pylab.savefig('GLMt_vs_RPIt_prediction_power.png')
    
    print 'WithoutTF'
    printCorrelationAnalysis(training_set,validation_set,glm_pred_act,glm_pred_val_act)
    print 'WithTF'
    printCorrelationAnalysis(training_set,validation_set,glm_pred_act_t,glm_pred_val_act_t)
                

    #db_node.add_data("Kernels",K,force=True)
    #db_node.add_data("GLM",glm,force=True)
    #db_node.add_data("ReversCorrelationPredictedActivities",glm_pred_act,force=True)
    #db_node.add_data("ReversCorrelationPredictedActivities+TF",glm_pred_act_t,force=True)
    #db_node.add_data("ReversCorrelationPredictedValidationActivities",glm_pred_val_act,force=True)
    #db_node.add_data("ReversCorrelationPredictedValidationActivities+TF",glm_pred_val_act_t,force=True)
    #return [K,validation_inputs, validation_set]
	
	
