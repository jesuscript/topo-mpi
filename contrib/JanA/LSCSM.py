from scipy.optimize import fmin_ncg, anneal, fmin_cg, fmin_bfgs, fmin_tnc
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


class LSCSM(object):
	
	def __init__(self,XX,YY,num_lgn):
	    (self.num_pres,self.image_size) = numpy.shape(XX)
	    self.num_lgn = num_lgn
	    self.ss = numpy.sqrt(self.image_size)

	    self.xx = theano.shared(numpy.repeat([numpy.arange(0,self.ss,1)],self.ss,axis=0).T.flatten())	
	    self.yy = theano.shared(numpy.repeat([numpy.arange(0,self.ss,1)],self.ss,axis=0).flatten())
	    self.Y = theano.shared(YY)
    	    self.X = theano.shared(XX)
	    
	    self.K = T.dvector('K')
	    #self.KK = theano.printing.Print(message='My mesasge')(self.K)
	    self.x = self.K[0:self.num_lgn]
	    self.y = self.K[self.num_lgn:2*self.num_lgn]
	    self.a = self.K[2*self.num_lgn:3*self.num_lgn]
	    self.s_c = self.K[3*self.num_lgn:4*self.num_lgn]
	    self.s_s = self.K[4*self.num_lgn:5*self.num_lgn]
	    self.n = self.K[5*self.num_lgn]
	    
	    self.output = T.dot(self.X,T.mul(self.a[0],T.exp(-T.div_proxy(((self.xx - self.x[0])**2 + (self.yy - self.y[0])**2),self.s_c[0])).T) - T.mul(self.a[0],T.exp(-T.div_proxy(((self.xx - self.x[0])**2 + (self.yy - self.y[0])**2),self.s_s[0] )).T))
	    
	    for i in xrange(1,self.num_lgn):
		self.output = self.output + T.dot(self.X,T.mul(self.a[i],T.exp(-T.div_proxy(((self.xx - self.x[i])**2 + (self.yy - self.y[i])**2),self.s_c[i] )).T) - T.mul(self.a[i],T.exp(-T.div_proxy(((self.xx - self.x[i])**2 + (self.yy - self.y[i])**2),self.s_s[i] )).T))
	    
	    self.model = T.exp(self.output-self.n)
	    self.loglikelyhood = T.sum(self.model) - T.sum(T.dot(self.Y.T,  T.log(self.model))) 

	def func(self):
	    return theano.function(inputs=[self.K], outputs=self.loglikelyhood) 
			
	def der(self):
	    g_K = T.grad(self.loglikelyhood, self.K)
	    return theano.function(inputs=[self.K], outputs=g_K)
 
 	def hess(self):
            g_K = T.grad(self.loglikelyhood, self.K,consider_constant=[self.Y,self.X])
	    H, updates = theano.scan(lambda i,v: T.grad(g_K[i],v), sequences= T.arange(g_K.shape[0]), non_sequences=self.K)
  	    return theano.function(inputs=[self.K], outputs=H)
	
	def response(self,X,kernels):
	    self.IN = theano.shared(X)	
	    
	    self.output = T.dot(self.IN,T.mul(self.a[0],T.exp(-T.div_proxy(((self.xx - self.x[0])**2 + (self.yy - self.y[0])**2),self.s_c[0])).T) - 		T.mul(self.a[0],T.exp(-T.div_proxy(((self.xx - self.x[0])**2 + (self.yy - self.y[0])**2),self.s_s[0] )).T))
	    
	    for i in xrange(1,self.num_lgn):
		self.output = self.output + T.dot(self.IN,T.mul(self.a[i],T.exp(-T.div_proxy(((self.xx - self.x[i])**2 + (self.yy - self.y[i])**2),self.s_c[i] )).T) - self.X,T.mul(self.a[i],T.exp(-T.div_proxy(((self.xx - self.x[i])**2 + (self.yy - self.y[i])**2),self.s_s[i] )).T))
			        
	    self.model = T.exp(self.output-self.n)
	    	
	    resp =  theano.function(inputs=[self.K], outputs=self.model)
	    
	    (a,b) = numpy.shape(kernels)
	    (c,d) = numpy.shape(X)
	    
	    responses = numpy.zeros((c,a))
	    
	    for i in xrange(a):
		responses[:,i] = resp(kernels[i,:]).T
	    
	    return responses
	    
	
	def returnRFs(self,kernels):
	    (k,l) = numpy.shape(kernels)
            rfs = numpy.zeros((k,self.image_size))
	    
	    xx = numpy.repeat([numpy.arange(0,self.ss,1)],self.ss,axis=0).T.flatten()	
	    yy = numpy.repeat([numpy.arange(0,self.ss,1)],self.ss,axis=0).flatten()
	    			    
			    
	    for j in xrange(k):
   		x = kernels[j,0:self.num_lgn]
	        y = kernels[j,self.num_lgn:2*self.num_lgn]
	        a = kernels[j,2*self.num_lgn:3*self.num_lgn]
		sc = kernels[j,3*self.num_lgn:4*self.num_lgn]
		ss = kernels[j,4*self.num_lgn:5*self.num_lgn]
		print x
		print y
		print a
	    	for i in xrange(0,self.num_lgn):
		    rfs[j,:] += a[i]*(numpy.exp(-((xx - x[i])**2 + (yy - y[i])**2)/sc[i]) - numpy.exp(-((xx - x[i])**2 + (yy - y[i])**2)/ss[i])) 
	    
	    return rfs
	
	
class LSCSM1(object):
	def __init__(self,XX,YY,num_lgn,num_neurons):
	    (self.num_pres,self.image_size) = numpy.shape(XX)
	    self.num_lgn = num_lgn
	    self.num_neurons = num_neurons
	    self.ss = numpy.sqrt(self.image_size)

	    self.xx = theano.shared(numpy.repeat([numpy.arange(0,self.ss,1)],self.ss,axis=0).T.flatten())	
	    self.yy = theano.shared(numpy.repeat([numpy.arange(0,self.ss,1)],self.ss,axis=0).flatten())
	    self.Y = theano.shared(YY)
    	    self.X = theano.shared(XX)
	    
	    self.K = T.dvector('K')
	    #self.KK = theano.printing.Print(message='My mesasge')(self.K)
	    self.x = self.K[0:self.num_lgn]
	    self.y = self.K[self.num_lgn:2*self.num_lgn]
	    self.s = self.K[2*self.num_lgn:3*self.num_lgn]
	    self.a = T.reshape(self.K[3*self.num_lgn:3*self.num_lgn+num_neurons*self.num_lgn],(self.num_lgn,self.num_neurons))
	    self.n = self.K[3*self.num_lgn+num_neurons*self.num_lgn:3*self.num_lgn+num_neurons*self.num_lgn+self.num_neurons]
	    
	    lgn_output,updates = theano.scan(lambda i,x,y,s: T.dot(self.X,T.exp(-T.div_proxy(((self.xx - x[i])**2 + (self.yy - y[i])**2),s[i] )).T), sequences= T.arange(self.num_lgn), non_sequences=[self.x,self.y,self.s])
	    
	    self.output = T.dot(lgn_output.T,self.a)
	    self.model = T.exp(self.output-self.n)
	    
	    self.loglikelyhood = T.sum(T.sum(self.model)) - T.sum(T.sum(self.Y * T.log(self.model))) 
	
	def func(self):
	    return theano.function(inputs=[self.K], outputs=self.loglikelyhood) 
			
	def der(self):
	    g_K = T.grad(self.loglikelyhood, self.K)
	    return theano.function(inputs=[self.K], outputs=g_K)
	
	def response(self,X,kernels):
	    self.X.value = X
	    
	    resp = theano.function(inputs=[self.K], outputs=self.model)
	    return resp(kernels)	
	
	def returnRFs(self,K):
	    x = K[0:self.num_lgn]
	    y = K[self.num_lgn:2*self.num_lgn]
	    s = K[2*self.num_lgn:3*self.num_lgn]
	    a = numpy.reshape(K[3*self.num_lgn:3*self.num_lgn+self.num_neurons*self.num_lgn],(self.num_lgn,self.num_neurons))
	    n = K[3*self.num_lgn+self.num_neurons*self.num_lgn:3*self.num_lgn+self.num_neurons*self.num_lgn+self.num_neurons]

            rfs = numpy.zeros((self.num_neurons,self.image_size))
	    
	    xx = numpy.repeat([numpy.arange(0,self.ss,1)],self.ss,axis=0).T.flatten()	
	    yy = numpy.repeat([numpy.arange(0,self.ss,1)],self.ss,axis=0).flatten()
	    			    
            print x
	    print y
	    print s
	    print a
	    print n			    
	    for j in xrange(self.num_neurons):
	    	for i in xrange(0,self.num_lgn):
		    rfs[j,:] += a[i,j]*numpy.exp(-((xx - x[i])**2 + (yy - y[i])**2)/s[i])
	    
	    return rfs
	
	
def fitLSCSM(X,Y,num_lgn,num_neurons_to_estimate):
    num_pres,num_neurons = numpy.shape(Y)
    num_pres,kernel_size = numpy.shape(X)
    
    Ks = numpy.zeros((num_neurons,num_lgn*5+1))
    laplace = 0.0001*laplaceBias(numpy.sqrt(kernel_size),numpy.sqrt(kernel_size))
    
    rpi = numpy.linalg.pinv(X.T*X + 10*laplace) * X.T * Y
    
    
    for i in xrange(0,num_neurons_to_estimate): 
	print i
	k0 = numpy.zeros((num_lgn*5+1,1)).flatten()
	z = numpy.reshape(k0[:-1],(5,num_lgn))
	l = numpy.ones(numpy.shape(z))
	u = numpy.ones(numpy.shape(z))
	z[0,:] = 2+numpy.random.rand(num_lgn)*(numpy.sqrt(kernel_size)-4)
	z[1,:] = 2+numpy.random.rand(num_lgn)*(numpy.sqrt(kernel_size)-4)
	z[2,:] = (numpy.random.rand(num_lgn)-0.5)
	z[3,:] = 2+numpy.random.rand(num_lgn)*5
	z[4,:] = 2+numpy.random.rand(num_lgn)*5
	l[0,:] = l[0,:]*2
	l[1,:] = l[1,:]*2
	l[2,:] = l[2,:]*-10
	l[3,:] = l[3,:]*0
	l[4,:] = l[4,:]*0
	u[0,:] = u[0,:]*numpy.sqrt(kernel_size)
	u[1,:] = u[1,:]*numpy.sqrt(kernel_size)
	u[2,:] = u[2,:]*8
	u[3,:] = u[3,:]*8
	u[4,:] = u[4,:]*8
	l = l.flatten().tolist() + [-10]
	u = u.flatten().tolist() + [10]
	
	print l
	print u
	k0[:-1] = z.flatten()
	k0=k0.tolist()
	lscsm = LSCSM(numpy.mat(X),numpy.mat(Y[:,i]),num_lgn)
	rf = lscsm.returnRFs(numpy.array([k0]))
	pylab.figure()
	m = numpy.max(numpy.abs(rf[0,0:kernel_size]))
	pylab.imshow(numpy.reshape(rf[0],(numpy.sqrt(kernel_size),numpy.sqrt(kernel_size))),vmin=-m,vmax=m,cmap=pylab.cm.RdBu,interpolation='nearest')
	pylab.colorbar()
	#K = fmin_ncg(lscsm.func(),numpy.array(k0),lscsm.der(),fhess = lscsm.hess(),avextol=0.00001,maxiter=100)
	#K = fmin_cg(lscsm.func(),numpy.array(k0),lscsm.der(),avextol=0.00001,maxiter=100)
	#K = anneal(lscsm.func(), numpy.array(k0), schedule='fast', lower=numpy.array(l),upper=numpy.array(u),full_output=0, maxiter=10000, boltzmann=1.0, learn_rate=0.5, feps=9.9999999999999995e-07)[0]
	#K = fmin_ncg(lscsm.func(),numpy.array(K),lscsm.der(),fhess = lscsm.hess(),avextol=0.00001,maxiter=100)
	from pybrain.optimization import CMAES

	l = CMAES(lscsm.func(),numpy.array(k0),maxEvaluations=10000,minimize=True)
	K = l.learn()
	
	print K
	Ks[i,:] = K[0]
	
    return [Ks,rpi,lscsm]

class GGEvo(object):
      def __init__(self,XX,YY,num_lgn,num_neurons,bounds):
		self.XX = XX
		self.YY = YY
		self.num_lgn = num_lgn
		self.num_neurons = num_neurons
		self.lscsm = LSCSM1(numpy.mat(XX),numpy.mat(YY),num_lgn,num_neurons)
		self.func = self.lscsm.func() 
		self.der = self.lscsm.der()
		self.num_eval = __main__.__dict__.get('NumEval',10)
		self.bounds = bounds
		
      def perform_gradient_descent(self,chromosome):
	  inp = numpy.array([v for v in chromosome])
	  #inp[0:3*self.num_lgn] = numpy.reshape(inp[0:3*self.num_lgn],(self.num_lgn,3)).T.flatten()
	  #new_K = inp 
	  #for i in xrange(0,self.num_eval):
	  #    new_K = new_K - 0.001*self.der(new_K)
	   
	  if self.num_eval != 0:
	  	(new_K,success,c)=fmin_tnc(self.func,inp[:],fprime=self.der,bounds=self.bounds,maxfun = self.num_eval,messages=0)
	  	for i in xrange(0,len(chromosome)):
	  		chromosome[i] = new_K[i]
		score = self.func(numpy.array(new_K))			
	  else:
	  	score = self.func(numpy.array(inp))
	  
	  #K = fmin_bfgs(self.func,numpy.array(inp),fprime=self.der,maxiter=2,full_output=0)
	  #score = self.func(K)
	  #(K,score,t1,t2,t3,t4,t5,t6) = fmin_ncg(self.func,numpy.array(inp),self.der,fhess = self.hess,avextol=0.00001,maxiter=2,full_output=1)
	  #print z
	  #(K,score,t1,t2,t3,t4,t5,t6) = fmin_ncg(self.func,numpy.array(inp),self.der,fhess = self.hess,avextol=0.00001,maxiter=2,full_output=1)
	  return score
	  
	

def fitLSCSMEvo(X,Y,num_lgn,num_neurons_to_estimate):
    from pyevolve import *
    num_pres,num_neurons = numpy.shape(Y)
    num_pres,kernel_size = numpy.shape(X)
    
    Ks = numpy.ones((num_neurons,num_lgn*4+1))
    
    laplace = laplaceBias(numpy.sqrt(kernel_size),numpy.sqrt(kernel_size))
    
    rpi = numpy.linalg.pinv(X.T*X + __main__.__dict__.get('RPILaplaceBias',0.0001)*laplace) * X.T * Y
	
    setOfAlleles = GAllele.GAlleles()
    bounds = []
    	
    for j in xrange(0,num_lgn):
	setOfAlleles.add(GAllele.GAlleleRange(6,(numpy.sqrt(kernel_size)-6),real=True))
	bounds.append((6,(numpy.sqrt(kernel_size)-6)))
	setOfAlleles.add(GAllele.GAlleleRange(6,(numpy.sqrt(kernel_size)-6),real=True))
	bounds.append((6,(numpy.sqrt(kernel_size)-6)))
	
    for j in xrange(0,num_lgn):	
	setOfAlleles.add(GAllele.GAlleleRange(3.0,10,real=True))
	bounds.append((3,10))
    
    for j in xrange(0,num_lgn):		
	for k in xrange(0,num_neurons_to_estimate):
	 	setOfAlleles.add(GAllele.GAlleleRange(-2000,2000,real=True))
		bounds.append((-2000,2000))
		
    for k in xrange(0,num_neurons_to_estimate):
    	setOfAlleles.add(GAllele.GAlleleRange(-100,100,real=True))
	bounds.append((-100,100))
    
    ggevo = GGEvo(X,Y,num_lgn,num_neurons_to_estimate,bounds)	
    
    
    genome = G1DList.G1DList(num_lgn*3+num_neurons_to_estimate*num_lgn+num_neurons_to_estimate)
    genome.setParams(allele=setOfAlleles)
    genome.evaluator.set(ggevo.perform_gradient_descent)
    genome.mutator.set(Mutators.G1DListMutatorAllele)
    genome.initializator.set(Initializators.G1DListInitializatorAllele)
    genome.crossover.set(Crossovers.G1DListCrossoverUniform) 

    ga = GSimpleGA.GSimpleGA(genome,1023)
    ga.minimax = Consts.minimaxType["minimize"]
    #ga.selector.set(Selectors.GRouletteWheel)
    ga.setElitism(True) 
    ga.setGenerations(__main__.__dict__.get('GenerationSize',100))
    ga.setPopulationSize(__main__.__dict__.get('PopulationSize',100))
    ga.setMutationRate(__main__.__dict__.get('MutationRate',0.05))
    ga.setCrossoverRate(__main__.__dict__.get('CrossoverRate',0.9))
     
    #pop = ga.getPopulation()
    #pop.scaleMethod.set(Scaling.SigmaTruncScaling)

    ga.evolve(freq_stats=1)
    best = ga.bestIndividual()
    
    
    
    #print best
    inp = [v for v in best]
    (new_K,success,c)=fmin_tnc(ggevo.func,inp[:],fprime=ggevo.der,bounds=bounds,maxfun = 1000,messages=0)
    #inp[:-1] = numpy.reshape(inp[:-1],(num_lgn,4)).T.flatten()
    Ks = new_K
    #rf= ggevo.lscsm.returnRFs(numpy.array([Ks[i,:]]))

    #pylab.figure()
    #m = numpy.max(numpy.abs(rf[0,0:kernel_size]))
    #pylab.imshow(numpy.reshape(rf[0],(numpy.sqrt(kernel_size),numpy.sqrt(kernel_size))),vmin=-m,vmax=m,cmap=pylab.cm.RdBu,interpolation='nearest')
    #pylab.colorbar()
    #pylab.show()	
    return [Ks,rpi,ggevo.lscsm]	
    
def runLSCSM():
    import noiseEstimation
    
    (sizex,sizey,training_inputs,training_set,validation_inputs,validation_set,ff,db_node) = contrib.JanA.dataimport.sortOutLoading(contrib.dd.DB(None))
    raw_validation_set = db_node.data["raw_validation_set"]
    
    num_pres,num_neurons = numpy.shape(training_set)
    num_pres,kernel_size = numpy.shape(training_inputs)
    size = numpy.sqrt(kernel_size)

    raw_validation_data_set=numpy.rollaxis(numpy.array(raw_validation_set),2)
    
    normalized_noise_power = [noiseEstimation.signal_and_noise_power(raw_validation_data_set[i])[2] for i in xrange(0,num_neurons)]
    
    #to_delete = numpy.array(numpy.nonzero((numpy.array(normalized_noise_power) > 85) * 1.0))[0]
    to_delete = [2,3,4,5,6,9,10,11,12,13,14,18,22,26,28,29,30,31,32,34,35,36,37,41,44,50,51,54,55,57,59,60,63,65,67,68,70,71,73,74,76,79,81,82,84,86,87,88,90,91,94,95,97,98,99,100,102]
    
    training_set = numpy.delete(training_set, to_delete, axis = 1)
    validation_set = numpy.delete(validation_set, to_delete, axis = 1)
    for i in xrange(0,10):
        raw_validation_set[i] = numpy.delete(raw_validation_set[i], to_delete, axis = 1)
    
    # creat history
    history_set = training_set[0:-1,:]
    history_validation_set = validation_set[0:-1,:]
    training_set = training_set[1:,:]
    validation_set = validation_set[1:,:]
    training_inputs= training_inputs[1:,:]
    validation_inputs= validation_inputs[1:,:]
    
    for i in xrange(0,len(raw_validation_set)):
	raw_validation_set[i] = raw_validation_set[i][1:,:]
    
    print numpy.shape(training_inputs[0])
    
    params={}
    params["LGN_NUM"] = __main__.__dict__.get('LgnNum',6)
    db_node1 = db_node
    db_node = db_node.get_child(params)
    
    
    
    num_neurons=__main__.__dict__.get('NumNeurons',103)

    sx,sy = numpy.shape(training_set)	
    
    training_set = training_set[:,:num_neurons]
    validation_set = validation_set[:,:num_neurons]
    for i in xrange(0,len(raw_validation_set)):
	raw_validation_set[i] = raw_validation_set[i][:,:num_neurons]
    
    raw_validation_data_set=numpy.rollaxis(numpy.array(raw_validation_set),2)
    
    if __main__.__dict__.get('Evo',True):
    	[K,rpi,glm]=  fitLSCSMEvo(numpy.mat(training_inputs),numpy.mat(training_set),params["LGN_NUM"],num_neurons)
    	rfs = glm.returnRFs(K)
    else:
    	[K,rpi,glm]=  fitLSCSM(numpy.mat(training_inputs),numpy.mat(training_set),params["LGN_NUM"],num_neurons)
        rfs = glm.returnRFs(K[:num_neurons,:])
 

    pylab.figure()
    m = numpy.max(numpy.abs(rfs))
    for i in xrange(0,num_neurons):
	pylab.subplot(11,11,i+1)    
    	pylab.imshow(numpy.reshape(rfs[i,0:kernel_size],(size,size)),vmin=-m,vmax=m,cmap=pylab.cm.RdBu,interpolation='nearest')
    pylab.savefig('GLM_rfs.png')	
    
    pylab.figure()
    m = numpy.max(numpy.abs(rpi))
    for i in xrange(0,num_neurons):
	pylab.subplot(11,11,i+1)
    	pylab.imshow(numpy.reshape(rpi[:,i],(size,size)),vmin=-m,vmax=m,cmap=pylab.cm.RdBu,interpolation='nearest')
    pylab.savefig('RPI_rfs.png')
    
    rpi_pred_act = training_inputs * rpi
    rpi_pred_val_act = validation_inputs * rpi
    
    glm_pred_act = glm.response(training_inputs,K)
    glm_pred_val_act = glm.response(validation_inputs,K)
    
    ofs = run_nonlinearity_detection(numpy.mat(training_set),numpy.mat(rpi_pred_act))
    rpi_pred_act_t = apply_output_function(numpy.mat(rpi_pred_act),ofs)
    rpi_pred_val_act_t = apply_output_function(numpy.mat(rpi_pred_val_act),ofs)
    
    ofs = run_nonlinearity_detection(numpy.mat(training_set),numpy.mat(glm_pred_act))
    glm_pred_act_t = apply_output_function(numpy.mat(glm_pred_act),ofs)
    glm_pred_val_act_t = apply_output_function(numpy.mat(glm_pred_val_act),ofs)
    
    
    pylab.figure()
    pylab.title('RPI')
    for i in xrange(0,num_neurons):
	pylab.subplot(11,11,i+1)    
    	pylab.plot(rpi_pred_val_act[:,i],validation_set[:,i],'o')
    pylab.savefig('RPI_val_relationship.png')	
	
    pylab.figure()
    pylab.title('GLM')
    for i in xrange(0,num_neurons):
	pylab.subplot(11,11,i+1)    
 	pylab.plot(glm_pred_val_act[:,i],validation_set[:,i],'o')   
    pylab.savefig('GLM_val_relationship.png')
    
    
    pylab.figure()
    pylab.title('RPI')
    for i in xrange(0,num_neurons):
	pylab.subplot(11,11,i+1)    
    	pylab.plot(rpi_pred_val_act_t[:,i],validation_set[:,i],'o')
    pylab.savefig('RPI_t_val_relationship.png')	
	
	
    pylab.figure()
    pylab.title('GLM')
    for i in xrange(0,num_neurons):
	pylab.subplot(11,11,i+1)    
 	pylab.plot(glm_pred_val_act_t[:,i],validation_set[:,i],'o')   
    pylab.savefig('GLM_t_val_relationship.png')
    
    
    print numpy.shape(validation_set)
    print numpy.shape(rpi_pred_val_act_t)
    print numpy.shape(glm_pred_val_act)
    
    pylab.figure()
    pylab.plot(numpy.mean(numpy.power(validation_set - rpi_pred_val_act_t,2),0),numpy.mean(numpy.power(validation_set - glm_pred_val_act,2),0),'o')
    pylab.hold(True)
    pylab.plot([0.0,1.0],[0.0,1.0])
    pylab.xlabel('RPI')
    pylab.ylabel('GLM')
    pylab.savefig('GLM_vs_RPI_MSE.png')
    
    
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
		
    signal_power,noise_power,normalized_noise_power,training_prediction_power,glm_validation_prediction_power,signal_power_variance = signal_power_test(raw_validation_data_set, numpy.array(training_set), numpy.array(validation_set), numpy.array(glm_pred_act), numpy.array(glm_pred_val_act))
    signal_power,noise_power,normalized_noise_power,training_prediction_power_t,validation_prediction_power_t,signal_power_variances = signal_power_test(raw_validation_data_set, numpy.array(training_set), numpy.array(validation_set), numpy.array(glm_pred_act_t), numpy.array(glm_pred_val_act_t))
	
    print "Prediction power on training set / validation set: ", numpy.mean(training_prediction_power) , " / " , numpy.mean(glm_validation_prediction_power)
    print "Prediction power after TF on training set / validation set: ", numpy.mean(training_prediction_power_t) , " / " , numpy.mean(validation_prediction_power_t)
    
    pylab.figure()
    pylab.plot(rpi_validation_prediction_power_t[:num_neurons],glm_validation_prediction_power[:num_neurons],'o')
    pylab.hold(True)
    pylab.plot([0.0,1.0],[0.0,1.0])
    pylab.xlabel('RPI')
    pylab.ylabel('GLM')
    pylab.savefig('GLM_vs_RPI_prediction_power.png')

    
    db_node.add_data("Kernels",K,force=True)
    db_node.add_data("GLM",glm,force=True)
    db_node.add_data("ReversCorrelationPredictedActivities",glm_pred_act,force=True)
    db_node.add_data("ReversCorrelationPredictedActivities+TF",glm_pred_act_t,force=True)
    db_node.add_data("ReversCorrelationPredictedValidationActivities",glm_pred_val_act,force=True)
    db_node.add_data("ReversCorrelationPredictedValidationActivities+TF",glm_pred_val_act_t,force=True)
    #return [K,validation_inputs, validation_set]
	
	
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
	
