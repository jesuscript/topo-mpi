from scipy.optimize import fmin_ncg, anneal, fmin_cg, fmin_bfgs, fmin_tnc, fmin_l_bfgs_b
import __main__
import numpy
import pylab
import sys
sys.path.append('/home/antolikjan/topographica/Theano/')
import theano
theano.config.floatX='float32' 
#theano.config.warn.sum_sum_bug=False
from theano import tensor as T
from topo.misc.filepath import normalize_path, application_path
from contrib.JanA.ofestimation import *
from contrib.modelfit import *
import contrib.dd
import contrib.JanA.dataimport
from contrib.JanA.regression import laplaceBias
from pyevolve import *
from contrib.JanA.visualization import printCorrelationAnalysis


class LSCTM(object):
	def __init__(self,XX1,XX2,XX3,YY,num_lgn,num_neurons,paramSpace):
	    (self.num_pres,self.image_size) = numpy.shape(XX1)
	    self.num_lgn = num_lgn
	    self.num_neurons = num_neurons
	    self.size = numpy.sqrt(self.image_size)
	    self.paramSpace = paramSpace

	    self.xx = theano.shared(numpy.repeat([numpy.arange(0,self.size,1)],self.size,axis=0).T.flatten())	
	    self.yy = theano.shared(numpy.repeat([numpy.arange(0,self.size,1)],self.size,axis=0).flatten())
	    self.Y = theano.shared(YY)
    	    self.X1 = theano.shared(XX1)
	    self.X2 = theano.shared(XX2)
	    self.X3 = theano.shared(XX3)
	    
	    self.hls = __main__.__dict__.get('HiddenLayerSize',1.0)
	    
	    self.v1of = __main__.__dict__.get('V1OF','Exp')
	    self.lgnof = __main__.__dict__.get('LGNOF','Exp')
	    
	    self.K = T.dvector('K')
	    
	    self.x1,self.y1,self.sc1,self.ss1,self.rc1,self.rs1,self.ln1,self.x2,self.y2,self.sc2,self.ss2,self.rc2,self.rs2,self.ln2,self.x3,self.y3,self.sc3,self.ss3,self.rc3,self.rs3,self.ln3,self.wl1t1,self.wl1t2,self.wl1t3,self.wl2,self.tl1,self.tl2 = chopVector(self.K,self.paramSpace)
	    
	    self.wl1t1 =  T.reshape(self.wl1t1 ,(self.num_lgn,int(self.num_neurons*self.hls)))
	    self.wl1t2 =  T.reshape(self.wl1t2 ,(self.num_lgn,int(self.num_neurons*self.hls)))
	    self.wl1t3 =  T.reshape(self.wl1t3 ,(self.num_lgn,int(self.num_neurons*self.hls)))
	    self.wl2 =  T.reshape(self.wl2 ,(int(self.num_neurons*self.hls),self.num_neurons))
	    
     	    lgn_kernel1 = lambda i,x,y,sc,ss,rc,rs: T.dot(self.X1,rc[i]*(T.exp(-((self.xx - x[i])**2 + (self.yy - y[i])**2)/sc[i]).T/ T.sqrt(sc[i]*numpy.pi)) - rs[i]*(T.exp(-((self.xx - x[i])**2 + (self.yy - y[i])**2)/ss[i]).T/ T.sqrt(ss[i]*numpy.pi)))
	    lgn_output1,updates = theano.scan(lgn_kernel1,sequences=T.arange(self.num_lgn),non_sequences=[self.x1,self.y1,self.sc1,self.ss1,self.rc1,self.rs1])
	    
	    lgn_kernel2 = lambda i,x,y,sc,ss,rc,rs: T.dot(self.X2,rc[i]*(T.exp(-((self.xx - x[i])**2 + (self.yy - y[i])**2)/sc[i]).T/ T.sqrt(sc[i]*numpy.pi)) - rs[i]*(T.exp(-((self.xx - x[i])**2 + (self.yy - y[i])**2)/ss[i]).T/ T.sqrt(ss[i]*numpy.pi)))
	    lgn_output2,updates = theano.scan(lgn_kernel2,sequences=T.arange(self.num_lgn),non_sequences=[self.x2,self.y2,self.sc2,self.ss2,self.rc2,self.rs2])
	    
	    lgn_kernel3 = lambda i,x,y,sc,ss,rc,rs: T.dot(self.X3,rc[i]*(T.exp(-((self.xx - x[i])**2 + (self.yy - y[i])**2)/sc[i]).T/ T.sqrt(sc[i]*numpy.pi)) - rs[i]*(T.exp(-((self.xx - x[i])**2 + (self.yy - y[i])**2)/ss[i]).T/ T.sqrt(ss[i]*numpy.pi)))
	    lgn_output3,updates = theano.scan(lgn_kernel3,sequences=T.arange(self.num_lgn),non_sequences=[self.x3,self.y3,self.sc3,self.ss3,self.rc3,self.rs3])
	    
	    
	    if __main__.__dict__.get('LGNTreshold',False):
	       lgn_output1 = lgn_output1 - self.ln1
	       lgn_output2 = lgn_output2 - self.ln2
	       lgn_output3 = lgn_output3 - self.ln3
            
	    lgn_output1 = self.construct_of(lgn_output1,self.lgnof).T
	    lgn_output2 = self.construct_of(lgn_output2,self.lgnof).T
	    lgn_output3 = self.construct_of(lgn_output3,self.lgnof).T
	       
	    self.output = T.dot(lgn_output1,self.wl1t1) + T.dot(lgn_output2,self.wl1t2) + T.dot(lgn_output2,self.wl1t2)
            self.model_output = self.construct_of(self.output-self.tl1,self.v1of)
	    self.model_output = self.construct_of(T.dot(self.model_output , self.wl2) - self.tl2,self.v1of)
	    
   	    if __main__.__dict__.get('LL',True):
	       ll = T.sum(self.model_output) - T.sum(self.Y * T.log(self.model_output+0.0000000000000000001))

	    self.loglikelyhood =  ll
	
	def func(self):
	    return theano.function(inputs=[self.K], outputs=self.loglikelyhood,mode='FAST_RUN')
	
	def der(self):
	    g_K = T.grad(self.loglikelyhood, self.K)
	    return theano.function(inputs=[self.K], outputs=g_K,mode='FAST_RUN')
	
	def response(self,X1,X2,X3,kernels):
	    self.X1.value = X1
	    self.X2.value = X2
	    self.X3.value = X3
	    
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

	    xx = numpy.repeat([numpy.arange(0,self.size,1)],self.size,axis=0).T.flatten()	
	    yy = numpy.repeat([numpy.arange(0,self.size,1)],self.size,axis=0).flatten()
	    			    
	    num_neurons_first_layer = int(self.num_neurons*self.hls)  
	    
	    x1,y1,sc1,ss1,rc1,rs1,ln1,x2,y2,sc2,ss2,rc2,rs2,ln2,x3,y3,sc3,ss3,rc3,rs3,ln3,wl1t1,wl1t2,wl1t3,wl2,tl1,tl2 = chopVector(K,self.paramSpace)
	    
    	    wl1t1 =  numpy.reshape(wl1t1 ,(self.num_lgn,int(self.num_neurons*self.hls)))
	    wl1t2 =  numpy.reshape(wl1t2 ,(self.num_lgn,int(self.num_neurons*self.hls)))
	    wl1t3 =  numpy.reshape(wl1t3 ,(self.num_lgn,int(self.num_neurons*self.hls)))
	    wl2 =  numpy.reshape(wl2 ,(int(self.num_neurons*self.hls),self.num_neurons))

	    
	    rfs1 = numpy.zeros((num_neurons_first_layer,self.image_size))		
	    rfs2 = numpy.zeros((num_neurons_first_layer,self.image_size))
	    rfs3 = numpy.zeros((num_neurons_first_layer,self.image_size))
	    
	    for j in xrange(num_neurons_first_layer):
	    	for i in xrange(0,self.num_lgn):
			rfs1[j,:] += wl1t1[i,j]*(rc1[i]*numpy.exp(-((xx - x1[i])**2 + (yy - y1[i])**2)/sc1[i])/numpy.sqrt((sc1[i]*numpy.pi)) - rs1[i]*numpy.exp(-((xx - x1[i])**2 + (yy - y1[i])**2)/ss1[i])/numpy.sqrt((ss1[i]*numpy.pi)))
			
	    for j in xrange(num_neurons_first_layer):
	    	for i in xrange(0,self.num_lgn):
			rfs2[j,:] += wl1t2[i,j]*(rc2[i]*numpy.exp(-((xx - x2[i])**2 + (yy - y2[i])**2)/sc2[i])/numpy.sqrt((sc2[i]*numpy.pi)) - rs2[i]*numpy.exp(-((xx - x2[i])**2 + (yy - y2[i])**2)/ss2[i])/numpy.sqrt((ss2[i]*numpy.pi)))
			
	    for j in xrange(num_neurons_first_layer):
	    	for i in xrange(0,self.num_lgn):
			rfs3[j,:] += wl1t3[i,j]*(rc3[i]*numpy.exp(-((xx - x3[i])**2 + (yy - y3[i])**2)/sc3[i])/numpy.sqrt((sc3[i]*numpy.pi)) - rs3[i]*numpy.exp(-((xx - x3[i])**2 + (yy - y3[i])**2)/ss3[i])/numpy.sqrt((ss3[i]*numpy.pi)))
			
	    return (rfs1,rfs2,rfs3)



      	
	
class GGEvo(object):
      def __init__(self,XX1,XX2,XX3,YY,num_lgn,num_neurons,bounds,paramSpace):
		self.XX1 = XX1
		self.XX2 = XX2
		self.XX3 = XX3
		self.YY = YY
		self.num_lgn = num_lgn
		self.num_neurons = num_neurons
		self.lscsm = LSCTM(numpy.mat(XX1),numpy.mat(XX2),numpy.mat(XX3),numpy.mat(YY),num_lgn,num_neurons,paramSpace)
		self.func = self.lscsm.func() 
		self.der = self.lscsm.der()
		self.num_eval = __main__.__dict__.get('NumEval',10)
		self.bounds = bounds
		
      def perform_gradient_descent(self,chromosome):
	  inp = numpy.array([v for v in chromosome])
	  
	  if self.num_eval != 0:
	  	(new_K,success,c)=fmin_tnc(self.func,inp[:],fprime=self.der,bounds=self.bounds,maxfun = self.num_eval,messages=0)
	  	for i in xrange(0,len(chromosome)):
	  		chromosome[i] = new_K[i]
		score = self.func(numpy.array(new_K))			
	  else:
		score=1
	  
	  return score

def returnAllelsAndBounds(paramSpace):
    setOfAlleles = GAllele.GAlleles()	
    bounds = []
    for (a,b,c) in paramSpace:
	for j in xrange(0,a):
            setOfAlleles.add(GAllele.GAlleleRange(b,c,real=True))		
	    bounds.append((b,c))
    return (bounds,setOfAlleles)

def chopVector(V,paramSpace):
    chops = []
    idx=0
    for (a,b,c) in paramSpace:
	chops.append(V[idx:idx+a])
	idx=idx+a
    return chops

def fitLSCTMEvo(X1,X2,X3,Y,num_lgn,num_neurons_to_estimate):
    num_pres,num_neurons = numpy.shape(Y)
    num_pres,kernel_size = numpy.shape(X1)
    
    if __main__.__dict__.get('NegativeLgn',True):
    	minw = -__main__.__dict__.get('MaxWL1',5000)
    else:
    	minw = 0
    maxw = __main__.__dict__.get('MaxWL1',5000)
    
    layer1size=int(num_neurons_to_estimate*__main__.__dict__.get('HiddenLayerSize',1.0))
    layer2size=num_neurons_to_estimate
 	
    paramSpace = [#T1
    		  (num_lgn,0,(numpy.sqrt(kernel_size)-1)), #x center coordinate
    		  (num_lgn,0,(numpy.sqrt(kernel_size)-1)), #y center coordinate
		  (num_lgn,1,25), #center size
    		  (num_lgn,1,25), #surr size
		  (num_lgn,0,1.0), #center strength
    		  (num_lgn,0,1.0), #surr strength
		  (num_lgn,0,20.0), #lgn threshold
		  #T2
		  (num_lgn,0,(numpy.sqrt(kernel_size)-1)), #x center coordinate
    		  (num_lgn,0,(numpy.sqrt(kernel_size)-1)), #y center coordinate
		  (num_lgn,1,25), #center size
    		  (num_lgn,1,25), #surr size
		  (num_lgn,0,1.0), #center strength
    		  (num_lgn,0,1.0), #surr strength
		  (num_lgn,0,20.0), #lgn threshold
		  #T3
		  (num_lgn,0,(numpy.sqrt(kernel_size)-1)), #x center coordinate
    		  (num_lgn,0,(numpy.sqrt(kernel_size)-1)), #y center coordinate
		  (num_lgn,1,25), #center size
    		  (num_lgn,1,25), #surr size
		  (num_lgn,0,1.0), #center strength
    		  (num_lgn,0,1.0), #surr strength
		  (num_lgn,0,20.0), #lgn threshold
		  (layer1size*num_lgn,minw,maxw), # 1st layer weights
		  (layer1size*num_lgn,minw,maxw), # 1st layer weights
		  (layer1size*num_lgn,minw,maxw), # 1st layer weights
		  (layer1size*layer2size,-__main__.__dict__.get('MaxWL2',5000),__main__.__dict__.get('MaxWL2',5000)), # 2nd layer weights
		  (layer1size,0,20.0), #L1 threshold
		  (layer2size,0,20.0) #L2 threshold
                 ]

    (bounds,setOfAlleles) = returnAllelsAndBounds(paramSpace)
    
			
    ggevo = GGEvo(X1,X2,X3,Y,num_lgn,num_neurons_to_estimate,bounds,paramSpace)
    genome_size = 3*7*num_lgn + 3*layer1size*num_lgn + layer2size*layer1size + layer1size + layer2size

    print 'Genome size and bounds size'
    print genome_size
    print len(bounds)
    
    genome = G1DList.G1DList(genome_size)
	    
    genome.setParams(allele=setOfAlleles)
    genome.evaluator.set(ggevo.perform_gradient_descent)
    genome.mutator.set(Mutators.G1DListMutatorAllele)
    genome.initializator.set(Initializators.G1DListInitializatorAllele)
    genome.crossover.set(Crossovers.G1DListCrossoverUniform) 

    ga = GSimpleGA.GSimpleGA(genome,__main__.__dict__.get('Seed',1023))
    ga.minimax = Consts.minimaxType["minimize"]
    #ga.selector.set(Selectors.GRouletteWheel)
    ga.setElitism(True) 
    ga.setGenerations(__main__.__dict__.get('GenerationSize',100))
    ga.setPopulationSize(__main__.__dict__.get('PopulationSize',100))
    ga.setMutationRate(__main__.__dict__.get('MutationRate',0.05))
    ga.setCrossoverRate(__main__.__dict__.get('CrossoverRate',0.9))
     
    pop = ga.getPopulation()
    ga.evolve(freq_stats=1)
    best = ga.bestIndividual()
    
    inp = [v for v in best]
    (new_K,success,c)=fmin_tnc(ggevo.func,inp[:],fprime=ggevo.der,bounds=bounds,maxfun = __main__.__dict__.get('FinalNumEval',10000),messages=0)
    
    print 'Final likelyhood'
    print ggevo.func(new_K)
    Ks = new_K
    rfs = ggevo.lscsm.returnRFs(Ks)
    return [Ks,ggevo.lscsm,rfs]	
    
def runLSCTM():
    import noiseEstimation
    res = contrib.dd.DB(None)
    (sizex,sizey,training_inputs,training_set,validation_inputs,validation_set,ff,db_node) = contrib.JanA.dataimport.sortOutLoading(res)
    raw_validation_set = db_node.data["raw_validation_set"]
    
    
    training_inputs1 = training_inputs[4:,:]
    validation_inputs1 = validation_inputs[4:,:]
    training_inputs2 = training_inputs[3:-1,:]
    validation_inputs2 = validation_inputs[3:-1,:]
    training_inputs3 = training_inputs[2:-2,:]
    validation_inputs3 = validation_inputs[2:-2,:]
    
    
    training_set = training_set[0:-4,:]
    validation_set = validation_set[0:-4,:]
	
    for i in xrange(0,len(raw_validation_set)):
	raw_validation_set[i] = raw_validation_set[i][4:,:]
    
    num_pres,num_neurons = numpy.shape(training_set)
    num_pres,kernel_size = numpy.shape(training_inputs)
    size = int(numpy.sqrt(kernel_size))

    raw_validation_data_set=numpy.rollaxis(numpy.array(raw_validation_set),2)
    
    
    params={}
    params["LSCTM"]=True
    db_node = db_node.get_child(params)
    
    params={}
    params["LaplacaBias"] = __main__.__dict__.get('LaplaceBias',0.0004)
    params["LGN_NUM"] = __main__.__dict__.get('LgnNum',6)
    params["num_neurons"] = __main__.__dict__.get('NumNeurons',103)
    params["V1OF"] = __main__.__dict__.get('V1OF','Exp')
    params["LGNOF"] = __main__.__dict__.get('LGNOF','Exp')
    params["LGNTreshold"] =  __main__.__dict__.get('LGNTreshold',False)
    params["HiddenLayerSize"] = __main__.__dict__.get('HiddenLayerSize',1.0)
    params["LogLossCoef"] = __main__.__dict__.get('LogLossCoef',1.0)
    params["NegativeLgn"] = __main__.__dict__.get('NegativeLgn',True)
    params["MaxW"] = __main__.__dict__.get('MaxW',5000)
    params["GenerationSize"] = __main__.__dict__.get('GenerationSize',100)
    params["PopulationSize"] = __main__.__dict__.get('PopulationSize',100)
    params["MutationRate"] = __main__.__dict__.get('MutationRate',0.05)
    params["CrossoverRate"] = __main__.__dict__.get('CrossoverRate',0.9)
    
    db_node1 = db_node
    db_node = db_node.get_child(params)
    
    num_neurons=params["num_neurons"]
    
    raw_validation_data_set=numpy.rollaxis(numpy.array(raw_validation_set),2)
    
    [K,lsctm,rfs]=  fitLSCTMEvo(numpy.mat(training_inputs1),numpy.mat(training_inputs2),numpy.mat(training_inputs3),numpy.mat(training_set),params["LGN_NUM"],num_neurons)
  
    (rfs1,rfs2,rfs3) = rfs
    
    
    pylab.figure()
    m = numpy.max(numpy.abs(rfs1))
    for i in xrange(0,int(num_neurons*__main__.__dict__.get('HiddenLayerSize',1.0))):
	pylab.subplot(11,11,i+1)
    	pylab.imshow(numpy.reshape(rfs1[i,0:kernel_size],(size,size)),vmin=-m,vmax=m,cmap=pylab.cm.RdBu,interpolation='nearest')
    pylab.savefig('GLM_rfs1.png')	
    
    pylab.figure()
    m = numpy.max(numpy.abs(rfs2))
    for i in xrange(0,int(num_neurons*__main__.__dict__.get('HiddenLayerSize',1.0))):
	pylab.subplot(11,11,i+1)
    	pylab.imshow(numpy.reshape(rfs2[i,0:kernel_size],(size,size)),vmin=-m,vmax=m,cmap=pylab.cm.RdBu,interpolation='nearest')
    pylab.savefig('GLM_rfs2.png')	

    pylab.figure()
    m = numpy.max(numpy.abs(rfs3))
    for i in xrange(0,int(num_neurons*__main__.__dict__.get('HiddenLayerSize',1.0))):
	pylab.subplot(11,11,i+1)
    	pylab.imshow(numpy.reshape(rfs3[i,0:kernel_size],(size,size)),vmin=-m,vmax=m,cmap=pylab.cm.RdBu,interpolation='nearest')
    pylab.savefig('GLM_rfs3.png')	
    

    lsctm_pred_act = lsctm.response(training_inputs1,training_inputs2,training_inputs3,K)
    lsctm_pred_val_act = lsctm.response(validation_inputs1,validation_inputs2,validation_inputs3,K)
    
    ofs = run_nonlinearity_detection(numpy.mat(training_set),numpy.mat(lsctm_pred_act),num_bins=10,display=False,name='RPI_piece_wise_nonlinearity.png')
    lsctm_pred_act_t = numpy.mat(apply_output_function(numpy.mat(lsctm_pred_act),ofs))
    lsctm_pred_val_act_t = numpy.mat(apply_output_function(numpy.mat(lsctm_pred_val_act),ofs))
    
    if len(raw_validation_set) != 1:
		print 'Without TF'
		(signal_power,noise_power,normalized_noise_power,training_prediction_power,rpi_validation_prediction_power,signal_power_variance) =performance_analysis(training_set,validation_set,lsctm_pred_act,lsctm_pred_val_act,raw_validation_set,85)
		print 'With TF'
		(signal_power_t,noise_power_t,normalized_noise_power_t,training_prediction_power_t,rpi_validation_prediction_power_t,signal_power_variance_t) =performance_analysis(training_set,validation_set,lsctm_pred_act_t,lsctm_pred_val_act_t,raw_validation_set,85)
		
		significant = numpy.array(numpy.nonzero((numpy.array(normalized_noise_power) < 85) * 1.0))[0]
		
		perf.append(numpy.mean(rpi_validation_prediction_power[significant]))
		perf_t.append(numpy.mean(rpi_validation_prediction_power_t[significant]))
    
    (train_c,val_c) = printCorrelationAnalysis(training_set,validation_set,lsctm_pred_act,lsctm_pred_val_act)
    (t_train_c,t_val_c) = printCorrelationAnalysis(training_set,validation_set,lsctm_pred_act_t,lsctm_pred_val_act_t)
    