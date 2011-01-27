import os;
import re;
import __main__
import pylab
import LSCSM
import contrib.dd
import numpy
from contrib.JanA.visualization import compareModelPerformanceWithRPI, showRFS
from contrib.JanA.ofestimation import *

def listdir_alpha(dirpath):
         import os
         reg_not = re.compile(".*AAA.*")
         os.stat_float_times=False
         files_dict = dict()
         for fname in os.listdir(dirpath):
                 mtime = fname #os.stat(dirpath+os.sep+fname).st_mtime
                 if not mtime in files_dict:
                         files_dict[mtime] = list()
                 files_dict[mtime].append(fname)
         
         mtimes = files_dict.keys()
         mtimes.sort()
         filenames = list()
         for mtime in mtimes:
                 fnames = files_dict[mtime]
                 fnames.sort()
                 for fname in fnames:
                         filenames.append(fname)
         return filenames

reg_params=re.compile(".*snapshot\=False(.*)")

def fetch_data():
    dirr= "/home/antolikjan/eddie_data/LSCSM_SEQ_LGN5_LLTrue_SL=True_LinearLGN/"
    reg=re.compile(".*FromWhichNeuron\=([^,]*)")
    import contrib.JanA.LSCSM
    import contrib
    
    (sizex,sizey,training_inputs,training_set,validation_inputs,validation_set,ff,db_node) = contrib.JanA.dataimport.sortOutLoading(contrib.dd.DB(None))
    raw_validation_set = db_node.data["raw_validation_set"]
    

    num_lgn=16
    
    pred_act=[]
    pred_val_act=[]
    
    idxs = []
    to_delete = []
    rfs = []
    Ks = []
    
    lscsm_new = contrib.JanA.LSCSM.LSCSM1(numpy.mat(training_inputs),numpy.mat(training_set)[:,0],num_lgn,1)
    
    
    for a in os.listdir(dirr):
	num = int(reg.search(a).group(1))
        print num
	idxs.append(num)
	if os.path.exists(dirr+a+'/res.dat'):
		res = contrib.dd.loadResults(dirr+a+'/res.dat')
		dataset_node = res.children[0].children[0]
	        
		print res.children[0].children[0].children[0].children[0].data.keys()
		K = res.children[0].children[0].children[0].children[0].data["Kernels"]
 		lscsm = res.children[0].children[0].children[0].children[0].data["LSCSM"][0]
		
		Ks.append(K[0,:])
		#rfs.append(lscsm_new.returnRFs(K[0,:]))
		
		
		#rfs=lscsm.returnRFs(numpy.array(K)[0])
		
		#print numpy.shape(numpy.array(rfs))
		#print sizex,sizey

		#showRFS(numpy.reshape(numpy.array(rfs),(-1,sizex,sizey)))
		#print 'A'
		#print numpy.shape(numpy.array(K)[0])
		
		#print numpy.array(K)[0]
		#lscsm.X.value = numpy.mat(training_inputs)
		#lscsm.Y.value = numpy.mat(training_set[:,int(num)]).T
		#lscsm1.X.value = numpy.mat(training_inputs)
		#lscsm1.Y.value = numpy.mat(training_set[:,int(num)]).T
		#fun = lscsm1.func()
		#print lscsm.func()(numpy.array(K)[0])
		#print fun(numpy.array(K)[0])
		
		pred_act.append(lscsm.response(training_inputs,numpy.array(K)[0]))
		pred_val_act.append(lscsm.response(validation_inputs,numpy.array(K)[0]))
	else:
		pred_act.append(numpy.zeros((1800,1)))
		pred_val_act.append(numpy.zeros((50,1)))
		rfs.append(numpy.zeros((1,sizex*sizey)))
		to_delete.append(num)
	
    
    print 'Number of missed neurons:', len(to_delete)
    from contrib.JanA.regression import laplaceBias
    
    pred_act = numpy.hstack(pred_act)	
    pred_val_act = numpy.hstack(pred_val_act)
    
    #print numpy.shape(rfs)
    #rfs = numpy.vstack(rfs)
    
    pred_act_new = pred_act[:,numpy.argsort(idxs)]
    pred_val_act_new = pred_val_act[:,numpy.argsort(idxs)]
    #rfs = rfs[numpy.argsort(idxs),:]
    
    print 'Y'
    print numpy.shape(pred_act_new)
    print numpy.shape(training_set)
    
    training_set = numpy.delete(training_set, to_delete, axis = 1)
    validation_set = numpy.delete(validation_set, to_delete, axis = 1)
    pred_act_new = numpy.delete(pred_act_new, to_delete, axis = 1)
    pred_val_act_new = numpy.delete(pred_val_act_new, to_delete, axis = 1)
    #rfs = numpy.delete(rfs, to_delete, axis = 0)
    for i in xrange(0,10):
    	raw_validation_set[i] = numpy.delete(raw_validation_set[i], to_delete, axis = 1)

    print 'X'
    print numpy.shape(pred_act_new)
    print numpy.shape(training_set)
    

    laplace = laplaceBias(sizex,sizey)
    LSCSM_rpi = numpy.linalg.pinv(numpy.mat(training_inputs).T*numpy.mat(training_inputs) + __main__.__dict__.get('RPILaplaceBias',0.0001)*laplace) * numpy.mat(training_inputs).T * numpy.mat(pred_act_new) 
    
    print 'A'
    print numpy.shape(LSCSM_rpi.T)
    print numpy.shape(numpy.array(LSCSM_rpi.T))
    print numpy.shape(numpy.reshape(numpy.array(LSCSM_rpi.T),(-1,sizex,sizey)))
    
    
    showRFS(numpy.reshape(numpy.array(LSCSM_rpi.T),(-1,sizex,sizey)))

    #print numpy.shape(rfs)
    #rfs = numpy.vstack(rfs)
    Ks = numpy.vstack(Ks)
    #print numpy.shape(rfs)
    
    #showRFS(numpy.reshape(rfs,(-1,sizex,sizey)))
    
    #a = Ks[:,7*num_lgn:7*num_lgn+1*num_lgn]
    #t = Ks[:,6*num_lgn:6*num_lgn+1*num_lgn]
    
    #print 'T'
    #print t
    
    #m = numpy.mat(numpy.tile(numpy.mean(abs(a),axis=1),(num_lgn,1))).T
    
    
    #print 'A'
    #print a
    
    #a = (abs(a) >= 0.3*m)*1.0
    #a = (t < 0.001)*1.0
    #Ks[:,0:num_lgn] = numpy.multiply(Ks[:,0:num_lgn],a)
    #Ks[:,num_lgn:2*num_lgn] = numpy.multiply(Ks[:,num_lgn:2*num_lgn],a)
    
    pylab.figure()
    pylab.plot(Ks[:,0:num_lgn].flatten(),Ks[:,num_lgn:2*num_lgn].flatten(),'bo')
    
    
    from contrib.JanA.sparsness_analysis import TrevesRollsSparsness
    
    pylab.figure()
    pylab.subplot(221)
    pylab.hist(TrevesRollsSparsness(numpy.mat(validation_set)).flatten())
    pylab.axis(xmin=0.0,xmax=1.0)
    pylab.title('Lifetime sparesness') 
    pylab.subplot(222)
    pylab.hist(TrevesRollsSparsness(numpy.mat(validation_set.T)).flatten())
    pylab.axis(xmin=0.0,xmax=1.0)
    pylab.title('Population sparesness')
    pylab.subplot(223) 
    pylab.hist(TrevesRollsSparsness(numpy.mat(pred_val_act)).flatten())
    pylab.axis(xmin=0.0,xmax=1.0) 
    pylab.subplot(224)
    pylab.hist(TrevesRollsSparsness(numpy.mat(pred_val_act.T)).flatten())
    pylab.axis(xmin=0.0,xmax=1.0) 
    
    print 'Life time sparsness measured:',numpy.mean(TrevesRollsSparsness(numpy.mat(validation_set)).flatten())
    print 'Life time sparsness predicted:',numpy.mean(TrevesRollsSparsness(numpy.mat(pred_val_act)).flatten())
    print 'Population sparsness measured:',numpy.mean(TrevesRollsSparsness(numpy.mat(validation_set.T)).flatten())
    print 'Population sparsness predicted:',numpy.mean(TrevesRollsSparsness(numpy.mat(pred_val_act.T)).flatten())
    

    #pred_act_new = numpy.mat(training_inputs) * numpy.mat(rfs).T 
    #pred_val_act_new = numpy.mat(validation_inputs) * numpy.mat(rfs).T
    
    print 'b'
    print numpy.shape(pred_act_new)
    print numpy.shape(training_set)
    
    return (pred_act_new,pred_val_act_new,training_set,validation_set,training_inputs,validation_inputs,numpy.array(raw_validation_set))
    
def analyse():
   
    (pred_act,pred_val_act,training_set,validation_set,training_inputs,validation_inputs,raw_validation_set)= fetch_data()
    size =numpy.sqrt(numpy.shape(training_inputs)[1])
    
    compareModelPerformanceWithRPI(training_set,validation_set,training_inputs,validation_inputs,numpy.mat(pred_act),numpy.mat(pred_val_act),raw_validation_set,size,size,modelname='LSCSM')	