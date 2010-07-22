import scipy
from scipy import linalg
import pickle
from contrib.modelfit import *
import numpy
import contrib.dd
import pylab

	
	
def CompareNaturalVSHartley():	
	f = open("modelfitDatabase1.dat",'rb')
	dd = pickle.load(f)
	node = dd.children[26]

	rfs  = node.children[0].data["ReversCorrelationRFs"][0:102]
	
	
	#params = fitGabor(rfs)
	#numpy.savetxt("params.txt", params)
	#return

	
	#b = numpy.reshape(numpy.array(rfs),(102,41*41)).T
	#numpy.savetxt("RFs.txt", b)

	
	
	pred_act  = numpy.array(node.children[0].data["ReversCorrelationPredictedActivities"][:,0:102])
	pred_val_act  = numpy.array(node.children[0].data["ReversCorrelationPredictedValidationActivities"][:,0:102])
	
	pred_act_t  = numpy.array(node.children[0].data["ReversCorrelationPredictedActivities+TF"][:,0:102])
	pred_val_act_t  = numpy.array(node.children[0].data["ReversCorrelationPredictedValidationActivities+TF"][:,0:102])
	
	
	training_set = node.data["training_set"][:,0:102]
	validation_set = node.data["validation_set"][:,0:102]
	
	training_inputs = node.data["training_inputs"]
	validation_inputs = node.data["validation_inputs"]
	raw_validation_set = node.data["raw_validation_set"]
	
	f = file("/home/antolikjan/topographica/topographica/Mice/2010_04_22/Hartley/imcutout.dat", "r")
    	hartley_inputs = [line.split() for line in f]
    	f.close()
	(a,b) = numpy.shape(hartley_inputs)
	for i in xrange(0,a):
	    for j in xrange(0,b):
		hartley_inputs[i][j] = float(hartley_inputs[i][j])
	hartley_in = numpy.array(numpy.mat(hartley_inputs).T)		
        hartley_inputs = []
			
	for i in xrange(0,b):
	    z = numpy.reshape(hartley_in[i,:],(numpy.sqrt(a),numpy.sqrt(a)))
	    hartley_inputs.append(z.T)			
	hartley_inputs_all = numpy.reshape(numpy.array(hartley_inputs),(900,41*41))
	hartley_inputs = numpy.array(hartley_inputs_all)[0:800,:]
	hartley_val_inputs = numpy.array(hartley_inputs_all)[801:850,:]
	
	
	f = file("/home/antolikjan/topographica/topographica/Mice/2010_04_22/Hartley/RFsubspace.dat", "r")
    	hartley_RFs = [line.split() for line in f]
    	f.close()
	(a,b) = numpy.shape(hartley_RFs)
	for i in xrange(0,a):
	    for j in xrange(0,b):
		hartley_RFs[i][j] = float(hartley_RFs[i][j])
	hartley_RFs = numpy.array(hartley_RFs)
	
	hartley_rfs = []		
	for i in xrange(0,b):
	    z = numpy.reshape(hartley_RFs[:,i],(numpy.sqrt(a),numpy.sqrt(a)))/800
	    hartley_rfs.append(z.T)			

	#params = fitGabor(hartley_rfs)
	#numpy.savetxt("Hart_params.txt", params)
	#return

	f = file("/home/antolikjan/topographica/topographica/Mice/2010_04_22/Hartley/responses.dat", "r")
    	hartley_set = [line.split() for line in f]
	f.close()
	
	(a,b) = numpy.shape(hartley_set)
	for i in xrange(0,a):
	    for j in xrange(0,b):
		hartley_set[i][j] = float(hartley_set[i][j])
	hartley_set_all = hartley_set	
	hartley_set = numpy.array(hartley_set_all)[0:800]
	hartley_val_set = numpy.array(hartley_set_all)[801:850]
	
	
	
	print numpy.shape(hartley_inputs)
	print numpy.shape(hartley_RFs)
	print numpy.shape(validation_inputs)
	
	hartley_pred_act = numpy.mat(training_inputs) * numpy.mat(hartley_RFs)
	hartley_pred_val_act = numpy.mat(validation_inputs) * numpy.reshape(numpy.array(hartley_rfs),(102,41*41)).T
	gratings_hartley_pred_act = numpy.mat(hartley_inputs) * numpy.reshape(numpy.array(hartley_rfs),(102,41*41)).T
	gratings_hartley_pred_val_act = numpy.mat(hartley_val_inputs) * numpy.reshape(numpy.array(hartley_rfs),(102,41*41)).T
	gratings_pred_act = numpy.mat(hartley_inputs) * numpy.reshape(numpy.array(rfs),(102,41*41)).T
	gratings_pred_val_act = numpy.mat(hartley_val_inputs) * numpy.reshape(numpy.array(rfs),(102,41*41)).T
	
	
	
	print numpy.shape(hartley_set)
	print numpy.var(hartley_set[:,1])
	print numpy.var(training_set[:,1])
	pylab.figure()
	pylab.subplot(2,1,1)
	pylab.plot(training_set[:,1])
	pylab.subplot(2,1,2)
	pylab.plot(hartley_set[:,1])
	rf_mag = [numpy.sum(numpy.power(r,2)) for r in rfs]	
	#discard ugly RFs          	
	pylab.figure()
	pylab.hist(rf_mag)

	
	#to_delete = numpy.nonzero((numpy.array(rf_mag) < 0.000000)*1.0)[0]
	#print to_delete
	#rfs = numpy.delete(rfs,to_delete,axis=0)
	#pred_act = numpy.delete(pred_act,to_delete,axis=1)
	#pred_val_act = numpy.delete(pred_val_act,to_delete,axis=1)
	#training_set = numpy.delete(training_set,to_delete,axis=1)
	#validation_set = numpy.delete(validation_set,to_delete,axis=1)
	
	#for i in xrange(0,len(raw_validation_set)):
	#    raw_validation_set[i] = numpy.delete(raw_validation_set[i],to_delete,axis=1)
	
	
	
	#(sx,sy) = numpy.shape(rfs[0])
	ofs = run_nonlinearity_detection(numpy.mat(hartley_set),numpy.mat(gratings_pred_act))
	pred_act_t = apply_output_function(numpy.mat(pred_act),ofs)
	pred_val_act_t= apply_output_function(numpy.mat(pred_val_act),ofs)
	gratings_pred_act_t = apply_output_function(numpy.mat(gratings_pred_act),ofs)
	gratings_pred_val_act_t= apply_output_function(numpy.mat(gratings_pred_val_act),ofs)
	
	
	pylab.figure()
	pylab.plot(hartley_set[:,1],gratings_hartley_pred_act[:,1],'o')
	
	pylab.figure()
	w = numpy.array(hartley_rfs[1])
	pylab.show._needmain=False
	pylab.imshow(w,interpolation='nearest',cmap=pylab.cm.RdBu)
	pylab.colorbar()
	
	pylab.figure()
	w = numpy.array(hartley_inputs[1,:])
	pylab.show._needmain=False
	pylab.imshow(numpy.reshape(w,(41,41)),interpolation='nearest',cmap=pylab.cm.RdBu)
	pylab.colorbar()
	
		
	ofs = run_nonlinearity_detection(numpy.mat(hartley_set),numpy.mat(gratings_hartley_pred_act))
	hartley_pred_act_t = apply_output_function(numpy.mat(hartley_pred_act),ofs)
	hartley_pred_val_act_t = apply_output_function(numpy.mat(hartley_pred_val_act),ofs)
	gratings_hartley_pred_act_t = apply_output_function(numpy.mat(gratings_hartley_pred_act),ofs) 
	gratings_hartley_pred_val_act_t = apply_output_function(numpy.mat(gratings_hartley_pred_val_act),ofs)
	
	pylab.figure()
	m = numpy.max([numpy.abs(numpy.min(rfs)),numpy.abs(numpy.max(rfs))])	
	for k in xrange(0,len(rfs)):		
		pylab.subplot(15,15,k+1)
		w = numpy.array(rfs[k])
		pylab.show._needmain=False
		pylab.imshow(w,vmin=-m,vmax=m,interpolation='nearest',cmap=pylab.cm.RdBu)
		pylab.axis('off')
	
	
	pylab.figure()
	m = numpy.max([numpy.abs(numpy.min(hartley_rfs)),numpy.abs(numpy.max(hartley_rfs))])	
	for k in xrange(0,len(rfs)):		
		pylab.subplot(15,15,k+1)
		w = numpy.array(hartley_rfs[k])
		pylab.show._needmain=False
		pylab.imshow(w,vmin=-m,vmax=m,interpolation='nearest',cmap=pylab.cm.RdBu)
		pylab.axis('off')
	
	pylab.figure()
	pylab.plot(validation_set[:,1],pred_val_act[:,1],'o')
	
	pylab.figure()
	pylab.plot(validation_set[:,1],hartley_pred_val_act_t[:,1],'o')
	
	pylab.figure()
	pylab.plot(validation_set[:,1],hartley_pred_val_act[:,1],'o')
	
	
	raw_validation_data_set=numpy.rollaxis(numpy.array(raw_validation_set),2)
	
	print 'NATURAL RF -> NATURAL'
	
	(ranks,correct,pred) = performIdentification(validation_set,pred_val_act)
	print "Natural:", correct , "Mean rank:", numpy.mean(ranks) , "MSE", numpy.mean(numpy.power(validation_set - pred_val_act,2))
	
	(ranks,correct,pred) = performIdentification(validation_set,pred_val_act_t)
	print "Natural+TF:", correct , "Mean rank:", numpy.mean(ranks) , "MSE", numpy.mean(numpy.power(validation_set - pred_val_act_t,2))
		
	signal_power,noise_power,normalized_noise_power,training_prediction_power,validation_prediction_power = signal_power_test(raw_validation_data_set, numpy.array(training_set), numpy.array(validation_set), pred_act, pred_val_act)
	signal_power,noise_power,normalized_noise_power,training_prediction_power_t,validation_prediction_power_t = signal_power_test(raw_validation_data_set, numpy.array(training_set), numpy.array(validation_set), pred_act_t, pred_val_act_t)
	
	print "Prediction power on training set / validation set: ", numpy.mean(training_prediction_power) , " / " , numpy.mean(validation_prediction_power)
	print "Prediction power after TF on training set / validation set: ", numpy.mean(training_prediction_power_t) , " / " , numpy.mean(validation_prediction_power_t)


        print 'HARTLEY RF -> NATURAL'
	 
	(ranks,correct,pred) = performIdentification(validation_set,hartley_pred_val_act)
	print "Natural:", correct , "Mean rank:", numpy.mean(ranks) , "MSE", numpy.mean(numpy.power(validation_set - hartley_pred_val_act,2))
	
	(ranks,correct,pred) = performIdentification(validation_set,hartley_pred_val_act_t)
	print "Natural+TF:", correct , "Mean rank:", numpy.mean(ranks) , "MSE", numpy.mean(numpy.power(validation_set - hartley_pred_val_act_t,2))

	signal_power,noise_power,normalized_noise_power,training_prediction_power,validation_prediction_power = signal_power_test(raw_validation_data_set, numpy.array(hartley_set), numpy.array(validation_set), numpy.array(gratings_hartley_pred_act), numpy.array(hartley_pred_val_act))
	signal_power,noise_power,normalized_noise_power,training_prediction_power_t,validation_prediction_power_t = signal_power_test(raw_validation_data_set, numpy.array(hartley_set), numpy.array(validation_set), numpy.array(gratings_hartley_pred_act_t), numpy.array(hartley_pred_val_act_t))
	print "Prediction power on training set / validation set: ", numpy.mean(training_prediction_power) , " / " , numpy.mean(validation_prediction_power)
	print "Prediction power after TF on training set / validation set: ", numpy.mean(training_prediction_power_t) , " / " , numpy.mean(validation_prediction_power_t)
	
	
	print 'NATURAL RF -> HARTLEY'
	
	(ranks,correct,pred) = performIdentification(hartley_val_set,gratings_pred_val_act)
	print "Natural:", correct , "Mean rank:", numpy.mean(ranks) , "MSE", numpy.mean(numpy.power(hartley_val_set - gratings_pred_val_act,2))
	
	(ranks,correct,pred) = performIdentification(hartley_val_set,gratings_pred_val_act_t)
	print "Natural+TF:", correct , "Mean rank:", numpy.mean(ranks) , "MSE", numpy.mean(numpy.power(hartley_val_set - gratings_pred_val_act_t,2))
		
        
	print 'HARTLEY RF -> HARTLEY'
	 
	(ranks,correct,pred) = performIdentification(hartley_val_set,gratings_hartley_pred_val_act)
	print "Natural:", correct , "Mean rank:", numpy.mean(ranks) , "MSE", numpy.mean(numpy.power(hartley_val_set - gratings_hartley_pred_val_act,2))
	
	(ranks,correct,pred) = performIdentification(hartley_val_set,gratings_hartley_pred_val_act_t)
	print "Natural+TF:", correct , "Mean rank:", numpy.mean(ranks) , "MSE", numpy.mean(numpy.power(hartley_val_set - gratings_hartley_pred_val_act_t,2))

	