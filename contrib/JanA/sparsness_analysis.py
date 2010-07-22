import __main__
import numpy
import pylab
import sys
from topo.misc.filepath import normalize_path, application_path
from contrib.modelfit import *
import contrib.dd
import noiseEstimation
import contrib.JanA.dataimport

def SparsnessAnalysis():
    import scipy
    import scipy.stats
    import numpy.random
    from scipy import linalg
    contrib.modelfit.save_fig_directory='/home/antolikjan/Doc/reports/Sparsness/'
    
    res = contrib.dd.loadResults("results.dat")
    
    node = res.children[0].children[0]
    
    rfs = node.children[0].data["ReversCorrelationRFs"]
    raw_validation_set = node.data["raw_validation_set"]
    training_set = node.data["training_set"]
    validation_set = node.data["validation_set"]

    
    #(sizex,sizey,training_inputs,training_set,validation_inputs,validation_set,ff,db_node) = contrib.modelfit.sortOutLoading(node)
    #raw_validation_set = db_node.data["raw_validation_set"]
    
    raw_validation_data_set=numpy.rollaxis(numpy.array(raw_validation_set),2)
    noiseEstimation.estimateNoise(raw_validation_data_set)
    	
    pred_act  = numpy.array(node.children[0].data["ReversCorrelationPredictedActivities"])
    val_pred_act  = numpy.array(node.children[0].data["ReversCorrelationPredictedValidationActivities"])
    

    # determine signal and noise power and created predictions with gaussian noise of corresponding variance 
    signal_power,noise_power,normalized_noise_power,training_prediction_power,validation_prediction_power = signal_power_test(raw_validation_data_set, training_set, validation_set, pred_act, val_pred_act)
    nt,nn = numpy.shape(pred_act) 
    numpy.random.randn(nt,nn)
     

    # LOAD NON-NORMALIZED DATA
    (sizex,sizey,training_inputs,training_set,validation_inputs,validation_set,ff,db_node) = contrib.JanA.dataimport.sortOutLoading(res)
    
    # fit the predicted activities to sigmoid again
    # this time note that we are fitting it to measured activities that were this time not zero meaned and unit variance scaled
    ofs = fit_sigmoids_to_of(numpy.mat(training_set),numpy.mat(pred_act),offset=False)
    pred_act = apply_sigmoid_output_function(numpy.mat(pred_act),ofs,offset=False)
    #pred_act_noise = apply_sigmoid_output_function(numpy.mat(pred_act_noise),ofs,offset=False)
    val_pred_act = apply_sigmoid_output_function(numpy.mat(val_pred_act),ofs,offset=False)
    (ranks,correct,pred) = performIdentification(validation_set,val_pred_act)
    print "Correct:", correct , "Mean rank:", numpy.mean(ranks) , "MSE", numpy.mean(numpy.power(val_pred_act - validation_set,2))

    pred_act_noise = pred_act + numpy.multiply(numpy.tile(numpy.sqrt(noise_power),(nt,1)) , numpy.random.randn(nt,nn))
    pred_act_noise = pred_act_noise * (pred_act_noise > 0)
    

    # Exclude neurons with weak RFs from further analysis
    rfs_mag=numpy.sum(numpy.power(numpy.reshape(numpy.abs(numpy.array(rfs)),(len(rfs),numpy.size(rfs[0]))),2),axis=1)
    pylab.figure()
    pylab.hist(rfs_mag)
    pylab.xlabel('RFs magnitued')
    to_delete = numpy.array(numpy.nonzero((rfs_mag < 0.031) * 1.0))[0]
    #to_delete=[]
    training_set = numpy.delete(training_set, to_delete, axis = 1)
    validation_set = numpy.delete(validation_set, to_delete, axis = 1)
    pred_act = numpy.delete(pred_act, to_delete, axis = 1)
    val_pred_act = numpy.delete(val_pred_act, to_delete, axis = 1)

    # SHOW MINIMUM FIRING RATES, HISTOGRAMS OF MEAN FIRING RATES AND HISTOGRAM OF FIRING RATES
    print "Minimum activity levels for triggered activities:", numpy.min(training_set,axis=0)
    print "Minimum activity levels for predicted activities:", numpy.min(pred_act,axis=0)
    pylab.figure(dpi=100,facecolor='w',figsize=(15,11))
    pylab.subplot(3,2,1)
    pylab.hist(numpy.mean(training_set,axis=0))
    pylab.xlabel('Mean triggered firing rate')
    pylab.subplot(3,2,2)
    pylab.hist(numpy.mean(pred_act,axis=0))
    pylab.xlabel('Mean predicted firing rate')
    
    pylab.subplot(3,2,3)
    nonz = training_set.flatten()
    nonz =nonz[numpy.nonzero(training_set.flatten())]
    bins,edge  =  numpy.histogram(nonz,50)
    pylab.plot(bins,'x',label='data')
    ex = numpy.random.standard_exponential((1, len(training_set.flatten()))).flatten() * numpy.mean(nonz)
    
    print numpy.mean(ex)
    print numpy.mean(nonz)
    bins,edge  =  numpy.histogram(ex,50)
    pylab.plot(bins,label='Exponential distribution')
    pylab.xscale('log')
    pylab.yscale('log')
    pylab.xlabel('Triggered non-zero firing rates (pooled across population)')
    
    pylab.subplot(3,2,4)
    nonz = pred_act.flatten()
    nonz =nonz[numpy.nonzero(pred_act.flatten())]
    bins,edge  =  numpy.histogram(nonz,50)
    pylab.plot(bins,'x')
    ex = numpy.random.standard_exponential((1, len(pred_act.flatten()))).flatten() * numpy.mean(nonz)
    bins,edge  =  numpy.histogram(ex,50)
    pylab.plot(bins,label='Exponential distribution')
    pylab.xlabel('Predicted non-zero firing rate (pooled across population)')
    pylab.xscale('log')
    pylab.yscale('log')
    
    pylab.subplot(3,2,5)
    nonz = training_set.flatten()
    nonz =nonz[numpy.nonzero(training_set.flatten())]
    bins,edge  =  numpy.histogram(nonz,50)
    pylab.plot(bins,'x')
    pylab.xlabel('Triggered non-zero firing rates (pooled across population)')
    
    pylab.subplot(3,2,6)
    nonz = pred_act.flatten()
    nonz =nonz[numpy.nonzero(pred_act.flatten())]
    bins,edge  =  numpy.histogram(nonz,50)
    pylab.plot(bins,'x')
    pylab.xlabel('Predicted non-zero firing rate (pooled across population)')
    release_fig('Firing_rate_distributions.pdf')
    
    
    
    ex = numpy.random.standard_exponential((1, len(training_set.flatten()))).flatten() * numpy.mean(training_set)
    norm = numpy.random.normal(loc=numpy.mean(training_set),scale=0.0001,size=(1, len(training_set.flatten()))).flatten() 

    
    nonz=ex.flatten()
    #nonz =nonz[numpy.nonzero(ex.flatten())]
    bins1,edge1  =  numpy.histogram(nonz,50)
    bins4,edge4  =  numpy.histogram(numpy.log(nonz),50)
    
    nonz=numpy.exp(norm.flatten())
    #nonz =nonz[numpy.nonzero(numpy.exp(norm.flatten()))]
    bins2,edge2  =  numpy.histogram(nonz,50)
    bins3,edge3  =  numpy.histogram(numpy.log(nonz),50)
    
    pylab.figure()
    pylab.plot(bins3)
    pylab.plot(bins4)
    pylab.figure()
    pylab.plot(bins1,label='exp')
    pylab.plot(bins2,label='lognormal')
    pylab.yscale('log')    
    pylab.xscale('log')
    pylab.legend()
    
    
    pylab.figure()
    (bins,tr,tr) = pylab.hist(training_set.flatten(),bins=100)
    
    pylab.figure()
    pylab.subplot(111, yscale='log')
    pylab.plot(bins)
    
    pylab.figure()
    pylab.subplot(111, xscale='log')
    pylab.plot(bins)
    
    
    pylab.figure()
    (bins,tr,tr) = pylab.hist(pred_act.flatten(),bins=100)
    
    pylab.figure()
    pylab.subplot(111, yscale='log')
    pylab.plot(bins)
    
    pylab.figure()
    pylab.subplot(111, xscale='log')
    pylab.plot(bins)
    
    
    
    mu = 1.0
    exponential = numpy.arange(39, dtype='float32') / 40.0 + 0.0125
    exponential = numpy.exp(- (1 / mu) * exponential) / mu
    
    pylab.figure()
    pylab.subplot(111, yscale='log')
    pylab.plot(exponential)
    
    pylab.figure()
    pylab.subplot(111, xscale='log')
    pylab.plot(exponential)
    
    
    lifetime_triggered_sparsness = TrevesRollsSparsness(numpy.mat(training_set))
    population_triggered_sparsness = TrevesRollsSparsness(numpy.mat(training_set).T)
    lifetime_predicted_sparsness = TrevesRollsSparsness(numpy.mat(pred_act))
    population_predicted_sparsness = TrevesRollsSparsness(numpy.mat(pred_act).T)
    lifetime_predicted_noise_sparsness = TrevesRollsSparsness(numpy.mat(pred_act_noise))
    population_predicted_noise_sparsness = TrevesRollsSparsness(numpy.mat(pred_act_noise).T)
    lifetime_triggered_averaged_sparsness = TrevesRollsSparsness(numpy.mat(validation_set))
    population_triggered_averaged_sparsness = TrevesRollsSparsness(numpy.mat(validation_set).T)


    pylab.figure(dpi=100,facecolor='w',figsize=(15,11))
    pylab.subplot(4,2,1)
    pylab.hist(lifetime_triggered_sparsness)
    pylab.xlabel('Life time sparsness of triggered activities')
    pylab.axis(xmin=0.0,xmax=1.0) 
    
    pylab.subplot(4,2,2)
    pylab.hist(population_triggered_sparsness)
    pylab.xlabel('Population sparsness of triggered activities')
    pylab.axis(xmin=0.0,xmax=1.0)
    
    pylab.subplot(4,2,3)
    pylab.hist(lifetime_predicted_sparsness)
    pylab.xlabel('Life time sparsness of predicted activities')
    pylab.axis(xmin=0.0,xmax=1.0)
    
    pylab.subplot(4,2,4)
    pylab.hist(population_predicted_sparsness)
    pylab.xlabel('Population sparsness of predicted activities')
    pylab.axis(xmin=0.0,xmax=1.0)
    
    pylab.subplot(4,2,5)
    pylab.hist(lifetime_predicted_noise_sparsness)
    pylab.xlabel('Life time sparsness of predicted activities with Gaussian noise')
    pylab.axis(xmin=0.0,xmax=1.0)
    
    pylab.subplot(4,2,6)
    pylab.hist(population_predicted_noise_sparsness)
    pylab.xlabel('Population sparsness of predicted activities with Gaussian noise')
    pylab.axis(xmin=0.0,xmax=1.0)
    
    pylab.subplot(4,2,7)
    pylab.hist(lifetime_triggered_averaged_sparsness)
    pylab.xlabel('Life time sparsness of triggered activities averaged over trials')
    pylab.axis(xmin=0.0,xmax=1.0) 
    
    pylab.subplot(4,2,8)
    pylab.hist(population_triggered_averaged_sparsness)
    pylab.xlabel('Population sparsness of triggered activities averaged over trials')
    pylab.axis(xmin=0.0,xmax=1.0)
    release_fig('Sparsness.pdf')	

def TrevesRollsSparsness(mat):
    # Computes Trevers Rolls Sparsness of data along columns
    x,y  = numpy.shape(mat)
    return numpy.array(1-(numpy.power(numpy.mean(mat,axis=0),2))/numpy.mean(numpy.power(mat,2),axis=0))[0]/(1.0 - 1.0/x)
