import __main__
import numpy
import pylab
import sys
from topo.misc.filepath import normalize_path, application_path
from contrib.modelfit import *
import contrib.dd
import noiseEstimation
import contrib.JanA.dataimport
import visualization
from contrib.JanA.ofestimation import *

def SparsnessAnalysis():
    pylab.rcParams['xtick.major.pad']='7'
    import scipy
    import scipy.stats
    import numpy.random
    from scipy import linalg
    from matplotlib.ticker import MaxNLocator, NullLocator

    contrib.modelfit.save_fig_directory='/home/antolikjan/Doc/reports/Sparsness/PieceWise/'
    
    #res = contrib.dd.loadResults("results3.dat")
    
    res = contrib.dd.loadResults("newest_dataset.dat")
    
    #dataset_node = res.children[0].children[3]
    dataset_node = res.children[0].children[0]

    #LOAD NON-NORMALIZED DATA
    
    training_set = dataset_node.data["training_set"]
    validation_set = dataset_node.data["validation_set"]
    training_inputs= dataset_node.data["training_inputs"]
    validation_inputs= dataset_node.data["validation_inputs"]
    raw_validation_set = dataset_node.data["raw_validation_set"]
    
    
    num_trials = numpy.shape(raw_validation_set)[0]
    
    node = dataset_node.children[0].children[0]
    print dataset_node.children[0].children_params
    #node = dataset_node.children[10]
    
    if __main__.__dict__.get('Alg','GLM') == 'GLM':
	sizex = numpy.sqrt(numpy.shape(validation_inputs)[1])	
	sizey = numpy.sqrt(numpy.shape(validation_inputs)[1])
	
    	K =  node.data["Kernels"]
    	print 'Treshold:',K[:,sizex*sizey]
	rfs = numpy.reshape(node.data["Kernels"][:,0:sizex*sizey],(numpy.shape(training_set)[1],sizex,sizey))
	
	training_set = training_set[1:,:]
    	validation_set = validation_set[1:,:]
    	training_inputs= training_inputs[1:,:]
    	validation_inputs= validation_inputs[1:,:]
    
    	#for i in xrange(0,len(raw_validation_set)):
        #	raw_validation_set[i] = raw_validation_set[i][1:,:]
    else:
	rfs = node.data["ReversCorrelationRFs"]
    
    
    raw_validation_data_set=numpy.rollaxis(numpy.array(raw_validation_set),2)
    
        
    pylab.figure()
    contrib.JanA.visualization.showRFS(rfs)
    pylab.colorbar()

    
    pred_act  = numpy.array(node.data["ReversCorrelationPredictedActivities"])
    val_pred_act  = numpy.array(node.data["ReversCorrelationPredictedValidationActivities"])
    
    if __main__.__dict__.get('Lateral',False):
	glm =  node.data["GLM"]
    	val_pred_act_raw = []
	for i in xrange(0,num_trials):
    		val_pred_act_raw.append(glm.sample_from_recurrent_model(validation_inputs,K))
    	val_pred_act_raw = numpy.array(val_pred_act_raw)

	
    	#val_pred_act=numpy.mean(val_pred_act_raw,axis=0)
	#val_pred_act_nl =  numpy.hstack([ glm.response(validation_inputs,numpy.delete(validation_set*0,[i],axis=1),numpy.array([K[i]])) for i in xrange(0,103)])
    	#val_pred_act =  numpy.hstack([ glm.response(validation_inputs,numpy.delete(val_pred_act_nl,[i],axis=1),numpy.array([K[i]])) for i in xrange(0,103)])
    else:
	val_pred_act_nl = val_pred_act
    
    
    # fit the predicted activities to sigmoid again
    # this time note that we are fitting it to measured activities that were this time not zero meaned and unit variance scaled
    ofs = run_nonlinearity_detection(numpy.mat(training_set),numpy.mat(pred_act),num_bins=10,display=True)
    release_fig('EstimatedOF.png')
    pred_act = apply_output_function(numpy.mat(pred_act),ofs)
    val_pred_act = apply_output_function(numpy.mat(val_pred_act),ofs)

    #ofs = fit_sigmoids_to_of(numpy.mat(training_set),numpy.mat(pred_act),offset=False)
    #release_fig('EstimatedOF.png')
    #pred_act = apply_sigmoid_output_function(numpy.mat(pred_act),ofs,offset=False)
    #val_pred_act = apply_sigmoid_output_function(numpy.mat(val_pred_act),ofs,offset=False)

    
    # determine signal and noise power and created predictions with gaussian noise of corresponding variance 
    signal_power,noise_power,normalized_noise_power,training_prediction_power,validation_prediction_power,signal_power_variance = signal_power_test(raw_validation_data_set, numpy.array(training_set), numpy.array(validation_set), pred_act, val_pred_act)
    nt,nn = numpy.shape(pred_act) 
    numpy.random.randn(nt,nn)
     
    print numpy.argmax(validation_prediction_power)
    print validation_prediction_power[numpy.argmin(validation_prediction_power)] 
     
    pylab.rc('font', size=9)
    pylab.figure(dpi=400,facecolor='w',figsize=(10,5))
    ax = pylab.subplot(111)
    ax.yaxis.set_major_locator(MaxNLocator(3))
    pylab.imshow(training_set[:600,:].T,cmap='gray',interpolation='nearest',vmin=0,vmax=10)
    pylab.xlabel('Image #')
    pylab.ylabel('Neuron #')
    pylab.colorbar(shrink=0.27,aspect=10,pad=0.02,ticks=[0,10,20])
    pylab.gca().yaxis.set_major_locator(MaxNLocator(4))
    pylab.gca().xaxis.set_ticks_position('top')
    pylab.gca().xaxis.set_label_position('top')
    release_fig('PopulationResponses.svg')
    pylab.rc('font', size=15)
    
    
    
    
    val_pred_act = val_pred_act * (val_pred_act > 0.0)
    pred_act = pred_act * (pred_act > 0.0)
    
    print 'Mean prediction power before neuron selection:',numpy.mean(validation_prediction_power)
    (ranks,correct,pred) = performIdentification(validation_set,val_pred_act)
    print "Correct:", correct , "Mean rank:", numpy.mean(ranks) , "MSE", numpy.mean(numpy.power(val_pred_act - validation_set,2))

    # Exclude neurons with weak RFs from further analysis
    
    rfs_mag=numpy.sum(numpy.power(numpy.reshape(numpy.abs(numpy.array(rfs)),(len(rfs),numpy.size(rfs[0]))),2),axis=1)
    
    visualization.showRFS(rfs)
    release_fig('RFsAll.png')
    
    pylab.figure()
    pylab.hist(rfs_mag)
    pylab.xlabel('RFs magnitued')
    
    pylab.figure()
    pylab.plot(normalized_noise_power ,validation_prediction_power,'ro')
    pylab.xlabel('normalized_noise_power')
    pylab.ylabel('validation_prediction_power')

    to_delete1 = numpy.array(numpy.nonzero((rfs_mag < 7000000) * 1.0))[0]
    to_delete2 = numpy.array(numpy.nonzero((numpy.array(normalized_noise_power) > 85) * 1.0))[0]
    to_delete = numpy.unique(numpy.append(to_delete1,to_delete2))
    
    print to_delete
    
    to_delete = [2, 3 ,  4 ,  5 ,  6 ,  7 ,  9 , 10 , 11 , 12 , 13 , 14 , 18 , 22 , 26 , 28 , 29 , 31, 32 , 34 , 35 , 36,  37 , 38 , 41 , 44 , 49 , 51 ,  54 , 55 , 57 , 59 , 60 , 63 , 67 , 68 , 70 , 72 , 73 , 74 , 76 , 77 , 79 , 81 , 82 , 84,  86,  87 , 88 , 90 , 94 , 95 , 97 , 98  , 99 ,100,  102]
    
    print 'Number of neurons excluded:', len(to_delete) 
    
    training_set = numpy.delete(training_set, to_delete, axis = 1)
    validation_set = numpy.delete(validation_set, to_delete, axis = 1)
    pred_act = numpy.delete(pred_act, to_delete, axis = 1)
    val_pred_act = numpy.delete(val_pred_act, to_delete, axis = 1)
    val_pred_act_nl = numpy.delete(val_pred_act_nl, to_delete, axis = 1)
    validation_prediction_power = numpy.delete(validation_prediction_power , to_delete)
    normalized_noise_power  = numpy.delete(normalized_noise_power , to_delete)
    rfs = numpy.delete(rfs, to_delete, axis = 0)
    
    for i in xrange(0,num_trials):
    	raw_validation_set[i] = numpy.delete(raw_validation_set[i], to_delete, axis = 1)

    
    print 'Mean prediction power after neuron selection:',numpy.mean(validation_prediction_power)
    (ranks,correct,pred) = performIdentification(validation_set,val_pred_act)
    print "Correct:", correct , "Mean rank:", numpy.mean(ranks) , "MSE", numpy.mean(numpy.power(val_pred_act - validation_set,2))

    visualization.showRFS(rfs)
    release_fig('RFs.png')
    
    (num_trials,num_pres,num_neurons) = numpy.shape(numpy.array(raw_validation_set))
    
    raw_validation_set_flattened = numpy.reshape(raw_validation_set,(num_trials*num_pres,num_neurons))
    
    pylab.figure(dpi=100,facecolor='w',figsize=(6,5))
    pylab.hist(validation_prediction_power)
    pylab.xlabel('Fraction of explained signal power ')
    pylab.ylabel('# neurons')
    
    # SHOW MINIMUM FIRING RATES, HISTOGRAMS OF MEAN FIRING RATES AND HISTOGRAM OF FIRING RATES
    #print "Minimum activity levels for raw validation activities:", numpy.min(raw_validation_data_set,axis=0)
    #print "Minimum activity levels for triggered activities:", numpy.min(training_set,axis=0)
    #print "Minimum activity levels for predicted activities:", numpy.min(pred_act,axis=0)
    #print "Maximum activity levels for triggered activities:", numpy.max(training_set,axis=0)
    #print "Maximum activity levels for predicted activities:", numpy.max(pred_act,axis=0)
    
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
    
    bins,edge  =  numpy.histogram(ex,50)
    pylab.plot(bins,label='Exponential distribution')
    pylab.xscale('log')
    pylab.yscale('log')
    pylab.xlabel('Triggered non-zero firing rates (pooled across population)')
    pylab.legend()
    
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
    
    
    
    ex = numpy.random.exponential(numpy.mean(raw_validation_data_set),(1, 10*len(training_set.flatten()))).flatten()  
    norm = numpy.random.normal(loc=numpy.exp(numpy.mean(raw_validation_data_set)),scale=0.5,size=(1, 10*len(training_set.flatten()))).flatten() 

    nonz=ex.flatten()
    bins1,edge1  =  numpy.histogram(nonz,50)
    bins1 = bins1*1.0/numpy.sum(bins1)
    bins3,edge3  =  numpy.histogram(numpy.log(nonz),50)
    bins3 = bins3*1.0/numpy.sum(bins3)
    
    nonz=numpy.exp(norm.flatten())
    bins2,edge2  =  numpy.histogram(nonz,50)
    bins2 = bins2*1.0/numpy.sum(bins2)
    bins4,edge4  =  numpy.histogram(numpy.log(nonz),50)
    bins4 = bins4*1.0/numpy.sum(bins4)
    
    pylab.figure()
    pylab.subplot(221)
    pylab.plot(bins3,label='logged exp')
    pylab.plot(bins4,label='logged lognormal')
    pylab.legend()
    pylab.subplot(222)
    pylab.plot(bins1,label='exp')
    pylab.plot(bins2,label='lognormal')
    pylab.legend()
    pylab.subplot(223,yscale='log')
    pylab.plot(bins3,label='logged exp')
    pylab.plot(bins4,label='logged lognormal')
    pylab.legend()
    pylab.subplot(224,xscale='log',yscale='log')
    pylab.plot(bins1,label='exp')
    pylab.plot(bins2,label='lognormal')
    pylab.legend()
    
    pylab.figure()
    pylab.hist(raw_validation_data_set.flatten()[numpy.nonzero(raw_validation_data_set.flatten())],bins=1000)
    pylab.figure()
    pylab.hist(validation_set.flatten()[numpy.nonzero(validation_set.flatten())],bins=1000)

    #perform sampling of the model prediction from poisson distribution
    if __main__.__dict__.get('Lateral',False):
	pred_act_singletrial = val_pred_act_raw
	pred_act_avg = val_pred_act 
    else:
	pred_act_avg = numpy.zeros(numpy.shape(val_pred_act))
	pred_act_singletrial = numpy.zeros(numpy.shape(raw_validation_set))
	for x in xrange(0,numpy.shape(val_pred_act)[0]):
		for y in xrange(0,numpy.shape(val_pred_act)[1]):
			pred_act_singletrial[:,x,y] = numpy.random.poisson(lam=val_pred_act[x,y],size=(num_trials,1)).flatten()
			pred_act_avg[x,y] = numpy.mean(pred_act_singletrial[:,x,y])

    
    pylab.figure(dpi=100,facecolor='w',figsize=(17,12))
    (bins,edges) = numpy.histogram(raw_validation_data_set.flatten()[numpy.nonzero(raw_validation_data_set.flatten())],bins=numpy.exp(numpy.arange(-4,10,0.4)))
    bins = bins*1.0/numpy.sum(bins)/(edges[1:]-edges[:-1])
    (bins_normal,edges_normal) = numpy.histogram(raw_validation_data_set.flatten()[numpy.nonzero(raw_validation_data_set.flatten())],bins=400)
    bins_normal = bins_normal*1.0/numpy.sum(bins_normal)/(edges_normal[1:]-edges_normal[:-1])
    
    print min(raw_validation_data_set.flatten())
    
    logged = numpy.log(raw_validation_data_set.flatten()[numpy.nonzero(raw_validation_data_set.flatten())])
    (bins_logged,edges_logged) = numpy.histogram(logged,bins=20)
    bins_logged = bins_logged*1.0/numpy.sum(bins_logged)/(edges_logged[1]-edges_logged[0])
    pylab.subplot(331)
    pylab.title('Raw validation set')
    xs = edges_normal[:-1]+(edges_normal[1]-edges_normal[0])/2
    pylab.bar(xs,bins_normal,width=(edges_normal[1]-edges_normal[0])/2,color='b')
    pylab.plot(xs,numpy.exp(-numpy.power(numpy.log(xs)-numpy.mean(logged),2)/(2*numpy.var(logged)))/(xs*numpy.sqrt(2*numpy.pi*numpy.var(logged))),'g',lw=2)
    pylab.plot(xs,numpy.exp(-xs/numpy.mean(raw_validation_data_set))/numpy.mean(raw_validation_data_set),'r',lw=2)
    pylab.axis(xmin=0,xmax=4)
    pylab.ylabel('Probability density of\n firing rates',multialignment='center')
    pylab.subplot(334)
    xs = edges_logged[:-1]+(edges_logged[1]-edges_logged[0])/2
    pylab.plot(xs,bins_logged,'o')
    xs = numpy.arange(edges_logged[0],edges_logged[-1],(edges_logged[-1]-edges_logged[0])/200)
    pylab.plot(xs,numpy.exp(-numpy.power(xs-numpy.mean(logged),2)/(2*numpy.var(logged)))/numpy.sqrt(2*numpy.pi*numpy.var(logged)),'g',lw=2)
    pylab.ylabel('Probability density of\n log firing rates',multialignment='center')
    pylab.subplot(337, xscale='log',yscale='log')
    xs = edges[:-1]+(edges[1:]-edges[:-1])/2
    pylab.plot(xs,bins,'o')
    xs = numpy.exp(numpy.arange(-4,10,0.05))
    pylab.plot(xs,numpy.exp(-numpy.power(numpy.log(xs)-numpy.mean(logged),2)/(2*numpy.var(logged)))/(xs*numpy.sqrt(2*numpy.pi*numpy.var(logged))),'g',lw=2)
    pylab.plot(xs,numpy.exp(-xs/numpy.mean(raw_validation_data_set))/numpy.mean(raw_validation_data_set),'r',lw=2)
    pylab.axis(xmin=10e-3,xmax=20,ymin=10e-4,ymax=10)
    pylab.ylabel('Probability density of\n firing rates',multialignment='center')
    pylab.xlabel('Firing rate')
    
    
    
    (bins,edges) = numpy.histogram(validation_set.flatten()[numpy.nonzero(validation_set.flatten())],bins=numpy.exp(numpy.arange(-4,10,0.4)))
    bins = bins*1.0/numpy.sum(bins)/(edges[1:]-edges[:-1])
    logged = numpy.log(validation_set.flatten()[numpy.nonzero(validation_set.flatten())])
    (bins_logged,edges_logged) = numpy.histogram(logged,bins=20)
    bins_logged = bins_logged*1.0/numpy.sum(bins_logged)/(edges_logged[1]-edges_logged[0])
    (bins_normal,edges_normal) = numpy.histogram(validation_set.flatten()[numpy.nonzero(validation_set.flatten())],bins=400)
    bins_normal = bins_normal*1.0/numpy.sum(bins_normal)/(edges_normal[1:]-edges_normal[:-1])
    pylab.subplot(332)
    pylab.title('Averaged validation set')
    xs = edges_normal[:-1]+(edges_normal[1]-edges_normal[0])/2
    pylab.bar(xs,bins_normal,width=(edges_normal[1]-edges_normal[0])/2,color='b')
    pylab.plot(xs,numpy.exp(-numpy.power(numpy.log(xs)-numpy.mean(logged),2)/(2*numpy.var(logged)))/(xs*numpy.sqrt(2*numpy.pi*numpy.var(logged))),'g',lw=2)
    pylab.plot(xs,numpy.exp(-xs/numpy.mean(validation_set))/numpy.mean(validation_set),'r',lw=2)
    pylab.axis(xmin=0,xmax=4)
    pylab.subplot(335)
    xs = edges_logged[:-1]+(edges_logged[1]-edges_logged[0])/2
    pylab.plot(xs,bins_logged,'o')
    xs = numpy.arange(edges_logged[0],edges_logged[-1],(edges_logged[-1]-edges_logged[0])/200)
    pylab.plot(xs,numpy.exp(-numpy.power(xs-numpy.mean(logged),2)/(2*numpy.var(logged)))/numpy.sqrt(2*numpy.pi*numpy.var(logged)),'g',lw=2)
    pylab.subplot(338, xscale='log',yscale='log')
    xs = edges[:-1]+(edges[1:]-edges[:-1])/2
    pylab.plot(xs,bins,'o')
    xs = numpy.exp(numpy.arange(-4,10,0.05))
    pylab.plot(xs,numpy.exp(-numpy.power(numpy.log(xs)-numpy.mean(logged),2)/(2*numpy.var(logged)))/(xs*numpy.sqrt(2*numpy.pi*numpy.var(logged))),'g',lw=2)
    pylab.plot(xs,numpy.exp(-xs/numpy.mean(validation_set))/numpy.mean(validation_set),'r',lw=2)
    pylab.axis(xmin=10e-3,xmax=20,ymin=10e-4,ymax=10)
    pylab.xlabel('Firing rate')
    
        
    (bins,edges) = numpy.histogram(val_pred_act.flatten()[numpy.nonzero(val_pred_act.flatten())],bins=numpy.exp(numpy.arange(-4,10,0.4)))
    bins = bins*1.0/numpy.sum(bins)/(edges[1:]-edges[:-1])
    logged = numpy.log(val_pred_act.flatten()[numpy.nonzero(val_pred_act.flatten())])
    (bins_logged,edges_logged) = numpy.histogram(logged,bins=20)
    bins_logged = bins_logged*1.0/numpy.sum(bins_logged)/(edges_logged[1]-edges_logged[0])
    (bins_normal,edges_normal) = numpy.histogram(val_pred_act.flatten()[numpy.nonzero(val_pred_act.flatten())],bins=400)
    bins_normal = bins_normal*1.0/numpy.sum(bins_normal)/(edges_normal[1:]-edges_normal[:-1])
    pylab.subplot(333)
    pylab.title('Model')
    xs = edges_normal[:-1]+(edges_normal[1]-edges_normal[0])/2
    pylab.bar(xs,bins_normal,width=(edges_normal[1]-edges_normal[0])/2,color='b')
    pylab.plot(xs,numpy.exp(-numpy.power(numpy.log(xs)-numpy.mean(logged),2)/(2*numpy.var(logged)))/(xs*numpy.sqrt(2*numpy.pi*numpy.var(logged))),'g',lw=2)
    pylab.plot(xs,numpy.exp(-xs/numpy.mean(val_pred_act))/numpy.mean(val_pred_act),'r',lw=2)
    pylab.axis(xmin=0,xmax=4)
    pylab.subplot(336)
    xs = edges_logged[:-1]+(edges_logged[1]-edges_logged[0])/2
    pylab.plot(xs,bins_logged,'o')
    xs = numpy.arange(edges_logged[0],edges_logged[-1],(edges_logged[-1]-edges_logged[0])/200)
    pylab.plot(xs,numpy.exp(-numpy.power(xs-numpy.mean(logged),2)/(2*numpy.var(logged)))/numpy.sqrt(2*numpy.pi*numpy.var(logged)),'g',lw=2)
    pylab.subplot(339, xscale='log',yscale='log')
    xs = edges[:-1]+(edges[1:]-edges[:-1])/2
    pylab.plot(xs,bins,'o')
    xs = numpy.exp(numpy.arange(-4,10,0.05))
    pylab.plot(xs,numpy.exp(-numpy.power(numpy.log(xs)-numpy.mean(logged),2)/(2*numpy.var(logged)))/(xs*numpy.sqrt(2*numpy.pi*numpy.var(logged))),'g',lw=2)
    pylab.plot(xs,numpy.exp(-xs/numpy.mean(val_pred_act))/numpy.mean(val_pred_act),'r',lw=2)
    pylab.axis(xmin=10e-3,xmax=20,ymin=10e-4,ymax=10)
    pylab.xlabel('Firing rate')
    
    release_fig('Firing_rate_distribution_comparison.png')
    
    
    pred_act_singletrial_flattened = numpy.reshape(pred_act_singletrial,(num_trials*num_pres,num_neurons))
    
    lifetime_triggered_sparsness = TrevesRollsSparsness(numpy.mat(raw_validation_set_flattened))
    population_triggered_sparsness = TrevesRollsSparsness(numpy.mat(raw_validation_set_flattened).T)
    lifetime_predicted_sparsness = TrevesRollsSparsness(numpy.mat(pred_act_singletrial_flattened))
    population_predicted_sparsness = TrevesRollsSparsness(numpy.mat(pred_act_singletrial_flattened).T)
    lifetime_predicted_averaged_sparsness = TrevesRollsSparsness(numpy.mat(pred_act_avg))
    population_predicted_averaged_sparsness = TrevesRollsSparsness(numpy.mat(pred_act_avg).T)
    lifetime_triggered_averaged_sparsness = TrevesRollsSparsness(numpy.mat(numpy.mean(raw_validation_set,axis=0)))
    population_triggered_averaged_sparsness = TrevesRollsSparsness(numpy.mat(numpy.mean(raw_validation_set,axis=0)).T)
    
    
    f = pylab.figure(dpi=100,facecolor='w',figsize=(3,3))
    pylab.hist(population_triggered_sparsness,bins=numpy.arange(0,1.0,0.025))
    release_fig('RawPopulationSparsness.png')
    
    

    f = pylab.figure(dpi=100,facecolor='w',figsize=(18,13))
    f.subplots_adjust(hspace=0.3)
    
    ax = pylab.subplot(4,2,1)
    pylab.hist(lifetime_triggered_sparsness,bins=numpy.arange(0,1.0,0.025))
    pylab.title('Lifetime sparseness')
    pylab.axis(xmin=0.0,xmax=1.0)
    ax.xaxis.set_major_locator(NullLocator())
    ax.yaxis.set_major_locator(MaxNLocator(4))
    pylab.ylabel('# neurons')

    ax = pylab.subplot(4,2,2)
    pylab.hist(population_triggered_sparsness,bins=numpy.arange(0,1.0,0.025))
    pylab.title('Population sparseness')
    pylab.axis(xmin=0.0,xmax=1.0)
    ax.xaxis.set_major_locator(NullLocator())
    ax.yaxis.set_major_locator(MaxNLocator(4))
    pylab.text(1.02,0.5,'triggered',horizontalalignment ='left',verticalalignment='center',rotation='vertical',transform=pylab.gca().transAxes,clip_on=False,fontsize=16)
    pylab.ylabel('# images')
    #ax2 = pylab.gcf().add_axes(ax.get_position(), sharex=ax, frameon=False)
    #ax.set_ylabel('triggered activities')
    
    ax = pylab.subplot(4,2,3)
    pylab.hist(lifetime_predicted_sparsness,bins=numpy.arange(0,1.0,0.025))
    pylab.axis(xmin=0.0,xmax=1.0)
    ax.xaxis.set_major_locator(NullLocator())
    ax.yaxis.set_major_locator(MaxNLocator(4))
    pylab.ylabel('# neurons')
    
    ax = pylab.subplot(4,2,4)
    pylab.hist(population_predicted_sparsness,bins=numpy.arange(0,1.0,0.025))
    pylab.axis(xmin=0.0,xmax=1.0)
    ax.xaxis.set_major_locator(NullLocator())
    ax.yaxis.set_major_locator(MaxNLocator(4))
    pylab.text(1.02,0.5,'predicted',horizontalalignment ='left',verticalalignment='center',rotation='vertical',transform=pylab.gca().transAxes,clip_on=False,fontsize=16)
    pylab.ylabel('# images')
    
    ax = pylab.subplot(4,2,5)
    pylab.hist(lifetime_triggered_averaged_sparsness,bins=numpy.arange(0,1.0,0.025))
    pylab.axis(xmin=0.0,xmax=1.0)
    ax.xaxis.set_major_locator(NullLocator())
    ax.yaxis.set_major_locator(MaxNLocator(4)) 
    pylab.ylabel('# neurons')
    
    ax = pylab.subplot(4,2,6)
    pylab.hist(population_triggered_averaged_sparsness,bins=numpy.arange(0,1.0,0.025))
    pylab.axis(xmin=0.0,xmax=1.0)
    ax.xaxis.set_major_locator(NullLocator())
    ax.yaxis.set_major_locator(MaxNLocator(4))
    pylab.text(1.02,0.5,'triggered\ntrial\naveraged\n',horizontalalignment ='left',verticalalignment='center',rotation='vertical',transform=pylab.gca().transAxes,clip_on=False,fontsize=16,multialignment='center')
    pylab.ylabel('# images')
    
    ax = pylab.subplot(4,2,7)
    pylab.hist(lifetime_predicted_averaged_sparsness,bins=numpy.arange(0,1.0,0.025))
    pylab.axis(xmin=0.0,xmax=1.0)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    pylab.xlabel('Treves Rolls sparseness')
    pylab.ylabel('# neurons')
    
    ax = pylab.subplot(4,2,8)
    pylab.hist(population_predicted_averaged_sparsness,bins=numpy.arange(0,1.0,0.025))
    pylab.axis(xmin=0.0,xmax=1.0)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    pylab.text(1.02,0.5,'predicted\ntrial\naveraged\n',horizontalalignment ='left',verticalalignment='center',rotation='vertical',transform=pylab.gca().transAxes,clip_on=False,fontsize=16,multialignment='center')
    pylab.xlabel('Treves Rolls sparseness')
    pylab.ylabel('# images')

    release_fig('Sparsness.png')	

    print numpy.shape(pred_act_avg)
    print numpy.shape(numpy.mean(raw_validation_set,axis=0))
    pylab.figure()
    pylab.subplot(131)
    pylab.imshow(validation_set,interpolation='nearest')
    pylab.colorbar()
    pylab.subplot(132)
    pylab.imshow(pred_act_avg,interpolation='nearest')
    pylab.colorbar()
    pylab.subplot(133)
    pylab.imshow(val_pred_act_nl,interpolation='nearest')
    pylab.colorbar()
    
    # analyze sparsification
    from numpy.random import shuffle
    print (num_trials,num_pres,num_neurons) 
    
    shuffled_raw_validation_set = numpy.zeros(numpy.shape(raw_validation_set))
    
    for x in xrange(0,num_pres):
	for y in xrange(0,num_neurons):
	    shuff_indexes = numpy.arange(0,num_trials,1)
	    shuffle(shuff_indexes)
	    shuffled_raw_validation_set[:,x,y] = numpy.array(raw_validation_set)[shuff_indexes,x,y]	
    
    shuffled_raw_validation_set_flattened = numpy.reshape(shuffled_raw_validation_set,(num_trials*num_pres,num_neurons))
    
    shuffled_population_triggered_sparsness = TrevesRollsSparsness(numpy.mat(shuffled_raw_validation_set_flattened).T)
    
    sums = numpy.sum(raw_validation_set,axis=2)
    sums_shuffled = numpy.sum(shuffled_raw_validation_set,axis=2)
    
    pylab.figure()
    pylab.subplot(1,2,1)
    pylab.title('Orignial data')
    pylab.hist(sums.flatten(),bins=numpy.arange(0,200,10))
    pylab.subplot(1,2,2)
    pylab.hist(sums_shuffled.flatten(),bins=numpy.arange(0,200,10))
    
    pylab.figure(dpi=100,facecolor='w',figsize=(6,5))    
    pylab.hist(numpy.array([population_triggered_sparsness.flatten(),shuffled_population_triggered_sparsness.flatten()]).T,bins=10,histtype='bar',label=['original','trial shuffled'])
    pylab.xlabel('Treves Rolls sparseness of measured activities')
    pylab.axis(xmin=0.0,xmax=1.0)
    pylab.legend(loc='upper left')    
    release_fig('ShuffledSparsness.png')
    
    # DISPERSAL ANALYSIS
    
    scree_data = numpy.sort(numpy.var(raw_validation_set_flattened,axis=0).flatten())[::-1]
    scree_data = scree_data/scree_data[0]
     
    scree_model = numpy.sort(numpy.var(pred_act_singletrial_flattened,axis=0).flatten())[::-1]
    scree_model = scree_model/scree_model[0]
    
    raw_validation_set_normalized_flattened = raw_validation_set_flattened / numpy.sqrt(numpy.tile(numpy.var(raw_validation_set_flattened,axis=0),(numpy.shape(raw_validation_set_flattened)[0],1)))   
    
    scree_normalized = numpy.sort(numpy.var(raw_validation_set_normalized_flattened ,axis=0).flatten())[::-1]
    scree_normalized  = scree_normalized /scree_normalized[0]
    
    population_normalized_triggered_sparsness = TrevesRollsSparsness(numpy.mat(raw_validation_set_normalized_flattened).T)
    
    pylab.figure(dpi=100,facecolor='w',figsize=(15,6))
    pylab.subplot(1,2,1)
    pylab.plot(scree_data,'bo-',label='measured data')
    pylab.plot(scree_model,'ro-',label='fitted model data')
    pylab.ylabel('Variance of response distribution')
    pylab.xlabel('Neuron identity')
    pylab.legend()
    
    pylab.subplot(1,2,2)
    pylab.hist(numpy.array([population_triggered_sparsness.flatten(),population_normalized_triggered_sparsness.flatten()]).T,bins=10,histtype='bar',label=['origninal','equal variance'])
    pylab.axis(xmin=0.0,xmax=1.0)
    pylab.xlabel('Treves Rolls sparseness')
    pylab.legend(loc='upper left')
    release_fig('Dispersal.png')
    

    
    node.add_data("validation_prediction_power",validation_prediction_power,force=True)

    node.add_data("lifetime_triggered_sparsness",lifetime_triggered_sparsness,force=True)
    node.add_data("population_triggered_sparsness",population_triggered_sparsness ,force=True)
    node.add_data("lifetime_predicted_sparsness",lifetime_predicted_sparsness,force=True)
    node.add_data("population_predicted_sparsness",population_predicted_sparsness,force=True)
    node.add_data("lifetime_predicted_averaged_sparsness",lifetime_predicted_averaged_sparsness,force=True)
    node.add_data("population_predicted_averaged_sparsness",population_predicted_averaged_sparsness,force=True)
    node.add_data("lifetime_triggered_averaged_sparsness",lifetime_triggered_averaged_sparsness,force=True)
    node.add_data("population_triggered_averaged_sparsness",population_triggered_averaged_sparsness,force=True)
    
    node.add_data("shuffled_population_triggered_sparsness",shuffled_population_triggered_sparsness,force=True)
    
    node.add_data("scree_data",scree_data,force=True)
    node.add_data("scree_model",scree_model,force=True)
    node.add_data("scree_normalized",scree_normalized,force=True)
    
    #contrib.dd.saveResults(res,"newest_dataset.dat")
    
 
def TrevesRollsSparsness(mat):
    # Computes Trevers Rolls Sparsness of data along columns
    x,y  = numpy.shape(mat)
    trs = numpy.array(1-(numpy.power(numpy.mean(mat,axis=0),2))/(numpy.mean(numpy.power(mat,2),axis=0)+0.0000000000000000000001))[0]/(1.0 - 1.0/x)
    return trs * (trs<=1.0)
    


def GenerateMultiplayerFigures():
    
    contrib.modelfit.save_fig_directory='/home/antolikjan/Doc/reports/Sparsness/PieceWise/'	
    
    res = contrib.dd.loadResults("newest_dataset.dat")
	
    animals = (
    		res.children[0].children[0].children[0].children[0],
    		res.children[4].children[0].children[0].children[0]
	      )
     
    vpps = numpy.concatenate([a.data["validation_prediction_power"] for a in animals])
    
    pylab.figure(dpi=100,facecolor='w',figsize=(6,5))
    pylab.hist(vpps)
    pylab.xlabel('Fraction of explained signal power ')
    pylab.ylabel('# neurons')
    release_fig('PredictionPower.png')
	
