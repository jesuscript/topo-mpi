from topo import numbergen
import numpy 


def SG(K,X,Y,lscsm,der,bounds,learning_rate=0.1,num_steps=100,batch_size=100):
    num_pres,num_neurons = numpy.shape(X)
    r = numbergen.UniformRandom(seed=513)	
    mins = numpy.array(zip(*bounds)[0])	
    maxs = numpy.array(zip(*bounds)[1])
    
    for i in xrange(0,num_steps):
	K = K - learning_rate*der(K[:])
	K = numpy.array(K)
	above_bounds = K > maxs
	below_bounds = K < mins
	  	
	if (numpy.sum(above_bounds)+numpy.sum(below_bounds)) > 0:
		K[numpy.nonzero(above_bounds)] = maxs[numpy.nonzero(above_bounds)]
		K[numpy.nonzero(below_bounds)] = mins[numpy.nonzero(below_bounds)]
	
	index = int(r()*(num_pres-batch_size))
	lscsm.X.value = X[index:index+batch_size,:]
	lscsm.Y.value = Y[index:index+batch_size,:]
	
    return K