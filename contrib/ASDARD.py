import numpy
import numpy.linalg
import __main__

def ASD(X,Y,dist):
# X - (p x q) matrix with inputs in rows
# Y - (p, 1) matrix with measurements
# dist - (q,q) matrix containing distances between input points
# Implelements the ASD regression descrived in:
#M. Sahani and J. F. Linden.
#Evidence optimization techniques for estimating stimulus-response functions.
#In S. Becker, S. Thrun, and K. Obermayer, eds., Advances in Neural Information Processing Systems, vol. 15, pp. 301-308, Cambridge, MA, 2003. 
 
	
	(p,q) = numpy.shape(X)
	
	#initialize parameters
	ro = 8
	delta_s = 2.0
	sigma_sq = 0.1#numpy.sum(numpy.power(Y - X*(numpy.linalg.qr(X)[1]*Y),2)) / p
	
	print sigma_sq
	
	step = 0.01
	
	C = X.T * X
	XY = X.T * Y
	start_flag = False
	
	der_ro_m=0
	der_delta_s_m=0
	der_sigma_sq_m=0
	
	for i in xrange(0,100):
		print i,ro,delta_s,sigma_sq
		
		S = numpy.exp(-ro-0.5*dist/(delta_s*delta_s))
		
		S_inv = numpy.linalg.inv(S)
		sigma =  numpy.linalg.inv(C /(sigma_sq) + S_inv)
		
		ni = sigma * (XY) /  (sigma_sq)
		
		Z = (S-sigma-(ni*ni.T)) * S_inv
		der_ro = numpy.trace(Z)
		der_delta_s = - numpy.trace(Z * numpy.multiply(S,dist/(numpy.power(delta_s,3))) * S_inv)
		
		if start_flag:
			der_ro_m=der_ro
			der_delta_s_m=der_delta_s
		else:
		   if der_ro_m*der_ro + der_delta_s_m * der_delta_s < 0:
		      step = step * 0.8
		      der_ro_m = der_ro
		      der_delta_s_m = der_delta_s
		   else:
		      
		      der_ro_m = der_ro +  (der_ro_m * der_ro > 0) * der_ro_m * 0.99	   
		      der_delta_s_m = der_delta_s +  (der_delta_s_m * der_delta_s > 0) *der_delta_s_m * 0.99
		
		ro = ro + step * der_ro_m
		delta_s = delta_s + step * der_delta_s_m
	
		sigma_sq = numpy.sum(numpy.power(Y - X*ni,2))/(p - numpy.trace(numpy.eye(q) - sigma*S_inv));
		delta_s = numpy.max([0.5, numpy.abs(delta_s)])
	
	S = numpy.exp(-ro-0.5*dist/(delta_s*delta_s)) 
	S_inv = numpy.linalg.inv(S)
	w = numpy.linalg.inv(C + sigma_sq * S_inv) * (XY)
	
	return (w,S)


def ARD(X,Y):
# X - (p x q) matrix with inputs in rows
# Y - (p, 1) matrix with measurements
# Implelements the ARD regression, adapted from:
#M. Sahani and J. F. Linden.
#Evidence optimization techniques for estimating stimulus-response functions.
#In S. Becker, S. Thrun, and K. Obermayer, eds., Advances in Neural Information Processing Systems, vol. 15, pp. 301-308, Cambridge, MA, 2003. 
	
	(p,q) = numpy.shape(X)
	
	#initialize parameters
	sigma_sq = 0.1
	CC = X.T * X
	XY = X.T * Y
	start_flag = False
		
	alpha = numpy.mat(numpy.zeros((q,1)))+2.0
	
	for i in xrange(0,100):
		sigma = numpy.linalg.inv(CC/sigma_sq + numpy.diagflat(alpha)) 
		ni = sigma * (XY) /  (sigma_sq)
		sigma_sq = numpy.sum(numpy.power(Y - X*ni,2))/(p - numpy.sum(1 - numpy.multiply(numpy.mat(numpy.diagonal(sigma)).T,alpha)));
		print numpy.min(numpy.abs(ni))
		alpha =  numpy.mat(numpy.divide((1 - numpy.multiply(alpha,numpy.mat(numpy.diagonal(sigma)).T)) , numpy.power(ni,2)))
		print  sigma_sq
		
	w = numpy.linalg.inv(CC + sigma_sq * numpy.diagflat(alpha)) * (XY)
	
	print alpha
	print  sigma_sq
	
	return w
	
def ASDRD(X,Y,S):
# X - (p x q) matrix with inputs in rows
# Y - (p, 1) matrix with measurements
# Implelements the ARD regression, adapted from:
#M. Sahani and J. F. Linden.
#Evidence optimization techniques for estimating stimulus-response functions.
#In S. Becker, S. Thrun, and K. Obermayer, eds., Advances in Neural Information Processing Systems, vol. 15, pp. 301-308, Cambridge, MA, 2003. 
	
	D,V = numpy.linalg.eigh(S)
	V = numpy.mat(V)
	D = numpy.diag(numpy.sqrt(D))
	R =  V*  D * V.T
	w = ARD(X*R,Y)
	w = R * w
	return w

	
	
	

def run():
	f = open("modelfitDatabase1.dat",'rb')
	import pickle
	import pylab
	
	dd = pickle.load(f)
	f.close()
	node = dd.children[0]
	activities = node.data["training_set"]
	training_inputs = node.data["training_inputs"]
	validation_activities = node.data["validation_set"]
	validation_inputs  = node.data["validation_inputs"]

	(p,q) = numpy.shape(training_inputs)
	q = int(numpy.sqrt(q))
	
	X = numpy.vstack([ i * numpy.ones((1,q)) for i in xrange(0,q)]).flatten()
	Y = numpy.hstack([ i * numpy.ones((q,1)) for i in xrange(0,q)]).flatten()
	
	params={}
	params["Method"] = __main__.__dict__.get('Method','ASD')

	dist = numpy.zeros((len(X),len(X)))
	for i in xrange(0,len(X)):
	    for j in xrange(0,len(X)):
		dist[i][j] = numpy.sqrt(numpy.power(X[i] - X[j],2) + numpy.power(Y[i] - Y[j],2))/q
		
		
	numpy.savetxt('/home/antolikjan/MATLAB/inputs.csv', training_inputs, fmt='%.6f', delimiter=';')		
	numpy.savetxt('/home/antolikjan/MATLAB/val_inputs.csv', validation_inputs, fmt='%.6f', delimiter=';')
	numpy.savetxt('/home/antolikjan/MATLAB/activities.csv', activities, fmt='%.6f', delimiter=';')
	numpy.savetxt('/home/antolikjan/MATLAB/distances.csv', dist, fmt='%.6f', delimiter=';')
	return
	#w,S = ASD(numpy.mat(training_inputs),numpy.mat(activities[:,0]).T,numpy.array(dist))
	#w = ARD(numpy.mat(training_inputs),numpy.mat(activities[:,0]).T)
	
	
	S = dd.children[0].children[1].data["S"][0]
	w = dd.children[0].children[1].data["RFs"][0]
	
	w = ASDRD(numpy.mat(training_inputs),numpy.mat(activities[:,0]).T,S)
	return w
	
	node = node.get_child(params)
	
	RFs = []
	S = []
	for i in xrange(0,103):
	    w,s = ASD(numpy.mat(training_inputs),numpy.mat(activities[:,i]).T,numpy.array(dist))			
	    RFs.append(w)
	    S.append(s)
	
	node.add_data("RFs",RFs,force=True)
	node.add_data("S",S,force=True)
	
	#pylab.figure()
        #m = numpy.max(numpy.abs(w))
        #pylab.imshow(w.reshape(q,q),vmin=-m,vmax=m,cmap=pylab.cm.jet,interpolation='nearest')
	#pylab.colorbar()
	#return w.reshape(q,q)
	f = open("modelfitDB2.dat",'wb')
    	pickle.dump(dd,f,-2)
	f.close()

	return RFs
	
	

