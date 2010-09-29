import theano
theano.config.floatX='float32' 
from theano import tensor as T
import numpy

Y = theano.shared(numpy.array([[1.0,1.0],[1.0,1.0]]))

A = 1.0 / (1.0 + T.exp(-Y))
Z = theano.printing.Print(message='My mesasge')(A)

f1 = T.sum(A - Z)

f1 = theano.function(inputs=[], outputs=f1)

print 'f1:', f1()

f2 = T.sum(A) - T.sum(Z)

f2 = theano.function(inputs=[], outputs=f2)

print 'f2:', f2()

print f1.maker.env.toposort()
print f2.maker.env.toposort()
