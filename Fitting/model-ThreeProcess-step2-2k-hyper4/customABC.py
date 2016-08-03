import numpy
import sys
from numpy import *
from numpy.linalg import *
# import numdifftools as nd

def distance(data1, data2, parameters, model):
  
			
		alldist = -1
		sig1 = numpy.sum([ data1[:,0], data1[:,1], data1[:,2], data1[:,4] ], axis=0 ) * 0.0285714857
		sig2 = numpy.sum([ data1[:,5], data1[:,6], data1[:,7], data1[:,44] ], axis=0 ) * 0.0285714857
		sig3 = numpy.sum([ data1[:,10], data1[:,11], data1[:,12], data1[:,45] ], axis=0 ) * 0.0285714857
		sig4 = numpy.sum([ data1[:,15], data1[:,16], data1[:,17], data1[:,41] ], axis=0 ) * 0.0285714857
		sig5 = numpy.sum([ data1[:,20], data1[:,21], data1[:,22] ], axis=0 ) * 0.0285714857
		sig6 = numpy.sum([ data1[:,25], data1[:,26], data1[:,27] ], axis=0 ) * 0.0285714857
		sig7 = numpy.sum([ data1[:,30], data1[:,31] ], axis=0 ) * 0.0285714857
		sig8 = numpy.sum([ data1[:,33], data1[:,34], data1[:,35] ], axis=0 ) * 0.0285714857
		#print sig, data2, numpy.shape(sig[:]), numpy.shape(data2[:,0])
 		#data1 and data2 are two numpy arrays
    		if(numpy.shape(sig1[:])!=numpy.shape(data2[:,0])):
        		print "\neuclidianDistance: data sets have different dimensions\n"
        		sys.exit()
    		else:
        		z1 = (sig1[:] - data2[:,0])*(sig1[:] - data2[:,0])
        		z2 = (sig2[:] - data2[:,1])*(sig2[:] - data2[:,1])
        		z3 = (sig3[:] - data2[:,2])*(sig3[:] - data2[:,2])
        		z4 = (sig4[:] - data2[:,3])*(sig4[:] - data2[:,3])
        		z5 = (sig5[:] - data2[:,4])*(sig5[:] - data2[:,4])
        		z6 = (sig6[:] - data2[:,5])*(sig6[:] - data2[:,5])
        		z7 = (sig7[:] - data2[:,6])*(sig7[:] - data2[:,6])
        		z8 = (sig8[:] - data2[:,7])*(sig8[:] - data2[:,7])
        		distance1 = numpy.sqrt(numpy.sum(z1))
        		distance2 = numpy.sqrt(numpy.sum(z2))
        		distance3 = numpy.sqrt(numpy.sum(z3))
        		distance4 = numpy.sqrt(numpy.sum(z4))
        		distance5 = numpy.sqrt(numpy.sum(z5))
        		distance6 = numpy.sqrt(numpy.sum(z6))
        		distance7 = numpy.sqrt(numpy.sum(z7))
        		distance8 = numpy.sqrt(numpy.sum(z8))
        		alldist = distance1
	
    		if alldist < 0:
        		return [None]
    		else:
        		return [distance1 + distance2 + distance3 + distance4 + distance5 + distance6 + distance7 + distance8]

    
