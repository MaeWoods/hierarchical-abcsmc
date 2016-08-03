# module defining all things kernel

# kernel_type = 1 : component-wise uniform kernels
# kernel_type = 2 : component-wise normal kernels
# kernel_type = 3 : multi-variate normal kernels
# kernel_type = 4 : multi-variate normal kernels using the nearest neighbours of the particles
# kernel_type = 4 : multi-variate normal kernels using an "optimal covariance matrix" 

import numpy
from numpy import random as rnd
from scipy.stats import norm
from abcsysbio import statistics


# kernel is a list of length 3 such that :
# kernel[0] contains the index of the non-constant paramameters
# kernel[1] contains the informations required to build the kernels in function getKernels - those informations are defined in the input file
# kernel[2] contains the kernel (list, matrix or dictionnary) once it has been built


# Compute the kernels WITHIN a model by examining the previous population of particles
# populations, weights refers to particles and weights from previous population for one model
def getKernel(kernel_type, kernel, population, weights):
    pop_size = population.shape[0]
    npar = population.shape[1]

    if kernel_type == 1:
    # component-wise uniform kernels
        tmp=list()
	if pop_size == 1:
            print "WARNING: getKernel : only one particle so adaptation is not possible"
            for param in kernel[0]:
                tmp.append([-1, 1])	
        else: 
            for param in kernel[0]:
                minimum=min(population[:,param])
                maximum=max(population[:,param])
                scale=(maximum-minimum)
                tmp.append([-scale/2.0,scale/2.0])
        kernel[2]=tmp
	# kernel[2] is a list of length the number of non-constant parameters. Each element of the list contains the inf and sup bound of the uniform kernel.
    
    return kernel




# Here params refers to one particle
# The function changes params in place and returns the probability (which may be zero)
def perturbParticle(params, priors, kernel, kernel_type, special_cases):
		np = len( priors )
		prior_prob = 1

		ind=0
			# n refers to the index of the parameter (integer between 0 and np-1)
			# ind is an integer between 0 and len(kernel[0])-1 which enables to determine the kernel to use
		for n in kernel[0]:
				params[n] = params[n] + rnd.uniform(low=kernel[2][ind][0],high=kernel[2][ind][1])
				ind+=1
		

		# compute the likelihood
		prior_prob=1
		for n in range(np):
			x = 1.0
			#if priors[n][0]==1: 
			#    x=statistics.getPdfGauss(priors[n][1], numpy.sqrt(priors[n][2]), params[n])
				# if we do not care about the value of prior_prob, then here: x=1.0

			if priors[n][0]==2: 
				x=statistics.getPdfUniform(priors[n][1],priors[n][2],params[n])

			#if priors[n][0]==3: 
			#    x=statistics.getPdfLognormal(priors[n][1],priors[n][2],params[n])
				# if we do not care about the value of prior_prob, then here: x=1.0 if params[n]>=0 and 0 otherwise

			if priors[n][0]==4: 
				x=statistics.getPdfUniform(priors[n][1],priors[n][2],params[n])
				
			if priors[n][0]==6: 
				x=statistics.getPdfUniform(priors[n][1],priors[n][2],params[n])

			prior_prob = prior_prob*x

		return prior_prob


# Here params and params0 refer to one particle each.
# Auxilliary is a vector size of nparameters
def getPdfParameterKernel(params, params0, priors, kernel, auxilliary, kernel_type):
    if kernel_type==1:
        prob=1
	# n refers to the index of the parameter (integer between 0 and np-1)
	# ind is an integer between 0 and len(kernel[0])-1 which enables to determine the kernel to use
	ind=0
        for n in kernel[0]:
	    kern = statistics.getPdfUniform(params0[n]+kernel[2][ind][0],params0[n]+kernel[2][ind][1], params[n])
            prob=prob*kern
	    ind += 1
        return prob

    

# Here models and parameters refer to the whole population
def getAuxilliaryInfo(kernel_type, models, parameters, model_objs, kernel ):
    nparticles = len(parameters)
    ret = []

    for k in range(nparticles):
        ##print 'model:', models[k]
        this_prior = model_objs[models[k]].prior
        this_kernel = kernel[models[k]]
        nparam = model_objs[models[k]].nparameters
        if kernel_type == 2:
            ret.append( [1.0 for n in range(nparam)] )
	    # n refers to the index of the parameter (integer between 0 and np-1)
	    # ind is an integer between 0 and len(kernel[0])-1 which enables to determine the kernel to use
	    ind=0
	    if not(len(this_kernel[2])==1):
	        for n in this_kernel[0]:
                    # if prior is uniform
		    if this_prior[n][0]==2:
		    	mean = parameters[k][n]
                        scale = numpy.sqrt(this_kernel[2][ind])
                        ret[k][n] = norm.cdf(this_prior[n][2],mean,scale) - norm.cdf(this_prior[n][1],mean,scale)
                    # if prior is normal, no truncation required
		    if this_prior[n][0]==1:
			ret[k][n] = 1
                    # if prior is lognormal, trucation for the negative values
		    if this_prior[n][0]==3:
		    	mean = parameters[k][n]
                        scale = numpy.sqrt(this_kernel[2][ind])
                        ret[k][n] = 1 - norm.cdf(0,mean,scale)			
		    ind+=1
        elif kernel_type==3:
            up=list()
            low=list()
            mean=list()
            for n in this_kernel[0]:
		if this_prior[n][0]==2:
                    low.append(this_prior[n][1])
                    up.append(this_prior[n][2])
		if this_prior[n][0]==1:
		    low.append(-float('inf'))
		    up.append(float('inf'))
		if this_prior[n][0]==3:
		    low.append(0)
		    up.append(float('inf'))
		mean.append(parameters[k][n])
            scale=this_kernel[2]
            ret.append(statistics.mvnormcdf(low,up,mean, scale))
        elif (kernel_type==4 or kernel_type==5):
            up=list()
            low=list()
            mean=list()
            for n in this_kernel[0]:
		if this_prior[n][0]==2:
                    low.append(this_prior[n][1])
                    up.append(this_prior[n][2])
		if this_prior[n][0]==1:
		    low.append(-float('inf'))
		    up.append(float('inf'))
		if this_prior[n][0]==3:
		    low.append(0)
		    up.append(float('inf'))
                mean.append(parameters[k][n])
            cur_part=list()
            for n in range(nparam):
                cur_part.append(parameters[k][n])
            D = this_kernel[2]
            scale=D[str(cur_part)]
            ret.append(statistics.mvnormcdf(low,up,mean, scale))                
        else:
            ret = [0 for i in  range(nparticles)]
    return ret






