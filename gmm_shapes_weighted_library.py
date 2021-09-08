import numpy as np
import numba
from numba import jit
import warnings
warnings.filterwarnings('ignore')
import random
import traj_tools


numericThresh = 1E-150
logNumericThresh = np.log(numericThresh)
gammaThresh = 1E-15
eigenValueThresh = 1E-10

#@jit
def maximum_likelihood_opt_weighted(lnWeights,lnLikelihood,centers,trajData,covar,kabschThresh, kabschMaxSteps):
    # get metadata from trajectory data
    nFrames = trajData.shape[0]
    nAtoms = trajData.shape[1]
    nDim = trajData.shape[2]
    nFeatures = nDim*nAtoms
    nClusters = lnWeights.size
    # Compute the normaliztion constant and overall loglikelihood
    logLikelihood = float(0.0)
    logNorm = []
    indeces = []
    count = 0
    for i in range(nFrames):
        normalization = np.float128(0.0)
        for k in range(nClusters):
            normalization += np.exp(np.float128(lnLikelihood[k,i] + lnWeights[k]))
        if (normalization > numericThresh) :
            logNorm.append(np.log(normalization))
            indeces.append(i)
            logLikelihood += np.float64(logNorm[count])
            count += 1
    nNonzeroFrames = len(indeces)
    #print("Number of nonzero frames:", nNonzeroFrames, " out of ", nFrames)
    indeces = np.array(indeces,dtype=np.int)
    logNorm = np.array(logNorm,dtype=np.float64)
    for k in range(nClusters):
        # use the current values for the parameters to evaluate the posterior
        # probabilities of the data to have been generanted by each gaussian
        # the following step can be numerically unstable
        loggamma = lnLikelihood[k,indeces] + lnWeights[k] - logNorm
        #newIndeces = np.argwhere(loggamma > logNumericThresh)
        gamma = np.exp(loggamma).astype(np.float64)
        #print(gamma)
        # gamma should be between 0 and 1
        gamma[np.argwhere(gamma > 1.0)] = 1.0
        # will only use frames that have greater than gammaThresh weight
        gamma_indeces = np.argwhere(gamma > gammaThresh).flatten()
        # update mean and variance
        #print(gamma)
        centers[k], covar[k] = traj_tools.traj_iterative_average_covar_weighted_weighted_kabsch(trajData[indeces[gamma_indeces]], gamma[gamma_indeces], centers[k], covar[k],thresh=kabschThresh,maxSteps=kabschMaxSteps)
        # update the weights
        lnWeights[k] = np.log(np.mean(gamma))
    return centers, covar, lnWeights, logLikelihood

# Expectation step
@jit
def expectation_weighted(trajData, centers, covar):
    # meta data
    nClusters = centers.shape[0]
    nFrames = trajData.shape[0]
#    nAtoms = trajData.shape[1]
#    nDim = trajData.shape[2]
#    nFeatures = nAtoms*nDim
    lnLikelihood = np.empty((nClusters,nFrames),dtype=np.float64)
    # compute likelihood of each frame at each Gaussian
    for k in range(nClusters):
        # align the entire trajectory to each cluster mean
        trajData = traj_tools.traj_align_weighted_kabsch(trajData,centers[k],covar[k])
        lnLikelihood[k,:] = ln_multivariate_NxN_gaussian_pdf(trajData, centers[k], covar[k])
    return lnLikelihood

@jit(nopython=True)
def pseudo_lpdet_inv(sigma):
    N = sigma.shape[0]
    e, v = np.linalg.eigh(sigma)
    #precision = np.zeros(sigma.shape,dtype=np.float128)
    precision = np.zeros(sigma.shape,dtype=np.float64)
    lpdet = 0.0
    for i in range(N):
        if (e[i] > eigenValueThresh):
            lpdet += np.log(e[i])
            precision += 1.0/e[i]*np.outer(v[:,i],v[:,i])
    #pdet = np.exp(lpdet)
    return lpdet, precision

@jit(nopython=True)
def ln_multivariate_NxN_gaussian_pdf(x, mu, sigma):
    # metadata from arrays
    nSamples = x.shape[0]
    nDim = x.shape[1]-1
    # compute pseudo determinant and inverse of sigma
    lpdet, precision = pseudo_lpdet_inv(sigma)
    # compute log of normalization constant
    lnnorm = -1.5*(nDim*np.log(2.0*np.pi)+lpdet)
    # declare array of log multivariate Gaussian values - one for each sample
    mvG = np.zeros(nSamples,dtype=np.float64)
    for i in range(nSamples):
        for j in range(3):
            diff = x[i,:,j] - mu[:,j]
            mvG[i] += np.dot(diff,np.dot(precision,diff))
        mvG[i] *= -0.5
        mvG[i] += lnnorm
    return mvG

@jit(nopython=True)
def weighted_sgmm_log_likelihood(trajData,clusters):
    # meta data from inputs
    nFrames = trajData.shape[0]
    nClusters = np.amax(clusters) + 1
    nAtoms = trajData.shape[1]
    nDim = trajData.shape[2]
    nFeatures = nAtoms*nDim
    # declare arrays 
    lnLikelihood = np.empty((nClusters,nFrames),dtype=np.float64)
    lnWeights = np.empty(nClusters,dtype=np.float64)
    # compute likelihood of each frame at each Gaussian
    for k in range(nClusters):
        indeces = np.argwhere(clusters == k).flatten()
        center, covar = traj_tools.traj_iterative_average_covar_weighted_kabsch_v02(trajData[indeces])[1:]
        # initialize weights as populations of clusters
        lnWeights[k] = np.log(indeces.size/nFrames)
        # align the entire trajectory to each cluster mean if requested
        trajData = traj_tools.traj_align_weighted_kabsch(trajData,center,covar)
        lnLikelihood[k,:] = ln_multivariate_NxN_gaussian_pdf(trajData, center, covar)
    # compute log likelihood
    logLikelihood = 0.0
    for i in range(nFrames):
        normalization = 0.0
        for k in range(nClusters):
            normalization += np.exp((lnLikelihood[k,i] + lnWeights[k]))
        logLikelihood += np.log(normalization)
    return logLikelihood


@jit(nopython=True)
def compute_bic(nAtoms, nDim, nClusters, nFrames, logLikelihood):
    k = nClusters*(nAtoms*nDim + nAtoms*(nAtoms+1)/2 + 1) - 1
    return k*np.log(nFrames) - 2*logLikelihood

@jit(nopython=True)
def compute_aic(nAtoms, nDim, nClusters, logLikelihood):
    k = nClusters*(nAtoms*nDim + nAtoms*(nAtoms+1)/2 + 1) - 1
    return 2*k - 2*logLikelihood

