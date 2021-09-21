import numpy as np
import numba
from numba import jit
import warnings
warnings.filterwarnings('ignore')
import random
import traj_tools
#from scipy.special import logsumexp

numericThresh = 1E-150
logNumericThresh = np.log(numericThresh)
gammaThresh = 1E-15
eigenValueThresh = 1E-10



@jit(nopython=True)
def logsumexp(x):
    c = x.max()
    return c + np.log(np.sum(np.exp(x - c)))

##
@jit(nopython=True)
def uniform_sgmm_log_likelihood(trajData,clusters):
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
        center, var = traj_tools.traj_iterative_average_var(trajData[indeces])
        # initialize weights as populations of clusters
        lnWeights[k] = np.log(indeces.size/nFrames)
        # align the entire trajectory to each cluster mean if requested
        trajData = traj_tools.traj_align(trajData,center)
        lnLikelihood[k,:] = ln_spherical_gaussian_pdf(trajData.reshape(nFrames,nFeatures), center.reshape(nFeatures), var)
    # compute log likelihood
    logLikelihood = 0.0
    for i in range(nFrames):
        logLikelihood += logsumexp(lnLikelihood[:,i]+lnWeights[k])
    return logLikelihood
##
@jit
def init_random(trajData, nClusters):
    # meta data from inputs
    nFrames = trajData.shape[0]
    # declare arrayes
    dists = np.empty((nFrames,nClusters))
    clustersPass = False
    while clustersPass == False:
        clustersPass = True
        randFrames = random.sample(range(nFrames),nClusters)
        centers = np.copy(trajData[randFrames])
        # make initial clustering based on RMSD distance from centers
        # measure distance to every center
        for i in range(nFrames):
            for k in range(nClusters):
                dists[i,k] = traj_tools.rmsd_kabsch(centers[k],trajData[i])
        # assign frame to nearest center
        clusters = np.argmin(dists, axis = 1)
        for k in range(nClusters):
            indeces = np.argwhere(clusters == k).flatten()
            if indeces.size == 0:
                clustersPass = False
                break
    return clusters

##
@jit(nopython=True)
def maximum_likelihood_opt_uniform(lnWeights,lnLikelihood,centers,trajData):
    # get metadata from trajectory data
    nFrames = trajData.shape[0]
    nAtoms = trajData.shape[1]
    nDim = trajData.shape[2]
    nFeatures = nDim*nAtoms
    nClusters = lnWeights.size
    # declare cluster variances
    var = np.empty(nClusters,dtype=np.float64)
    logLikelihood = float(0.0)
    logNorm = np.empty(nFrames,dtype=np.float64)
    count = 0
    for i in range(nFrames):
        #logLikelihood += logsumexp(lnLikelihood[:,i]+lnWeights[k])
        logNorm[i] = logsumexp(lnLikelihood[:,i]+lnWeights)
        logLikelihood += logNorm[i]
#        normalization = np.float128(0.0)
#        for k in range(nClusters):
#            normalization += np.exp(np.float128(lnLikelihood[k,i] + lnWeights[k]))
#        if (normalization > numericThresh) :
#            logNorm.append(np.log(normalization))
#            indeces.append(i)
#            logLikelihood += np.float64(logNorm[count])
#            count += 1
#    nNonzeroFrames = len(indeces)
    #print("Number of nonzero frames:", nNonzeroFrames, " out of ", nFrames)
#    indeces = np.array(indeces,dtype=np.int)
#    logNorm = np.array(logNorm,dtype=np.float64)
    for k in range(nClusters):
        # use the current values for the parameters to evaluate the posterior
        # probabilities of the data to have been generanted by each gaussian
        # the following step can be numerically unstable
        loggamma = lnLikelihood[k] + lnWeights[k] - logNorm
        #newIndeces = np.argwhere(loggamma > logNumericThresh)
        gamma = np.exp(loggamma).astype(np.float64)
        #print(gamma)
        # gamma should be between 0 and 1
#        gamma[np.argwhere(gamma > 1.0)] = 1.0
        # will only use frames that have greater than gammaThresh weight
        gamma_indeces = np.argwhere(gamma > gammaThresh).flatten()
        # update mean and variance
        centers[k], var[k] = traj_tools.traj_iterative_average_var_weighted(trajData[gamma_indeces], gamma[gamma_indeces], centers[k])
        # update the weights
        lnWeights[k] = np.log(np.mean(gamma))
        #
        #print(gamma)
    return centers, var, lnWeights, logLikelihood

# Expectation step
@jit
def expectation_uniform(trajData, centers, var):
    # meta data
    nClusters = centers.shape[0]
    nFrames = trajData.shape[0]
    nAtoms = trajData.shape[1]
    nDim = trajData.shape[2]
    nFeatures = nAtoms*nDim
    lnLikelihood = np.empty((nClusters,nFrames),dtype=np.float64)
    # compute likelihood of each frame at each Gaussian
    for k in range(nClusters):
        # align the entire trajectory to each cluster mean if requested
        trajData = traj_tools.traj_align(trajData,centers[k])
        lnLikelihood[k,:] = ln_spherical_gaussian_pdf(trajData.reshape(nFrames,nFeatures), centers[k].reshape(nFeatures), var[k])
    return lnLikelihood


@jit(nopython=True)
def ln_spherical_gaussian_pdf(x, mu, sigma):
    nSamples = x.shape[0]
    nDim = x.shape[1]-3
#    lnnorm = -0.5*nDim*(np.log(2.0*np.pi*sigma))
    lnnorm = -0.5*nDim*(np.log(sigma))
    mvG = np.empty(nSamples,dtype=np.float64)
    multiplier = -0.5/sigma
    for i in range(nSamples):
        diffV = x[i] - mu
        mvG[i] = multiplier*np.dot(diffV,diffV) + lnnorm
    return mvG

@jit(nopython=True)
def compute_bic_uniform(nFeatures, nClusters, nFrames, logLikelihood):
    k = nClusters*(nFeatures + 1 + 1) - 1
    return k*np.log(nFrames) - 2*logLikelihood

@jit(nopython=True)
def compute_aic_uniform(nFeatures, nClusters, logLikelihood):
    k = nClusters*(nFeatures + 1 + 1) - 1
    return 2*k - 2*logLikelihood

