import numpy as np
import numba
from numba import jit
import warnings
warnings.filterwarnings('ignore')
import random
import traj_tools


numericThresh = 1E-150
logNumericThresh = np.log(numericThresh)
GammaThresh = 1E-15
eigenValueThresh = 1E-10

@jit(nopython=True)
def logsumexp(x):
    c = x.max()
    return c + np.log(np.sum(np.exp(x - c)))

@jit(nopython=True)
def maximum_likelihood_opt_weighted(ln_weights, cluster_frame_ln_likelihoods, centers, traj_data, precisions, lpdets, kabsch_thresh, kabsch_max_steps):
    # get metadata from trajectory data
    n_frames = traj_data.shape[0]
    n_atoms = traj_data.shape[1]
    n_dim = traj_data.shape[2]
    n_features = n_dim*n_atoms
    n_clusters = ln_weights.size
    # Compute the normaliztion constant and overall loglikelihood of current clustering
    log_likelihood = float(0.0)
    logNorm = np.empty(n_frames,dtype=np.float64)
    count = 0
    for i in range(n_frames):
        logNorm[i] = logsumexp(cluster_frame_ln_likelihoods[:,i]+ln_weights)
        log_likelihood += logNorm[i]
    # maximize averages, covariances/precisions, and weights
    for k in range(n_clusters):
        # use the current values for the parameters to evaluate the posterior
        # probabilities of the data to have been generanted by each gaussian
        # the following step can be numerically unstable
        loggamma = cluster_frame_ln_likelihoods[k] + ln_weights[k] - logNorm
        gamma = np.exp(loggamma).astype(np.float64)
        # gamma should be between 0 and 1
#        gamma[np.argwhere(gamma > 1.0)] = 1.0
        # will only use frames that have greater than GammaThresh weight
        gamma_indeces = np.argwhere(gamma > GammaThresh).flatten()
        # update mean and covariance/precision
        #print(gamma)
        centers[k], precisions[k], lpdets[k] = traj_tools.traj_iterative_average_precision_weighted_weighted_kabsch(traj_data[gamma_indeces], gamma[gamma_indeces], centers[k], precisions[k], lpdets[k], thresh=kabsch_thresh, max_steps=kabsch_max_steps)
        # update the weights
        ln_weights[k] = np.log(np.mean(gamma))
    return centers, precisions, lpdets, ln_weights, log_likelihood

# Expectation step
@jit(nopython=True)
def expectation_weighted(traj_data, centers, precisions, lpdets):
    # meta data
    n_clusters = centers.shape[0]
    n_frames = traj_data.shape[0]
    cluster_frame_ln_likelihoods = np.empty((n_clusters,n_frames),dtype=np.float64)
    # compute likelihood of each frame at each Gaussian
    for k in range(n_clusters):
        # align the entire trajectory to each cluster mean
        traj_data = traj_tools.traj_align_weighted_kabsch(traj_data, centers[k], precisions[k])
        cluster_frame_ln_likelihoods[k,:] = ln_multivariate_NxN_gaussian_pdf(traj_data, centers[k], precisions[k], lpdets[k])
    return cluster_frame_ln_likelihoods

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
def ln_multivariate_NxN_gaussian_pdf(x, mu, precision, lpdet):
    # metadata from arrays
    nSamples = x.shape[0]
    n_dim = x.shape[1]-1
    # compute pseudo determinant and inverse of sigma
    #lpdet, precision = pseudo_lpdet_inv(sigma)
    # compute log of normalization constant
    lnnorm = -1.5*(lpdet)
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
def weighted_sgmm_log_likelihood(traj_data, clusters):
    # meta data from inputs
    n_frames = traj_data.shape[0]
    n_clusters = np.amax(clusters) + 1
    n_atoms = traj_data.shape[1]
    n_dim = traj_data.shape[2]
    n_features = n_atoms*n_dim
    # declare arrays 
    cluster_frame_ln_likelihoods = np.empty((n_clusters,n_frames),dtype=np.float64)
    ln_weights = np.empty(n_clusters,dtype=np.float64)
    # compute likelihood of each frame at each Gaussian
    for k in range(n_clusters):
        indeces = np.argwhere(clusters == k).flatten()
        center, precision, lpdet = traj_tools.traj_iterative_average_precision_weighted_kabsch(traj_data[indeces])[1:]
        # initialize weights as populations of clusters
        ln_weights[k] = np.log(indeces.size/n_framesprecision, lpdet)
        # align the entire trajectory to each cluster mean if requested
        traj_data = traj_tools.traj_align_weighted_kabsch(traj_data, center, precision)
        cluster_frame_ln_likelihoods[k,:] = ln_multivariate_NxN_gaussian_pdf(traj_data, center, precision, lpdet)
    # compute log likelihood
    log_likelihood = 0.0
    for i in range(n_frames):
        log_likelihood += logsumexp(cluster_frame_ln_likelihoods[:,i]+ln_weights)
    return log_likelihood


@jit(nopython=True)
def compute_bic(n_atoms, n_dim, n_clusters, n_frames, log_likelihood):
    k = n_clusters*(n_atoms*n_dim + n_atoms*(n_atoms+1)/2 + 1) - 1
    return k*np.log(n_frames) - 2*log_likelihood

@jit(nopython=True)
def compute_aic(n_atoms, n_dim, n_clusters, log_likelihood):
    k = n_clusters*(n_atoms*n_dim + n_atoms*(n_atoms+1)/2 + 1) - 1
    return 2*k - 2*log_likelihood

