import numpy as np
import numba
from numba import jit
import warnings
warnings.filterwarnings('ignore')
import random
from . import _traj_tools as traj_tools
#from scipy.special import logsumexp

NUMERIC_THRESH = 1E-150
LOG_NUMERIC_THRESH = np.log(NUMERIC_THRESH)
GAMMA_THRESH = 1E-15
eigenValueThresh = 1E-10



@jit(nopython=True)
def logsumexp(x):
    c = x.max()
    return c + np.log(np.sum(np.exp(x - c)))

##
@jit(nopython=True)
def uniform_sgmm_log_likelihood(traj_data,clusters):
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
        center, var = traj_tools.traj_iterative_average_var(traj_data[indeces])
        # initialize weights as populations of clusters
        ln_weights[k] = np.log(indeces.size/n_frames)
        # align the entire trajectory to each cluster mean if requested
        traj_data = traj_tools.traj_align(traj_data,center)
        cluster_frame_ln_likelihoods[k,:] = ln_spherical_gaussian_pdf(traj_data.reshape(n_frames,n_features), center.reshape(n_features), var)
    # compute log likelihood
    log_likelihood = 0.0
    for i in range(n_frames):
        log_likelihood += logsumexp(cluster_frame_ln_likelihoods[:,i]+ln_weights[k])
    return log_likelihood
##
@jit
def init_random(traj_data, n_clusters):
    # meta data from inputs
    n_frames = traj_data.shape[0]
    # declare arrayes
    dists = np.empty((n_frames,n_clusters))
    clustersPass = False
    while clustersPass == False:
        clustersPass = True
        randFrames = random.sample(range(n_frames),n_clusters)
        centers = np.copy(traj_data[randFrames])
        # make initial clustering based on RMSD distance from centers
        # measure distance to every center
        for i in range(n_frames):
            for k in range(n_clusters):
                dists[i,k] = traj_tools.rmsd_kabsch(centers[k],traj_data[i])
        # assign frame to nearest center
        clusters = np.argmin(dists, axis = 1)
        for k in range(n_clusters):
            indeces = np.argwhere(clusters == k).flatten()
            if indeces.size == 0:
                clustersPass = False
                break
    return clusters

##
@jit(nopython=True)
def maximum_likelihood_opt_uniform(ln_weights,cluster_frame_ln_likelihoods,centers,traj_data):
    # get metadata from trajectory data
    n_frames = traj_data.shape[0]
    n_atoms = traj_data.shape[1]
    n_dim = traj_data.shape[2]
    n_features = n_dim*n_atoms
    n_clusters = ln_weights.size
    # declare cluster variances
    var = np.empty(n_clusters,dtype=np.float64)
    log_likelihood = float(0.0)
    log_norm = np.empty(n_frames,dtype=np.float64)
    count = 0
    # compute log likelihood and gamma normalization
    for i in range(n_frames):
        #log_likelihood += logsumexp(cluster_frame_ln_likelihoods[:,i]+ln_weights[k])
        log_norm[i] = logsumexp(cluster_frame_ln_likelihoods[:,i]+ln_weights)
        log_likelihood += log_norm[i]
    # update averages, variances and weights
    for k in range(n_clusters):
        # use the current values for the parameters to evaluate the posterior
        # probabilities of the data to have been generanted by each gaussian
        # the following step can be numerically unstable
        loggamma = cluster_frame_ln_likelihoods[k] + ln_weights[k] - log_norm
        #newIndeces = np.argwhere(loggamma > LOG_NUMERIC_THRESH)
        gamma = np.exp(loggamma).astype(np.float64)
        # gamma should be between 0 and 1
#        gamma[np.argwhere(gamma > 1.0)] = 1.0
        # will only use frames that have greater than GAMMA_THRESH weight
        gamma_indeces = np.argwhere(gamma > GAMMA_THRESH).flatten()
        # update mean and variance
        centers[k], var[k] = traj_tools.traj_iterative_average_var_weighted(traj_data[gamma_indeces], gamma[gamma_indeces], centers[k])
        # update the weights
        ln_weights[k] = np.log(np.mean(gamma))
        #
        #print(gamma)
    return centers, var, ln_weights, log_likelihood

# Expectation step
@jit
def expectation_uniform(traj_data, centers, var):
    # meta data
    n_clusters = centers.shape[0]
    n_frames = traj_data.shape[0]
    n_atoms = traj_data.shape[1]
    n_dim = traj_data.shape[2]
    n_features = n_atoms*n_dim
    cluster_frame_ln_likelihoods = np.empty((n_clusters,n_frames),dtype=np.float64)
    # compute likelihood of each frame at each Gaussian
    for k in range(n_clusters):
        # align the entire trajectory to each cluster mean if requested
        traj_data = traj_tools.traj_align(traj_data,centers[k])
        cluster_frame_ln_likelihoods[k,:] = ln_spherical_gaussian_pdf(traj_data.reshape(n_frames,n_features), centers[k].reshape(n_features), var[k])
    return cluster_frame_ln_likelihoods


@jit(nopython=True)
def ln_spherical_gaussian_pdf(x, mu, sigma):
    n_samples = x.shape[0]
    n_dim = x.shape[1]-3
#    lnnorm = -0.5*n_dim*(np.log(2.0*np.pi*sigma))
    lnnorm = -0.5*n_dim*(np.log(sigma))
    mvG = np.empty(n_samples,dtype=np.float64)
    multiplier = -0.5/sigma
    for i in range(n_samples):
        diffV = x[i] - mu
        mvG[i] = multiplier*np.dot(diffV,diffV) + lnnorm
    return mvG

@jit(nopython=True)
def compute_bic_uniform(n_features, n_clusters, n_frames, log_likelihood):
    k = n_clusters*(n_features + 1 + 1) - 1
    return k*np.log(n_frames) - 2*log_likelihood

@jit(nopython=True)
def compute_aic_uniform(n_features, n_clusters, log_likelihood):
    k = n_clusters*(n_features + 1 + 1) - 1
    return 2*k - 2*log_likelihood

