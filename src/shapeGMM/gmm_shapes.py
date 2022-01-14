import numpy as np
import numba
from numba import jit
import warnings
warnings.filterwarnings('ignore')
from sklearn import metrics
import random
# the following are local libraries
from . import _traj_tools as traj_tools
from . import _kmeans_shapes as kmeans_shape
from . import _gmm_shapes_uniform_library as gmm_shapes_uniform_library
from . import _gmm_shapes_weighted_library as gmm_shapes_weighted_library

# class
class ShapeGMM:
    """
    ShapeGMM is a class that can be used to perform Gaussian Mixture Model clustering in size-and-shape space.
    The class is designed to mimic similar clustering methods implemented in sklearn.  The model is first initialized
    and then fit with supplied data.  Fit parameters for the model include average structures and (co)variances.
    Once fit, the model can be used to predict clustering on an alternative (but same feature space size) data set.


    Author: Martin McCullagh
    Date: 9/27/2021
    """
    def __init__(self, n_clusters, log_thresh=1E-3,max_steps=200, init_cluster_method="random", init_iter=5, kabsch_thresh=1E-1, kabsch_max_steps=500, verbose=False):
        """
        Initialize size-and-shape GMM.
        n_clusters (required)   - integer number of clusters must be input
        log_thresh              - float threshold in log likelihood difference to determine convergence. Default value is 1e-3.
        max_steps               - integer maximum number of steps that the GMM procedure will do.  Default is 200.
        init_cluster_method     - string dictating how to initialize clusters.  Understood values include 'uniform', 'kmeans' and 'random'.  Default is 'random'.
        init_iter               - integer dictating number of iterations done to initialize for 'random'.  Default is 5.
        kabsch_thresh           - float dictating convergence criteria for each alignment step.  Default value is 1e-1.
        kabsch_max_steps        - integer dictating maximum number of allowed iterations in each alignment step. Default is 500.
        verbose                 - boolean dictating whether to print various things at every step. Defulat is False.
        """
        
        self.n_clusters = n_clusters                            # integer
        self.log_thresh = log_thresh                            # float
        self.max_steps = max_steps                              # integer
        self.init_cluster_method = init_cluster_method          # string
        self.init_iter = init_iter                              # integer
        self.kabsch_thresh = kabsch_thresh                      # float
        self.kabsch_max_steps = kabsch_max_steps                # integer
        self.verbose = verbose                                  # boolean
        self.init_clusters_flag = False                         # boolean tracking if clusters have been initialized or not.
        self.gmm_uniform_flag = False                           # boolean tracking if uniform GMM has been fit.
        self.gmm_weighted_flag = False                          # boolean tracking if weighted GMM has been fit.

    # initialize clusters
    def init_clusters(self,traj_data, clusters=[]):
        
        # get metadata
        self.n_frames = int(traj_data.shape[0])
        self.n_atoms = traj_data.shape[1]
        self.n_dim = traj_data.shape[2]
        self.n_features = self.n_dim*self.n_atoms
        if (self.verbose == True):
            # print metadata to log
            print("Number of frames being analyzed:", self.n_frames)
            print("Number of particles being analyzed:", self.n_atoms)
            print("Number of dimensions (must be 3):", self.n_dim)
            print("Initializing clustering using method:", self.init_cluster_method)
        # declare clusters
        self.clusters = np.zeros(self.n_frames,dtype=np.int)

        # Center and align the entire trajectory to start using uniform Kabsch
        traj_data = traj_tools.traj_iterative_average(traj_data)[1]

        # make initial clustering based on input user choice (default is random)
        if (self.init_cluster_method == "uniform"):
            for i in range(self.n_frames):
                self.clusters[i] = i*self.n_clusters // self.n_frames
        elif (self.init_cluster_method == "kmeans"):
            kmeans = kmeans_shapes.kmeans_shape(self.n_clusters,max_steps=self.max_steps)
            kmeans.fit(traj_data)
            self.clusters = kmeans.clusters
        elif (self.init_cluster_method == "read"):
            # should affirm that there are n_frames clusters
            self.clusters = clusters
        else: # default is random
            for i in range(self.init_iter):
                init_clusters = gmm_shapes_uniform_library.init_random(traj_data,self.n_clusters)
                logLik = gmm_shapes_uniform_library.uniform_sgmm_log_likelihood(traj_data,init_clusters)
                if (i==0 or logLik > maxLogLik):
                    maxLogLik = logLik
                    self.clusters = init_clusters
        # clusters have been initialized
        self.init_clusters_flag = True
        # 
        return traj_data
        
    # uniform fit
    def fit_uniform(self,traj_data, clusters = []):
        # Initialize clusters if they have not been already
        if (self.init_clusters_flag == False):
            traj_data = self.init_clusters(traj_data, clusters)
        # declare some important arrays for the model
        self.centers = np.empty((self.n_clusters,self.n_atoms,self.n_dim),dtype=np.float64)
        self.var = np.empty(self.n_clusters,dtype=np.float64)
        self.likelihood = np.empty((self.n_clusters,self.n_frames),dtype=np.float64)
        self.weights = np.empty(self.n_clusters,dtype=np.float64)
        self.cluster_frame_ln_likelihoods = np.empty((self.n_clusters,self.n_frames),dtype=np.float64)
        self.ln_weights = np.empty(self.n_clusters,dtype=np.float64)
    
        # compute average and covariance of initial clustering
        for k in range(self.n_clusters):
            indeces = np.argwhere(self.clusters == k).flatten()
            # initialize weights as populations of clusters
            self.weights[k] = indeces.size/self.n_frames
            self.centers[k],self.var[k] = traj_tools.traj_iterative_average_var(traj_data[indeces])
        if (self.verbose == True):
            print("Weights from initial clusters in fit_uniform:", self.weights)
        self.ln_weights = np.log(self.weights)
    
        # perform Expectation Maximization
        logLikeDiff = 2*self.log_thresh
        step = 0
        log_likelihoodArray = np.empty(self.max_steps,dtype=np.float64)
        while step < self.max_steps and logLikeDiff > self.log_thresh:
            # Expectation step
            self.cluster_frame_ln_likelihoods = gmm_shapes_uniform_library.expectation_uniform(traj_data, self.centers, self.var)
            # Maximum Likelihood Optimization
            self.centers, self.var, self.ln_weights, log_likelihoodArray[step] = gmm_shapes_uniform_library.maximum_likelihood_opt_uniform(self.ln_weights, self.cluster_frame_ln_likelihoods, self.centers, traj_data)
            if (self.verbose == True):
                print(step, np.exp(self.ln_weights), log_likelihoodArray[step])
            # compute convergence criteria
            if step>0:
                logLikeDiff = np.abs(log_likelihoodArray[step] - log_likelihoodArray[step-1])
            step += 1
        # recompute weights
        self.weights = np.exp(self.ln_weights).astype(np.float64)
        self.weights_uniform = np.copy(self.weights)
        # assign clusters based on largest likelihood 
        self.clusters = np.argmax(self.cluster_frame_ln_likelihoods, axis = 0)
        self.clusters_uniform = np.copy(self.clusters)
        # save log_likelihood
        self.log_likelihood = log_likelihoodArray[step-1] 
        self.log_likelihood_uniform = self.log_likelihood
        # iteratively align averages
        self.centers, self.globalCenter = traj_tools.traj_iterative_average_weighted(self.centers,self.weights)
        self.centers_uniform = np.copy(self.centers)
        for k in range(self.n_clusters):
            indeces = np.argwhere(self.clusters == k).flatten()
            traj_data[indeces] = traj_tools.traj_align(traj_data[indeces],self.centers[k])
        # Compute clustering scores
        self.silhouette_score = metrics.silhouette_score(traj_data.reshape(self.n_frames,self.n_features), self.clusters)
        self.ch_score = metrics.calinski_harabasz_score(traj_data.reshape(self.n_frames,self.n_features), self.clusters)
        self.db_score = metrics.davies_bouldin_score(traj_data.reshape(self.n_frames,self.n_features), self.clusters)
        self.bic = gmm_shapes_uniform_library.compute_bic_uniform(self.n_features-6, self.n_clusters, self.n_frames, self.log_likelihood_uniform)
        self.bic_uniform = self.bic 
        self.aic = gmm_shapes_uniform_library.compute_aic_uniform(self.n_features-6, self.n_clusters, self.log_likelihood_uniform)
        self.aic_uniform = self.aic
        # uniform has been performed
        self.gmm_uniform_flag = True
        # return aligned trajectory
        return traj_data

    # weighted fit
    def fit_weighted(self,traj_data,clusters=[]):
        
        # make sure clusters have been initialized
        if (self.init_clusters_flag == False):
            traj_data = self.init_clusters(traj_data, clusters)
            # declare some arrays 
            self.centers = np.empty((self.n_clusters,self.n_atoms,self.n_dim),dtype=np.float64)
            self.likelihood = np.empty((self.n_clusters,self.n_frames),dtype=np.float64)
            self.weights = np.empty(self.n_clusters,dtype=np.float64)
            self.cluster_frame_ln_likelihoods = np.empty((self.n_clusters,self.n_frames),dtype=np.float64)
            self.ln_weights = np.empty(self.n_clusters,dtype=np.float64)
            
        # declare precision matrices (inverse covariances)
        self.precisions = np.empty((self.n_clusters,self.n_atoms,self.n_atoms),dtype=np.float64)
        # declare array for log determinants for each clusters
        self.lpdets = np.empty(self.n_clusters, dtype=np.float64)

        # compute averages and precisions of initial clustering
        for k in range(self.n_clusters):
            indeces = np.argwhere(self.clusters == k).flatten()
            self.weights[k] = indeces.size / self.n_frames
            self.centers[k], self.precisions[k], self.lpdets[k] = traj_tools.traj_iterative_average_precision_weighted_kabsch(traj_data[indeces],thresh=self.kabsch_thresh,max_steps=self.kabsch_max_steps)[1:]
        # 
        if (self.verbose == True):
            print("Weights from initial clusters in fit_weighted:", self.weights)
        self.ln_weights = np.log(self.weights)
        # perform Expectation Maximization
        logLikeDiff = 2*self.log_thresh
        step = 0
        log_likelihoodArray = np.empty(self.max_steps,dtype=np.float64)
        while step < self.max_steps and logLikeDiff > self.log_thresh:
            # Expectation step
            self.cluster_frame_ln_likelihoods = gmm_shapes_weighted_library.expectation_weighted(traj_data, self.centers, self.precisions, self.lpdets)
            # Maximum Likelihood Oprecisionptimization
            self.centers, self.precisions, self.lpdets, self.ln_weights, log_likelihoodArray[step] = gmm_shapes_weighted_library.maximum_likelihood_opt_weighted(self.ln_weights, self.cluster_frame_ln_likelihoods, self.centers, traj_data, self.precisions, self.lpdets, self.kabsch_thresh, self.kabsch_max_steps)
            if (self.verbose == True):
                print(step, np.exp(self.ln_weights), log_likelihoodArray[step])
            # compute convergence criteria
            if step>0:
                logLikeDiff = np.abs(log_likelihoodArray[step] - log_likelihoodArray[step-1])
            step += 1
        # recompute weights
        self.weights = np.exp(self.ln_weights).astype(np.float64)
        self.weights_weighted = np.copy(self.weights)
        # assign clusters based on largest likelihood 
        self.clusters = np.argmax(self.cluster_frame_ln_likelihoods, axis = 0)
        self.clusters_weighted = np.copy(self.clusters)
        # save log_likelihood
        self.log_likelihood = log_likelihoodArray[step-1] 
        self.log_likelihood_weighted = self.log_likelihood
        # iteratively align averages (note this in using uniform Kabsch)
        self.centers, self.globalCenter = traj_tools.traj_iterative_average_weighted(self.centers,self.weights)
        self.centers_weighted = np.copy(self.centers)
        for k in range(self.n_clusters):
            indeces = np.argwhere(self.clusters == k).flatten()
            traj_data[indeces] = traj_tools.traj_align_weighted_kabsch(traj_data[indeces],self.centers[k],self.precisions[k])
        # Compute clustering scores
        self.silhouette_score = metrics.silhouette_score(traj_data.reshape(self.n_frames,self.n_features), self.clusters)
        self.ch_score = metrics.calinski_harabasz_score(traj_data.reshape(self.n_frames,self.n_features), self.clusters)
        self.db_score = metrics.davies_bouldin_score(traj_data.reshape(self.n_frames,self.n_features), self.clusters)
        self.bic = gmm_shapes_weighted_library.compute_bic(self.n_atoms, self.n_dim, self.n_clusters, self.n_frames, self.log_likelihood)
        self.bic_weighted = self.bic
        self.aic = gmm_shapes_weighted_library.compute_aic(self.n_atoms, self.n_dim, self.n_clusters, self.log_likelihood)
        self.aic_weighted = self.aic
        # weighted has been performed
        self.gmm_weighted_flag = True
        # return aligned trajectory
        return traj_data

    # predict clustering of provided data based on prefit parameters from fit_weighted
    def predict_weighted(self,traj_data):
        if self.gmm_weighted_flag == True:
            # get metadata from trajectory data
            n_frames = traj_data.shape[0]
            # declare likelihood array
            cluster_frame_ln_likelihoods = np.empty((self.n_clusters,n_frames),dtype=np.float64)
            # make sure trajectory is centered
            traj_data = traj_tools.traj_remove_cog_translation(traj_data)
            # Expectation step
            for k in range(self.n_clusters):
                # align the entire trajectory to each cluster mean
                traj_data = traj_tools.traj_align_weighted_kabsch(traj_data,self.centers[k],self.precisions[k])
                cluster_frame_ln_likelihoods[k,:] = gmm_shapes_weighted_library.ln_multivariate_NxN_gaussian_pdf(traj_data, self.centers[k], self.precisions[k], self.lpdets[k])
            # compute log likelihood
            log_likelihood = 0.0
            for i in range(n_frames):
                log_likelihood += gmm_shapes_weighted_library.logsumexp(cluster_frame_ln_likelihoods[:,i] + self.ln_weights)
            # assign clusters based on largest likelihood (probability density)
            clusters = np.argmax(cluster_frame_ln_likelihoods, axis = 0)
            # center trajectory around averages
            for k in range(self.n_clusters):
                indeces = np.argwhere(clusters == k).flatten()
                traj_data[indeces] = traj_tools.traj_align_weighted_kabsch(traj_data[indeces],self.centers[k],self.precisions[k])
            return clusters, traj_data, log_likelihood
        else:
            print("Weighted shape-GMM must be fitted before you can predict.")

    # predict clustering of provided data based on prefit parameters from fit_uniform
    def predict_uniform(self,traj_data):
        if self.gmm_uniform_flag == True:
            # get metadata from trajectory data
            n_frames = traj_data.shape[0]
            # declare likelihood array
            cluster_frame_ln_likelihoods = np.empty((self.n_clusters,n_frames),dtype=np.float64)
            # make sure trajectory is centered
            traj_data = traj_tools.traj_remove_cog_translation(traj_data)
            # Expectation step
            for k in range(self.n_clusters):
                # align the entire trajectory to each cluster mean
                traj_data = traj_tools.traj_align(traj_data,self.centers[k])
                cluster_frame_ln_likelihoods[k,:] = gmm_shapes_uniform_library.ln_spherical_gaussian_pdf(traj_data.reshape(n_frames,self.n_features), self.centers[k].reshape(self.n_features), self.var[k])
            # compute log likelihood
            log_likelihood = 0.0
            for i in range(n_frames):
                log_likelihood += gmm_shapes_uniform_library.logsumexp(cluster_frame_ln_likelihoods[:,i] + self.ln_weights)
            # assign clusters based on largest likelihood (probability density)
            clusters = np.argmax(cluster_frame_ln_likelihoods, axis = 0)
            # center trajectory around averages
            for k in range(self.n_clusters):
                indeces = np.argwhere(clusters == k).flatten()
                traj_data[indeces] = traj_tools.traj_align(traj_data[indeces],self.centers[k])
            return clusters, traj_data, log_likelihood
        else:
            print("Uniform shape-GMM must be fitted before you can predict.")

