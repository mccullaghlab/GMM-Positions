import numpy as np
import pickle
import numba
from numba import jit
import warnings
warnings.filterwarnings('ignore')
from sklearn import mixture
from sklearn import metrics
from scipy.stats import multivariate_normal
from scipy import spatial
import random
import traj_tools
import kmeans_shapes
import gmm_shapes_uniform_library
import gmm_shapes_weighted_library

numericThresh = 1E-150
logNumericThresh = np.log(numericThresh)
gammaThresh = 1E-15
eigenValueThresh = 1E-10

# class
class gmm_shape:

    def __init__(self, nClusters, logThresh=1E-3,maxSteps=200,align=True, initClustersMethod="random", initIter=5, kabschThresh=1E-10, kabschMaxSteps=500):
        
        self.nClusters = nClusters
        self.logThresh = logThresh
        self.maxSteps = maxSteps
        self.align = align
        self.initClustersMethod = initClustersMethod
        self.initIter = initIter
        self.kabschThresh = kabschThresh
        self.kabschMaxSteps = kabschMaxSteps
        self.initClusters = False
        self.gmmUniform = False
        self.gmmWeighted = False

    def fit_both(self,trajData):
        # initialize clusterings
        self.init_clusters(trajData)
        # first uniform
        self.fit_uniform(self.trajData)
        # followed by weighted
        self.fit_weighted(self.trajData)
    
    # initialize clusters
    def init_clusters(self,trajData):
        
        # get metadata
        self.nFrames = int(trajData.shape[0])
        self.nAtoms = trajData.shape[1]
        self.nDim = trajData.shape[2]
        self.nFeatures = self.nDim*self.nAtoms
        # declare clusters
        self.clusters = np.zeros(self.nFrames,dtype=np.int)

        # Center and align the entire trajectory to start using uniform Kabsch
        self.trajData = traj_tools.traj_iterative_average(trajData)[1]

        # make initial clustering based on input user choice (default is random)
        if (self.initClustersMethod == "uniform"):
            for i in range(self.nFrames):
                self.clusters[i] = i*self.nClusters // self.nFrames
        elif (self.initClustersMethod == "kmeans"):
            kmeans = kmeans_shapes.kmeans_shape(self.nClusters,maxSteps=self.maxSteps)
            kmeans.fit(self.trajData)
            self.clusters = kmeans.clusters
        else: # default is random
            for i in range(self.initIter):
                initClusters = gmm_shapes_uniform_library.init_random(self.trajData,self.nClusters)
                logLik = gmm_shapes_uniform_library.uniform_sgmm_log_likelihood(self.trajData,initClusters)
                if (i==0 or logLik > maxLogLik):
                    maxLogLik = logLik
                    self.clusters = initClusters
        # clusters have been initialized
        self.initClusters = True
        
    # uniform fit
    def fit_uniform(self,trajData):
        # make sure clusters have been initialized
        if (self.initClusters == False):
            self.init_clusters(trajData)
        # declare some important arrays for the model
        self.centers = np.empty((self.nClusters,self.nAtoms,self.nDim),dtype=np.float64)
        self.var = np.empty(self.nClusters,dtype=np.float64)
        self.likelihood = np.empty((self.nClusters,self.nFrames),dtype=np.float64)
        self.weights = np.empty(self.nClusters,dtype=np.float64)
        self.lnLikelihood = np.empty((self.nClusters,self.nFrames),dtype=np.float64)
        self.lnWeights = np.empty(self.nClusters,dtype=np.float64)
    
        # compute average and covariance of initial clustering
        for k in range(self.nClusters):
            indeces = np.argwhere(self.clusters == k).flatten()
            # initialize weights as populations of clusters
            self.weights[k] = indeces.size/self.nFrames
            self.centers[k],self.var[k] = traj_tools.traj_iterative_average_var(self.trajData[indeces])
        print("Weights from initial clusters in fit_uniform:", self.weights)
        self.lnWeights = np.log(self.weights)
    
        # perform Expectation Maximization
        logLikeDiff = 2*self.logThresh
        step = 0
        logLikelihoodArray = np.empty(self.maxSteps,dtype=np.float64)
        while step < self.maxSteps and logLikeDiff > self.logThresh:
            # Expectation step
            self.lnLikelihood = gmm_shapes_uniform_library.expectation_uniform(self.trajData, self.centers, self.var)
            # Maximum Likelihood Optimization
            self.centers, self.var, self.lnWeights, logLikelihoodArray[step] = gmm_shapes_uniform_library.maximum_likelihood_opt_uniform(self.lnWeights, self.lnLikelihood, self.centers, self.trajData)
            print(step, np.exp(self.lnWeights), logLikelihoodArray[step])
            # compute convergence criteria
            if step>0:
                logLikeDiff = np.abs(logLikelihoodArray[step] - logLikelihoodArray[step-1])
            step += 1
        # recompute weights
        self.weights = np.exp(self.lnWeights).astype(np.float64)
        self.weights_uniform = np.copy(self.weights)
        # assign clusters based on largest likelihood 
        self.clusters = np.argmax(self.lnLikelihood, axis = 0)
        self.clusters_uniform = np.copy(self.clusters)
        # save logLikelihood
        self.logLikelihood_uniform = logLikelihoodArray[step-1] 
        # iteratively align averages
        self.centers, self.globalCenter = traj_tools.traj_iterative_average_weighted(self.centers,self.weights)
        self.centers_uniform = np.copy(self.centers)
        for k in range(self.nClusters):
            indeces = np.argwhere(self.clusters == k).flatten()
            self.trajData[indeces] = traj_tools.traj_align(self.trajData[indeces],self.centers[k])
        # Compute clustering scores
        self.silhouette_score = metrics.silhouette_score(self.trajData.reshape(self.nFrames,self.nFeatures), self.clusters)
        self.ch_score = metrics.calinski_harabasz_score(self.trajData.reshape(self.nFrames,self.nFeatures), self.clusters)
        self.db_score = metrics.davies_bouldin_score(self.trajData.reshape(self.nFrames,self.nFeatures), self.clusters)
        self.bic = gmm_shapes_uniform_library.compute_bic_uniform(self.nFeatures-6, self.nClusters, self.nFrames, self.logLikelihood_uniform)
        self.bic_uniform = self.bic 
        self.aic = gmm_shapes_uniform_library.compute_aic_uniform(self.nFeatures-6, self.nClusters, self.logLikelihood_uniform)
        self.aic_uniform = self.aic
        # uniform has been performed
        self.gmmUniform = True

    def fit_weighted(self,trajData):
        
        # make sure clusters have been initialized
        if (self.initClusters == False):
            self.init_clusters(trajData)
            # declare some arrays 
            self.centers = np.empty((self.nClusters,self.nAtoms,self.nDim),dtype=np.float64)
            self.likelihood = np.empty((self.nClusters,self.nFrames),dtype=np.float64)
            self.weights = np.empty(self.nClusters,dtype=np.float64)
            self.lnLikelihood = np.empty((self.nClusters,self.nFrames),dtype=np.float64)
            self.lnWeights = np.empty(self.nClusters,dtype=np.float64)
            
        # declare covariance
        self.covar = np.empty((self.nClusters,self.nAtoms,self.nAtoms),dtype=np.float64)

        # compute average and covariance of initial clustering
        for k in range(self.nClusters):
            indeces = np.argwhere(self.clusters == k).flatten()
            self.weights[k] = indeces.size / self.nFrames
            self.centers[k], self.covar[k] = traj_tools.traj_iterative_average_covar_weighted_kabsch_v02(self.trajData[indeces],thresh=self.kabschThresh,maxSteps=self.kabschMaxSteps)[1:]
        # 
        print("Weights from initial clusters in fit_weighted:", self.weights)
        self.lnWeights = np.log(self.weights)
        # perform Expectation Maximization
        logLikeDiff = 2*self.logThresh
        step = 0
        logLikelihoodArray = np.empty(self.maxSteps,dtype=np.float64)
        while step < self.maxSteps and logLikeDiff > self.logThresh:
            # Expectation step
            self.lnLikelihood = gmm_shapes_weighted_library.expectation_weighted(self.trajData, self.centers, self.covar)
            #print(self.lnLikelihood)
            # Maximum Likelihood Optimization
            self.centers, self.covar, self.lnWeights, logLikelihoodArray[step] = gmm_shapes_weighted_library.maximum_likelihood_opt_weighted(self.lnWeights, self.lnLikelihood, self.centers, self.trajData, self.covar,self.kabschThresh, self.kabschMaxSteps)
            print(step, np.exp(self.lnWeights), logLikelihoodArray[step])
            # compute convergence criteria
            if step>0:
                logLikeDiff = np.abs(logLikelihoodArray[step] - logLikelihoodArray[step-1])
            step += 1
        # recompute weights
        self.weights = np.exp(self.lnWeights).astype(np.float64)
        self.weights_weighted = np.copy(self.weights)
        # assign clusters based on largest likelihood 
        self.clusters = np.argmax(self.lnLikelihood, axis = 0)
        self.clusters_weighted = np.copy(self.clusters)
        # save logLikelihood
        self.logLikelihood = logLikelihoodArray[step-1] 
        self.logLikelihood_weighted = self.logLikelihood
        # iteratively align averages
        self.centers, self.globalCenter = traj_tools.traj_iterative_average_weighted(self.centers,self.weights)
        self.centers_weighted = np.copy(self.centers)
        for k in range(self.nClusters):
            indeces = np.argwhere(self.clusters == k).flatten()
            self.trajData[indeces] = traj_tools.traj_align_weighted_kabsch(self.trajData[indeces],self.centers[k],self.covar[k])
        # Compute clustering scores
        self.silhouette_score = metrics.silhouette_score(self.trajData.reshape(self.nFrames,self.nFeatures), self.clusters)
        self.ch_score = metrics.calinski_harabasz_score(self.trajData.reshape(self.nFrames,self.nFeatures), self.clusters)
        self.db_score = metrics.davies_bouldin_score(self.trajData.reshape(self.nFrames,self.nFeatures), self.clusters)
        self.bic = gmm_shapes_weighted_library.compute_bic(self.nAtoms, self.nDim, self.nClusters, self.nFrames, self.logLikelihood)
        self.bic_weighted = self.bic
        self.aic = gmm_shapes_weighted_library.compute_aic(self.nAtoms, self.nDim, self.nClusters, self.logLikelihood)
        self.aic_weighted = self.aic
        # weighted has been performed
        self.gmmWeighted = True

    def predict_weighted(self,trajData):
        if self.gmmWeighted == True:
            # get metadata from trajectory data
            nFrames = trajData.shape[0]
            # declare likelihood array
            lnLikelihood = np.empty((self.nClusters,nFrames),dtype=np.float64)
            # make sure trajectory is centered
            trajData = traj_tools.traj_remove_cog_translation(trajData)
            # Expectation step
            for k in range(self.nClusters):
                # align the entire trajectory to each cluster mean
                trajData = traj_tools.traj_align_weighted_kabsch(trajData,self.centers[k],self.covar[k])
                lnLikelihood[k,:] = gmm_shapes_weighted_library.ln_multivariate_NxN_gaussian_pdf(trajData, self.centers[k], self.covar[k])
            # assign clusters based on largest likelihood (probability density)
            clusters = np.argmax(lnLikelihood, axis = 0)
            # center trajectory around averages
            for k in range(self.nClusters):
                indeces = np.argwhere(clusters == k).flatten()
                trajData[indeces] = traj_tools.traj_align_weighted_kabsch(trajData[indeces],self.centers[k],self.covar[k])
            return clusters, trajData
        else:
            print("Weighted shape-GMM must be fitted before you can predict.")

    def predict_uniform(self,trajData):
        if self.gmmUniform == True:
            # get metadata from trajectory data
            nFrames = trajData.shape[0]
            # declare likelihood array
            lnLikelihood = np.empty((self.nClusters,nFrames),dtype=np.float64)
            # make sure trajectory is centered
            trajData = traj_tools.traj_remove_cog_translation(trajData)
            # Expectation step
            for k in range(self.nClusters):
                # align the entire trajectory to each cluster mean
                trajData = traj_tools.traj_align(trajData,self.centers[k])
                lnLikelihood[k,:] = gmm_shapes_uniform_library.ln_spherical_gaussian_pdf(trajData.reshape(nFrames,self.nFeatures), self.centers[k].reshape(self.nFeatures), self.var[k])
            # assign clusters based on largest likelihood (probability density)
            clusters = np.argmax(lnLikelihood, axis = 0)
            # center trajectory around averages
            for k in range(self.nClusters):
                indeces = np.argwhere(clusters == k).flatten()
                trajData[indeces] = traj_tools.traj_align(trajData[indeces],self.centers[k])
            return clusters, trajData
        else:
            print("Uniform shape-GMM must be fitted before you can predict.")

