import numpy as np
import pickle
import numba
from numba import jit
from sklearn import mixture
from sklearn import metrics
from scipy.stats import multivariate_normal
import traj_tools
import kmeans_shapes

numericThresh = 1E-150
logNumericThresh = np.log(numericThresh)
gammaThresh = 1E-15
eigenValueThresh = 1E-10

#@jit
def maximum_likelihood_opt(lnWeights,lnLikelihood,centers,trajData):
    # get metadata from trajectory data
    nFrames = trajData.shape[0]
    nAtoms = trajData.shape[1]
    nDim = trajData.shape[2]
    nFeatures = nDim*nAtoms
    nClusters = lnWeights.size
    # declare cluster variances
    var = np.empty(nClusters,dtype=np.float64)
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
#    nNonzeroFrames = len(indeces)
#    print("Number of nonzero frames:", nNonzeroFrames, " out of ", nFrames)
    indeces = np.array(indeces,dtype=np.int)
    logNorm = np.array(logNorm,dtype=np.float64)
    for k in range(nClusters):
        # use the current values for the parameters to evaluate the posterior
        # probabilities of the data to have been generanted by each gaussian
        # the following step can be numerically unstable
        loggamma = lnLikelihood[k,indeces] + lnWeights[k] - logNorm
        #newIndeces = np.argwhere(loggamma > logNumericThresh)
        gamma = np.exp(loggamma).astype(np.float64)
        # gamma should be between 0 and 1
        gamma[np.argwhere(gamma > 1.0)] = 1.0
        # will only use frames that have greater than gammaThresh weight
        gamma_indeces = np.argwhere(gamma > gammaThresh).flatten()
        # update mean and variance
        centers[k], var[k] = traj_tools.traj_iterative_average_var_weighted(trajData[indeces[gamma_indeces]], gamma[gamma_indeces], centers[k])
        # update the weights
        lnWeights[k] = np.log(np.mean(gamma))
    return centers, var, lnWeights, logLikelihood

# Expectation step
@jit
def expectation(trajData, centers, var):
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
    nDim = x.shape[1]
    lnnorm = -0.5*nDim*(np.log(2.0*np.pi*sigma))
    mvG = np.empty(nSamples,dtype=np.float64)
    multiplier = -0.5/sigma
    for i in range(nSamples):
        diffV = x[i] - mu
        mvG[i] = multiplier*np.dot(diffV,diffV) + lnnorm
    return mvG

@jit(nopython=True)
def compute_bic(nFeatures, nClusters, nFrames, logLikelihood):
    k = nClusters*(nFeatures + 1 + 1) - 1
    return k*np.log(nFrames) - 2*logLikelihood

@jit(nopython=True)
def compute_aic(nFeatures, nClusters, logLikelihood):
    k = nClusters*(nFeatures + 1 + 1) - 1
    return 2*k - 2*logLikelihood

# class
class gmm_shape:
    
    def __init__(self, nClusters, logThresh=1E-3,maxSteps=50,align=True, initClusters="GMM", initIter=5):
        
        self.nClusters = nClusters
        self.logThresh = logThresh
        self.maxSteps = maxSteps
        self.align = align
        self.initClusters = initClusters
        self.initIter = initIter

    def fit(self,trajData):
        # get metadata from trajectory data
        self.nFrames = int(trajData.shape[0])
        self.nAtoms = trajData.shape[1]
        self.nDim = trajData.shape[2]
        self.nFeatures = self.nDim*self.nAtoms
        # center and align the entire trajectory to start
        self.trajData = traj_tools.traj_iterative_average(trajData)[1]
        # declare some important arrays for the model
        self.centers = np.empty((self.nClusters,self.nAtoms,self.nDim),dtype=np.float64)
        self.var = np.empty(self.nClusters,dtype=np.float64)
        self.clusters = np.zeros(self.nFrames,dtype=np.int)
        self.likelihood = np.empty((self.nClusters,self.nFrames),dtype=np.float128)
        self.weights = np.empty(self.nClusters,dtype=np.float128)
        self.lnLikelihood = np.empty((self.nClusters,self.nFrames),dtype=np.float64)
        self.lnWeights = np.empty(self.nClusters,dtype=np.float64)
        # make initial clustering based on input user choice (default is GMM without alignment)
        if (self.initClusters == "uniform"):
            for i in range(self.nFrames):
                self.clusters[i] = i*self.nClusters // self.nFrames
        elif (self.initClusters == "random"):
            dists = np.empty((self.nFrames,self.nClusters))
            randFrames = np.random.randint(self.nFrames,size=self.nClusters) # can this give two of the same values?
            self.centers = np.copy(self.trajData[randFrames,:,:])
            # make initial clustering based on RMSD distance from centers
            # measure distance to every center
            for i in range(self.nFrames):
                for k in range(self.nClusters):
                    dists[i,k] = traj_tools.rmsd_kabsch(self.centers[k],self.trajData[i])
            # assign frame to nearest center
            self.clusters = np.argmin(dists, axis = 1)
        elif (self.initClusters == "kmeans"):
            kmeans = kmeans_shapes.kmeans_shape(self.nClusters,maxSteps=self.maxSteps)
            kmeans.fit(self.trajData)
            self.clusters = kmeans.clusters
        elif (self.initClusters == "BGMM"): 
            bgmm = mixture.BayesianGaussianMixture(n_components=self.nClusters, max_iter=200, covariance_type='full', init_params="kmeans").fit(self.trajData.reshape(self.nFrames,self.nFeatures))
            self.clusters = bgmm.predict(self.trajData.reshape(self.nFrames,self.nFeatures))
            score = bgmm.score(self.trajData.reshape(self.nFrames,self.nFeatures))
            for i in range(self.initIter-1):
                bgmm = mixture.BayesianGaussianMixture(n_components=self.nClusters, max_iter=200, covariance_type='full', init_params="kmeans").fit(self.trajData.reshape(self.nFrames,self.nFeatures))
                tempScore = bgmm.score(self.trajData.reshape(self.nFrames,self.nFeatures))
                if (tempScore > score):
                    self.clusters = bgmm.predict(self.trajData.reshape(self.nFrames,self.nFeatures))
                    score = tempScore
        else: # use GMM without alignment
            gmm = mixture.GaussianMixture(n_components=self.nClusters, max_iter=200, covariance_type='full', init_params="kmeans").fit(self.trajData.reshape(self.nFrames,self.nFeatures))
            self.clusters = gmm.predict(self.trajData.reshape(self.nFrames,self.nFeatures))
            score = gmm.score(self.trajData.reshape(self.nFrames,self.nFeatures))
            #self.weights = gmm.weights_
            #self.centers = gmm.means_.reshape(self.nClusters,self.nAtoms,self.nDim)
            for i in range(self.initIter-1):
                gmm = mixture.GaussianMixture(n_components=self.nClusters, max_iter=200, covariance_type='full', init_params="kmeans").fit(self.trajData.reshape(self.nFrames,self.nFeatures))  
                tempScore = gmm.score(self.trajData.reshape(self.nFrames,self.nFeatures))
                if (tempScore > score):
                    self.clusters = gmm.predict(self.trajData.reshape(self.nFrames,self.nFeatures))
                    score = tempScore
                  #  self.weights = gmm.weights_
                  #  self.centers = gmm.means_.reshape(self.nClusters,self.nAtoms,self.nDim)

        # compute average and covariance of initial clustering
        for k in range(self.nClusters):
            indeces = np.argwhere(self.clusters == k).flatten()
            self.centers[k],self.var[k] = traj_tools.traj_iterative_average_var(self.trajData[indeces])
            # initialize weights as populations of clusters
            self.weights[k] = indeces.size
        self.weights /= np.sum(self.weights)
        self.lnWeights = np.log(self.weights)
    
        # perform Expectation Maximization
        logLikeDiff = 2*self.logThresh
        step = 0
        logLikelihoodArray = np.empty(self.maxSteps,dtype=np.float64)
        while step < self.maxSteps and logLikeDiff > self.logThresh:
            # Expectation step
            self.lnLikelihood = expectation(self.trajData, self.centers, self.var)
            # Maximum Likelihood Optimization
            self.centers, self.var, self.lnWeights, logLikelihoodArray[step] = maximum_likelihood_opt(self.lnWeights, self.lnLikelihood, self.centers, self.trajData)
            print(step, np.exp(self.lnWeights), logLikelihoodArray[step])
            # compute convergence criteria
            if step>0:
                logLikeDiff = np.abs(logLikelihoodArray[step] - logLikelihoodArray[step-1])
            step += 1
        # recompute weights
        self.weights = np.exp(self.lnWeights).astype(np.float64)
        # assign clusters based on largest likelihood 
        self.clusters = np.argmax(self.lnLikelihood, axis = 0)
        # save logLikelihood
        self.logLikelihood = logLikelihoodArray[step-1] 
        # iteratively align averages
        self.centers, self.globalCenter = traj_tools.traj_iterative_average_weighted(self.centers,self.weights)
        for k in range(self.nClusters):
            indeces = np.argwhere(self.clusters == k).flatten()
            self.trajData[indeces] = traj_tools.traj_align(self.trajData[indeces],self.centers[k])
        # Compute clustering scores
        self.silhouette_score = metrics.silhouette_score(self.trajData.reshape(self.nFrames,self.nFeatures), self.clusters)
        self.ch_score = metrics.calinski_harabasz_score(self.trajData.reshape(self.nFrames,self.nFeatures), self.clusters)
        self.db_score = metrics.davies_bouldin_score(self.trajData.reshape(self.nFrames,self.nFeatures), self.clusters)
        self.bic = compute_bic(self.nFeatures-6, self.nClusters, self.nFrames, self.logLikelihood)
        self.aic = compute_aic(self.nFeatures-6, self.nClusters, self.logLikelihood)

    def predict(self,trajData):
        # get metadata from trajectory data
        nFrames = trajData.shape[0]
        # declare likelihood array
        likelihood = np.empty((self.nClusters,nFrames),dtype=np.float128)
        # make sure trajectory is centered
        trajData = traj_tools.traj_remove_cog_translation(trajData)
        # Expectation step
        for k in range(self.nClusters):
            # align the entire trajectory to each cluster mean
            trajData = traj_tools.traj_align(trajData,self.centers[k])
            likelihood[k,:] = multivariate_normal.pdf(x=trajData.reshape(nFrames,self.nFeatures), mean=self.centers[k].reshape(self.nFeatures), cov=self.var[k],allow_singular=True)
        # assign clusters based on largest likelihood (probability density)
        clusters = np.argmax(likelihood, axis = 0)
        # center trajectory around averages
        for k in range(self.nClusters):
            indeces = np.argwhere(clusters == k).flatten()
            trajData[indeces] = traj_tools.traj_align(trajData[indeces],self.centers[k])
        return clusters, trajData


