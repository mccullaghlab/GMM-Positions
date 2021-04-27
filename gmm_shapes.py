import numpy as np
from numba import jit
from sklearn import mixture
from sklearn import metrics
from scipy.stats import multivariate_normal

@jit(nopython=True)
def rmsd_kabsch(xyz1, xyz2):
    xyz1_prime = kabsch_rotate(xyz1, xyz2)
    delta = xyz1_prime - xyz2
    rmsd = (delta ** 2.0).sum(1).mean() ** 0.5
    return rmsd

@jit(nopython=True)
def kabsch_rotate(mobile, target):
    correlation_matrix = np.dot(np.transpose(mobile), target)
    V, S, W_tr = np.linalg.svd(correlation_matrix)
    if np.linalg.det(V) * np.linalg.det(W_tr) < 0.0:
        V[:, -1] = -V[:, -1]
    rotation = np.dot(V, W_tr)
    mobile_prime = np.dot(mobile,rotation) 
    return mobile_prime

@jit(nopython=True)
def kabsch_transform(mobile, target):
    translation, rotation = compute_translation_and_rotation(mobile, target)
    #mobile_prime = mobile.dot(rotation) + translation
    mobile_prime = np.dot(mobile,rotation) #+ translation
    return mobile_prime

@jit(nopython=True)
def compute_translation_and_rotation(mobile, target):
    #meta data
    nAtoms = mobile.shape[0]
    nDim = mobile.shape[1]
    mu1 = np.zeros(nDim)
    mu2 = np.zeros(nDim)
    for i in range(nAtoms):
        for j in range(nDim):
            mu1[j] += mobile[i,j]
            mu2[j] += target[i,j]
    mu1 /= nAtoms
    mu2 /= nAtoms
    mobile = mobile - mu1
    target = target - mu2

    correlation_matrix = np.dot(np.transpose(mobile), target)
    V, S, W_tr = np.linalg.svd(correlation_matrix)
    #is_reflection = (np.linalg.det(V) * np.linalg.det(W_tr)) < 0.0
    if np.linalg.det(V) * np.linalg.det(W_tr) < 0.0:
        V[:, -1] = -V[:, -1]
    rotation = np.dot(V, W_tr)

    translation = mu2 - np.dot(mu1,rotation)

    return translation, rotation

# compute the average structure from trajectory data
@jit(nopython=True)
def traj_iterative_average(trajData,thresh=1E-10):
    # trajectory metadata
    nFrames = trajData.shape[0]
    nAtoms = trajData.shape[1]
    nDim = trajData.shape[2]
    # create numpy array of aligned positions
    alignedPos = np.copy(trajData)
    # start be removing COG translation
    for ts in range(nFrames):
        mu = np.zeros(nDim)
        for atom in range(nAtoms):
            mu += alignedPos[ts,atom]
        mu /= nAtoms
        alignedPos[ts] -= mu
    # Initialize average as first frame
    avg = np.copy(alignedPos[0]).astype(np.float64)
    # perform iterative alignment and average to converge average
    avgRmsd = 2*thresh
    while avgRmsd > thresh:
        # rezero new average
        newAvg = np.zeros((nAtoms,nDim),dtype=np.float64)
        # align trajectory to average and accumulate new average
        for ts in range(nFrames):
            alignedPos[ts] = kabsch_rotate(alignedPos[ts], avg)
            newAvg += alignedPos[ts]
        # finish average
        newAvg /= nFrames
        #avgRmsd = rmsd(avg,newAvg,center=False,superposition=False)
        avgRmsd = rmsd_kabsch(avg,newAvg)
        avg = np.copy(newAvg)
    return avg, alignedPos

# compute the average structure from trajectory data
@jit(nopython=True)
def traj_iterative_average_covar(trajData,thresh=1E-10):
    # trajectory metadata
    nFrames = trajData.shape[0]
    nAtoms = trajData.shape[1]
    nDim = trajData.shape[2]
    # create numpy array of aligned positions
    alignedPos = np.copy(trajData)
    # Initialize average as first frame
    avg = np.copy(alignedPos[0]).astype(np.float64)
    # perform iterative alignment and average to converge average
    avgRmsd = 2*thresh
    while avgRmsd > thresh:
        # rezero new average
        newAvg = np.zeros((nAtoms,nDim),dtype=np.float64)
        # align trajectory to average and accumulate new average
        for ts in range(nFrames):
            # align positions
            alignedPos[ts] = kabsch_rotate(alignedPos[ts], avg)
            newAvg += alignedPos[ts]
        # finish average
        newAvg /= nFrames
        avgRmsd = rmsd_kabsch(avg,newAvg)
        avg = np.copy(newAvg)
    covar = np.zeros((nAtoms*nDim,nAtoms*nDim),dtype=np.float64)
    # loop over trajectory and compute average and covariance
    for ts in range(nFrames):
        disp = alignedPos[ts].flatten()-avg.flatten()
        covar += np.outer(disp,disp)
    # finish average
    covar /= nFrames
    return avg, covar

# compute the average structure from trajectory data
@jit(nopython=True)
def traj_iterative_average_covar_weighted(trajData, weights, prevAvg, thresh=1E-10):
    # trajectory metadata
    nFrames = trajData.shape[0]
    nAtoms = trajData.shape[1]
    nDim = trajData.shape[2]
    # determine normalization
    norm = np.sum(weights)
    # create numpy array of aligned positions
    alignedPos = np.copy(trajData)
    # Initialize average with previous average
    avg = np.copy(prevAvg)
    # perform iterative alignment and average to converge average
    avgRmsd = 2*thresh
    while avgRmsd > thresh:
        # rezero new average
        newAvg = np.zeros((nAtoms,nDim),dtype=np.float64)
        # align trajectory to average and accumulate new average
        for ts in range(nFrames):
            # align to average
            alignedPos[ts] = kabsch_rotate(alignedPos[ts], avg)
            newAvg += weights[ts]*alignedPos[ts]
        # finish average
        newAvg /= norm
        avgRmsd = rmsd_kabsch(avg,newAvg)
        # copy new avg
        avg = np.copy(newAvg)
    # loop over trajectory and compute covariance
    covar = np.zeros((nAtoms*nDim,nAtoms*nDim),dtype=np.float64)
    for ts in range(nFrames):
        disp = alignedPos[ts].flatten()-avg.flatten()
        covar += weights[ts]*np.outer(disp,disp)
    # finish covar
    covar /= norm
    return avg, covar

# align trajectory data to a reference structure
@jit(nopython=True)
def traj_align(trajData,ref):
    # trajectory metadata
    nFrames = trajData.shape[0]
    # create numpy array of aligned positions
    alignedPos = np.copy(trajData)
    for ts in range(nFrames):
        # make sure positions are centered
        alignedPos[ts] = kabsch_rotate(alignedPos[ts], ref)
    return alignedPos

# compute the covariance from trajectory data
# we assume the trajectory is aligned here
@jit(nopython=True)
def traj_covar(trajData):
    # trajectory metadata
    nFrames = trajData.shape[0]
    nAtoms = trajData.shape[1]
    nDim = trajData.shape[2]
    # Initialize average and covariance arrays
    avg = np.zeros((nAtoms*nDim))
    covar = np.zeros((nAtoms*nDim,nAtoms*nDim))
    # loop over trajectory and compute average and covariance
    for ts in range(nFrames):
        flat = trajData[ts].flatten()
        avg += flat
        covar += np.outer(flat,flat)
    # finish averages
    avg /= nFrames
    covar /= nFrames
    # finish covar
    covar -= np.outer(avg,avg)
    return covar


@jit(nopython=True)
def maximum_likelihood_opt(weights,likelihood,centers,trajData):
    # get metadata from trajectory data
    nFrames = trajData.shape[0]
    nAtoms = trajData.shape[1]
    nDim = trajData.shape[2]
    nFeatures = nDim*nAtoms
    nClusters = weights.size
    # declare cluster covariance matrices
    covar = np.empty((nClusters,nFeatures,nFeatures),dtype=np.float64)
    # declare cluster centers
    #centers = np.empty((nClusters,nAtoms,nDim),dtype=np.float64)
    # Compute the normaliztion constant
    normalization = np.zeros(nFrames,dtype=np.float64)
    for i in range(nFrames):
        for k in range(nClusters):
            normalization[i] += likelihood[k,i]*weights[k]
    normalization = np.power(normalization,-1)
    #normalization = np.power(np.sum([likelihood[i] * weights[i] for i in range(nClusters)], axis=0),-1)
    for k in range(nClusters):
        # use the current values for the parameters to evaluate the posterior
        # probabilities of the data to have been generanted by each gaussian
        b = (likelihood[k] * weights[k])*normalization
        # update mean and variance
        centers[k], covar[k] = traj_iterative_average_covar_weighted(trajData, b, centers[k])
        # update the weights
        weights[k] = np.mean(b)
    return centers, covar, weights
    
# Expectation step
@jit
def expectation(trajData, centers, covar):
    # meta data
    nClusters = centers.shape[0]
    nFrames = trajData.shape[0]
    nAtoms = trajData.shape[1]
    nDim = trajData.shape[2]
    nFeatures = nAtoms*nDim
    likelihood = np.empty((nClusters,nFrames))
    #trajData = trajData.reshape(nFrames,nFeatures)
    # compute likelihood of each frame at each Gaussian
    for k in range(nClusters):
        # align the entire trajectory to each cluster mean if requested
        trajData = traj_align(trajData,centers[k])
        likelihood[k,:] = multivariate_normal.pdf(x=trajData.reshape(nFrames,nFeatures), mean=centers[k].reshape(nFeatures), cov=covar[k],allow_singular=True)
    return likelihood

@jit(nopython=True)
def compute_log_likelihood(weights,likelihood):
    logLikelihood = 0.0
    for i in range(likelihood.shape[1]): # data points
        temp = 0
        for j in range(likelihood.shape[0]): # clusters
            temp += weights[j]*likelihood[j,i]
        logLikelihood += np.log(temp)
    return logLikelihood

@jit(nopython=True)
def compute_bic(nFeatures, nClusters, nFrames, logLikelihood):
    k = nClusters*(nFeatures + nFeatures*(nFeatures+1)/2 + 1) - 1
    return k*np.log(nFrames) - 2*logLikelihood

@jit(nopython=True)
def compute_aic(nFeatures, nClusters, logLikelihood):
    k = nClusters*(nFeatures + nFeatures*(nFeatures+1)/2 + 1) - 1
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
        
    @jit
    def fit(self,trajData):
        # get metadata from trajectory data
        self.nFrames = trajData.shape[0]
        self.nAtoms = trajData.shape[1]
        self.nDim = trajData.shape[2]
        self.nFeatures = self.nDim*self.nAtoms
        # center and align the entire trajectory to start
        self.trajData = traj_iterative_average(trajData)[1]
        # declare some important arrays for the model
        self.centers = np.empty((self.nClusters,self.nAtoms,self.nDim),dtype=np.float64)
        self.covar = np.empty((self.nClusters,self.nFeatures,self.nFeatures),dtype=np.float64)
        self.clusters = np.zeros(self.nFrames,dtype=np.int)
        self.likelihood = np.empty((self.nClusters,self.nFrames),dtype=np.float64)
        self.weights = np.empty(self.nClusters,dtype=np.float64)
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
                    dists[i,k] = rmsd_kabsch(self.centers[k],self.trajData[i])
            # assign frame to nearest center
            self.clusters = np.argmin(dists, axis = 1)
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
            for i in range(self.initIter-1):
                gmm = mixture.GaussianMixture(n_components=self.nClusters, max_iter=200, covariance_type='full', init_params="kmeans").fit(self.trajData.reshape(self.nFrames,self.nFeatures))  
                tempScore = gmm.score(self.trajData.reshape(self.nFrames,self.nFeatures))
                if (tempScore > score):
                    self.clusters = gmm.predict(self.trajData.reshape(self.nFrames,self.nFeatures))
                    score = tempScore
        # compute average and covariance of initial clustering
        for k in range(self.nClusters):
            indeces = np.where(self.clusters == k)[0]
            self.centers[k],self.covar[k] = traj_iterative_average_covar(self.trajData[indeces])
            # initialize weights as populations of clusters
            self.weights[k] = indeces.size
        self.weights /= np.sum(self.weights)
    
        # perform Expectation Maximization
        logLikeDiff = 2*self.logThresh
        step = 0
        logLikelihoodArray = np.empty(self.maxSteps,dtype=np.float64)
        while step < self.maxSteps and logLikeDiff > self.logThresh:
            # Expectation step
            self.likelihood = expectation(self.trajData, self.centers, self.covar)
            # compute log likelihood of this step
            logLikelihoodArray[step] = compute_log_likelihood(self.weights,self.likelihood)
            print(step, self.weights, logLikelihoodArray[step])
            # Maximum Likelihood Optimization
            self.centers, self.covar, self.weights = maximum_likelihood_opt(self.weights, self.likelihood, self.centers, self.trajData)
            # compute convergence criteria
            if step>0:
                logLikeDiff = np.abs(logLikelihoodArray[step] - logLikelihoodArray[step-1])
            step += 1
        # assign clusters based on largest likelihood (probability density)
        self.clusters = np.argmax(self.likelihood, axis = 0)
        # save logLikelihood
        self.logLikelihood = logLikelihoodArray[step-1] 
        # align averages to the first average
        for k in range(1,self.nClusters):
            self.centers[k] = kabsch_rotate(self.centers[k], self.centers[0])
        # align clusters to averages
        for k in range(self.nClusters):
            indeces = np.where(self.clusters == k)[0]
            self.trajData[indeces] = traj_align(self.trajData[indeces],self.centers[k])
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
        likelihood = np.empty((self.nClusters,nFrames),dtype=np.float64)
        # Expectation step
        for k in range(self.nClusters):
            # align the entire trajectory to each cluster mean if requested
            if self.align==True:
                trajData = traj_align(trajData,self.centers[k])
            likelihood[k,:] = multivariate_normal.pdf(x=trajData.reshape(nFrames,self.nFeatures), mean=self.centers[k].reshape(self.nFeatures), cov=self.covar[k],allow_singular=True)
        # assign clusters based on largest likelihood (probability density)
        clusters = np.argmax(likelihood, axis = 0)
        # center trajectory around averages
        for k in range(self.nClusters):
            indeces = np.where(clusters == k)[0]
            trajData[indeces] = traj_align(trajData[indeces],self.centers[k])
        return clusters, trajData

