import numpy as np
import numba
from numba import jit
from sklearn import metrics
import traj_tools

numericThresh = 1E-150
logNumericThresh = np.log(numericThresh)
gammaThresh = 1E-10
eigenValueThresh = 1E-10

@jit(nopython=True)
def centers_frames_rmsds(centers,trajData):
    # meta data from arrays
    nClusters = centers.shape[0]
    nFrames = trajData.shape[0]
    # declare RMSD distance array
    dists = np.empty((nFrames,nClusters),dtype=np.float64)
    # measure distance to every center
    for i in range(nFrames):
        for k in range(nClusters):
            dists[i,k] = traj_tools.rmsd_kabsch(centers[k],trajData[i])
    return dists

@jit(nopython=True)
def center_frames_rmsds(center,trajData):
    # meta data from arrays
    nFrames = trajData.shape[0]
    # declare RMSD distance array
    dists = np.empty(nFrames,dtype=np.float64)
    # measure distance to every center
    for i in range(nFrames):
        dists[i] = traj_tools.rmsd_kabsch(center,trajData[i])
    return dists

@jit(nopython=True)
def centers_error(centers1,centers2):
    error = np.float64(0.0)
    nCenters = centers1.shape[0]
    for i in range(nCenters):
        error += traj_tools.rmsd_kabsch(centers1[i],centers2[i])
    return error

class kmeans_shape:

    def __init__(self, nClusters, thresh=1E-3,maxSteps=50):
        
        self.nClusters = nClusters
        self.thresh = thresh
        self.maxSteps = maxSteps

    def fit(self,trajData):
    # perform K-means clustering and trajData.  
    # Inputs:
    #         trajData : float array of size nFrames x nAtoms x 3
    #   from self:
    #         nClusters : integer scalar dictating number of clusters
    #         thresh : float that dictates the allowed deviation of centroids
    # Returns:
    #         clusters: integer array of size nAtoms with cluster assignments (values 0 through nClusters-1)
    #         centersNew: float array containing centroids 
    #         WCSS: float scalar containing the within cluster sum of squares

        # get meta data from trajectory data
        nFrames = trajData.shape[0]
        nAtoms = trajData.shape[1]
        # center trajectory to start
        self.trajData = traj_tools.traj_remove_cog_translation(trajData)
        # declare some arrays
        self.clusters = np.zeros(nFrames,dtype=np.int)
        #diff = True
        #while diff == True:
        #    randFrames = np.random.randint(nFrames,size=self.nClusters)
        #    if np.unique(randFrames).size == randFrames.size:
        #        diff = False
        # initialize using a first random frame and then the frames farthest away..
        randFrames = np.empty(self.nClusters,dtype=np.int)
        dists = np.empty((nFrames,self.nClusters),dtype=np.float64)
        self.centers = np.empty((self.nClusters,nAtoms,3),dtype=np.float64)
        frames = np.arange(0,nFrames,1,dtype=np.int)
        randFrames[0] = np.random.randint(nFrames)
        self.centers[0] = self.trajData[randFrames[0]]
        frames = np.delete(frames,np.argwhere(frames==randFrames[0]))
        for i in range(1,self.nClusters):
            # compute distances
            dists[:,i-1] = center_frames_rmsds(self.centers[i-1], self.trajData)
            randFrames[i] = frames[np.argmax(dists[frames,i-1:i])]
            #randFrames[i] = np.argmax(np.linalg.norm(dists[frames,i-1:i],axis=1))
            frames = np.delete(frames,np.argwhere(frames==randFrames[i]))
            self.centers[i] = self.trajData[randFrames[i]]
        #self.centers = np.copy(self.trajData[randFrames])
        print(randFrames)
        centersOld = np.zeros(self.centers.shape)
        # 
        error = 2.0*self.thresh
        iteration = 0
        while error > self.thresh and iteration < self.maxSteps:
            # compute RMSD between centers and frames
            dists = centers_frames_rmsds(self.centers,self.trajData)
            # assign frame to nearest center
            self.clusters = np.argmin(dists, axis = 1)
            # copy new centers into Old
            centersOld = np.copy(self.centers)
            # compute new centers as average structure of a cluster
            for k in range(self.nClusters):
                indeces = np.argwhere(self.clusters == k).flatten()
                self.centers[k] = traj_tools.traj_iterative_average(self.trajData[indeces])[0]
            error = centers_error(self.centers,centersOld)
            iteration += 1
        # center trajectory around means
        self.weights = np.empty(self.nClusters,dtype=np.float32)
        for k in range(self.nClusters):
            indeces = np.argwhere(self.clusters == k).flatten()
            self.weights[k] = indeces.size / nFrames
            self.trajData[indeces,:,:] = traj_tools.traj_align(self.trajData[indeces],self.centers[k])
        print(self.weights, iteration, error)
        # Compute clustering scores
        self.silhouette_score = metrics.silhouette_score(self.trajData.reshape(nFrames,nAtoms*3), self.clusters)
        self.ch_score = metrics.calinski_harabasz_score(self.trajData.reshape(nFrames,nAtoms*3), self.clusters)
        self.db_score = metrics.davies_bouldin_score(self.trajData.reshape(nFrames,nAtoms*3), self.clusters)
