import numpy as np
import numba
from numba import jit
from sklearn import metrics
import random
import traj_tools

numericThresh = 1E-150
logNumericThresh = np.log(numericThresh)
gammaThresh = 1E-10
eigenValueThresh = 1E-10

@jit(nopython=True)
def centers_frames_rmsds(centers,traj_data):
    # meta data from arrays
    n_clusters = centers.shape[0]
    n_frames = traj_data.shape[0]
    # declare RMSD distance array
    dists = np.empty((n_frames,n_clusters),dtype=np.float64)
    # measure distance to every center
    for i in range(n_frames):
        for k in range(n_clusters):
            dists[i,k] = traj_tools.rmsd_kabsch(centers[k],traj_data[i])
    return dists

@jit(nopython=True)
def center_frames_rmsds(center,traj_data):
    # meta data from arrays
    n_frames = traj_data.shape[0]
    # declare RMSD distance array
    dists = np.empty(n_frames,dtype=np.float64)
    # measure distance to every center
    for i in range(n_frames):
        dists[i] = traj_tools.rmsd_kabsch(center,traj_data[i])
    return dists

@jit(nopython=True)
def centers_error(centers1,centers2):
    error = np.float64(0.0)
    nCenters = centers1.shape[0]
    for i in range(nCenters):
        error += traj_tools.rmsd_kabsch(centers1[i],centers2[i])
    return error

class ShapeKmeans:
    # perform K-means clustering and traj_data.  
    # Inputs:
    #         traj_data : float array of size n_frames x n_atoms x 3
    #   from self:
    #         n_clusters : integer scalar dictating number of clusters
    #         thresh : float that dictates the allowed deviation of centroids
    # Returns:
    #         clusters: integer array of size n_atoms with cluster assignments (values 0 through n_clusters-1)
    #         centersNew: float array containing centroids 

    def __init__(self, n_clusters, thresh=1E-3,max_steps=50):
        
        self.n_clusters = n_clusters
        self.thresh = thresh
        self.max_steps = max_steps

    def fit(self,traj_data):

        # get meta data from trajectory data
        n_frames = traj_data.shape[0]
        n_atoms = traj_data.shape[1]
        # center trajectory to start
        self.traj_data = traj_tools.traj_remove_cog_translation(traj_data)
        # declare some arrays
        self.clusters = np.zeros(n_frames,dtype=np.int)
        rand_frames = np.empty(self.n_clusters,dtype=np.int)
        dists = np.empty((n_frames,self.n_clusters),dtype=np.float64)
        self.centers = np.empty((self.n_clusters,n_atoms,3),dtype=np.float64)
        # initialize centers with random frames
        rand_frames = random.sample(range(n_frames),n_clusters)
        self.centers = self.traj_data[rand_frames]
        # 
        error = 2.0*self.thresh
        iteration = 0
        while error > self.thresh and iteration < self.max_steps:
            # compute RMSD between centers and frames
            dists = centers_frames_rmsds(self.centers,self.traj_data)
            # assign frame to nearest center
            self.clusters = np.argmin(dists, axis = 1)
            # copy new centers into Old
            centers_old = np.copy(self.centers)
            # compute new centers as average structure of a cluster
            for k in range(self.n_clusters):
                indeces = np.argwhere(self.clusters == k).flatten()
                self.centers[k] = traj_tools.traj_iterative_average(self.traj_data[indeces])[0]
            error = centers_error(self.centers,centers_old)
            iteration += 1
        # center trajectory around means
        self.weights = np.empty(self.n_clusters,dtype=np.float32)
        for k in range(self.n_clusters):
            indeces = np.argwhere(self.clusters == k).flatten()
            self.weights[k] = indeces.size / n_frames
            self.traj_data[indeces,:,:] = traj_tools.traj_align(self.traj_data[indeces],self.centers[k])
        print(self.weights, iteration, error)
        # Compute clustering scores
        self.silhouette_score = metrics.silhouette_score(self.traj_data.reshape(n_frames,n_atoms*3), self.clusters)
        self.ch_score = metrics.calinski_harabasz_score(self.traj_data.reshape(n_frames,n_atoms*3), self.clusters)
        self.db_score = metrics.davies_bouldin_score(self.traj_data.reshape(n_frames,n_atoms*3), self.clusters)
