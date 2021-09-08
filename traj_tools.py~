import numpy as np
import pickle
import numba
from numba import jit
from scipy import spatial

numericThresh = 1E-150
logNumericThresh = np.log(numericThresh)
gammaThresh = 1E-10
eigenValueThresh = 1E-10

@jit(nopython=True)
def sample_variance(zeroMeanDataArray,norm):
    # meta data from array
    nDataPoints = zeroMeanDataArray.shape[0]
    # zero variance
    var = np.float64(0.0)
    for i in range(nDataPoints):
        var += zeroMeanDataArray[i]**2
    return var/norm


@jit(nopython=True)
def weight_kabsch_dist_align(x1, x2, weights):
    x1_prime = weight_kabsch_rotate(x1, x2, weights)
    dist = 0.0
    for i in range(3):
        disp = x1_prime[:,i] - x2[:,i]
        dist += np.dot(disp,np.dot(weights,disp))
    return dist

@jit(nopython=True)
def weight_kabsch_dist(x1, x2, weights):
    dist = 0.0
    for i in range(3):
        disp = x1[:,i] - x2[:,i]
        dist += np.dot(disp,np.dot(weights,disp))
    return dist

@jit(nopython=True)
def pseudo_lpdet_inv(sigma):
    N = sigma.shape[0]
    e, v = np.linalg.eigh(sigma)
    precision = np.zeros(sigma.shape,dtype=np.float64)
    lpdet = 0.0
    rank = 0
    for i in range(N):
        if (e[i] > eigenValueThresh):
            lpdet += np.log(e[i])
            precision += 1.0/e[i]*np.outer(v[:,i],v[:,i])
            rank += 1
    return lpdet, precision, rank

@jit(nopython=True)
def weight_kabsch_log_lik(x, mu, covar):
    # meta data
    nFrames = x.shape[0]
    # determine precision and pseudo determinant 
    lpdet, precision, rank = pseudo_lpdet_inv(covar)
    # compute log Likelihood for all points
    logLik = 0.0
    for i in range(nFrames):
        #disp = x[i] - mu
        for j in range(3):
            disp = x[i,:,j] - mu[:,j]
            logLik += np.dot(disp,np.dot(precision,disp))
    logLik += 3 * nFrames * lpdet
    logLik *= -0.5
    return logLik, precision

# old version
# @jit(nopython=True)
# def weight_kabsch_log_lik_v02(x, mu, covar):
#    # meta data
#    nFrames = x.shape[0]
#    # get covar 3Nx3N
#    covar3N = np.kron(covar,np.identity(3))
#    # determine precision and pseudo determinant 
#    lpdet, precision = pseudo_lpdet_inv(covar3N)
#    # compute log Likelihood for all points
#    logLik = 0.0
#    for i in range(nFrames):
#        disp = (x[i] - mu).flatten()
#        logLik += np.dot(disp,np.dot(precision,disp))
#    logLik +=  nFrames * lpdet
#    logLik *= -0.5
#    return logLik, precision

@jit(nopython=True)
def weight_kabsch_rotate(mobile, target, weights):
    correlation_matrix = np.dot(np.transpose(mobile), np.dot(weights, target))
    V, S, W_tr = np.linalg.svd(correlation_matrix)
    if np.linalg.det(V) * np.linalg.det(W_tr) < 0.0:
        V[:, -1] = -V[:, -1]
    rotation = np.dot(V, W_tr)
    mobile_prime = np.dot(mobile,rotation)
    return mobile_prime

@jit(nopython=True)
def weight_kabsch_rmsd(mobile, target, weights):
    xyz1_prime = weight_kabsch_rotate(mobile, target, weights)
    delta = xyz1_prime - target
    rmsd = (delta ** 2.0).sum(1).mean() ** 0.5
    return rmsd

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

# remove COG translation
@jit(nopython=True)
def traj_remove_cog_translation(trajData):
    # trajectory metadata
    nFrames = trajData.shape[0]
    nAtoms = trajData.shape[1]
    nDim = trajData.shape[2]
    # start be removing COG translation
    for ts in range(nFrames):
        mu = np.zeros(nDim)
        for atom in range(nAtoms):
            mu += trajData[ts,atom]
        mu /= nAtoms
        trajData[ts] -= mu
    return trajData

# compute the average structure and covariance from trajectory data
@jit(nopython=True)
def traj_iterative_average_covar_weighted_kabsch_v02(trajData,thresh=1E-8,maxSteps=300):
    # trajectory metadata
    nFrames = trajData.shape[0]
    nAtoms = trajData.shape[1]
    nDim = trajData.shape[2]
    # Initialize with uniform weighted Kabsch
    avg, alignedPos = traj_iterative_average(trajData,thresh)
    # Compute Kabsch Weights
    disp = alignedPos - avg
    covar = np.zeros((nAtoms,nAtoms),dtype=np.float64)
    for ts in range(nFrames):
        covar += np.dot(disp[ts],disp[ts].T)
    covar /= nFrames
    kabschWeights = np.linalg.pinv(covar,rcond=1e-10)
    # perform iterative alignment and average to converge average
    avgRmsd = 2*thresh
    step = 0
    while avgRmsd > thresh and step < maxSteps:
        # rezero new average
        newAvg = np.zeros((nAtoms,nDim),dtype=np.float64)
        # align trajectory to average and accumulate new average
        for ts in range(nFrames):
            alignedPos[ts] = weight_kabsch_rotate(alignedPos[ts], avg, kabschWeights)
            newAvg += alignedPos[ts]
        # finish average
        newAvg /= nFrames
        # compute new Kabsch Weights
        covar = np.zeros((nAtoms,nAtoms),dtype=np.float64)
        for ts in range(nFrames):
            disp = alignedPos[ts] - newAvg
            covar += np.dot(disp,disp.T)    
        covar /= nDim*(nFrames-1)
        # compute log likelihood
        logLik, kabschWeights = weight_kabsch_log_lik(alignedPos, newAvg, covar)
        #kabschWeights = np.linalg.pinv(covar,rcond=1e-10)
        # compute Distance between averages
        avgRmsd = weight_kabsch_dist_align(avg,newAvg,kabschWeights)
        avg = np.copy(newAvg)
        step += 1
        #print(step, avgRmsd, logLik)
    return alignedPos, avg, covar

# compute the average structure and covariance from trajectory data
@jit(nopython=True)
def traj_iterative_average_weighted_kabsch_v02(trajData,thresh=1E-4,maxSteps=200):
    # trajectory metadata
    nFrames = trajData.shape[0]
    nAtoms = trajData.shape[1]
    nDim = trajData.shape[2]
    # Initialize with uniform weighted Kabsch
    avg, alignedPos = traj_iterative_average(trajData,thresh)
    # Compute Kabsch Weights
    disp = alignedPos.reshape(nFrames,nAtoms*nDim) - avg.reshape(nAtoms*nDim)
    covar = np.dot(disp.T,disp)/nFrames
    kabschWeights = np.linalg.pinv(trace_3Nx3N(covar),rcond=1e-10)
    # perform iterative alignment and average to converge average
    avgRmsd = 1.0 + thresh
    step = 0
    while avgRmsd > thresh and step < maxSteps:
        # rezero new average
        newAvg = np.zeros((nAtoms,nDim),dtype=np.float64)
        # align trajectory to average and accumulate new average
        for ts in range(nFrames):
            alignedPos[ts] = weight_kabsch_rotate(alignedPos[ts], avg, kabschWeights)
            newAvg += alignedPos[ts]
        # finish average
        newAvg /= nFrames
        # compute new Kabsch Weights
        kabschWeights = np.zeros((nAtoms,nAtoms),dtype=np.float64)
        for ts in range(nFrames):
            disp = alignedPos[ts] - newAvg
            kabschWeights += np.dot(disp,disp.T)
#        for atom1 in range(nAtoms):
#            for atom2 in range(atom1,nAtoms):
#                for ts in range(nFrames):
#                    disp1 = alignedPos[ts,atom1] - newAvg[atom1]
#                    disp2 = alignedPos[ts,atom2] - newAvg[atom2]
#                    kabschWeights[atom1,atom2] += np.dot(disp1,disp2)
#                kabschWeights[atom2,atom1] = kabschWeights[atom1,atom2]
        kabschWeights /= 3*(nFrames-1)
        kabschWeights = np.linalg.pinv(kabschWeights,rcond=1e-10)
        # compute RMSD between averages
        avgRmsd = weight_kabsch_dist_align(avg,newAvg,kabschWeights)
        #print(step,avgRmsd)
        avg = np.copy(newAvg)
        step += 1
    # return average structure and aligned trajectory
    return avg, alignedPos

# compute the average structure from trajectory data
@jit(nopython=True)
def traj_iterative_average(trajData,thresh=1E-10):
    # trajectory metadata
    nFrames = trajData.shape[0]
    nAtoms = trajData.shape[1]
    nDim = trajData.shape[2]
    # create numpy array of aligned positions
    alignedPos = np.copy(trajData).astype(np.float64)
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
        disp = (alignedPos[ts]-avg).flatten()
        covar += np.outer(disp,disp)
    # finish average
    covar /= (nFrames-1)
    return avg, covar

# compute the average structure from weighted trajectory data
@jit(nopython=True)
def traj_iterative_average_weighted(trajData, weights, prevAvg=None, thresh=1E-10):
    # trajectory metadata
    nFrames = trajData.shape[0]
    nAtoms = trajData.shape[1]
    nDim = trajData.shape[2]
    nFeatures = nAtoms*nDim
    # determine normalization
    norm = np.power(np.sum(weights),-1)
    weights *= norm
    # create numpy array of aligned positions
    alignedPos = np.copy(trajData)
    # Initialize average 
    if prevAvg == None:
        avg = np.copy(trajData[np.argmax(weights)])
    else:
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
        # compute RMSD between averages
        avgRmsd = rmsd_kabsch(avg,newAvg)
        # copy new avg
        avg = np.copy(newAvg)
    return alignedPos, avg

# compute the average structure and covariance from weighted trajectory data
@jit(nopython=True)
def traj_iterative_average_covar_weighted_weighted_kabsch(trajData, weights, prevAvg, preCovar, thresh=1E-4, maxSteps=100):
    # trajectory metadata
    nFrames = trajData.shape[0]
    nAtoms = trajData.shape[1]
    nDim = trajData.shape[2]
    nFeatures = nAtoms*nDim
    # determine normalization
    norm = np.power(np.sum(weights),-1)
    weights *= norm
    # create numpy array of aligned positions
    alignedPos = np.copy(trajData)
    # Initialize average with previous average
    avg = np.copy(prevAvg)
    # set kasbch weights to inverse of current NxN covars
    kabschWeights = np.linalg.pinv(preCovar,rcond=1e-10)
    # perform iterative alignment and average to converge average
    avgRmsd = 2*thresh
    step = 0
    while avgRmsd > thresh and step < maxSteps:
        # rezero new average
        newAvg = np.zeros((nAtoms,nDim),dtype=np.float64)
        # align trajectory to average and accumulate new average
        for ts in range(nFrames):
            # align to average
            alignedPos[ts] = weight_kabsch_rotate(alignedPos[ts], avg, kabschWeights)
            newAvg += weights[ts]*alignedPos[ts]
        # compute new Kabsch Weights
        covar = np.zeros((nAtoms,nAtoms),dtype=np.float64)
        for ts in range(nFrames):
            disp = alignedPos[ts] - newAvg
            covar += weights[ts]*np.dot(disp,disp.T)
        kabschWeights = np.linalg.pinv(covar/3,rcond=1e-10)
        # determine rmsd between consecutive average structures
        avgRmsd = weight_kabsch_dist_align(avg,newAvg,kabschWeights)
        # copy new avg
        avg = np.copy(newAvg)
        step += 1

    return avg, covar

# align trajectory data to a reference structure
@jit(nopython=True)
def traj_align_weighted_kabsch(trajData,ref, covar):
    # trajectory metadata
    nFrames = trajData.shape[0]
    # kabsch weights
    kabschWeights = np.linalg.pinv(covar,rcond=1e-10)
    # create numpy array of aligned positions
    alignedPos = np.copy(trajData)
    for ts in range(nFrames):
        # align positions based on weighted Kabsch
        alignedPos[ts] = weight_kabsch_rotate(alignedPos[ts], ref, kabschWeights)
    return alignedPos

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

# compute the time separated covariance matrix
@jit(nopython=True)
def traj_time_covar(traj1, traj2, mean1, mean2, lag):
    # trajectory metadata
    nFrames = traj1.shape[0]
    nAtoms = traj1.shape[1]
    nDim = traj1.shape[2]
    # declare covar
    covar = np.zeros((nAtoms*nDim,nAtoms*nDim),dtype=np.float64)
    # loop over trajectory and compute average and covariance
    for ts in range(nFrames-lag):
        disp1 = traj1[ts].flatten()-mean1.flatten()
        disp2 = traj2[ts+lag].flatten()-mean2.flatten()
        covar += np.outer(disp1,disp2)
    # finish average
    covar /= (nFrames-lag)
    return covar

# compute the average structure and variance from trajectory data
@jit(nopython=True)
def traj_iterative_average_var(trajData,thresh=1E-10):
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
    #var = np.mean(np.power(alignedPos-avg,2))
    # Compute variance correcting for normalizing by 3(nAtoms-1)*(nFrames-1)
    var = sample_variance((alignedPos-avg).flatten(),nDim*(nAtoms-1)*(nFrames-1))  
    return avg, var

# compute the average structure and variance from weighted trajectory data
@jit(nopython=True)
def traj_iterative_average_var_weighted(trajData, weights, prevAvg, thresh=1E-10):
    # trajectory metadata
    nFrames = trajData.shape[0]
    nAtoms = trajData.shape[1]
    nDim = trajData.shape[2]
    nFeatures = nAtoms*nDim
    # determine normalization
    norm = np.power(np.sum(weights),-1)
    weights *= norm
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
        # compute RMSD between averages
        avgRmsd = rmsd_kabsch(avg,newAvg)
        # copy new avg
        avg = np.copy(newAvg)
    # loop over trajectory and compute variance
    var = np.float64(0.0)
    varNorm = 3*nAtoms-3
    for ts in range(nFrames):
        #var += weights[ts]*np.mean(np.power(alignedPos[ts].flatten()-avg.flatten(),2))
        var += weights[ts]*sample_variance((alignedPos[ts]-avg).flatten(),varNorm)
    return avg, var

