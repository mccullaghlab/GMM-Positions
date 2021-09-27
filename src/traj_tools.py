import numpy as np
import numba
from numba import jit

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
def lpdet_inv(sigma):
    N = sigma.shape[0]
    e, v = np.linalg.eigh(sigma)
    lpdet = 0.0
    rank = 0
    for i in range(N):
        if (e[i] > eigenValueThresh):
            lpdet -= np.log(e[i])
    return lpdet

@jit(nopython=True)
def uniform_kabsch_log_lik(x, mu):
    # meta data
    nFrames = x.shape[0]
    nAtoms = x.shape[1]
    # compute log Likelihood for all points
    logLik = 0.0
    sampleVar = 0.0
    for i in range(nFrames):
        for j in range(3):
            disp = x[i,:,j] - mu[:,j]
            temp = np.dot(disp,disp)
            sampleVar += temp
            logLik += temp
    # finish variance
    sampleVar /= (nFrames-1)*3*(nAtoms-1)
    logLik /= sampleVar
    logLik +=  nFrames * 3 * (nAtoms-1) * np.log(sampleVar)
    logLik *= -0.5
    return logLik

@jit(nopython=True)
def intermediate_kabsch_log_lik(x, mu, kabschWeights):
    # meta data
    nFrames = x.shape[0]
    # determine precision and pseudo determinant 
    lpdet = lpdet_inv(kabschWeights)
    # compute log Likelihood for all points
    logLik = 0.0
    for i in range(nFrames):
        #disp = x[i] - mu
        for j in range(3):
            disp = x[i,:,j] - mu[:,j]
            logLik += np.dot(disp,np.dot(kabschWeights,disp))
    logLik += 3 * nFrames * lpdet
    logLik *= -0.5
    return logLik

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

@jit(nopython=True)
def particle_variances_from_trajectory(trajData, avg):
    # meta data
    nFrames = trajData.shape[0]
    nAtoms = trajData.shape[1]
    # 
    disp = trajData - avg
    particleVariances = np.zeros(nAtoms,dtype=np.float64)
    for ts in range(nFrames):
        for atom in range(nAtoms):
            particleVariances[atom] += np.dot(disp[ts,atom],disp[ts,atom])
    particleVariances /= 3*(nFrames-1)
    return particleVariances

@jit(nopython=True)
def intermediate_kabsch_weights(variances):
    # meta data
    nAtoms = variances.shape[0]
    # kasbch weights are inverse of variances
    inverseVariances = np.power(variances,-1)
    kabschWeights = np.zeros((nAtoms,nAtoms),dtype=np.float64)
    # force constant vector to be null space of kabsch weights
    wsum = np.sum(inverseVariances)
    for i in range(nAtoms):
        # Populate diagonal elements
        kabschWeights[i,i] = inverseVariances[i]
        for j in range(nAtoms):
            kabschWeights[i,j] -= inverseVariances[i]*inverseVariances[j]/wsum
    # return the weights
    return kabschWeights

# compute the average structure and covariance from trajectory data
@jit(nopython=True)
def traj_iterative_average_vars_intermediate_kabsch(trajData,thresh=1E-3,maxSteps=300):
    # trajectory metadata
    nFrames = trajData.shape[0]
    nAtoms = trajData.shape[1]
    nDim = trajData.shape[2]
    # Initialize with uniform weighted Kabsch
    avg, alignedPos = traj_iterative_average(trajData,thresh)
    # Compute Kabsch Weights
    particleVariances = particle_variances_from_trajectory(alignedPos, avg)
    kabschWeights = intermediate_kabsch_weights(particleVariances)
    logLik = intermediate_kabsch_log_lik(alignedPos,avg,kabschWeights)
    # perform iterative alignment and average to converge average
    logLikDiff = 10
    step = 0
    while logLikDiff > thresh and step < maxSteps:
        # rezero new average
        newAvg = np.zeros((nAtoms,nDim),dtype=np.float64)
        # align trajectory to average and accumulate new average
        for ts in range(nFrames):
            alignedPos[ts] = weight_kabsch_rotate(alignedPos[ts], avg, kabschWeights)
            newAvg += alignedPos[ts]
        # finish average
        newAvg /= nFrames
        # compute log likelihood
        newLogLik = intermediate_kabsch_log_lik(alignedPos,avg,kabschWeights)
        logLikDiff = np.abs(newLogLik-logLik)
        logLik = newLogLik
        # compute new Kabsch Weights
        particleVariances = particle_variances_from_trajectory(alignedPos,newAvg)
        kabschWeightes = intermediate_kabsch_weights(particleVariances)
        # compute Distance between averages
        avgRmsd = weight_kabsch_dist_align(avg,newAvg,kabschWeights)
        avg = np.copy(newAvg)
        step += 1
        print(step, avgRmsd,logLik)
    return alignedPos, avg, particleVariances

# compute the average structure and covariance from trajectory data
@jit(nopython=True)
def traj_iterative_average_covar_weighted_kabsch(trajData,thresh=1E-3,maxSteps=300):
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
    covar /= nDim*(nFrames-1)
    # compute log likelihood
    logLik, kabschWeights = weight_kabsch_log_lik(alignedPos, avg, covar)
    # perform iterative alignment and average to converge average
    logLikDiff = 10
    step = 0
    while logLikDiff > thresh and step < maxSteps:
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
        newLogLik, kabschWeights = weight_kabsch_log_lik(alignedPos, newAvg, covar)
        logLikDiff = np.abs(newLogLik-logLik)
        logLik = newLogLik
        #kabschWeights = np.linalg.pinv(covar,rcond=1e-10)
        # compute Distance between averages
#        avgRmsd = weight_kabsch_dist_align(avg,newAvg,kabschWeights)
        avg = np.copy(newAvg)
        step += 1
#        print(step, logLik)
    return alignedPos, avg, covar

# compute the average structure and covariance from trajectory data
@jit(nopython=True)
def traj_iterative_average_weighted_kabsch(trajData,thresh=1E-3,maxSteps=200):
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
    covar /= nDim*(nFrames-1)
    # perform iterative alignment and average to converge average
    logLik, kabschWeights = weight_kabsch_log_lik(alignedPos, avg, covar)
    logLikDiff = 10
    step = 0
    while logLikDiff > thresh and step < maxSteps:
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
            covar += np.dot(disp[ts],disp[ts].T)
        covar /= nDim*(nFrames-1)
        newLogLik, kabschWeights = weight_kabsch_log_lik(alignedPos, newAvg, covar)
        logLikDiff = np.abs(newLogLik-logLik)
        logLik = newLogLik
        # compute RMSD between averages
#        avgRmsd = weight_kabsch_dist_align(avg,newAvg,kabschWeights)
        #print(step,avgRmsd)
        avg = np.copy(newAvg)
        step += 1
    # return average structure and aligned trajectory
    return avg, alignedPos

# compute the average structure from trajectory data
@jit(nopython=True)
def traj_iterative_average(trajData,thresh=1E-3):
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
    logLik = uniform_kabsch_log_lik(alignedPos,avg)
    # perform iterative alignment and average to converge log likelihood
    logLikDiff = 10
    count = 1
    while logLikDiff > thresh:
        # rezero new average
        newAvg = np.zeros((nAtoms,nDim),dtype=np.float64)
        # align trajectory to average and accumulate new average
        for ts in range(nFrames):
            alignedPos[ts] = kabsch_rotate(alignedPos[ts], avg)
            newAvg += alignedPos[ts]
        # finish average
        newAvg /= nFrames
        # compute log likelihood
        newLogLik = uniform_kabsch_log_lik(alignedPos,avg)
        logLikDiff = np.abs(newLogLik-logLik)
        logLik = newLogLik
        # copy new average
        avg = np.copy(newAvg)
        count += 1
    return avg, alignedPos

# compute the average structure from trajectory data
@jit(nopython=True)
def traj_iterative_average_covar(trajData,thresh=1E-3):
    # trajectory metadata
    nFrames = trajData.shape[0]
    nAtoms = trajData.shape[1]
    nDim = trajData.shape[2]
    # create numpy array of aligned positions
    alignedPos = np.copy(trajData)
    # Initialize average as first frame
    avg = np.copy(alignedPos[0]).astype(np.float64)
    logLik = uniform_kabsch_log_lik(alignedPos,avg)
    # perform iterative alignment and average to converge average
    logLikDiff = 10
    while logLikDiff > thresh:
        # rezero new average
        newAvg = np.zeros((nAtoms,nDim),dtype=np.float64)
        # align trajectory to average and accumulate new average
        for ts in range(nFrames):
            # align positions
            alignedPos[ts] = kabsch_rotate(alignedPos[ts], avg)
            newAvg += alignedPos[ts]
        # finish average
        newAvg /= nFrames
        # compute log likelihood
        newLogLik = uniform_kabsch_log_lik(alignedPos,avg)
        logLikDiff = np.abs(newLogLik-logLik)
        logLik = newLogLik
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
def traj_iterative_average_weighted(trajData, weights, prevAvg=None, thresh=1E-3):
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
    logLik = uniform_kabsch_log_lik(alignedPos,avg)
    # perform iterative alignment and average to converge average
    logLikDiff = 10
    while logLikDiff > thresh:
        # rezero new average
        newAvg = np.zeros((nAtoms,nDim),dtype=np.float64)
        # align trajectory to average and accumulate new average
        for ts in range(nFrames):
            # align to average
            alignedPos[ts] = kabsch_rotate(alignedPos[ts], avg)
            newAvg += weights[ts]*alignedPos[ts]
        # compute log likelihood
        newLogLik = uniform_kabsch_log_lik(alignedPos,avg)
        logLikDiff = np.abs(newLogLik-logLik)
        logLik = newLogLik
        # copy new avg
        avg = np.copy(newAvg)
    return alignedPos, avg

# compute the average structure and covariance from weighted trajectory data
@jit(nopython=True)
def traj_iterative_average_covar_weighted_weighted_kabsch(trajData, weights, prevAvg, preCovar, thresh=1E-3, maxSteps=100):
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
    logLik, kabschWeights = weight_kabsch_log_lik(alignedPos, avg, preCovar)
    # perform iterative alignment and average to converge average
    logLikDiff = 10
    step = 0
    while logLikDiff > thresh and step < maxSteps:
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
        covar /= 3.0
        newLogLik, kabschWeights = weight_kabsch_log_lik(alignedPos, newAvg, covar)
        logLikDiff = np.abs(newLogLik-logLik)
        logLik = newLogLik
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
def traj_iterative_average_var(trajData,thresh=1E-3):
    # trajectory metadata
    nFrames = trajData.shape[0]
    nAtoms = trajData.shape[1]
    nDim = trajData.shape[2]
    # create numpy array of aligned positions
    alignedPos = np.copy(trajData)
    # Initialize average as first frame
    avg = np.copy(alignedPos[0]).astype(np.float64)
    logLik = uniform_kabsch_log_lik(alignedPos,avg)
    # perform iterative alignment and average to converge average
    logLikDiff = 10
    while logLikDiff > thresh:
        # rezero new average
        newAvg = np.zeros((nAtoms,nDim),dtype=np.float64)
        # align trajectory to average and accumulate new average
        for ts in range(nFrames):
            # align positions
            alignedPos[ts] = kabsch_rotate(alignedPos[ts], avg)
            newAvg += alignedPos[ts]
        # finish average
        newAvg /= nFrames
        # compute log likelihood
        newLogLik = uniform_kabsch_log_lik(alignedPos,avg)
        logLikDiff = np.abs(newLogLik-logLik)
        logLik = newLogLik
        avg = np.copy(newAvg)
    # Compute variance correcting for normalizing by 3(nAtoms-1)*(nFrames-1)
    var = sample_variance((alignedPos-avg).flatten(),nDim*(nAtoms-1)*(nFrames-1))  
    return avg, var

# compute the average structure and variance from weighted trajectory data
@jit(nopython=True)
def traj_iterative_average_var_weighted(trajData, weights, prevAvg, thresh=1E-3):
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
    logLik = uniform_kabsch_log_lik(alignedPos,avg)
    # perform iterative alignment and average to converge average
    logLikDiff = 10
    while logLikDiff > thresh:
        # rezero new average
        newAvg = np.zeros((nAtoms,nDim),dtype=np.float64)
        # align trajectory to average and accumulate new average
        for ts in range(nFrames):
            # align to average
            alignedPos[ts] = kabsch_rotate(alignedPos[ts], avg)
            newAvg += weights[ts]*alignedPos[ts]
        # compute log likelihood
        newLogLik = uniform_kabsch_log_lik(alignedPos,avg)
        logLikDiff = np.abs(newLogLik-logLik)
        logLik = newLogLik
        # copy new avg
        avg = np.copy(newAvg)
    # loop over trajectory and compute variance
    var = np.float64(0.0)
    varNorm = 3*nAtoms-3
    for ts in range(nFrames):
        var += weights[ts]*sample_variance((alignedPos[ts]-avg).flatten(),varNorm)
    return avg, var

