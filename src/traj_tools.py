import numpy as np
import numba
from numba import jit

NumericThresh = 1E-150
LogNumericThresh = np.log(NumericThresh)
EigenValueThresh = 1E-10

@jit(nopython=True)
def sample_variance(zero_mean_data_array,norm):
    """ 
    Compute the variance of a zero meaned array.  Divide by normalization factor.  
    zero_mean_data_array    (required)  : float64 array of data
    norm                    (required)  : float64 value to divide variance by - supplied so one can substract appropriate values etc from normalization
    """
    # meta data from array
    nDataPoints = zero_mean_data_array.shape[0]
    # zero variance
    var = np.float64(0.0)
    # compute sum of variances over data
    for i in range(nDataPoints):
        var += zero_mean_data_array[i]**2
    # returned averaged variance
    return var/norm


@jit(nopython=True)
def weight_kabsch_dist_align(x1, x2, weights):
    """
    Compute the Mahalabonis distance between positions x1 and x2 after aligned x1 to x2 given Kabsch weights (inverse variance)
    x1                      (required)  : float64 array with dimensions (n_atoms,3) of one molecular configuration
    x2                      (required)  : float64 array with dimensions (n_atoms,3) of another molecular configuration
    weights                 (required)  : float64 matrix with dimensions (n_atoms, n_atoms) of inverse (n_atoms, n_atoms) covariance
    """
    # rotate x1 to x2 given Kabsch weights
    x1_prime = weight_kabsch_rotate(x1, x2, weights)
    # zero distance
    dist = 0.0
    # compute distance as sum over indepdent (because covar is n_atoms x n_atoms) dimensions
    for i in range(3):
        disp = x1_prime[:,i] - x2[:,i]
        dist += np.dot(disp,np.dot(weights,disp))
    # return distance - this is actually the squared Mahalabonis distance
    return dist

@jit(nopython=True)
def weight_kabsch_dist(x1, x2, weights):
    """
    Compute the Mahalabonis distance between positions x1 and x2 given Kabsch weights (inverse variance)
    x1                      (required)  : float64 array with dimensions (n_atoms,3) of one molecular configuration
    x2                      (required)  : float64 array with dimensions (n_atoms,3) of another molecular configuration
    weights                 (required)  : float64 matrix with dimensions (n_atoms, n_atoms) of inverse (n_atoms, n_atoms) covariance
    """
    # zero distance
    dist = 0.0
    # compute distance as sum over indepdent (because covar is n_atoms x n_atoms) dimensions
    for i in range(3):
        disp = x1[:,i] - x2[:,i]
        dist += np.dot(disp,np.dot(weights,disp))
    # return value
    return dist

@jit(nopython=True)
def pseudo_lpdet_inv(sigma):
    N = sigma.shape[0]
    e, v = np.linalg.eigh(sigma)
    precision = np.zeros(sigma.shape,dtype=np.float64)
    lpdet = 0.0
    rank = 0
    for i in range(N):
        if (e[i] > EigenValueThresh):
            lpdet += np.log(e[i])
            precision += 1.0/e[i]*np.outer(v[:,i],v[:,i])
            rank += 1
    return lpdet, precision, rank

@jit(nopython=True)
def lpdet_inv(sigma):
    N = sigma.shape[0]
    e, v = np.linalg.eigh(sigma)
    lpdet = 0.0
    for i in range(N):
        if (e[i] > EigenValueThresh):
            lpdet -= np.log(e[i])
    return lpdet

@jit(nopython=True)
def uniform_kabsch_log_lik(x, mu):
    # meta data
    n_frames = x.shape[0]
    n_atoms = x.shape[1]
    # compute log Likelihood for all points
    log_lik = 0.0
    sampleVar = 0.0
    for i in range(n_frames):
        for j in range(3):
            disp = x[i,:,j] - mu[:,j]
            temp = np.dot(disp,disp)
            sampleVar += temp
            log_lik += temp
    # finish variance
    sampleVar /= (n_frames-1)*3*(n_atoms-1)
    log_lik /= sampleVar
    log_lik +=  n_frames * 3 * (n_atoms-1) * np.log(sampleVar)
    log_lik *= -0.5
    return log_lik

@jit(nopython=True)
def intermediate_kabsch_log_lik(x, mu, kabsch_weights):
    # meta data
    n_frames = x.shape[0]
    # determine precision and pseudo determinant 
    lpdet = lpdet_inv(kabsch_weights)
    # compute log Likelihood for all points
    log_lik = 0.0
    for i in range(n_frames):
        #disp = x[i] - mu
        for j in range(3):
            disp = x[i,:,j] - mu[:,j]
            log_lik += np.dot(disp,np.dot(kabsch_weights,disp))
    log_lik += 3 * n_frames * lpdet
    log_lik *= -0.5
    return log_lik

@jit(nopython=True)
def weight_kabsch_log_lik(x, mu, covar):
    # meta data
    n_frames = x.shape[0]
    # determine precision and pseudo determinant 
    lpdet, precision, rank = pseudo_lpdet_inv(covar)
    # compute log Likelihood for all points
    log_lik = 0.0
    for i in range(n_frames):
        #disp = x[i] - mu
        for j in range(3):
            disp = x[i,:,j] - mu[:,j]
            log_lik += np.dot(disp,np.dot(precision,disp))
    log_lik += 3 * n_frames * lpdet
    log_lik *= -0.5
    return log_lik, precision

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
    n_atoms = mobile.shape[0]
    nDim = mobile.shape[1]
    mu1 = np.zeros(nDim)
    mu2 = np.zeros(nDim)
    for i in range(n_atoms):
        for j in range(nDim):
            mu1[j] += mobile[i,j]
            mu2[j] += target[i,j]
    mu1 /= n_atoms
    mu2 /= n_atoms
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
    n_frames = trajData.shape[0]
    n_atoms = trajData.shape[1]
    nDim = trajData.shape[2]
    # start be removing COG translation
    for ts in range(n_frames):
        mu = np.zeros(nDim)
        for atom in range(n_atoms):
            mu += trajData[ts,atom]
        mu /= n_atoms
        trajData[ts] -= mu
    return trajData

@jit(nopython=True)
def particle_variances_from_trajectory(trajData, avg):
    # meta data
    n_frames = trajData.shape[0]
    n_atoms = trajData.shape[1]
    # 
    disp = trajData - avg
    particleVariances = np.zeros(n_atoms,dtype=np.float64)
    for ts in range(n_frames):
        for atom in range(n_atoms):
            particleVariances[atom] += np.dot(disp[ts,atom],disp[ts,atom])
    particleVariances /= 3*(n_frames-1)
    return particleVariances

@jit(nopython=True)
def intermediate_kabsch_weights(variances):
    # meta data
    n_atoms = variances.shape[0]
    # kasbch weights are inverse of variances
    inverseVariances = np.power(variances,-1)
    kabsch_weights = np.zeros((n_atoms,n_atoms),dtype=np.float64)
    # force constant vector to be null space of kabsch weights
    wsum = np.sum(inverseVariances)
    for i in range(n_atoms):
        # Populate diagonal elements
        kabsch_weights[i,i] = inverseVariances[i]
        for j in range(n_atoms):
            kabsch_weights[i,j] -= inverseVariances[i]*inverseVariances[j]/wsum
    # return the weights
    return kabsch_weights

# compute the average structure and covariance from trajectory data
@jit(nopython=True)
def traj_iterative_average_vars_intermediate_kabsch(trajData,thresh=1E-3,maxSteps=300):
    # trajectory metadata
    n_frames = trajData.shape[0]
    n_atoms = trajData.shape[1]
    nDim = trajData.shape[2]
    # Initialize with uniform weighted Kabsch
    avg, alignedPos = traj_iterative_average(trajData,thresh)
    # Compute Kabsch Weights
    particleVariances = particle_variances_from_trajectory(alignedPos, avg)
    kabsch_weights = intermediate_kabsch_weights(particleVariances)
    log_lik = intermediate_kabsch_log_lik(alignedPos,avg,kabsch_weights)
    # perform iterative alignment and average to converge average
    log_lik_diff = 10
    step = 0
    while log_lik_diff > thresh and step < maxSteps:
        # rezero new average
        newAvg = np.zeros((n_atoms,nDim),dtype=np.float64)
        # align trajectory to average and accumulate new average
        for ts in range(n_frames):
            alignedPos[ts] = weight_kabsch_rotate(alignedPos[ts], avg, kabsch_weights)
            newAvg += alignedPos[ts]
        # finish average
        newAvg /= n_frames
        # compute log likelihood
        newLogLik = intermediate_kabsch_log_lik(alignedPos,avg,kabsch_weights)
        log_lik_diff = np.abs(newLogLik-log_lik)
        log_lik = newLogLik
        # compute new Kabsch Weights
        particleVariances = particle_variances_from_trajectory(alignedPos,newAvg)
        kabschWeightes = intermediate_kabsch_weights(particleVariances)
        # compute Distance between averages
        avgRmsd = weight_kabsch_dist_align(avg,newAvg,kabsch_weights)
        avg = np.copy(newAvg)
        step += 1
        print(step, avgRmsd,log_lik)
    return alignedPos, avg, particleVariances

# compute the average structure and covariance from trajectory data
@jit(nopython=True)
def traj_iterative_average_covar_weighted_kabsch(trajData,thresh=1E-3,maxSteps=300):
    # trajectory metadata
    n_frames = trajData.shape[0]
    n_atoms = trajData.shape[1]
    nDim = trajData.shape[2]
    # Initialize with uniform weighted Kabsch
    avg, alignedPos = traj_iterative_average(trajData,thresh)
    # Compute Kabsch Weights
    disp = alignedPos - avg
    covar = np.zeros((n_atoms,n_atoms),dtype=np.float64)
    for ts in range(n_frames):
        covar += np.dot(disp[ts],disp[ts].T)
    covar /= nDim*(n_frames-1)
    # compute log likelihood
    log_lik, kabsch_weights = weight_kabsch_log_lik(alignedPos, avg, covar)
    # perform iterative alignment and average to converge average
    log_lik_diff = 10
    step = 0
    while log_lik_diff > thresh and step < maxSteps:
        # rezero new average
        newAvg = np.zeros((n_atoms,nDim),dtype=np.float64)
        # align trajectory to average and accumulate new average
        for ts in range(n_frames):
            alignedPos[ts] = weight_kabsch_rotate(alignedPos[ts], avg, kabsch_weights)
            newAvg += alignedPos[ts]
        # finish average
        newAvg /= n_frames
        # compute new Kabsch Weights
        covar = np.zeros((n_atoms,n_atoms),dtype=np.float64)
        for ts in range(n_frames):
            disp = alignedPos[ts] - newAvg
            covar += np.dot(disp,disp.T)    
        covar /= nDim*(n_frames-1)
        # compute log likelihood
        newLogLik, kabsch_weights = weight_kabsch_log_lik(alignedPos, newAvg, covar)
        log_lik_diff = np.abs(newLogLik-log_lik)
        log_lik = newLogLik
        #kabsch_weights = np.linalg.pinv(covar,rcond=1e-10)
        # compute Distance between averages
#        avgRmsd = weight_kabsch_dist_align(avg,newAvg,kabsch_weights)
        avg = np.copy(newAvg)
        step += 1
#        print(step, log_lik)
    return alignedPos, avg, covar

# compute the average structure and covariance from trajectory data
@jit(nopython=True)
def traj_iterative_average_weighted_kabsch(trajData,thresh=1E-3,maxSteps=200):
    # trajectory metadata
    n_frames = trajData.shape[0]
    n_atoms = trajData.shape[1]
    nDim = trajData.shape[2]
    # Initialize with uniform weighted Kabsch
    avg, alignedPos = traj_iterative_average(trajData,thresh)
    # Compute Kabsch Weights
    disp = alignedPos - avg
    covar = np.zeros((n_atoms,n_atoms),dtype=np.float64)
    for ts in range(n_frames):
        covar += np.dot(disp[ts],disp[ts].T)
    covar /= nDim*(n_frames-1)
    # perform iterative alignment and average to converge average
    log_lik, kabsch_weights = weight_kabsch_log_lik(alignedPos, avg, covar)
    log_lik_diff = 10
    step = 0
    while log_lik_diff > thresh and step < maxSteps:
        # rezero new average
        newAvg = np.zeros((n_atoms,nDim),dtype=np.float64)
        # align trajectory to average and accumulate new average
        for ts in range(n_frames):
            alignedPos[ts] = weight_kabsch_rotate(alignedPos[ts], avg, kabsch_weights)
            newAvg += alignedPos[ts]
        # finish average
        newAvg /= n_frames
        # compute new Kabsch Weights
        covar = np.zeros((n_atoms,n_atoms),dtype=np.float64)
        for ts in range(n_frames):
            covar += np.dot(disp[ts],disp[ts].T)
        covar /= nDim*(n_frames-1)
        newLogLik, kabsch_weights = weight_kabsch_log_lik(alignedPos, newAvg, covar)
        log_lik_diff = np.abs(newLogLik-log_lik)
        log_lik = newLogLik
        # compute RMSD between averages
#        avgRmsd = weight_kabsch_dist_align(avg,newAvg,kabsch_weights)
        #print(step,avgRmsd)
        avg = np.copy(newAvg)
        step += 1
    # return average structure and aligned trajectory
    return avg, alignedPos

# compute the average structure from trajectory data
@jit(nopython=True)
def traj_iterative_average(trajData,thresh=1E-3):
    # trajectory metadata
    n_frames = trajData.shape[0]
    n_atoms = trajData.shape[1]
    nDim = trajData.shape[2]
    # create numpy array of aligned positions
    alignedPos = np.copy(trajData).astype(np.float64)
    # start be removing COG translation
    for ts in range(n_frames):
        mu = np.zeros(nDim)
        for atom in range(n_atoms):
            mu += alignedPos[ts,atom]
        mu /= n_atoms
        alignedPos[ts] -= mu
    # Initialize average as first frame
    avg = np.copy(alignedPos[0]).astype(np.float64)
    log_lik = uniform_kabsch_log_lik(alignedPos,avg)
    # perform iterative alignment and average to converge log likelihood
    log_lik_diff = 10
    count = 1
    while log_lik_diff > thresh:
        # rezero new average
        newAvg = np.zeros((n_atoms,nDim),dtype=np.float64)
        # align trajectory to average and accumulate new average
        for ts in range(n_frames):
            alignedPos[ts] = kabsch_rotate(alignedPos[ts], avg)
            newAvg += alignedPos[ts]
        # finish average
        newAvg /= n_frames
        # compute log likelihood
        newLogLik = uniform_kabsch_log_lik(alignedPos,avg)
        log_lik_diff = np.abs(newLogLik-log_lik)
        log_lik = newLogLik
        # copy new average
        avg = np.copy(newAvg)
        count += 1
    return avg, alignedPos

# compute the average structure from trajectory data
@jit(nopython=True)
def traj_iterative_average_covar(trajData,thresh=1E-3):
    # trajectory metadata
    n_frames = trajData.shape[0]
    n_atoms = trajData.shape[1]
    nDim = trajData.shape[2]
    # create numpy array of aligned positions
    alignedPos = np.copy(trajData)
    # Initialize average as first frame
    avg = np.copy(alignedPos[0]).astype(np.float64)
    log_lik = uniform_kabsch_log_lik(alignedPos,avg)
    # perform iterative alignment and average to converge average
    log_lik_diff = 10
    while log_lik_diff > thresh:
        # rezero new average
        newAvg = np.zeros((n_atoms,nDim),dtype=np.float64)
        # align trajectory to average and accumulate new average
        for ts in range(n_frames):
            # align positions
            alignedPos[ts] = kabsch_rotate(alignedPos[ts], avg)
            newAvg += alignedPos[ts]
        # finish average
        newAvg /= n_frames
        # compute log likelihood
        newLogLik = uniform_kabsch_log_lik(alignedPos,avg)
        log_lik_diff = np.abs(newLogLik-log_lik)
        log_lik = newLogLik
        avg = np.copy(newAvg)
    covar = np.zeros((n_atoms*nDim,n_atoms*nDim),dtype=np.float64)
    # loop over trajectory and compute average and covariance
    for ts in range(n_frames):
        disp = (alignedPos[ts]-avg).flatten()
        covar += np.outer(disp,disp)
    # finish average
    covar /= (n_frames-1)
    return avg, covar

# compute the average structure from weighted trajectory data
@jit(nopython=True)
def traj_iterative_average_weighted(trajData, weights, prevAvg=None, thresh=1E-3):
    # trajectory metadata
    n_frames = trajData.shape[0]
    n_atoms = trajData.shape[1]
    nDim = trajData.shape[2]
    nFeatures = n_atoms*nDim
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
    log_lik = uniform_kabsch_log_lik(alignedPos,avg)
    # perform iterative alignment and average to converge average
    log_lik_diff = 10
    while log_lik_diff > thresh:
        # rezero new average
        newAvg = np.zeros((n_atoms,nDim),dtype=np.float64)
        # align trajectory to average and accumulate new average
        for ts in range(n_frames):
            # align to average
            alignedPos[ts] = kabsch_rotate(alignedPos[ts], avg)
            newAvg += weights[ts]*alignedPos[ts]
        # compute log likelihood
        newLogLik = uniform_kabsch_log_lik(alignedPos,avg)
        log_lik_diff = np.abs(newLogLik-log_lik)
        log_lik = newLogLik
        # copy new avg
        avg = np.copy(newAvg)
    return alignedPos, avg

# compute the average structure and covariance from weighted trajectory data
@jit(nopython=True)
def traj_iterative_average_covar_weighted_weighted_kabsch(trajData, weights, prevAvg, preCovar, thresh=1E-3, maxSteps=100):
    # trajectory metadata
    n_frames = trajData.shape[0]
    n_atoms = trajData.shape[1]
    nDim = trajData.shape[2]
    nFeatures = n_atoms*nDim
    # determine normalization
    norm = np.power(np.sum(weights),-1)
    weights *= norm
    # create numpy array of aligned positions
    alignedPos = np.copy(trajData)
    # Initialize average with previous average
    avg = np.copy(prevAvg)
    # set kasbch weights to inverse of current NxN covars
    log_lik, kabsch_weights = weight_kabsch_log_lik(alignedPos, avg, preCovar)
    # perform iterative alignment and average to converge average
    log_lik_diff = 10
    step = 0
    while log_lik_diff > thresh and step < maxSteps:
        # rezero new average
        newAvg = np.zeros((n_atoms,nDim),dtype=np.float64)
        # align trajectory to average and accumulate new average
        for ts in range(n_frames):
            # align to average
            alignedPos[ts] = weight_kabsch_rotate(alignedPos[ts], avg, kabsch_weights)
            newAvg += weights[ts]*alignedPos[ts]
        # compute new Kabsch Weights
        covar = np.zeros((n_atoms,n_atoms),dtype=np.float64)
        for ts in range(n_frames):
            disp = alignedPos[ts] - newAvg
            covar += weights[ts]*np.dot(disp,disp.T)
        covar /= 3.0
        newLogLik, kabsch_weights = weight_kabsch_log_lik(alignedPos, newAvg, covar)
        log_lik_diff = np.abs(newLogLik-log_lik)
        log_lik = newLogLik
        # copy new avg
        avg = np.copy(newAvg)
        step += 1

    return avg, covar

# align trajectory data to a reference structure
@jit(nopython=True)
def traj_align_weighted_kabsch(trajData,ref, covar):
    # trajectory metadata
    n_frames = trajData.shape[0]
    # kabsch weights
    kabsch_weights = np.linalg.pinv(covar,rcond=1e-10)
    # create numpy array of aligned positions
    alignedPos = np.copy(trajData)
    for ts in range(n_frames):
        # align positions based on weighted Kabsch
        alignedPos[ts] = weight_kabsch_rotate(alignedPos[ts], ref, kabsch_weights)
    return alignedPos

# align trajectory data to a reference structure
@jit(nopython=True)
def traj_align(trajData,ref):
    # trajectory metadata
    n_frames = trajData.shape[0]
    # create numpy array of aligned positions
    alignedPos = np.copy(trajData)
    for ts in range(n_frames):
        # make sure positions are centered
        alignedPos[ts] = kabsch_rotate(alignedPos[ts], ref)
    return alignedPos

# compute the covariance from trajectory data
# we assume the trajectory is aligned here
@jit(nopython=True)
def traj_covar(trajData):
    # trajectory metadata
    n_frames = trajData.shape[0]
    n_atoms = trajData.shape[1]
    nDim = trajData.shape[2]
    # Initialize average and covariance arrays
    avg = np.zeros((n_atoms*nDim))
    covar = np.zeros((n_atoms*nDim,n_atoms*nDim))
    # loop over trajectory and compute average and covariance
    for ts in range(n_frames):
        flat = trajData[ts].flatten()
        avg += flat
        covar += np.outer(flat,flat)
    # finish averages
    avg /= n_frames
    covar /= n_frames
    # finish covar
    covar -= np.outer(avg,avg)
    return covar

# compute the time separated covariance matrix
@jit(nopython=True)
def traj_time_covar(traj1, traj2, mean1, mean2, lag):
    # trajectory metadata
    n_frames = traj1.shape[0]
    n_atoms = traj1.shape[1]
    nDim = traj1.shape[2]
    # declare covar
    covar = np.zeros((n_atoms*nDim,n_atoms*nDim),dtype=np.float64)
    # loop over trajectory and compute average and covariance
    for ts in range(n_frames-lag):
        disp1 = traj1[ts].flatten()-mean1.flatten()
        disp2 = traj2[ts+lag].flatten()-mean2.flatten()
        covar += np.outer(disp1,disp2)
    # finish average
    covar /= (n_frames-lag)
    return covar

# compute the average structure and variance from trajectory data
@jit(nopython=True)
def traj_iterative_average_var(trajData,thresh=1E-3):
    # trajectory metadata
    n_frames = trajData.shape[0]
    n_atoms = trajData.shape[1]
    nDim = trajData.shape[2]
    # create numpy array of aligned positions
    alignedPos = np.copy(trajData)
    # Initialize average as first frame
    avg = np.copy(alignedPos[0]).astype(np.float64)
    log_lik = uniform_kabsch_log_lik(alignedPos,avg)
    # perform iterative alignment and average to converge average
    log_lik_diff = 10
    while log_lik_diff > thresh:
        # rezero new average
        newAvg = np.zeros((n_atoms,nDim),dtype=np.float64)
        # align trajectory to average and accumulate new average
        for ts in range(n_frames):
            # align positions
            alignedPos[ts] = kabsch_rotate(alignedPos[ts], avg)
            newAvg += alignedPos[ts]
        # finish average
        newAvg /= n_frames
        # compute log likelihood
        newLogLik = uniform_kabsch_log_lik(alignedPos,avg)
        log_lik_diff = np.abs(newLogLik-log_lik)
        log_lik = newLogLik
        avg = np.copy(newAvg)
    # Compute variance correcting for normalizing by 3(n_atoms-1)*(n_frames-1)
    var = sample_variance((alignedPos-avg).flatten(),nDim*(n_atoms-1)*(n_frames-1))  
    return avg, var

# compute the average structure and variance from weighted trajectory data
@jit(nopython=True)
def traj_iterative_average_var_weighted(trajData, weights, prevAvg, thresh=1E-3):
    # trajectory metadata
    n_frames = trajData.shape[0]
    n_atoms = trajData.shape[1]
    nDim = trajData.shape[2]
    nFeatures = n_atoms*nDim
    # determine normalization
    norm = np.power(np.sum(weights),-1)
    weights *= norm
    # create numpy array of aligned positions
    alignedPos = np.copy(trajData)
    # Initialize average with previous average
    avg = np.copy(prevAvg)
    log_lik = uniform_kabsch_log_lik(alignedPos,avg)
    # perform iterative alignment and average to converge average
    log_lik_diff = 10
    while log_lik_diff > thresh:
        # rezero new average
        newAvg = np.zeros((n_atoms,nDim),dtype=np.float64)
        # align trajectory to average and accumulate new average
        for ts in range(n_frames):
            # align to average
            alignedPos[ts] = kabsch_rotate(alignedPos[ts], avg)
            newAvg += weights[ts]*alignedPos[ts]
        # compute log likelihood
        newLogLik = uniform_kabsch_log_lik(alignedPos,avg)
        log_lik_diff = np.abs(newLogLik-log_lik)
        log_lik = newLogLik
        # copy new avg
        avg = np.copy(newAvg)
    # loop over trajectory and compute variance
    var = np.float64(0.0)
    varNorm = 3*n_atoms-3
    for ts in range(n_frames):
        var += weights[ts]*sample_variance((alignedPos[ts]-avg).flatten(),varNorm)
    return avg, var

