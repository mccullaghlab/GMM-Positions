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
def weight_kabsch_log_lik(x, mu, precision, lpdet):
    # meta data
    n_frames = x.shape[0]
    # compute log Likelihood for all points
    log_lik = 0.0
    for i in range(n_frames):
        #disp = x[i] - mu
        for j in range(3):
            disp = x[i,:,j] - mu[:,j]
            log_lik += np.dot(disp,np.dot(precision,disp))
    log_lik += 3 * n_frames * lpdet
    log_lik *= -0.5
    return log_lik

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
def fast_weight_kabsch_rotate(mobile, weights_target):
    correlation_matrix = np.dot(np.transpose(mobile), weights_target)
    V, S, W_tr = np.linalg.svd(correlation_matrix)
    if np.linalg.det(V) * np.linalg.det(W_tr) < 0.0:
        V[:, -1] = -V[:, -1]
    rotation = np.dot(V, W_tr)
    mobile_prime = np.dot(mobile,rotation)
    return mobile_prime

@jit(nopython=True)
def covar_NxN_from_traj(disp):
    # trajectory metadata
    n_frames = disp.shape[0]
    n_atoms = disp.shape[1]
    # declare covar
    covar = np.zeros((n_atoms,n_atoms),np.float64)
    # loop and compute
    for ts in range(n_frames):
        covar += np.dot(disp[ts],disp[ts].T)
    # symmetrize and average covar
    covar /= 3*(n_frames-1)
    # done, return
    return covar

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
def traj_remove_cog_translation(traj_data):
    # trajectory metadata
    n_frames = traj_data.shape[0]
    n_atoms = traj_data.shape[1]
    nDim = traj_data.shape[2]
    # start be removing COG translation
    for ts in range(n_frames):
        mu = np.zeros(nDim)
        for atom in range(n_atoms):
            mu += traj_data[ts,atom]
        mu /= n_atoms
        traj_data[ts] -= mu
    return traj_data

@jit(nopython=True)
def particle_variances_from_trajectory(traj_data, avg):
    # meta data
    n_frames = traj_data.shape[0]
    n_atoms = traj_data.shape[1]
    # 
    disp = traj_data - avg
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
def traj_iterative_average_vars_intermediate_kabsch(traj_data,thresh=1E-3,max_steps=300):
    # trajectory metadata
    n_frames = traj_data.shape[0]
    n_atoms = traj_data.shape[1]
    nDim = traj_data.shape[2]
    # Initialize with uniform weighted Kabsch
    avg, aligned_pos = traj_iterative_average(traj_data,thresh)
    # Compute Kabsch Weights
    particleVariances = particle_variances_from_trajectory(aligned_pos, avg)
    kabsch_weights = intermediate_kabsch_weights(particleVariances)
    log_lik = intermediate_kabsch_log_lik(aligned_pos,avg,kabsch_weights)
    # perform iterative alignment and average to converge average
    log_lik_diff = 10
    step = 0
    while log_lik_diff > thresh and step < max_steps:
        # rezero new average
        new_avg = np.zeros((n_atoms,nDim),dtype=np.float64)
        # align trajectory to average and accumulate new average
        for ts in range(n_frames):
            aligned_pos[ts] = weight_kabsch_rotate(aligned_pos[ts], avg, kabsch_weights)
            new_avg += aligned_pos[ts]
        # finish average
        new_avg /= n_frames
        # compute log likelihood
        new_log_lik = intermediate_kabsch_log_lik(aligned_pos,avg,kabsch_weights)
        log_lik_diff = np.abs(new_log_lik-log_lik)
        log_lik = new_log_lik
        # compute new Kabsch Weights
        particleVariances = particle_variances_from_trajectory(aligned_pos,new_avg)
        kabschWeightes = intermediate_kabsch_weights(particleVariances)
        # compute Distance between averages
        avgRmsd = weight_kabsch_dist_align(avg,new_avg,kabsch_weights)
        avg = np.copy(new_avg)
        step += 1
        print(step, avgRmsd,log_lik)
    return aligned_pos, avg, particleVariances

# compute the average structure and covariance from trajectory data
@jit(nopython=True)
def traj_iterative_average_precision_weighted_kabsch(traj_data,thresh=1E-3,max_steps=300):
    # trajectory metadata
    n_frames = traj_data.shape[0]
    n_atoms = traj_data.shape[1]
    nDim = traj_data.shape[2]
    # Initialize with uniform weighted Kabsch
    avg, aligned_pos = traj_iterative_average(traj_data,thresh)
    # compute NxN covar
    covar = covar_NxN_from_traj(aligned_pos-avg)
    # determine precision and pseudo determinant 
    lpdet, precision, rank = pseudo_lpdet_inv(covar)
    # compute log likelihood
    log_lik = weight_kabsch_log_lik(aligned_pos, avg, precision, lpdet)
    # perform iterative alignment and average to converge average
    log_lik_diff = 10+thresh
    step = 0
    while log_lik_diff > thresh and step < max_steps:
        # rezero new average
        new_avg = np.zeros((n_atoms,nDim),dtype=np.float64)
        # align trajectory to average and accumulate new average
        weights_target = np.dot(precision,avg)
        for ts in range(n_frames):
            aligned_pos[ts] = fast_weight_kabsch_rotate(aligned_pos[ts], weights_target)
            new_avg += aligned_pos[ts]
        # finish average
        new_avg /= n_frames
        # compute new Kabsch Weights
        covar = covar_NxN_from_traj(aligned_pos-new_avg)
        # determine precision and pseudo determinant 
        lpdet, precision, rank = pseudo_lpdet_inv(covar)
        # compute log likelihood
        new_log_lik = weight_kabsch_log_lik(aligned_pos, new_avg, precision, lpdet)
        log_lik_diff = np.abs(new_log_lik-log_lik)
        log_lik = new_log_lik
        avg = np.copy(new_avg)
        step += 1
#        print(step, log_lik)
    return aligned_pos, avg, precision, lpdet

# compute the average structure and covariance from trajectory data
@jit(nopython=True)
def traj_iterative_average_weighted_kabsch(traj_data,thresh=1E-3,max_steps=200):
    # trajectory metadata
    n_frames = traj_data.shape[0]
    n_atoms = traj_data.shape[1]
    nDim = traj_data.shape[2]
    # Initialize with uniform weighted Kabsch
    avg, aligned_pos = traj_iterative_average(traj_data,thresh)
    # Compute Kabsch Weights
    disp = aligned_pos - avg
    covar = np.zeros((n_atoms,n_atoms),dtype=np.float64)
    for ts in range(n_frames):
        covar += np.dot(disp[ts],disp[ts].T)
    covar /= nDim*(n_frames-1)
    # determine precision and pseudo determinant 
    lpdet, precision, rank = pseudo_lpdet_inv(covar)
    # perform iterative alignment and average to converge average
    log_lik = weight_kabsch_log_lik(aligned_pos, avg, precision, lpdet)
    log_lik_diff = 10
    step = 0
    while log_lik_diff > thresh and step < max_steps:
        # rezero new average
        new_avg = np.zeros((n_atoms,nDim),dtype=np.float64)
        # align trajectory to average and accumulate new average
        for ts in range(n_frames):
            aligned_pos[ts] = weight_kabsch_rotate(aligned_pos[ts], avg, precision)
            new_avg += aligned_pos[ts]
        # finish average
        new_avg /= n_frames
        # compute new Kabsch Weights
        covar = np.zeros((n_atoms,n_atoms),dtype=np.float64)
        disp = aligned_pos - new_avg
        for ts in range(n_frames):
            covar += np.dot(disp[ts],disp[ts].T)
        covar /= nDim*(n_frames-1)
        # determine precision and pseudo determinant 
        lpdet, precision, rank = pseudo_lpdet_inv(covar)
        # compute new log likelihood
        new_log_lik = weight_kabsch_log_lik(aligned_pos, new_avg, precision, lpdet)
        log_lik_diff = np.abs(new_log_lik-log_lik)
        log_lik = new_log_lik
        avg = np.copy(new_avg)
        step += 1
    # return average structure and aligned trajectory
    return avg, aligned_pos

# compute the average structure from trajectory data
@jit(nopython=True)
def traj_iterative_average(traj_data,thresh=1E-3):
    # trajectory metadata
    n_frames = traj_data.shape[0]
    n_atoms = traj_data.shape[1]
    nDim = traj_data.shape[2]
    # create numpy array of aligned positions
    aligned_pos = np.copy(traj_data).astype(np.float64)
    # start be removing COG translation
    for ts in range(n_frames):
        mu = np.zeros(nDim)
        for atom in range(n_atoms):
            mu += aligned_pos[ts,atom]
        mu /= n_atoms
        aligned_pos[ts] -= mu
    # Initialize average as first frame
    avg = np.copy(aligned_pos[0]).astype(np.float64)
    log_lik = uniform_kabsch_log_lik(aligned_pos,avg)
    # perform iterative alignment and average to converge log likelihood
    log_lik_diff = 10
    count = 1
    while log_lik_diff > thresh:
        # rezero new average
        new_avg = np.zeros((n_atoms,nDim),dtype=np.float64)
        # align trajectory to average and accumulate new average
        for ts in range(n_frames):
            aligned_pos[ts] = kabsch_rotate(aligned_pos[ts], avg)
            new_avg += aligned_pos[ts]
        # finish average
        new_avg /= n_frames
        # compute log likelihood
        new_log_lik = uniform_kabsch_log_lik(aligned_pos,avg)
        log_lik_diff = np.abs(new_log_lik-log_lik)
        log_lik = new_log_lik
        # copy new average
        avg = np.copy(new_avg)
        count += 1
    return avg, aligned_pos

# compute the average structure from trajectory data
@jit(nopython=True)
def traj_iterative_average_covar(traj_data,thresh=1E-3):
    # trajectory metadata
    n_frames = traj_data.shape[0]
    n_atoms = traj_data.shape[1]
    nDim = traj_data.shape[2]
    # create numpy array of aligned positions
    aligned_pos = np.copy(traj_data)
    # Initialize average as first frame
    avg = np.copy(aligned_pos[0]).astype(np.float64)
    log_lik = uniform_kabsch_log_lik(aligned_pos,avg)
    # perform iterative alignment and average to converge average
    log_lik_diff = 10
    while log_lik_diff > thresh:
        # rezero new average
        new_avg = np.zeros((n_atoms,nDim),dtype=np.float64)
        # align trajectory to average and accumulate new average
        for ts in range(n_frames):
            # align positions
            aligned_pos[ts] = kabsch_rotate(aligned_pos[ts], avg)
            new_avg += aligned_pos[ts]
        # finish average
        new_avg /= n_frames
        # compute log likelihood
        new_log_lik = uniform_kabsch_log_lik(aligned_pos,avg)
        log_lik_diff = np.abs(new_log_lik-log_lik)
        log_lik = new_log_lik
        avg = np.copy(new_avg)
    covar = np.zeros((n_atoms*nDim,n_atoms*nDim),dtype=np.float64)
    # loop over trajectory and compute average and covariance
    for ts in range(n_frames):
        disp = (aligned_pos[ts]-avg).flatten()
        covar += np.outer(disp,disp)
    # finish average
    covar /= (n_frames-1)
    return avg, covar

# compute the average structure from weighted trajectory data
@jit(nopython=True)
def traj_iterative_average_weighted(traj_data, weights, prev_avg=None, thresh=1E-3):
    # trajectory metadata
    n_frames = traj_data.shape[0]
    n_atoms = traj_data.shape[1]
    nDim = traj_data.shape[2]
    nFeatures = n_atoms*nDim
    # determine normalization
    norm = np.power(np.sum(weights),-1)
    weights *= norm
    # create numpy array of aligned positions
    aligned_pos = np.copy(traj_data)
    # Initialize average 
    if prev_avg == None:
        avg = np.copy(traj_data[np.argmax(weights)])
    else:
        avg = np.copy(prev_avg)
    log_lik = uniform_kabsch_log_lik(aligned_pos,avg)
    # perform iterative alignment and average to converge average
    log_lik_diff = 10
    while log_lik_diff > thresh:
        # rezero new average
        new_avg = np.zeros((n_atoms,nDim),dtype=np.float64)
        # align trajectory to average and accumulate new average
        for ts in range(n_frames):
            # align to average
            aligned_pos[ts] = kabsch_rotate(aligned_pos[ts], avg)
            new_avg += weights[ts]*aligned_pos[ts]
        # compute log likelihood
        new_log_lik = uniform_kabsch_log_lik(aligned_pos,avg)
        log_lik_diff = np.abs(new_log_lik-log_lik)
        log_lik = new_log_lik
        # copy new avg
        avg = np.copy(new_avg)
    return aligned_pos, avg

# compute the average structure and covariance from weighted trajectory data
@jit(nopython=True)
def traj_iterative_average_precision_weighted_weighted_kabsch(traj_data, weights, prev_avg, prev_precision, prev_lpdet, thresh=1E-3, max_steps=100):
    # trajectory metadata
    n_frames = traj_data.shape[0]
    n_atoms = traj_data.shape[1]
    nDim = traj_data.shape[2]
    nFeatures = n_atoms*nDim
    # determine normalization
    norm = np.power(np.sum(weights),-1)
    weights *= norm
    # create numpy array of aligned positions
    aligned_pos = np.copy(traj_data)
    # Initialize average with previous average
    avg = np.copy(prev_avg)
    # compute log likelihood of current trajectory alignment
    log_lik = weight_kabsch_log_lik(aligned_pos, avg, prev_precision, prev_lpdet)
    # perform iterative alignment and average to converge average
    log_lik_diff = 10 + thresh
    precision = prev_precision
    step = 0
    while log_lik_diff > thresh and step < max_steps:
        # rezero new average
        new_avg = np.zeros((n_atoms,nDim),dtype=np.float64)
        # align trajectory to average and accumulate new average
        weights_target = np.dot(precision,avg)
        for ts in range(n_frames):
            # align to average
            aligned_pos[ts] = fast_weight_kabsch_rotate(aligned_pos[ts], weights_target)
            new_avg += weights[ts]*aligned_pos[ts]
        # compute covar using weights
        covar = np.zeros((n_atoms,n_atoms),dtype=np.float64)
        for ts in range(n_frames):
            disp = aligned_pos[ts] - new_avg
            covar += weights[ts]*np.dot(disp,disp.T)
        covar /= 3.0 # covar still needs to be averaged over x, y, z
        # determine precision and pseudo determinant 
        lpdet, precision, rank = pseudo_lpdet_inv(covar)
        # compute log likelihood
        new_log_lik = weight_kabsch_log_lik(aligned_pos, new_avg, precision, lpdet)
        log_lik_diff = np.abs(new_log_lik-log_lik)
        log_lik = new_log_lik
        # copy new avg
        avg = np.copy(new_avg)
        step += 1

    return avg, precision, lpdet

# align trajectory data to a reference structure
@jit(nopython=True)
def traj_align_weighted_kabsch(traj_data, ref, precision):
    # trajectory metadata
    n_frames = traj_data.shape[0]
    # create numpy array of aligned positions
    aligned_pos = np.copy(traj_data)
    weights_target = np.dot(precision,ref)
    for ts in range(n_frames):
        # align positions based on weighted Kabsch
        aligned_pos[ts] = fast_weight_kabsch_rotate(aligned_pos[ts], weights_target)
    return aligned_pos

# align trajectory data to a reference structure
@jit(nopython=True)
def traj_align(traj_data,ref):
    # trajectory metadata
    n_frames = traj_data.shape[0]
    # create numpy array of aligned positions
    aligned_pos = np.copy(traj_data)
    for ts in range(n_frames):
        # make sure positions are centered
        aligned_pos[ts] = kabsch_rotate(aligned_pos[ts], ref)
    return aligned_pos

# compute the covariance from trajectory data
# we assume the trajectory is aligned here
@jit(nopython=True)
def traj_covar(traj_data):
    # trajectory metadata
    n_frames = traj_data.shape[0]
    n_atoms = traj_data.shape[1]
    nDim = traj_data.shape[2]
    # Initialize average and covariance arrays
    avg = np.zeros((n_atoms*nDim))
    covar = np.zeros((n_atoms*nDim,n_atoms*nDim))
    # loop over trajectory and compute average and covariance
    for ts in range(n_frames):
        flat = traj_data[ts].flatten()
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
def traj_iterative_average_var(traj_data,thresh=1E-3):
    # trajectory metadata
    n_frames = traj_data.shape[0]
    n_atoms = traj_data.shape[1]
    nDim = traj_data.shape[2]
    # create numpy array of aligned positions
    aligned_pos = np.copy(traj_data)
    # Initialize average as first frame
    avg = np.copy(aligned_pos[0]).astype(np.float64)
    log_lik = uniform_kabsch_log_lik(aligned_pos,avg)
    # perform iterative alignment and average to converge average
    log_lik_diff = 10
    while log_lik_diff > thresh:
        # rezero new average
        new_avg = np.zeros((n_atoms,nDim),dtype=np.float64)
        # align trajectory to average and accumulate new average
        for ts in range(n_frames):
            # align positions
            aligned_pos[ts] = kabsch_rotate(aligned_pos[ts], avg)
            new_avg += aligned_pos[ts]
        # finish average
        new_avg /= n_frames
        # compute log likelihood
        new_log_lik = uniform_kabsch_log_lik(aligned_pos,avg)
        log_lik_diff = np.abs(new_log_lik-log_lik)
        log_lik = new_log_lik
        avg = np.copy(new_avg)
    # Compute variance correcting for normalizing by 3(n_atoms-1)*(n_frames-1)
    var = sample_variance((aligned_pos-avg).flatten(),nDim*(n_atoms-1)*(n_frames-1))  
    return avg, var

# compute the average structure and variance from weighted trajectory data
@jit(nopython=True)
def traj_iterative_average_var_weighted(traj_data, weights, prev_avg, thresh=1E-3):
    # trajectory metadata
    n_frames = traj_data.shape[0]
    n_atoms = traj_data.shape[1]
    nDim = traj_data.shape[2]
    nFeatures = n_atoms*nDim
    # enforce normalized weights
    norm = np.power(np.sum(weights),-1)
    weights *= norm
    # create numpy array of aligned positions
    aligned_pos = np.copy(traj_data)
    # Initialize average with previous average
    avg = np.copy(prev_avg)
    log_lik = uniform_kabsch_log_lik(aligned_pos,avg)
    # perform iterative alignment and average to converge average
    log_lik_diff = 10
    while log_lik_diff > thresh:
        # rezero new average
        new_avg = np.zeros((n_atoms,nDim),dtype=np.float64)
        # align trajectory to average and accumulate new average
        for ts in range(n_frames):
            # align to average
            aligned_pos[ts] = kabsch_rotate(aligned_pos[ts], avg)
            new_avg += weights[ts]*aligned_pos[ts]
        # compute log likelihood
        new_log_lik = uniform_kabsch_log_lik(aligned_pos,avg)
        log_lik_diff = np.abs(new_log_lik-log_lik)
        log_lik = new_log_lik
        # copy new avg
        avg = np.copy(new_avg)
    # loop over trajectory and compute variance
    var = np.float64(0.0)
    varNorm = 3*n_atoms-3
    for ts in range(n_frames):
        var += weights[ts]*sample_variance((aligned_pos[ts]-avg).flatten(),varNorm)
    return avg, var

