# GMM Positions

## Overview

This is a package to perform Gaussian Mixture Model (GMM) clustering on particle positions (in <img src="https://render.githubusercontent.com/render/math?math=\mathbb{R}^3">). Like other GMM schemes, the user must specify the number of clusters and a cluster initialization scheme (defaults to random).  This is specified in the object initialization line, analagous to how it is done for the sklean GaussianMixture package.  There are two choices for the form of the covariance but those are specified by calling different fit functions.  See preprint (https://arxiv.org/abs/2112.11424) for additional details.

## Installation

The package can be installed using pip

`pip install shapeGMM`

or downloaded and installed with 

`python setup.py install`

## Usage 

This package is designed to mimic the usage of the sklearn package.  You first initiliaze the object and then fit.  Predict can be done once the model is fit.  Fit and ppredict functions take particle position trajectories as input in the form of a (n_frames, n_atoms, 3) numpy array.

### Initialize:

`from shapeGMM import gmm_shapes`

`sgmm_object = gmm_shapes.ShapeGMM(n_clusters,verbose=True)`

### Fit:

Uniform (spherical, uncorrelated) covariance:

`aligned_trajectory = sgmm_object.fit_uniform(training_set_positions)`

Weighted (Kronecker product) covariance:

`aligned_trajectory = sgmm_object.fit_weighted(training_set_positions)`

### Predict:

Uniform (spherical, uncorrelated) covariance:

`clusters, aligned_traj, log_likelihood = sgmm_object.predict_uniform(full_trajectory_positions)`

Weighted (Kronecker product) covariance:

`clusters, aligned_traj, log_likelihood = sgmm_object.predict_weighted(full_trajectory_positions)`

## Description of Contents

## Test Cases

