# GMM Positions

## Overview

This is a package to perform Gaussian Mixture Model (GMM) clustering on particle positions (in R3). Like other GMM schemes, the user must specify the number of clusters and a cluster initialization scheme (defaults to random).  This is specified in the object initialization line, analagous to how it is done for the sklean GaussianMixture package.  There are two choices for the form of the covariance but those are specified by calling different fit functions.  See pending preprint (citation to be added) for additional details.

## Usage 

This package is designed to mimic the usage of the sklearn package.  You first initiliaze the object and then fit.  Predict can be done once the model is fit.

Initialize:

from shapeGMM import gmm_shapes
sgmm_object = gmm_shapes.ShapeGMM(3,init_cluster_method=`random',verbose=True)

Fit uniform:
aligned_trajectory = sgmm_object.fit_uniform(trajectory_positions)

Fit weighted:
aligned_trajectory = sgmm_object.fit_weighted(trajectory_positions)


## Description of Contents

## Test Cases

