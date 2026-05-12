# Topological Fusion Lidar Embankments

## Overview
This repo provides a minimal, easy-to-follow implementation of the core ideas introduced in the paper "Topological Data Fusion of Multi-Season LiDAR Observations for Monitoring of Expansive Clay Embankments". It is intended to help readers understand, reproduce, and experiment with the proposed methods.

## files description:
The main file associated with this paper is "Fusion_paper," which contains different folders for each analysis in the paper. The associated explanations for each code are included in this folder as README.md.

# Fusion_paper

## Overview
This folder contains the majority of the code used to write the paper as follows:

### Main_TDA_code:
This folder includes the primary code for calculating key features associated with Topological Data Analysis (TDA). It presents methods for persistence homology (PH) and persistent diagrams (PD).


### Feature_importance:
This folder contains the code that receives feature values and ranks them by importance in principal component analysis (PCA).
In this paper, we have two sets of features in a CSV file. One file is related to the simulation analysis, and another one is related to the experimental analysis.

### Feature_over_sampling points:
This folder contains code that calculates feature values using different numbers of sampling points. This demonstrates how quickly the features converge as dataset sample sizes increase, aiming to reduce computational costs.

### Feature Over-Index:
This folder contains code that computes feature values based on different sizes of abnormalities in the simulation study. This is done to check the sensitivity of topological data analysis (TDA) in capturing varying sizes and shapes of defects.

### Logistic_regression:
This folder contains code that computes the solo-logistic regression for experimental analysis, where the dry and wet regions are classified feature-wise.

### PCA_logistic_regression:
This folder contains the main analysis of Principal Component Analysis (PCA), along with logistic regression for both simulation and experimental analyses. High-dimensional features (16 dimensions in this case) are reduced to two dimensions (PC1 and PC2). Logistic regression is then employed to classify different types of abnormalities in the simulation study and to distinguish between wet and dry conditions in the experimental analysis.



## Licensing and Citation

This work is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License [cc-by-sa 4.0].

[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC_BY--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)


Cite this as: 

@misc{Zardian2026TDAFusionRepo,
	author = {Zardian, Mohsen and Feng, Weicong and Zohuruzzaman, A. Q. M. and Downey, Austin R. J. Wei, Jie and Khan, Sadik and Schrader, Paul T. and Blasch, Erik and Ardiles-Cruz, Erika and Bakos, Jason and Imran, Jasim},
	title = {Paper: Topological Data Fusion of Multi-Season LiDAR Observations for Embankment Monitoring},
	year = {2026},
	howpublished = {\url{https://github.com/ARTS-Laboratory/paper-topological-fusion-lidar-embankments}},
	note  = {GitHub repository, ARTS Laboratory}
}













