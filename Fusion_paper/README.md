# Code:

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


 






