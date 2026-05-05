
The strategies are:

1- To create different synthetic point clouds (data sets) using simple slop with abnormalities such as humps and cavities.  
2- To run the TDA over created point clouds.
3- To extract the prime features from the dataset.
4- To demonstrate the features' trend over different sample points.
5- To showcase how the features change when layered on top of one another.

# Toy-example:

## Codes:

### meshed slope with cavity_V2.py
* This code, as run once, demonstrates:
  create a slope with cavities. The cavities' dimensions and location can be adjusted by the user.
* The output of this code is a shape in the .obj format.
Note: The .obj will be imported into the CloudCompare software so that the point clouds will be created as.las files.

### meshed slope with hump.py
* This code, as run once, demonstrates:
  create a slope with humps. The humps' dimensions and location can be adjusted by the user.
* The output of this code is a shape in the .obj format.
Note: The .obj will be imported into the CloudCompare software so that the point clouds will be created as.las files.
  

### Abnormalities_Features-V3.py
* This code, as run once, demonstrates:
  The different plots (features vs. features) are 16 * (4*4 grid) plots, and the features over indexes
* The input of this code is an .xlsx " maim/Data/Abnormalities_Features-V3.xlsx" file that comes from the " TDA-feature extraction.py".
* This code gets the input and generates 17 separate figures.   
            

### TDA-feature extraction-V4.py
* This code, as run once, demonstrates:
  The features ( as a vector) for each specific abnormality with a fixed number of points (m) @ K=10 iterations.
* The input of this code is the.las file ( any arbitrary point cloud which is saved in the Data folder). The input can be called through the "file_path=" line in the code.
  Then, a dataframe will be created from those vectors as a matrix to be fitted as input into "Abnormalities_Features.py".
* This code calculates the median of the features at the end and saves it as a .CSV file.
  Note: This code needs to be developed to make a datafarme once as run.   

### TDA-all features with graph-V3.py
* This code, as run once, demonstrates:
  The overall trend of all features crosses the different number of points (e.g., m_values = [50, 200, 500, 1000, 1500, 2500, 3500, 4500]) @ k=10.
* * The input of this code is the.las file ( any arbitrary point cloud which is saved in the Data folder). The input can be called through the "file_path=" line in the code.
* This code normalizes all features value [0,1] then plots all of them in the single graph.
  

# Toy-example:

## plots:

The plots folder includes the 16 figures from "Abnormalities_Features-V3.py" and the accumulated feature value across the different number of points from " TDA-all features with graph-V3.py".
 
