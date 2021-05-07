# Data Sets and Source Codes for Early Research on a Neural Network to Compute Dimensionless Mean Curvature

Originally, I was computing the numerical mean curvature at the center nodes of stencils without bilinearly
interpolating it to the interface.  On the other hand, the neural network was approximating the dimensionless
curvature but at the interface.  When comparing these two quantities, the neural network was always better.

The updated versions of the files I used for generating the flower (i.e. 3-petaled polar rose) data sets are
located in this folder, where the numerical curvature is bilinearly interpolated at Gamma (i.e. in the negative
direction of the unit gradient that starts at the center of the 9-point stencil or sample).

Also, I'm using the utilities I wrote for collecting the samples directly from the grid, which do not rely
on any a priori mathematical information about the shape of the interface.

Furthermore, the updated codes produce ParaView files so that one can visualize the level-set function values 
for the two levels of curvature steepness in the polar rose interface.  These files are prefixed `old_mlds_`.

For the old implementation, please refer to the `ML_Datasets` branch.