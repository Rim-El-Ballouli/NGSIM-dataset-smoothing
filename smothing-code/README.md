# Smoothing Code

### Description 
The python code performs the following:

   1. Smoothes out the noise in the local x and local y values in the NGSIM Dataset using Savitzky-Golay filter
   2. Recomputes the velocites and accelerations
   3. Saves smoothed dataset to three separate csv files

 In version 2.0 step 2 is optimized using matrix expressions. The x and y values are
 represented as a single matrix A and the next value for each x and y in matrix A is
 represented in matrix B. Next the distance is computed using numpy.expressions.
 
 
### Speed-up Benchmark 
 **BEFOR v1.0:** 768949652 function calls (767703838 primitive calls) in 1299.943 seconds
 
 **AFTER v2.0:** 12877249 function calls (12764933 primitive calls) in 179.409 seconds

The new version is upto 7 times faster
 
 
### Library Requirments
- python
- pandas
- scipy
- numexpr

You can add the corresponding libraries to your working environment using either pip or anaconda.


### Matrix Expression Animation 
The animation below depicts the process of smoothing using matrix expression. 

![smoothing process](https://github.com/Rim-El-Ballouli/NGSIM-US-101-trajectory-dataset-smoothing/blob/master/smothing-code/velocity_smoothing_process_2.gif)
 


