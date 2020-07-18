## opencv_util
opencv utility

It includes some implementation of computer vision papers using Eigen and OpenCV.
The implementation is for easy undersatanding of algorithms.


# How to use it  
mkdir build  
cd build  
cmake ..  
make -j4

Note, the CONDA_DIR in the CMakeLists.txt is hard-coded. You have to change it   
to the right directory.  


# Dependences  
conda install -c conda-forge opencv  
conda install -c conda-forge eigen  

