-------------
 Introduction
-------------
This example was written for the graduate course 
Data-parallel computing (dpc) held at Aarhus University. 
The code was originally written by Christian P. V. Christoffersen
and subsequently adapted by Thomas Sangild Sørensen.

-----------------
Cuda Installation
-----------------
On the department machines in Zuse 
(a01660 a01661 a01662 a01663 a01665 a01666 a01667 a01668 
a01669 a01670 a01672 a01674 tata tesla toyota) 
Cuda is preinstalled and you should be ready to go.

If you want to use your own machine, then Nvidia's CUDA 
toolkit and corresponding drivers must be installed:
http://developer.nvidia.com/cuda/cuda-downloads

-------------------
 Build instructions
-------------------
Cmake has been chosen as the overall compilation framework and 
supports both Linux and OSX with the instructions below.

The exercises use some common libraries, which must be compiled first. 

To compile these libraries execute the following commands 
starting in the DPC folder: 

> cd libs
> mkdir build
> cd build
> cmake ..
> make install

The compiled libraries and headers have now been installed in the
newly created lib and include folders in the DPC root folder.
The common lib folder (probably) needs to be added to your LD_LIBRARY_PATH, e.g.:
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:~/DPC/lib

To compile exercise01 we again use Cmake. 
Execute the following commands starting in the DPC folder: 

> cd exercise01
> mkdir build
> cd build
> cmake ..
> make install

The compiled exercise01 binary can now be found in DPC/bin.

--------------------
Executing exercise01
--------------------
This exercise's main purpose is to have you 
compiling and running your first Cuda program -- 
and get whatever hurdles you might encounter out of the way.

The program simply loads an image from disk, 
applies thresholding to the image, and saves the result back to disk. 

A raw image is necessary for input. 
Go to the DPC/exercise01/data folder.
Convert (using imagemagick) the provided lena.png to the raw format:

> convert lena.png gray:lena.raw

Run the program:

> ../../bin/exercise01 lena.raw 512 512 lena_filtered.raw 128

And convert back to png for more easy viewing:

> convert -size 512x512 -depth 8 gray:lena_filtered.raw lena_filtered.png
