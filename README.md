DPC
===
A partly data parallelized photon mapper calculating subsurface scattering 


Prerequisites
-------------
1. Cuda Toolkit. Version 4.1 should be supported, but it had only been tested with 5.0 and 5.5
2. CMake >= 2.8
3. A C++ compiler (both GCC and Clang have been tested to work)
4. An Nvidia GPU capable for at least Compute Model 2.0


How to install
--------------
1. Clone this repository
2. Compile libraries: In `DPC/libs` do

  ```
  mkdir build
  cd build
  cmake ..
  make install
  ```
3. Compile project: Do the same, just in `DPC/miro`.

How to run
----------
Run `miro` in `DPC/miro/build/`
