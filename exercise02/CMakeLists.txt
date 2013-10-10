cmake_minimum_required(VERSION 2.8)
project(exercise02)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/../cmake)
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/..)

find_package(CUDA 4.1 REQUIRED)

# Compile kernels for shader models 1.0 and 2.0, 3.0, and 3.5 as default
set(CUDA_NVCC_FLAGS "-gencode=arch=compute_10,code=\"sm_10,compute_10\" -gencode=arch=compute_20,code=\"sm_20,compute_20\" -gencode=arch=compute_30,code=\"sm_30,compute_30\" -gencode=arch=compute_35,code=\"sm_35,compute_35\"")

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)

include_directories(
  ${CMAKE_SOURCE_DIR}/../include
  ${CUDA_INCLUDE_DIRS}
  ${OPENGL_INCLUDE_DIR}
  ${GLUT_INCLUDE_DIR}
  ${GLEW_INCLUDE_DIR}
  )

link_directories(
  ${CMAKE_SOURCE_DIR}/../lib
  )

cuda_add_executable(exercise02 main.cpp rd_kernel.cu)

target_link_libraries(exercise02 
  common
  cuda_utils
  opengl_utils
  ${CUDA_LIBRARIES}
  ${OPENGL_gl_LIBRARY}
  ${OPENGL_glu_LIBRARY}
  ${GLUT_LIBRARY}
  ${GLEW_LIBRARY}
  )

install(TARGETS exercise02 DESTINATION bin)
