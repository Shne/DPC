cmake_minimum_required(VERSION 2.8)
project(msd)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/../cmake)
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/..)

find_package(CUDA 4.1 REQUIRED)

# Compile kernels for shader models 1.0 and 2.0 as default
set(CUDA_NVCC_FLAGS "-gencode=arch=compute_10,code=\"sm_10,compute_10\" -gencode=arch=compute_20,code=\"sm_20,compute_20\"")

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

cuda_add_executable(msd main.cu)

target_link_libraries(msd 
  common
  cuda_utils
  opengl_utils
  ${CUDA_LIBRARIES}
  ${OPENGL_gl_LIBRARY}
  ${OPENGL_glu_LIBRARY}
  ${GLUT_LIBRARY}
  ${GLEW_LIBRARY}
  )

install(TARGETS msd DESTINATION bin)
