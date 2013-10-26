cmake_minimum_required(VERSION 2.8)
project(miro)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/../cmake)
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/..)

find_package(CUDA 4.1 REQUIRED)

# Compile kernels for shader models 1.0 and 2.0 as default
set(CUDA_NVCC_FLAGS "-gencode=arch=compute_10,code=\"sm_10,compute_10\" -gencode=arch=compute_20,code=\"sm_20,compute_20\" -G -g -use_fast_math")

SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fomit-frame-pointer -Wno-deprecated -use_fast_math")
SET (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -O3")

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

SET(SOURCES
	BLPatch
	BVH
	BuildScenes
	Camera
	Clock
	Console
	EnvMapMaterial
	GlossyHighlights
	HashGrid
	Image
	Lambert
	Material
	MiroWindow
	PFMLoader
	Plane
	Scene
	SpecularReflection
	SpecularRefraction
	Sphere
	TranslucentGlossyHighlights
	Triangle
	TriangleMesh
	TriangleMeshLoad
	main
	kernel.cu
	Vector3.cu
	FinalPassKernel.cu
)
#SET(SOURCES
#	kernel.cu
#	Vector3.cu
#	FinalPassKernel.cu
#)

link_directories(
	${CMAKE_SOURCE_DIR}/../lib
)

cuda_add_executable(miro
	${SOURCES}
)

target_link_libraries(miro 
	common
	cuda_utils
	opengl_utils
	${CUDA_LIBRARIES}
	${OPENGL_gl_LIBRARY}
	${OPENGL_glu_LIBRARY}
	${GLUT_LIBRARY}
	${GLEW_LIBRARY}
)

install(TARGETS miro DESTINATION bin)
