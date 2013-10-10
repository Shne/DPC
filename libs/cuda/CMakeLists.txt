include_directories( 
  ${CMAKE_SOURCE_DIR}/common		     
  ${CUDA_INCLUDE_DIRS}
  )

add_library(cuda_utils SHARED 
  cuda_utilities.c 
  )

target_link_libraries(cuda_utils
  common
  ${CUDA_LIBRARIES}
  )

install(TARGETS cuda_utils DESTINATION lib)

install(FILES
  cuda_utilities.h
  float_util.hcu
  float_util_device.hcu
  uint_util.hcu
  DESTINATION include)
