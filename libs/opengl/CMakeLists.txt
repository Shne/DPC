include_directories( 
  ../common
  ${OPENGL_INCLUDE_DIR}
  )

add_library(opengl_utils SHARED 
  glsl_shader.c 
  image2d_opengl.c
  opengl_utilities.c
  )

target_link_libraries(opengl_utils
  common
  ${OPENGL_gl_LIBRARY}
  )

install(TARGETS opengl_utils DESTINATION lib)

install(FILES
  glsl_shader.h 
  image2d_opengl.h
  opengl_utilities.h
  DESTINATION include)
