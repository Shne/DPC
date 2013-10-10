add_library(common SHARED 
  error.c 
  image2d.c
  )

install(TARGETS common DESTINATION lib)

install(FILES
  error.h
  image2d.h
  DESTINATION include)
