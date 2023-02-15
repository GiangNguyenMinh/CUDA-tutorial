set(PART1_LIB part1_programing_model CACHE INTERNAL "")
file(GLOB PART1_SRC *.cu)
#set(PART1_SRC Kernels.cu ThreadHierarchy.cu)

add_library(${PART1_LIB} ${PART1_SRC})

