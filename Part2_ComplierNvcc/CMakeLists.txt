set(PART2_LIB part2_complier_nvcc CACHE INTERNAL "")
file(GLOB PART2_SRC *.cu)

add_library(${PART2_LIB} ${PART2_SRC})