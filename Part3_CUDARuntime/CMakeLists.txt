set(PART3_LIB part3_cudaruntime CACHE INTERNAL "")
file(GLOB PART3_SRC *.cu)

add_library(${PART3_LIB} ${PART3_SRC})