/*
 * Complier nvcc build from PTX code or hight-level (as C++) --> binary code --> execute in device
 * System code (-arch) for set PTX compute capability, (-code) for set height-level
*/

// for example:
/* ```bash
$ nvcc program.cu -gencode arch=compute_xx, code=sm_xx //arch and code is gencode option in compute capability
// or the same is
$ nvcc program.cu -arch=sm_xx
// or the same is
$ nvcc program.cu - arch=compute_xx -code=sm_xx
 ```*/

// can use __CUDA_ARCH__ : print compute capability (or SM)