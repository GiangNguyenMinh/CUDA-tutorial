#include<iostream>
#include<vector>

/*
 * CUDA C++ extend C++ bằng cách định nghĩa C++ funtions hay còn được gọi là kernels
 * Khi kernel được gọi sẽ thực hiện song song N lần trong N threads CUDA
 *
 * Kernel được định nghĩa với __global__ declaration specifier
 * Khi được gọi thì cần truyền vào số lượng thread CUDA sẽ thực thi trong <<<...>>>
 * Mỗi thread thực thi kernel sẽ có một thread ID độc nhất và có thể truy cập trong kernel thông qua biến built-in "threadIdx"
 */

/*
// Cấu trúc chung 1 kernel thường có
// Device
__global__ typeReturn nameFC(inputPtr, outputPtr, sizeOutput){
    ...
}
// Host
// N là output size
nameFC<<<blocksPerGrid = (N + threadPerBlock - 1)/threadPerBlock, threadPerBlock>>>(inputPtr, outputPtr, sizeOutput);
*/


//For example vecAdd kernel cộng 2 vector
__global__ void vecAdd(float *A, float *B, float *C){
    int i = threadIdx.x; //get thread idx của thread đang thực thi kernel (thread có thể có 3 chiều x, y, z tùy theo định nghĩa trong <<<...>>>
    C[i] = A[i] + B[i];
}

/*
int main()
{
    ...
    vecAdd<<<1, N>>>(A.data(), B.data(), C.data());
    ...
}
*/