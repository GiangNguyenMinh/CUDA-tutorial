#include<iostream>
#include<vector>

/*
 * Heterogeneous programing giả định rắng hệ thống thực thi trên cả một host và một device, mỗi cái có 1 memory riếng biêt.
 * Runtime cung cấp các hàm allocate, deallocate, copy device memory, tranfer data giữa host memory và device memory
 * Device memory có thể được cấp phá như 'linear memory' hoặc 'CUDA arrays':
 *  - 'CUDA arrays' là một lớp bộ nhớ được tối ưu cho texture fetching
 *  - 'linear memory' được cấp phát trong không gian địa chỉ thống nhất duy nhất, nghĩa là các đối tượng được cấp phát
 *    riêng biệt và có thể tham chiếu đến nhau thông qua con trỏ. Kích thước không gian địa chỉ dựa vào host system (CPU)
 *    và khả năng tính toán của GPU.
 */

/*
 * Linear memory:
 *   + cudaMalloc(&ptr, size): Cấp phát bộ nhớ
 *   + cudaFree(ptr): Free bộ nhớ
 *   + cudaMemcpy(dsPtr, srcPtr, size, cudaMemcpyHostToDevice): copy data từ host sang device
 *   + cudaMemcpy(dsPtr, srcPtr, size, cudamemcpyDeviceToHost): copy data từ device sang host
 *   + cudaMemcpy(dsPtr, srcPtr, size, cudamemcpyDeviceToDevice): copy data từ device sang device
 *   + cudaMallocPitch(**devPtr, *pitch, width * sizeof(float), height): 2D allocate
 *   + cudaMalloc3D(): 3D allocate
 *   + cudaMallocHost() và cudaHostRegister() là cấp phát cho bộ nhớ châm hơn (trong PageLockHostMemory.cu)
 */

// ----------------------------------------------------------------------------------------------------------
// For example VecAdd cộng 2 vector với cudaMalloc(), cudaFree(), cudaMemcpy()
// Device
__global__ void VecAdd(float *A, float *B, float*C, int outputSize){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < outputSize){
        C[idx] = A[idx] + B[idx];
    }
}

/*
// Host
int main(){
    int N = 5;

//    size_t size = N*sizeof(float);
//
//    float *h_A = (float*)malloc(size);
//    float *h_B = (float*)malloc(size);
//    float *h_C = (float*)malloc(size);

    // create vector on host
    std::vector<float> h_A(N);
    std::vector<float> h_B(N);
    std::vector<float> h_C(N);

    size_t size = h_A.size()*sizeof(float);

    // initialize device ptr
    float *d_A;
    float *d_B;
    float *d_C;

    // allocate in device memory
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // copy data from host to device
    cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);

    int threadPerBlock = 256;
    int blockPerGrid = (N + threadPerBlock - 1) / threadPerBlock;

    VecAdd<<<blockPerGrid, threadPerBlock>>>(d_A, d_B, d_C, N);

    // copy data from device to host
    cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToDevice);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
*/

// ---------------------------------------------------------------------------------------------------------------
/*
 * cudaMallocPitch() và cudaMalloc3D() được đề xuất để cấp phát  mảng 2D và 3D vì nó đảm bảo rằng sự cấp phát được đệm
 ** tương ứng để đạt được các yêu cầu căn chỉnh trong (Part5 DeviceMemoryAccesses.cu).
 * Để đinh truy cập vào địa chỉ hàng hoặc copies mảng 2D với các vùng device memory khác dùng:
 *   + cudaMemcpy2D(*dst, size_t dpitch, *src, size_t spitch, width*sizeof(float), height, cudamemcpyDeviceToHost!!option)
 *   + cudaMemcpy3D()
 */


// For example sử dụng cudaMallocPitch để câp phát cho mảng width x height 2D và các lặp qua các giá trị trong mảng
// Device
__global__ void PitchKernel(float *devPtr, size_t pitch, int width, int height){
    for (int r = 0; r < height; ++r){
        float *row = (float*)((char*)devPtr + r * pitch);
        for (int c = 0; c < width; ++c){
            float element = row[c];
            printf("element[%d][%d]: %f", r, c, element);
        }
    }
}

/*
// Host
int main(){
    int width = 64;
    int height = 64;
    float *devPtr;
    size_t pitch;
    cudaMallocPitch(&devPtr, &pitch, width*sizeof(float), height); // devPtr is pointer
    PitchKernel<<<10, 512>>>(devPtr, pitch, width, height);
    return 0;
}
*/

// For example sử dụng cudaMalloc3D để cấp phát một mảng 3D width x height x depth và cách lặp quá các phần tử
// Device
__global__ void Kernel3D(cudaPitchedPtr devPitchedPtr, int width, int height, int depth){
    char *devPtr = (char *)devPitchedPtr.ptr;
    size_t pitch = devPitchedPtr.pitch;
    size_t slicePitch = pitch * height;
    for (int z = 0; z < depth; ++z){
        char* slice = devPtr + z * slicePitch;
        for (int y = 0; y < height; ++y){
            float* row = (float*)(slice + y * pitch);
            for (int x = 0; x < width; ++x){
                float element = row[x];
                printf("element[%d][%d][%d]: %f", x, y, z, element);
            }
        }
    }
}

/*
//Host
int main(){
    int width = 64, height = 64, depth = 64;
    cudaExtent extent = make_cudaExtent(width*sizeof(float), height, depth);
    cudaPitchedPtr devPitchedPtr;
    cudaMalloc3D(&devPitchedPtr, extent);
    Kernel3D<<<64, 512>>>(devPitchedPtr, width, height, depth);
    return 0;
}
*/
