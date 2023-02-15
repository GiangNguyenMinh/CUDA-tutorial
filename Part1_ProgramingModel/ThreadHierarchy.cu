#include<iostream>
#include<vector>

/*
 * "threadIdx" là 1 vector 3 thành phần, tùy vào định nghĩa cho sử dụng vị trí của thread theo 1, 2 hay 3 chiều
 * Kết hợp các threads theo với só chiều cụ thể tạo ra các thread block từ đó có thể tính toán trên phần tử, vector hay ma trận
 *
 * Index của một thread và thread ID liên quan mật thiết với nhau:
 *  - Đối với block 1 chiều (Dx): thread idx và thread ID là như nhau
 *  - Đối với block 2 chiều (Dx, Dy): thread ID đối với thread idx (x, y) được tính bằng (x + y*Dx)
 *  - Đối với block 3 chiều (Dx, Dy, Dz): thread ID đối với thread idx (x, y, z) được tính bằng (x + y*Dx + z*Dx*Dy)
 */

//For example matAdd kernel thực hiện cộng 2 ma trân A, B kích thức NxN và lưu kết quả vào C
__global__ void matAdd(float A[3][3], float B[3][3], float C[3][3]){
    int i = threadIdx.x;
    int j = threadIdx.y;
    C[i][j] = A[i][j] + B[i][j];
}

/*
int main()
{
    ...
    int numBlocks = 1;
    dim3 threadPerBlock(3, 3);
    matAdd<<<numBlocks, threadPerBlock>>>(A, B, C);
    ...
}
*/

//----------------------------------------------------------------------------------------------------------------
/*
 * Việc số lượng threads trong block là có giới hạn, và tất cả các thread trong block sẽ shared tài nguyên bộ nhớ giới hạn
 *  Có thể sử dụng nhiều block để nhiều threads block
 *  Số lượng của threads per block và block per grid được chỉ định bên trong <<<...>>> có thể là dạng int hoặc dim3
 *  Mỗi block trong grid có thể xác định bằng cách truy cập index 1, 2 hoặc 3 chiều duy nhất trong kernel với biến build-in "blockIdx"
 *  Kích thước của block có thể truy cập trong kernel với biến build-in "blockDim"
 */

//For example matAddMultiBlock
__global__ void matAddMultiBlock(float A[16][16], float B[16][16], float C[16][16]){ // N = 16
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < 16 && j < 16){
        C[i][j] = A[i][j] + B[i][j];
    }
}

/*
int main(){
    ...
    dim3 threadsPerBlock(3, 3);
    dim3 numBlocks((16 + 3 - 1)/threadsPerBlock.x, (16 + 3 -1)/threadsPerBlock.y); // (N + threadPerBlock - 1)/threadPerBlock
    matAddMultiBlock<<<numBlocks, threadsPerBlock>>>(A, B, C);
    ...
}
*/

//Trong một block các thread có thể dùng chung bộ nhớ Shared memory
// __syscthreads() được gọi đóng vai trò như một barrier, tại đó tất cả các threads trong block phải đợi trước khi được cho phép xử lý