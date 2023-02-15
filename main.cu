#include <iostream>
#include "Part1_ProgramingModel/Kernels.cu"


int main() {
    std::vector<float> A {1, 2, 3, 4, 5};
    std::vector<float> B {2, 2, 2, 2, 2};
    std::vector<float> C;
    vecAdd<<<1, 5>>>(A.data(), B.data(), C.data());
    return 0;
}
