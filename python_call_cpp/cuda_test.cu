#include <cuda_runtime.h>
#include <stdio.h>


__global__ void kernel() {

    int tid = threadIdx.x;

    if (tid < 8) {
        printf("inside the kernel\n");
    }
    else {
        printf("outside the kernel\n");
    }

}

int cuda(int a, int b) {
    kernel<<<1, 10>>>();
    cudaDeviceSynchronize();

    return 0;
}


