#include <stdio.h>

__global__ void sum_test(float *a, float *b) {
    
    int tid = threadIdx.x;

    b[0] = 0;
    __syncthreads();
    // b[0] = a[tid] + 2;
    // printf("a[%d] is %2.2f\n", tid, a[tid]);
    // printf("the thread id is %d\n", tid);
    // printf("b[0] is: %2.1f\n", b[0]);
    // atomicAdd(&a[tid], 1);
    atomicAdd(&a[tid], 1);
    b[0] += a[tid];
}


__global__ void hist_compute(int *a, int *hist) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = tid + bid * blockDim.x;

    // printf("a[%d] is %d\n", idx, a[idx]);

    // hist[a[idx]] += 1;
    atomicAdd(&hist[a[idx]], 1);
}

int main(int argc, char* argv[]) {

    int pixel_num = 5120;
    int a[pixel_num];
    int length = 10;
    
    for (int i = 0; i < pixel_num; i++) {
        a[i] = i * (i + 1) % length;
	// printf("a[%d]=%d\n", i, a[i]);
    }

    int *hist = new int[length]();
    
    for (int i = 0; i < pixel_num; i++) {
        hist[a[i]] += 1;
    }

    for (int i = 0; i < length; i++) {
        printf("hist[%d]=%d\n", i, hist[i]);
    }

    int *aGpu, *histGpu;
    int hist2[length];
    cudaMalloc((void**)&aGpu, pixel_num * sizeof(int));
    cudaMalloc((void**)&histGpu, length * sizeof(int));
    cudaMemcpy(aGpu, a, pixel_num * sizeof(int), cudaMemcpyHostToDevice);

    hist_compute<<<pixel_num / 512, 512>>>(aGpu, histGpu);

    cudaMemcpy(hist2, histGpu, length * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < length; i++) {
        printf("hist[%d]=%d\n", i, hist2[i]);
    }

    return 0;
}
