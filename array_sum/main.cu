#include <stdio.h>

static void HandleError(cudaError_t err,
		       const char *file,
		       int line) {
                           if (err != cudaSuccess) {
			       printf("%s in %s at line %d\n",
			       cudaGetErrorString(err),
			       file, line);
			       exit(EXIT_FAILURE);
			   }
                       }

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

int getThreadNum() {
    cudaDeviceProp prop;
    int count;

    HANDLE_ERROR(cudaGetDeviceCount(&count));
    printf("gpu num %d\n", count);
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
    printf("max thread num: %d\n", prop.maxThreadsPerBlock);
    printf("max grid dimensions: (%d, %d, %d)\n",
        prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

    return prop.maxThreadsPerBlock;
}

__global__ void sum(float *a, float *b, int num_p2) {
    int tid = threadIdx.x;

    __shared__ float sData[1024];
    sData[tid] = a[tid];
    __syncthreads();

    /*
    if (tid < 8) {
        sData[tid] = sData[tid] + sData[tid + 8];
    }
    __syncthreads();

    if (tid < 4) {
        sData[tid] = sData[tid] + sData[tid + 4];
    }
    __syncthreads();

    if (tid < 2) {
        sData[tid] = sData[tid] + sData[tid + 2];
    }
    __syncthreads();

    if (tid < 1) {
        sData[tid] = sData[tid] + sData[tid + 1];
    }
    __syncthreads();

    b[0] = sData[0];
    */
    for (int i = num_p2 / 2; i > 0; i /= 2) {
        if (tid < i) {
	    sData[tid] = sData[tid] + sData[tid + i];
	}
	__syncthreads();
    }

    *b = sData[0];     
}

inline int next_p2(int a) {
    int rval = 1;
    while (rval < a) {
        rval <<= 1;
    }
    return rval;
}

int main(int argc, char* argv[]) {
    
    int num = 16;
    int num_p2 = next_p2(num);
    printf("%d's p2 is %d\n", num, num_p2);
    float a[num];
    float a_tmp[num_p2];

    for (int i = 0; i < num; i++) {
        a[i] = i * (i + 1);
    }

    for (int i = 0; i < num_p2; i++) {
        if (i < num) {
	    a_tmp[i] = a[i];
	}
	else {
	    a_tmp[i] = 0.0;
	}
    }

    float *aGpu;
    cudaMalloc((void**)&aGpu, num_p2 * sizeof(float));
    cudaMemcpy(aGpu, a_tmp, num_p2 * sizeof(float), cudaMemcpyHostToDevice);

    float *bGpu;
    cudaMalloc((void**)&bGpu, 1 * sizeof(float));
    sum<<<1, 1024>>>(aGpu, bGpu, num_p2);

    float b[1];
    cudaMemcpy(b, bGpu, 1 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("the result is: %2.0f\n", b[0]);

    return 0;

}
