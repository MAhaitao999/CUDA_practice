#include <stdio.h>

#define LENGTH 16
#define THREADNUM 4
#define BLOCKNUM 2

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

__global__ void dot_product(float *a, float *b, float *r) {
    
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int total_thread_nnum = THREADNUM * BLOCKNUM;

	__shared__ float sData[THREADNUM];
	sData[tid] = 0;
	int global_id = tid + bid * blockDim.x;

	while (global_id < LENGTH) {
	    sData[tid] += a[global_id] * b[global_id];
		global_id += total_thread_nnum;
	}
	__syncthreads();
	
	for (int i = THREADNUM / 2; i > 0; i /= 2) {
	
		if (tid < i) {
		    sData[tid] = sData[tid] + sData[tid + i];
		}
		__syncthreads();
	}

	if (tid == 0) {
	    r[bid] = sData[0];
	}

}

int main(int argc, char* argv[]) {
    
	float a[LENGTH];
	float b[LENGTH];
	for (int i = 0; i < LENGTH; i++) {
	    a[i] = i * (i + 1);
		b[i] = i * (i - 2);
	}

	float *aGpu;
	cudaMalloc((void**)&aGpu, LENGTH * sizeof(float));
	cudaMemcpy(aGpu, a, LENGTH * sizeof(float), cudaMemcpyHostToDevice);

	float *bGpu;
	cudaMalloc((void**)&bGpu, LENGTH * sizeof(float));
	cudaMemcpy(bGpu, b, LENGTH * sizeof(float), cudaMemcpyHostToDevice);

	float *rGpu;
	cudaMalloc((void**)&rGpu, BLOCKNUM * sizeof(float));

	dot_product<<<BLOCKNUM, THREADNUM>>>(aGpu, bGpu, rGpu);

	float r[BLOCKNUM];
	cudaMemcpy(r, rGpu, BLOCKNUM * sizeof(float), cudaMemcpyDeviceToHost);

	float result = 0.0;
	for (int i = 0; i < BLOCKNUM; i++) {
	    printf("r[%d]: %f\n", i, r[i]);
		result += r[i];
	}

	printf("result is: %f\n", result);

	return 0;

}
