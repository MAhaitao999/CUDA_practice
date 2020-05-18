#include <stdio.h>
#include <ctime>

#define CUDA_KERNEL_LOOP(i, n) \
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
		 i < (n); \
		 i += blockDim.x * gridDim.x)

__global__ void distance(float *xSquare, float *ySquare, int *result, int testNum) {
    // int tid = threadIdx.x;
	// int bid = blockIdx.x;
	// int idx = tid + blockDim.x * bid;
	// printf("idx is: %d\n", idx);
	CUDA_KERNEL_LOOP(index, testNum) {
	    // if (index < num) 
		{
		    float temp = (xSquare[index] - 1) * (xSquare[index] - 1) + (ySquare[index] - 1) * (ySquare[index] - 1);
			if (temp < 1) {
			    result[index] = 1;
			}
			else {
			    result[index] = 0;
				// printf("**********************************************************");
			}
		}
	}
}

__global__ void sum_array(int *a, int *b, int testNum) {
    
	int tid = threadIdx.x;
	b[0] = 0;
	__shared__ float sData[512];

	for (int count = 0; count < ceilf(testNum / 512); count++) {
	    if (tid + count * 512 < testNum) {
		    sData[tid] = a[tid + count * 512];
			__syncthreads();
		}

		for (int i = 512 / 2; i > 0; i /= 2) {
		    if (tid < i) {
			    sData[tid] = sData[tid] + sData[tid + i];
			}
			__syncthreads();
		}

		if (tid == 0) {
		    b[0] += sData[0];
		}
	}
}

int main(int argc, char* argv[]) {

	int testNum = 100000000;
	srand((int)time(0));

	float *xSquare = new float[testNum];
	float *ySquare = new float[testNum];

	for (int i = 0; i < testNum; i++) {
	    xSquare[i] = rand() % 10000 * 1.0 / 10000;
		ySquare[i] = rand() % 10000 * 1.0 / 10000;
	}

	float *xSquareGpu;
	cudaMalloc((void**)&xSquareGpu, testNum * sizeof(float));
	cudaMemcpy(xSquareGpu, xSquare, testNum * sizeof(float), cudaMemcpyHostToDevice);

	float *ySquareGpu;
	cudaMalloc((void**)&ySquareGpu, testNum * sizeof(float));
	cudaMemcpy(ySquareGpu, ySquare, testNum * sizeof(float), cudaMemcpyHostToDevice);

	int threadNum = 1024;
	int blockNum = 512;
	int *resultGpu;
	cudaMalloc((void**)&resultGpu, testNum * sizeof(int));
	distance<<<blockNum, threadNum>>>(xSquareGpu, ySquareGpu, resultGpu, testNum);

	int *result = new int[testNum];
	cudaMemcpy(result, resultGpu, testNum * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < 10; i++) {
	    printf("result[%d] is %d\n", i, result[i]);
	}

	int *bGpu;
	cudaMalloc((void**)&bGpu, 1 * sizeof(int));

	sum_array<<<1, 512>>>(resultGpu, bGpu, testNum);

	int b[1];
	cudaMemcpy(b, bGpu, 1 * sizeof(int), cudaMemcpyDeviceToHost);

	printf("the b[0] is: %d\n", b[0]);
	printf("the appropriate pi is: %2.6f\n", float(b[0])/testNum*4);
    
	return 0;
}
