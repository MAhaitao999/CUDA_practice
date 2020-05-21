#include <iostream>
#include <sstream>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


template <typename T>
__global__ void matrixMultiplicationKernel(T *A, T *B, T *C, int N) {
    
	int ROW = blockIdx.y * blockDim.y + threadIdx.y;
	int COL = blockIdx.x * blockDim.x + threadIdx.x;

	T tmpSum = 0;

	if (ROW < N && COL < N) {
	    for (int i = 0; i < N; i++) {
		    tmpSum += A[ROW * N + i] * B[i * N + COL];
		}
	}
	C[ROW * N + COL] = tmpSum;
}


template <typename T>
void matrixMultiplication(float *A, float *B, float *C, int M, int K, int N) {

	// declare the number of blocks per grid and the number of threads per block
	// use 1 to 512 threads per block
	dim3 threadsPerBlock(N, N);
	dim3 blocksPerGrid(1, 1);
	if (N*N > 512) {
	    threadsPerBlock.x = 512;
		threadsPerBlock.y = 512;
		blocksPerGrid.x = ceil(double(N)/double(threadsPerBlock.x));
		blocksPerGrid.y = ceil(double(N)/double(threadsPerBlock.y));
	}

	matrixMultiplicationKernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
	    std::stringstream strstr;
		strstr << "matrixMultiplicationKernel launch failed" << std::endl;
		strstr << "dimBlock: " << blocksPerGrid.x << ", " << blocksPerGrid.y << std::endl;
		strstr << "dimGrid: " << threadsPerBlock.x << ", " << threadsPerBlock.y << std::endl;
		strstr << cudaGetErrorString(error);
		throw strstr.str();
	}
}

template <typename T>
void map_matrix(pybind11::array_t<T> A, pybind11::array_t<T> B, pybind11::array_t<T> C, int M, int K, int N) {
	pybind11::buffer_info AA = A.request();
	pybind11::buffer_info BB = B.request();
	pybind11::buffer_info CC = C.request();

	if (AA.ndim != 1) {
	    std::stringstream strstr;
		strstr << "AA.ndim != 1" << std::endl;
		strstr << "AA.ndim: " << AA.ndim << std::endl;
		throw std::runtime_error(strstr.str());
	}

	if (BB.ndim != 1) {
	    std::stringstream strstr;
		strstr << "BB.ndim != 1" << std::endl;
		strstr << "BB.ndim: " << BB.ndim << std::endl;
		throw std::runtime_error(strstr.str());
	}

	if (CC.ndim != 1) {
	    std::stringstream strstr;
		strstr << "CC.ndim != 1" << std::endl;
		strstr << "CC.ndim: " << CC.ndim << std::endl;
		throw std::runtime_error(strstr.str());
	}

	int size_A = AA.shape[0];
	if (M*N != size_A) {
	    std::stringstream strstr;
		strstr << "Matrix A size != M * N" << std::endl;
		strstr << "Please confirm the shape" << std::endl;
		throw std::runtime_error(strstr.str());
	}

	int size_B = BB.shape[0];
	if (N*K != size_B) {
        std::stringstream strstr;
		strstr << "Matrix B size != N * K" << std::endl;
		strstr << "Please confirm the shape" << std::endl;
		throw std::runtime_error(strstr.str());
	}

	int size_C = CC.shape[0];
	if (M*N != size_C) {
	    std::stringstream strstr;
		strstr << "Matrix C size != M * K" << std::endl;
		strstr << "Please confirm the shape" << std::endl;
		throw std::runtime_error(strstr.str());
	}

	int size_bytes_A = size_A * sizeof(T);
	int size_bytes_B = size_B * sizeof(T);
	int size_bytes_C = size_C * sizeof(T);

	T *AGpu_ptr;
	T *BGpu_ptr;
	T *CGpu_ptr;

	cudaError_t error = cudaMalloc((void**)&AGpu_ptr, size_bytes_A);
	if (error != cudaSuccess) {
	    throw std::runtime_error(cudaGetErrorString(error));
	}

	error = cudaMalloc((void**)&BGpu_ptr, size_bytes_B);
	if (error != cudaSuccess) {
	    throw std::runtime_error(cudaGetErrorString(error));
	}

	error = cudaMalloc((void**)&CGpu_ptr, size_bytes_C);
	if (error != cudaSuccess) {
	    throw std::runtime_error(cudaGetErrorString(error));
	}

	T* ptr_A = reinterpret_cast<T*>(AA.ptr);
	T* ptr_B = reinterpret_cast<T*>(BB.ptr);
	T* ptr_C = reinterpret_cast<T*>(CC.ptr);

	error = cudaMemcpy(AGpu_ptr, ptr_A, size_bytes_A, cudaMemcpyHostToDevice);

	if (error != cudaSuccess) {
	    throw std::runtime_error(cudaGetErrorString(error));
	}

	error = cudaMemcpy(BGpu_ptr, ptr_B, size_bytes_B, cudaMemcpyHostToDevice);

	if (error != cudaSuccess) {
	    throw std::runtime_error(cudaGetErrorString(error));
	}

	matrixMultiplication<T>(AGpu_ptr, BGpu_ptr, CGpu_ptr, M, K, N);

	error = cudaMemcpy(ptr_C, CGpu_ptr, size_bytes_C, cudaMemcpyDeviceToHost);

	if (error != cudaSuccess) {
	    throw std::runtime_error(cudaGetErrorString(error));
	}
}

int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(matrix_multi_library, m) {
    m.doc() = "matrix multi using GPU";
	m.def("add", &add, "multi two matrix");
	m.def("matrix_multi", map_matrix<float>, "multi two matrix");
}
