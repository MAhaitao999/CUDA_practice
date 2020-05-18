#include <thrust/sort.h>
#include <thrust/copy.h>
#include <iostream>

int main(int argc, char *argv[]) {

	const int N = 6;
    int A[N] = {1, 4, 2, 8, 5, 7};

	int keys[N]    = {1, 4, 3, 8, 5, 7};
	char values[N] = {'a', 'b', 'c', 'd', 'e', 'f'};

    // thrust::sort(A, A + N);
	thrust::sort_by_key(keys, keys + N, values);

	for (int i = 0; i < N; i++) {
        std::cout << keys[i] << " ";	
	}
	std::cout << std::endl;

	for (int i = 0; i < N; i++) {
        std::cout << values[i] << " ";	
	}
	std::cout << std::endl;

	thrust::stable_sort(A, A + N, thrust::greater<int>());

	std::cout << "A:\n";
	for (int i = 0; i < N; i++) {
	    std::cout << A[i] << " ";
	}
	std::cout << std::endl;
	

}
