#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <iostream>

int main(int argc, char* argv[]) {

	// create iterators
    // thrust::constant_iterator<int> first(10);
    // thrust::constant_iterator<int> last = first + 3;
	// thrust::counting_iterator<int> first(1);
    // thrust::counting_iterator<int> last = first + 100;



    // std::cout << first[100] << std::endl;

    // sum of [first, last)
	// std::cout << thrust::reduce(first, last) << std::endl; // returns 33 (i.e. 10 + 11 + 12)

	// thrust::device_vector<int> vec(3);
	// vec[0] = 10; vec[1] = 20; vec[2] = 30;

	// std::cout << thrust::make_transform_iterator(vec.begin(), negate<int>()) << std::endl;

	thrust::device_vector<int> A(3);
	thrust::device_vector<char> B(3);

	A[0] = 10; A[1] = 20; A[2] = 30;
    B[0] = 'x'; B[1] = 'y'; B[2] = 'z';

	// create iterator (type omitted)
    auto first = thrust::make_zip_iterator(thrust::make_tuple(A.begin(), B.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(A.end(), B.end()));

	// std::cout << std::get<0>(first[0]) << std::endl;
	// std::cout << std::get<0>(first[1]) << std::endl;
	// std::cout << std::get<0>(first[2]) << std::endl;

	return 0;
}
