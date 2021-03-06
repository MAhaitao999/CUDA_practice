#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <iostream>


int main(int argc, char* argv[]) {

	// put three 1s in a device_vector
    thrust::device_vector<int> vec(5,0);
    vec[1] = 1;
    vec[3] = 1;
    vec[4] = 1;

	// count the 1s
    int result = thrust::count(vec.begin(), vec.end(), 1);
    // result is three

	std::cout << result << std::endl;

	return 0;

}

