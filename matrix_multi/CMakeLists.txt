cmake_minimum_required(VERSION 3.6)
find_package(CUDA REQUIRED)
find_package(PythonInterp 3.6 REQUIRED)
find_package(PythonLibs 3.6 REQUIRED)

include_directories(
    ${PYTHON_INCLUDE_DIRS}
	/usr/local/include/pybind11
	)

link_directories(
	/usr/local/cuda/lib64
	)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

cuda_add_library(matrix_multi_library SHARED
  matrix.cu)

target_link_libraries(matrix_multi_library
	${PYTHON_LIBRARIES}
	cudart)

set_target_properties(matrix_multi_library PROPERTIES PREFIX "")
