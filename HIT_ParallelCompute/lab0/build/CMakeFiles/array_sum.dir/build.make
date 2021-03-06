# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.9

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /workspace/CUDA_practice/HIT_ParallelCompute/lab0

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /workspace/CUDA_practice/HIT_ParallelCompute/lab0/build

# Include any dependencies generated for this target.
include CMakeFiles/array_sum.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/array_sum.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/array_sum.dir/flags.make

CMakeFiles/array_sum.dir/main.cu.o: CMakeFiles/array_sum.dir/flags.make
CMakeFiles/array_sum.dir/main.cu.o: ../main.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/CUDA_practice/HIT_ParallelCompute/lab0/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/array_sum.dir/main.cu.o"
	/usr/local/cuda/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /workspace/CUDA_practice/HIT_ParallelCompute/lab0/main.cu -o CMakeFiles/array_sum.dir/main.cu.o

CMakeFiles/array_sum.dir/main.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/array_sum.dir/main.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/array_sum.dir/main.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/array_sum.dir/main.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/array_sum.dir/main.cu.o.requires:

.PHONY : CMakeFiles/array_sum.dir/main.cu.o.requires

CMakeFiles/array_sum.dir/main.cu.o.provides: CMakeFiles/array_sum.dir/main.cu.o.requires
	$(MAKE) -f CMakeFiles/array_sum.dir/build.make CMakeFiles/array_sum.dir/main.cu.o.provides.build
.PHONY : CMakeFiles/array_sum.dir/main.cu.o.provides

CMakeFiles/array_sum.dir/main.cu.o.provides.build: CMakeFiles/array_sum.dir/main.cu.o


# Object files for target array_sum
array_sum_OBJECTS = \
"CMakeFiles/array_sum.dir/main.cu.o"

# External object files for target array_sum
array_sum_EXTERNAL_OBJECTS =

CMakeFiles/array_sum.dir/cmake_device_link.o: CMakeFiles/array_sum.dir/main.cu.o
CMakeFiles/array_sum.dir/cmake_device_link.o: CMakeFiles/array_sum.dir/build.make
CMakeFiles/array_sum.dir/cmake_device_link.o: CMakeFiles/array_sum.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/workspace/CUDA_practice/HIT_ParallelCompute/lab0/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles/array_sum.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/array_sum.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/array_sum.dir/build: CMakeFiles/array_sum.dir/cmake_device_link.o

.PHONY : CMakeFiles/array_sum.dir/build

# Object files for target array_sum
array_sum_OBJECTS = \
"CMakeFiles/array_sum.dir/main.cu.o"

# External object files for target array_sum
array_sum_EXTERNAL_OBJECTS =

array_sum: CMakeFiles/array_sum.dir/main.cu.o
array_sum: CMakeFiles/array_sum.dir/build.make
array_sum: CMakeFiles/array_sum.dir/cmake_device_link.o
array_sum: CMakeFiles/array_sum.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/workspace/CUDA_practice/HIT_ParallelCompute/lab0/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA executable array_sum"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/array_sum.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/array_sum.dir/build: array_sum

.PHONY : CMakeFiles/array_sum.dir/build

CMakeFiles/array_sum.dir/requires: CMakeFiles/array_sum.dir/main.cu.o.requires

.PHONY : CMakeFiles/array_sum.dir/requires

CMakeFiles/array_sum.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/array_sum.dir/cmake_clean.cmake
.PHONY : CMakeFiles/array_sum.dir/clean

CMakeFiles/array_sum.dir/depend:
	cd /workspace/CUDA_practice/HIT_ParallelCompute/lab0/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /workspace/CUDA_practice/HIT_ParallelCompute/lab0 /workspace/CUDA_practice/HIT_ParallelCompute/lab0 /workspace/CUDA_practice/HIT_ParallelCompute/lab0/build /workspace/CUDA_practice/HIT_ParallelCompute/lab0/build /workspace/CUDA_practice/HIT_ParallelCompute/lab0/build/CMakeFiles/array_sum.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/array_sum.dir/depend

