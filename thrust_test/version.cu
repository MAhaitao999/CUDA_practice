#include <thrust/version.h>
#include <iostream>

int main(int argc, char* argv[]) {
    int major = THRUST_MAJOR_VERSION;
    int minor = THRUST_MINOR_VERSION;

    std::cout << "Thrust v" << major << "." << minor << std::endl;

    return 0;
}
