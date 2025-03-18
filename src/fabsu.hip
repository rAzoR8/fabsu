#include <hip/hip_runtime.h>
#include <iostream>

int main(){

    int devices{0};
    hipGetDeviceCount(&devices);

    hipDeviceProp_t props;

    for(int d = 0; d < devices; ++d){
        hipGetDeviceProperties(&props, 0);
        std::cout << props.name << std::endl;
    }

    return devices;
}