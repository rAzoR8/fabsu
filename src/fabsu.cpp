#include <hip/hip_runtime.h>
#include <iostream>

int test(){
    int i = 0;
    ++i;
    return i;
}

int main(){

    // https://code.visualstudio.com/docs/cpp/launch-json-reference#_launchcompletecommand

    int devices{0};
    hipGetDeviceCount(&devices);

    hipDeviceProp_t props;

    for(int d = 0; d < devices; ++d){
        hipGetDeviceProperties(&props, 0);
        if (d == test()){
            std::cout << props.name << std::endl;
        }
    }

    return devices;
}