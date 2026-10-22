// Small CUDA driver-API execution check. No CUDA toolkit or Python dependency.
#include <dlfcn.h>
#include <nlohmann/json.hpp>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

using DevicePtr = unsigned long long;
template<class T> T symbol(void* library, const char* name) {
    auto address = dlsym(library, name);
    if (!address) throw std::runtime_error(std::string("Missing CUDA symbol: ") + name);
    return reinterpret_cast<T>(address);
}
void checked(int result, const char* operation) {
    if (result != 0) throw std::runtime_error(std::string(operation) + " CUDA error " + std::to_string(result));
}
int main() {
    void* library = dlopen("libcuda.so.1", RTLD_NOW);
    void* context = nullptr;
    try {
        if (!library) throw std::runtime_error("NVIDIA driver library is unavailable");
        checked(symbol<int(*)(unsigned)>(library,"cuInit")(0),"cuInit");
        int device = 0, count = 0, version = 0;
        checked(symbol<int(*)(int*)>(library,"cuDeviceGetCount")(&count),"cuDeviceGetCount");
        if (count < 1) throw std::runtime_error("No CUDA devices visible");
        checked(symbol<int(*)(int*,int)>(library,"cuDeviceGet")(&device,0),"cuDeviceGet");
        char name[256]{};
        checked(symbol<int(*)(char*,int,int)>(library,"cuDeviceGetName")(name,256,device),"cuDeviceGetName");
        checked(symbol<int(*)(int*)>(library,"cuDriverGetVersion")(&version),"cuDriverGetVersion");
        checked(symbol<int(*)(void**,unsigned,int)>(library,"cuCtxCreate_v2")(&context,0,device),"cuCtxCreate");
        constexpr auto ptx = R"ptx(
.version 6.4
.target sm_52
.address_size 64
.visible .entry helicopter_add(.param .u64 input_ptr, .param .u64 output_ptr) {
    .reg .b64 %a, %b;
    .reg .f32 %x, %y, %z;
    ld.param.u64 %a, [input_ptr];
    ld.param.u64 %b, [output_ptr];
    ld.global.f32 %x, [%a];
    ld.global.f32 %y, [%a+4];
    add.f32 %z, %x, %y;
    st.global.f32 [%b], %z;
    ret;
}
)ptx";
        void* module = nullptr;
        checked(symbol<int(*)(void**,const void*,unsigned,void*,void*)>(library,"cuModuleLoadDataEx")(&module,ptx,0,nullptr,nullptr),"cuModuleLoadDataEx");
        void* kernel = nullptr;
        checked(symbol<int(*)(void**,void*,const char*)>(library,"cuModuleGetFunction")(&kernel,module,"helicopter_add"),"cuModuleGetFunction");
        DevicePtr inputs{}, output{};
        const float values[2]{2.0f,3.0f};
        checked(symbol<int(*)(DevicePtr*,size_t)>(library,"cuMemAlloc_v2")(&inputs,sizeof(values)),"cuMemAlloc input");
        checked(symbol<int(*)(DevicePtr*,size_t)>(library,"cuMemAlloc_v2")(&output,sizeof(float)),"cuMemAlloc output");
        checked(symbol<int(*)(DevicePtr,const void*,size_t)>(library,"cuMemcpyHtoD_v2")(inputs,values,sizeof(values)),"cuMemcpyHtoD");
        void* parameters[]{&inputs,&output};
        checked(symbol<int(*)(void*,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,void*,void**,void**)>(library,"cuLaunchKernel")(
            kernel,1,1,1,1,1,1,0,nullptr,parameters,nullptr),"cuLaunchKernel");
        checked(symbol<int(*)()>(library,"cuCtxSynchronize")(),"cuCtxSynchronize");
        float result = 0.0f;
        checked(symbol<int(*)(void*,DevicePtr,size_t)>(library,"cuMemcpyDtoH_v2")(&result,output,sizeof(float)),"cuMemcpyDtoH");
        if (result != 5.0f) throw std::runtime_error("CUDA arithmetic result mismatch");
        checked(symbol<int(*)(void*)>(library,"cuCtxDestroy_v2")(context),"cuCtxDestroy");
        context = nullptr;
        std::cout << nlohmann::json{{"passed",true},{"device",name},{"device_count",count},
            {"driver_version",version},{"kernel_result",result},{"expected",5.0},
            {"scope","CUDA arithmetic smoke test; flight physics and feedback control run on CPU"}}.dump(2) << '\n';
        dlclose(library);
        return 0;
    } catch (const std::exception& error) {
        if (context && library) {
            auto destroy = reinterpret_cast<int(*)(void*)>(dlsym(library,"cuCtxDestroy_v2"));
            if (destroy) destroy(context);
        }
        if (library) dlclose(library);
        std::cerr << nlohmann::json{{"passed",false},{"error",error.what()}}.dump(2) << '\n';
        return 1;
    }
}
