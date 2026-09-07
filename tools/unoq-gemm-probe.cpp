// Standalone, fixed-shape K-quant kernel probe. No model loading or GGML build required.
// Uses unit weights/activations: every output must equal K exactly. The full GGML
// operator suite remains the randomized correctness test before model validation.
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

static void check(cl_int e) { if (e != CL_SUCCESS) { fprintf(stderr, "OpenCL error %d\n", e); exit(1); } }
int main(int argc, char ** argv) {
    if ((argc != 3 && argc != 6) || (std::string(argv[1]) != "q4" && std::string(argv[1]) != "q6")) {
        fprintf(stderr, "usage: unoq-gemm-probe q4|q6 kernel.cl [TM TN BK]\n"); return 2;
    }
    const int tm=argc==6?atoi(argv[3]):2, tn=argc==6?atoi(argv[4]):8, bk=argc==6?atoi(argv[5]):16;
    if (tm<1 || tn<1 || tm>32 || tn>32 || 32%tm || 32%tn ||
        (bk!=8 && bk!=16 && bk!=32 && bk!=64)) return 2;
    const bool q4 = std::string(argv[1]) == "q4";
    const int probe_threads = 1024/(tm*tn);
    // Each loader must cover whole rows without addressing beyond the local tile.
    if (probe_threads*(q4?4:2)<bk || probe_threads*4>32*bk) return 2;
    std::ifstream input(argv[2]);
    if (!input) return 2;
    std::string source = "#define GGML_OPENCL_RUSTICL_SUBGROUP32_COMPAT 1\n";
    source += "#define TM " + std::to_string(tm) + "\n#define TN " + std::to_string(tn) + "\n#define BK " + std::to_string(bk) + "\n";
    if (const char * value=getenv("SA3_PROBE_K_UNROLL")) {
        const int count=atoi(value);
        if (count<1 || count>16) return 2;
        source += "#define K_UNROLL _Pragma(\"unroll " + std::to_string(count) + "\")\n";
    }
    source.append(std::istreambuf_iterator<char>(input), {});
    cl_platform_id platform; cl_device_id device; cl_int err;
    check(clGetPlatformIDs(1, &platform, nullptr));
    check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr));
    char name[256]; check(clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, nullptr));
    printf("device=%s kernel=%s\n", name, argv[2]);
    cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err); check(err);
    const cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    cl_command_queue queue = clCreateCommandQueueWithProperties(ctx, device, props, &err); check(err);
    const char * ptr = source.c_str();
    cl_program program = clCreateProgramWithSource(ctx, 1, &ptr, nullptr, &err); check(err);
    err = clBuildProgram(program, 1, &device,
        "-cl-std=CL3.0 -cl-mad-enable -cl-unsafe-math-optimizations -cl-finite-math-only -cl-fast-relaxed-math", nullptr, nullptr);
    if (err) {
        size_t size; clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &size);
        std::vector<char> log(size); clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, size, log.data(), nullptr);
        fprintf(stderr, "%s\n", log.data()); check(err);
    }
    cl_kernel kernel = clCreateKernel(program, q4 ? "kernel_mul_mm_q4_k_f32_l4_lm" : "kernel_mul_mm_q6_k_f32_l4_lm", &err); check(err);
    size_t wg; cl_ulong local_bytes, private_bytes;
    check(clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(wg), &wg, nullptr));
    check(clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(local_bytes), &local_bytes, nullptr));
    check(clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(private_bytes), &private_bytes, nullptr));
    printf("max_wg=%zu local_bytes=%llu private_bytes=%llu\n", wg, (unsigned long long)local_bytes, (unsigned long long)private_bytes);
    const size_t threads=1024/(tm*tn);
    printf("TM=%d TN=%d BK=%d threads=%zu\n",tm,tn,bk,threads);
    if (threads>wg) { fprintf(stderr,"variant exceeds kernel workgroup limit\n"); return 2; }
    const int M=768, N=256, K=768, blocks=M*K/256, one=1;
    std::vector<cl_mem> buffers;
    auto buffer = [&](const void * data, size_t bytes) {
        cl_mem m=clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bytes, const_cast<void *>(data), &err);
        check(err); buffers.push_back(m); return m;
    };
    auto bytes = [&](size_t n, unsigned char v) { std::vector<unsigned char> x(n,v); return buffer(x.data(),x.size()); };
    std::vector<unsigned short> d(blocks, 0x3c00);
    cl_mem a[4];
    a[0]=bytes(blocks*128,0x11);
    if (q4) {
        std::vector<unsigned char> scales(blocks*12);
        for (int i=0;i<blocks;i++) for (int j : {0,1,2,3,8,9,10,11}) scales[i*12+j]=1;
        a[1]=buffer(scales.data(),scales.size()); a[2]=buffer(d.data(),d.size()*2); a[3]=bytes(blocks*2,0);
    } else {
        a[1]=bytes(blocks*64,0xaa); a[2]=bytes(blocks*16,1); a[3]=buffer(d.data(),d.size()*2);
    }
    std::vector<float> b(K*N,1), output(M*N,0);
    cl_mem bm=buffer(b.data(),b.size()*4), cm=buffer(output.data(),output.size()*4);
    const cl_ulong zero=0;
    auto arg = [&](cl_uint i, const auto & v) { check(clSetKernelArg(kernel,i,sizeof(v),&v)); };
    for (int i=0;i<4;i++) arg(i,a[i]);
    arg(4,bm); arg(5,zero); arg(6,cm); arg(7,zero);
    arg(8,K); arg(9,M); arg(10,one); arg(11,N); arg(12,one);
    arg(13,K); arg(14,K); arg(15,M);
    const int as=M*K, bs=N*K, cs=M*N;
    arg(16,as); arg(17,bs); arg(18,cs); arg(19,one); arg(20,one);
    const size_t global[]={M/32*threads,N/32,1}, local[]={threads,1,1};
    for (int i=0;i<6;i++) {
        cl_event event;
        check(clEnqueueNDRangeKernel(queue,kernel,3,nullptr,global,local,0,nullptr,&event));
        check(clWaitForEvents(1,&event));
        cl_ulong start,end;
        check(clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_START,sizeof(start),&start,nullptr));
        check(clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_END,sizeof(end),&end,nullptr));
        printf("repeat=%d device_ms=%.6f\n",i,double(end-start)*1e-6);
        clReleaseEvent(event);
    }
    check(clEnqueueReadBuffer(queue,cm,CL_TRUE,0,output.size()*4,output.data(),0,nullptr,nullptr));
    const bool ok=std::all_of(output.begin(),output.end(),[&](float v){return v==K;});
    printf("unit-weight-check=%s\n",ok?"PASS":"FAIL");
    for (auto m:buffers) clReleaseMemObject(m);
    clReleaseKernel(kernel); clReleaseProgram(program); clReleaseCommandQueue(queue); clReleaseContext(ctx);
    return ok?0:1;
}
