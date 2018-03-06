#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <string>
#include <cmath>

#include "CL/opencl.h"
#include "AOCLUtils/opencl.h"

#define W 32
#define R 16
#define P 0
#define M1 13
#define M2 9
#define M3 5

#define MAT0POS(t,v) (v^(v>>t))
#define MAT0NEG(t,v) (v^(v<<(-(t))))
#define MAT3NEG(t,v) (v<<(-(t)))
#define MAT4NEG(t,b,v) (v ^ ((v<<(-(t))) & b))

#define V0            STATE[state_i                   ]
#define VM1           STATE[(state_i+M1) & 0x0000000fU]
#define VM2           STATE[(state_i+M2) & 0x0000000fU]
#define VM3           STATE[(state_i+M3) & 0x0000000fU]
#define VRm1          STATE[(state_i+15) & 0x0000000fU]
#define VRm2          STATE[(state_i+14) & 0x0000000fU]
#define newV0         STATE[(state_i+15) & 0x0000000fU]
#define newV1         STATE[state_i                 ]
#define newVRm1       STATE[(state_i+14) & 0x0000000fU]

#define FACT 2.32830643653869628906e-10

static unsigned int state_i = 0;
static unsigned int STATE[R];
static unsigned int z0, z1, z2;

void InitWELLRNG512a (unsigned int seed){
    STATE[0] =      seed + 72852922; // Numeros "aleatorios" arbritários para inicialização 
    STATE[1] =      seed + 41699578; 
    STATE[2] =      seed + 56707026;
    STATE[3] =      seed + 33717249;
    STATE[4] =      seed + 18306974;
    STATE[5] =      seed + 30824004;
    STATE[6] =      seed + 42901955;
    STATE[7] =      seed + 80465302;
    STATE[8] =      94968136;
    STATE[9] =      41480876;
    STATE[10] =      57870066;
    STATE[11] =      37220400;
    STATE[12] =      14597146;
    STATE[13] =      1165159;
    STATE[14] =      99349121;
    STATE[15] =      68083911;
}

double WELLRNG512a (void){
  z0    = VRm1;
  z1    = MAT0NEG (-16,V0)    ^ MAT0NEG (-15, VM1);
  z2    = MAT0POS (11, VM2)  ;
  newV1 = z1                  ^ z2; 
  newV0 = MAT0NEG (-2,z0)     ^ MAT0NEG(-18,z1)    ^ MAT3NEG(-28,z2) ^ MAT4NEG(-5,0xda442d24U,newV1) ;
  state_i = (state_i + 15) & 0x0000000fU;
  return ((double) STATE[state_i]) * FACT;
}

using namespace aocl_utils;

#define AOCL_ALIGNMENT 64
//#define SIZE 1024*1024*1024
#define SIZE 1024*1024*2
#define PRECOMPILED_BINARY "well_kernel_emulation"

static cl_context my_context;

// OpenCL runtime configuration
static cl_platform_id platform = NULL;
static cl_device_id device = NULL;
static cl_command_queue queue = NULL;
static cl_kernel kernel = NULL;
static cl_program program = NULL;
static cl_int status;

static const size_t work_group_size = 1;  // 8 threads in the demo workgroup

int main(int argc, char *argv[] )
{
    printf("Begin!!!\n");
    int REPEAT;
    if( argc == 2 ) {
      REPEAT = atoi(argv[1]);
    }
    else{
        REPEAT = 2;
    }
    
    double accum, accum2;
    clock_t beginConf = clock();
    accum = 0.0;
    accum2 = 0.0;
    cl_int status;
    cl_platform_id platform;
    cl_uint num_platforms;
    status = clGetPlatformIDs(1, &platform, &num_platforms);
    cl_device_id device;
    cl_uint num_devices;
    status = clGetDeviceIDs(platform,
                            CL_DEVICE_TYPE_ALL,
                            1,
                            &device,
                            &num_devices);

    printf("Programming Device(s)\n");
    printf("Num Devices = %d\n", num_devices);
    
    // create a context
    my_context = clCreateContext(0, num_devices, &device, &oclContextCallback, NULL, &status);
    checkError(status,"Failed clCreateContext.");

    // Create the command queue.
    queue = clCreateCommandQueue(my_context, device, CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status, "Failed to create command queue");

    // Create the program.
    std::string binary_file = getBoardBinaryFile(PRECOMPILED_BINARY, device);
    printf("Using AOCX: %s\n", binary_file.c_str());
    program = createProgramFromBinary(my_context, binary_file.c_str(), &device, num_devices);
    status = clBuildProgram(program, num_devices, &device, "", NULL, NULL);
    cl_kernel well_kernel = clCreateKernel(program, "WELLRNG512a_generate", &status);
    cl_kernel pi_kernel = clCreateKernel(program, "Pi_kernel", &status);

    //uint *_STATE = (uint *)malloc(sizeof(uint)*16);

    void *_STATE = NULL;
    posix_memalign(&_STATE, AOCL_ALIGNMENT,sizeof(uint)*16 );
    cl_mem result_buffer =  clCreateBuffer(my_context,
                                       CL_MEM_WRITE_ONLY ,
                                       sizeof(float)*SIZE,
                                       NULL,
                                       &status);

    cl_mem STATE_buffer = clCreateBuffer(my_context,
                                      CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                      sizeof(uint) * 16,
                                      _STATE,
                                      &status);

    //uint *state_i_ptr = (uint *)malloc(sizeof(uint));
    void *state_i_ptr = NULL;
    posix_memalign(&state_i_ptr, AOCL_ALIGNMENT,sizeof(uint));
    cl_mem state_i_buffer = clCreateBuffer(my_context,
                                      CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                      sizeof(uint) ,
                                      state_i_ptr,
                                      &status);
   
    uint seed = 0;
    uint init = 0;
    int size = SIZE;

    status = clSetKernelArg(well_kernel, 0, sizeof(cl_mem), &result_buffer);
    checkError(status, "Failed to set kernel arg 0");
    status = clSetKernelArg(well_kernel, 1, sizeof(cl_mem), &STATE_buffer);
    checkError(status, "Failed to set kernel arg 1");
    status = clSetKernelArg(well_kernel, 2, sizeof(cl_mem), &state_i_buffer);
    checkError(status, "Failed to set kernel arg 2");
    status = clSetKernelArg(well_kernel, 3, sizeof(uint), &seed);
    checkError(status, "Failed to set kernel arg 3");
    status = clSetKernelArg(well_kernel, 4, sizeof(uint), &init);
    checkError(status, "Failed to set kernel arg 4");
    status = clSetKernelArg(well_kernel, 5, sizeof(int), &size);
    checkError(status, "Failed to set kernel arg 5");

     // Configure work set over which the kernel will execute
    size_t wgSize[3] = {work_group_size, 1, 1};
    size_t gSize[3] = {work_group_size, 1, 1};
    clock_t endConf = clock();
    double time_spent_conf = (double)(endConf - beginConf) / CLOCKS_PER_SEC;
    InitWELLRNG512a(0);
    clock_t begin2 = clock();
    for(int j = 0; j < REPEAT ; j++) {
         //Launch the kernel
         printf("Run = %d \n", j);
        status = clEnqueueNDRangeKernel(queue, well_kernel, 1, NULL, gSize, wgSize, 0, NULL, NULL);
        checkError(status, "Failed to launch kernel");

        status = clFinish(queue);
        void *output = NULL;
        posix_memalign(&output, AOCL_ALIGNMENT, sizeof(float)*SIZE); // isso é importante
        status = clEnqueueReadBuffer(queue,
                                    result_buffer,
                                    CL_TRUE,
                                    0,
                                    sizeof(float)*SIZE,
                                    output,
                                    0,
                                    NULL,
                                    NULL);

        int golden = 0, result = 0;
        float *data = (float *)malloc(sizeof(float)*SIZE); //gambiarra para copiar um void em um float
        memcpy(data,output, sizeof(float)*SIZE);
        
        for (int i = 0; i < SIZE; ++i)
        {
           double _output = data[i];
           //printf("%.8f -- %.8f\n", _output, WELLRNG512a());
           if(std::isnan(_output)){
               printf("NAN!!\n");
           }
           else{
              accum += _output; 
           }   
        } 
        free(data);
        printf("passou 1\n");
        cl_mem pi_result_buffer =  clCreateBuffer(my_context,
                                       CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                       sizeof(float),
                                       NULL,
                                       &status);
        checkError(status, "Create Buffer failed");
        printf("passou 2\n");

        void *rand_input = NULL;
        posix_memalign(&rand_input, AOCL_ALIGNMENT,sizeof(float)*SIZE);
        memcpy(rand_input, output, sizeof(float)*SIZE);
        cl_mem input_buffer = clCreateBuffer(my_context,  // TODO : será possivel usar o ponteiro void do output?
                                      CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                      sizeof(float) * SIZE,
                                      rand_input,
                                     &status);
        checkError(status, "Create Buffer failed");
        printf("passou 3.5\n");
        free(rand_input);
        free(output);        

        status = clSetKernelArg(pi_kernel, 0, sizeof(cl_mem), &pi_result_buffer);
        checkError(status, "Failed to set kernel arg 0");
        status = clSetKernelArg(pi_kernel, 1, sizeof(cl_mem), &input_buffer);
        checkError(status, "Failed to set kernel arg 1");
        status = clSetKernelArg(pi_kernel, 2, sizeof(int), &size);
        checkError(status, "Failed to set kernel arg 2");
        
        status = clEnqueueNDRangeKernel(queue, pi_kernel, 1, NULL, gSize, wgSize, 0, NULL, NULL);
        checkError(status, "Failed to launch kernel");
        printf("passou 4\n");
        status = clFinish(queue);
        printf("passou 5\n");                           
        void *pi_output = NULL;
        posix_memalign(&pi_output, AOCL_ALIGNMENT, sizeof(float)); // isso é importante
        status = clEnqueueReadBuffer(queue,
                                    pi_result_buffer,
                                    CL_TRUE,
                                    0,
                                    sizeof(float),
                                    pi_output,
                                    0,
                                    NULL,
                                    NULL);
        checkError(status, "Read buffer fail");
        printf("passou 6\n");
        float *pi_accum = (float *)pi_output;
        
        
        
        printf("final pi = %.9f\n", *pi_accum);
        printf("Total Hard = %.9f\n", accum);
        printf("Total Box = %.9f\n", accum2);
                
        
        //free(box_result_buffer);
        //free(final_size_buffer);
        clReleaseMemObject(input_buffer); 
        clReleaseMemObject(pi_result_buffer);     
        printf("passou 7\n"); 
        init = 1;  
        status = clSetKernelArg(well_kernel, 4, sizeof(cl_uint), &init);
        checkError(status, "Failed to set kernel arg 4");
        printf("passou 8\n");
    }
    clock_t end2 = clock();
    double time_spent2 = (double)(end2 - begin2) / CLOCKS_PER_SEC;
    //printf("Total Hard = %.9f\n", accum);
    //printf("Total Box = %.9f\n", accum2);
    //printf("Processing Time = %.5f\n", time_spent2);
    //printf("Total FPGA TIME = %.5f\n", time_spent2 + time_spent_conf);
    printf("PASSED!");
    
    printf("\n");
    return 0;
}
