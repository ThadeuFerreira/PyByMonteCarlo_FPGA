/* ***************************************************************************** */
/* Copyright:      Francois Panneton and Pierre L'Ecuyer, University of Montreal */
/*                 Makoto Matsumoto, Hiroshima University                        */
/* Notice:         This code can be used freely for personal, academic,          */
/*                 or non-commercial purposes. For commercial purposes,          */
/*                 please contact P. L'Ecuyer at: lecuyer@iro.UMontreal.ca       */
/* ***************************************************************************** */

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

#define V0           _STATE[_state_i                   ]
#define VM1          _STATE[(_state_i+M1) & 0x0000000fU]
#define VM2          _STATE[(_state_i+M2) & 0x0000000fU]
#define VM3          _STATE[(_state_i+M3) & 0x0000000fU]
#define VRm1         _STATE[(_state_i+15) & 0x0000000fU]
#define VRm2         _STATE[(_state_i+14) & 0x0000000fU]
#define newV0        _STATE[(_state_i+15) & 0x0000000fU]
#define newV1        _STATE[_state_i                 ]
#define newVRm1      _STATE[(_state_i+14) & 0x0000000fU]

#define FACT 2.32830643653869628906e-10
typedef float16 vec_float_ty;
typedef uint16 vec_uint_ty;

// 4 channels of random numbers

//unsigned int z0, z1, z2;

__kernel void Pi_kernel(__global float *restrict result,                        
                        __global float *restrict input_buffer,
                        const int size )
{

    int accum = 0;
    for(int i = 0; i < size -1 ; i+=2){            
        
            float x1 = input_buffer[i];

            float y1 = input_buffer[i+1];
            
            if(sqrt((x1*x1)+(y1*y1))<1.0f){
               accum = accum + 1;
              // printf("OPENCL: Total = %.3f\n", accum);
           }
    }
    *result = 8*((float)accum)/((size -1));
    //printf("OPENCL: Total = %.5f\n", accum);
    //printf("OPENCL: Total = %.5f\n", *result);
    printf("OPENCL: Pi = %.5f\n", 8*((float)accum)/((size -1)));
    
} 

__kernel void WELLRNG512a_generate(__global float *restrict result,
                                   __global unsigned int *restrict STATE, 
                                   __global unsigned int *restrict state_i,
                                   const uint seed,
                                   const uint init,
                                   const int size)
{
    
   unsigned int _STATE[16];
   
   if(init == 0)
    {
       _STATE[0] =      seed + 72852922; // Numeros "aleatorios" arbritários para inicialização 
       _STATE[1] =      seed + 41699578; 
       _STATE[2] =      seed + 56707026;
       _STATE[3] =      seed + 33717249;
       _STATE[4] =      seed + 18306974;
       _STATE[5] =      seed + 30824004;
       _STATE[6] =      seed + 42901955;
       _STATE[7] =      seed + 80465302;
       _STATE[8] =      94968136;
       _STATE[9] =      41480876;
       _STATE[10] =      57870066;
       _STATE[11] =      37220400;
       _STATE[12] =      14597146;
       _STATE[13] =      1165159;
       _STATE[14] =      99349121;
       _STATE[15] =      68083911;
    }
    else{
        for(int i = 0; i < 16; ++i){
            _STATE[i] = STATE[i];
        }
    }

    unsigned int z0;
    unsigned int z1;
    unsigned int z2;
    uint _state_i = *state_i;

    for(int i = 0; i < size; i++){
        z0    = VRm1;
        z1    = MAT0NEG (-16,V0)    ^ MAT0NEG (-15, VM1);
        z2    = MAT0POS (11, VM2)  ;
        newV1 = z1 ^ z2; 
        newV0 = MAT0NEG (-2,z0)     ^ MAT0NEG(-18,z1)    ^ MAT3NEG(-28,z2) ^ MAT4NEG(-5,0xda442d24U,newV1) ;
        _state_i = (_state_i + 15) & 0x0000000fU;
        result[i] =  ((float) _STATE[_state_i]) * FACT;
        
    }
    *state_i = _state_i;

    for(int i = 0; i < 16; ++i){
            STATE[i] = _STATE[i];            
        }
}

/*
// Box-Muller transform. Create pairs of independent normally distributed
// random numbers from a pair of uniformly distributed random numbers
//
float2 box_muller(float a, float b)
{
   float radius = sqrt(-2.0f * log(a));
   float angle = 2.0f*b;
   float2 result;
   result.x = radius*cospi(angle);
   result.y = radius*sinpi(angle);
   return result;
}*/