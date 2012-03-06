import os
os.environ["CUDA_DEVICE"] = "0"
os.environ["CUDA_PROFILE"] = "1"

import numpy

#import cuda
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

#main
mod = SourceModule("""
#include <math.h>

__constant__ float PI = 3.1415926535;

__device__ inline double fast_exp(double y)
{
	double d;
	*((int*)(&d) + 0) = 0;
	*((int*)(&d) + 1) = (int)(1512775 * y + 1072632447);
	return d;
}

__device__ inline float* get_diff_coefficents(float *w, int stab)
{
    float e[2];
    return e;
}

__global__ void get_gpu_conc(float *dest,           //concentration [out, width*width]
							 float *pos,            //center (x,y,z) position of each smoke. [in, smoke_count*3]
							 float *diffc,          //diffusion coefficents of x,z of each smoke. [in, smoke_count*2]
							 float *mass,           //mass of each smoke [in, smoke_count]
							 float height,          //concentration at given height [in]
							 int width,             //the width of matrix of concentration [in]
							 int gridw,             //the width of each grid in matrix of concentration [in]
							 int smoke_count        // the count of smokes [in]
							 )
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	int n = j * width + i;
	int halfw = width / 2;
	float x = gridw * (-1.0f * halfw) + 0.5f * gridw + i * gridw;
	float y = gridw * halfw - 0.5f * gridw - j * gridw;
	for(int q = 0; q < smoke_count; q++)
	{
		float temp_conc = mass[q] / (pow(2*PI, 1.5f)*diffc[2*q]*diffc[2*q]*diffc[2*q+1]) * exp(-0.5f*pow((x - pos[3*q])/diffc[2*q],2)) * exp(-0.5f*pow((y - pos[3*q+1])/diffc[2*q],2)) * (exp(-0.5f*pow((pos[3*q+2] - height)/diffc[2*q+1],2)) + exp(-0.5f*pow((pos[3*q+2] + height)/diffc[2*q+1],2)));
		if(q == 0)	dest[n] = temp_conc;
		else		dest[n] += temp_conc;
	}
}
""")

get_gpu_conc = mod.get_function("get_gpu_conc")

block_size = 16

def run_gpu_conc(pos, diffc, mass, height, GRID_WIDTH, gridw, smoke_count):
    dest = numpy.zeros(GRID_WIDTH*GRID_WIDTH).astype(numpy.float32)
    get_gpu_conc(drv.Out(dest), drv.In(pos), drv.In(diffc), drv.In(mass), numpy.float32(height), numpy.int32(GRID_WIDTH), numpy.int32(gridw), numpy.int32(smoke_count),
                 block=(block_size,block_size,1), grid=(GRID_WIDTH / block_size,GRID_WIDTH / block_size))
    return dest


