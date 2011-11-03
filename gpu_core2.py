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

__global__ void get_gpu_conc_point(float *dest,     //concentration [out, count=smoke_group]
                             float *point,          //concentration [in, given (x,y,z) to compute concentrations, 3]
							 float *pos,            //center (x,y,z) positions of all smokes at all ticks. [in, varied count]
							 float *diffc,          //diffusion coefficents of x,z of all smokes at all ticks. [in, varied count]
							 float *mass,           //mass of all smokes at all ticks [in, varied count. NOTE: pos,diffc,mass have same count]
                             int *smoke_count,      // accumulated count of smokes in each group[in, smoke_group]
							 int smoke_group        //count of groups of smokes [in, it should always be 720 in current version]
							 )
{
	int n = blockDim.x * blockIdx.x + threadIdx.x;
    if(n > smoke_group)
        return;
    int start, end;
    if(n == 0)
    {
        start = 0;
        end = smoke_count[n];
    }
    else
    {
        start = smoke_count[n-1];
        end = smoke_count[n];
    }
    if(end - start == 0)
        return;
	for(int q = start; q < end; q++)
	{
		float temp_conc = mass[q] / (pow(2*PI, 1.5f)*diffc[2*q]*diffc[2*q]*diffc[2*q+1]) * exp(-0.5*pow((point[0] - pos[3*q])/diffc[2*q],2)) * exp(-0.5*pow((point[1] - pos[3*q+1])/diffc[2*q],2)) * (exp(-0.5*pow((pos[3*q+2] - point[2])/diffc[2*q+1],2)) + exp(-0.5*pow((pos[3*q+2] + point[2])/diffc[2*q+1],2)));
		if(q == start)
        {
            dest[n] = temp_conc;
        }
		else
		{
            dest[n] += temp_conc;
        }
	}
}
""")

get_gpu_conc_point = mod.get_function("get_gpu_conc_point")

block_size = 1024

def run_gpu_conc_point(point, pos, diffc, mass, smoke_count, smoke_group):
    dest = numpy.zeros(smoke_group).astype(numpy.float32)
    get_gpu_conc_point(drv.Out(dest), drv.In(point), drv.In(pos), drv.In(diffc), drv.In(mass), drv.In(smoke_count), numpy.int32(smoke_group),
                 block=(block_size,1,1), grid=(smoke_group // block_size + 1, 1))
    return dest


