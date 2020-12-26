__global__ void addKernel(int* c, int* a, int* b, unsigned int size)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size)
	{
		c[index] = a[index] + b[index];
	}
}


#define kernel addKernel
#include "GPUCalculation.h"