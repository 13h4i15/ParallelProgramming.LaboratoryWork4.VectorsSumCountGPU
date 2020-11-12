#include <cublas_v2.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[])
{
	int array_size = atoi(argv[1]);
	printf("n = %d\n", array_size);

	// Сpu memory allocation
	int* first_vector = (int*)calloc(array_size, sizeof(int));
	int* second_vector = (int*)calloc(array_size, sizeof(int));
	int* result_vector = (int*)calloc(array_size, sizeof(int));

	// Vectors filling 
	int i;
	for (i = 0; i < array_size; ++i)
	{
		first_vector[i] = 1;
		second_vector[i] = 1;
	}


	// Gpu memory allocation
	int* first_vector_gpu = NULL;
	cudaError_t cuda_error = cudaMalloc((void**)&first_vector_gpu, array_size * sizeof(int));
	if (cuda_error != cudaSuccess)
	{
		fprintf(stderr, "Cannot allocate device array for first vector: %s\n", cudaGetErrorString(cuda_error));
		return 0;
	}

	int* second_vector_gpu = NULL;
	cuda_error = cudaMalloc((void**)&second_vector_gpu, array_size * sizeof(int));
	if (cuda_error != cudaSuccess)
	{
		fprintf(stderr, "Cannot allocate device array for second vector: %s\n", cudaGetErrorString(cuda_error));
		return 0;
	}

	int* result_vector_gpu = NULL;
	cuda_error = cudaMalloc((void**)&result_vector_gpu, array_size * sizeof(int));
	if (cuda_error != cudaSuccess)
	{
		fprintf(stderr, "Cannot allocate device array for result vector: %s\n", cudaGetErrorString(cuda_error));
		return 0;
	}


	// Start/stop events registration
	cudaEvent_t start, stop;
	cuda_error = cudaEventCreate(&start);
	if (cuda_error != cudaSuccess)
	{
		fprintf(stderr, "Cannot create CUDA start event: %s\n", cudaGetErrorString(cuda_error));
		return 0;
	}

	cuda_error = cudaEventCreate(&stop);
	if (cuda_error != cudaSuccess)
	{
		fprintf(stderr, "Cannot create CUDA end event: %s\n", cudaGetErrorString(cuda_error));
		return 0;
	}


	// Data copying: from cpu to gpu
	cuda_error = cudaMemcpy(first_vector_gpu, first_vector, array_size * sizeof(int), cudaMemcpyHostToDevice);
	if (cuda_error != cudaSuccess)
	{
		fprintf(stderr, "Cannot copy first vector array from host to device: %s\n", cudaGetErrorString(cuda_error));
		return 0;
	}

	cuda_error = cudaMemcpy(second_vector_gpu, second_vector, array_size * sizeof(int), cudaMemcpyHostToDevice);
	if (cuda_error != cudaSuccess)
	{
		fprintf(stderr, "Cannot copy second vector array from host to device: %s\n", cudaGetErrorString(cuda_error));
		return 0;
	}


	// Start event
	cuda_error = cudaEventRecord(start, 0);
	if (cuda_error != cudaSuccess)
	{
		fprintf(stderr, "Cannot record CUDA start event: %s\n", cudaGetErrorString(cuda_error));
		return 0;
	}


	int GRID_SIZE = (array_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
	// Start kernel
	kernel << < GRID_SIZE, BLOCK_SIZE >> >(result_vector_gpu, first_vector_gpu, second_vector_gpu, array_size);

	cuda_error = cudaGetLastError();
	if (cuda_error != cudaSuccess)
	{
		fprintf(stderr, "Cannot launch CUDA kernel: %s\n", cudaGetErrorString(cuda_error));
		return 0;
	}


	// Synchronization
	cuda_error = cudaDeviceSynchronize();
	if (cuda_error != cudaSuccess)
	{
		fprintf(stderr, "Cannot synchronize CUDA kernel: %s\n", cudaGetErrorString(cuda_error));
		return 0;
	}


	// Stop event
	cuda_error = cudaEventRecord(stop, 0);
	if (cuda_error != cudaSuccess)
	{
		fprintf(stderr, "Cannot record CUDA stop event: %s\n", cudaGetErrorString(cuda_error));
		return 0;
	}


	// Result copying: from gpu to cpu
	cuda_error = cudaMemcpy(result_vector, result_vector_gpu, array_size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cuda_error != cudaSuccess)
	{
		fprintf(stderr, "Cannot copy result array from device to host: %s\n", cudaGetErrorString(cuda_error));
		return 0;
	}


	// Time calculation
	float gpu_time = 0.0f;
	cuda_error = cudaEventElapsedTime(&gpu_time, start, stop);
	printf("time spent executing %s: %.9f seconds\n", "kernel", gpu_time / 1000);


	// Memory cleaning
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(first_vector_gpu);
	cudaFree(second_vector_gpu);
	cudaFree(result_vector_gpu);
	free(first_vector);
	free(second_vector);
	free(result_vector);

	return 0;
}
