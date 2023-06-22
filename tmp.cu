#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>

__global__ void cuda_hello(){
	printf("Hello World du GPU\n");
}

int main(void) {
	printf("Hello World du CPU\n");
	cuda_hello<<<1,1>>>();
	cudaDeviceSynchronize();
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize error: %s\n", cudaGetErrorString(cudaStatus));
		// Additional error handling if needed
	}
	return 0;
}