#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include <math.h>
#include <cuComplex.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "processing_gpu.cuh"
//#include "matplotlibcpp.h"
//namespace plt = matplotlibcpp;
int main()
{
																			
	int M = 64, N = 4000;																//M行 N列
	dev_setup(M, N);
	size_t memSize = M * N * sizeof(cuDoubleComplex);
	cuDoubleComplex* signal, * ori;
	cudaMallocHost((void**)&signal, memSize);												//用固定内存能快点, 在cpy的时候
	cudaMallocHost((void**)&ori, memSize/M);
	readData(signal, ori, M, N);															//读matlab的回波信号, 然后直接传给doGpuProcessing, 在那里面cudaMalloc
	//printf("%lf", cuCreal(ori[1]));
	float time = 0;
	int runNum = 1000;
	for (int i = 0; i < runNum; i++)
	{
		time+=doGpuProcessing(signal, ori, M, N);									
	}
	time = time / runNum;
	cudaFree(signal);
	cudaFree(ori);
	printf("\n\naverage run time is: %.6f   \n", time);
	return 0;
}

