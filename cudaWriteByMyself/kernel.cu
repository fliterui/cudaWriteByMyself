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
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;
int main()
{
	/*long int M = 64, N = 4000;																//M行 N列
	size_t memSize = M * N * sizeof(cuFloatComplex);
	cuFloatComplex* signal, * ori;
	cudaMallocHost((void**)&signal, memSize);												//用固定内存能快点, 在cpy的时候
	cudaMallocHost((void**)&ori, memSize/M);
	readData(signal, ori, M, N);
	doGpuProcessing(signal, ori, M, N);*/
	plt::plot({ 1,2,3,4,5 });
	plt::show();
	return 0;
}