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
	int runNum = 1;
	for (int i = 0; i < runNum; i++)
	{
		time+=doGpuProcessing(signal, ori, M, N);									
	}
	time = time / runNum;
	printf("\n\naverage run time is: %.6f   \n", time);
	return 0;
}
 
/*warmup
* 你别说, 传数据也可以异步, 把signal分成几块, 弄多个流, 同时传是不是能快点
* 好像cudaMalloc ori和cudaMemcpy ori可以和pc的d_signal的fft并行
* 块和网格结构
* 瘦块
* 看看那些规约, 循环展开
* 读数据(特别是转置), 还有延迟隐藏
* 还有指令的延迟隐藏
* 师兄的库
* 流
* nsight
* 共享内存有用吗?
* 师兄发的那个fpga的论文
* 那个原子级啥啥啥的有用吗		->这一章涉及了精确度
* 内存类型
*/