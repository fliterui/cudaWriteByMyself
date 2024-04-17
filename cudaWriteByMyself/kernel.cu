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
	float ori_time = 0;
	float best_time = 1000;
	int runNum = 100;
	dim3 block[5], grid[5], bestBlock, bestGrid;
	block[0].x = 1024;
	block[1].x = 1024;
	block[2].x = 1024;
	block[3].x = 1024;
	block[4].x = 1024;
	for (int k = 0; k < 5; k++)
	{
		grid[k] = (M * N + block[1].x - 1) / block[1].x;
	}
	


	float ttttt = doGpuProcessing(signal, ori, M, N, grid, block);
	printf("warmup %f", ttttt);


	printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%");
	runNum = 1000;
	for (int i = 0; i < runNum; i++)
	{
		ori_time += doGpuProcessing(signal, ori, M, N, grid, block);
	}
	ori_time = ori_time / runNum;
	printf("\naverage run time is: %.6f  \n", ori_time);
	printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%");
	runNum = 100;


	for (int m = 0; m < 5; m++)
	{
		bestBlock.x = 1024;
		bestGrid.x = (M * N + bestBlock.x - 1) / bestBlock.x;					//这两行好像没啥用, 因为肯定在1024的时候best_time > time
		best_time = 1000;
		for(int n = 0; n < 8; n++)												//每个global是各自独立的所以一个一个找就行了
		{
			
			for (int i = 0; i < runNum; i++)
			{
				time+=doGpuProcessing(signal, ori, M, N, grid, block);									
			}
			time = time / runNum;
			printf("%d   ", block[m].x);
			if (best_time > time)
			{
				bestBlock = block[m];
				bestGrid = grid[m];
				best_time = time;
			}
			printf("%d   ", bestBlock.x);
			printf("average run time is: %.6f  \n", time);
			block[m] = block[m].x / 2;
			grid[m] = (M * N + block[m].x - 1) / block[m].x;
			
		}
		block[m] = bestBlock;
		grid[m] = bestGrid;
		printf("index: %d, bestBlock: %d, bestGrid: %d, bestTime: %f \n\n\n", m, bestBlock.x, bestGrid.x, best_time);
	}
	printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%");
	runNum = 1000;
	for (int i = 0; i < runNum; i++)
	{
		time += doGpuProcessing(signal, ori, M, N, grid, block);
	}
	time = time / runNum;
	printf("\naverage run time is: %.6f  \n", time);
	for (int i = 0; i < 5; i++)
	{
		printf("%d, block: %d, grid: %d\n", i, block[i].x, grid[i].x);
	}

	printf("快了: %f", ori_time - time);
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