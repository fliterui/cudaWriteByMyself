#ifndef TRANSPOSER_H
#define TRANSPOSER_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include <math.h>
#include <cuComplex.h>
#include <windows.h>
//#include "nvtx3.hpp"
//#include "nvToolsExt.h"

#define IPAD 1
/*
* blockx= BDIMX
* blocky=BDIMY
* dim3e block3(blockx, blocky);
* dim3 grid3((N + block3.x - 1) / block3.x, (M + block3.y - 1) / block3.y);
*/
__global__ void naiveGmem(cuDoubleComplex* out, cuDoubleComplex* in, int nx, int ny);				//for a M*N matric . nx is N, ny is M
__global__ void copyGmem(cuDoubleComplex* out, cuDoubleComplex* in, int nx, int ny);
__global__ void transposeSmem(cuDoubleComplex* out, cuDoubleComplex* in, const int nx, const int ny);		//BDIMY * BDIMX*sizeof(cuDoubleComplex)
__global__ void transposeSmemPad(cuDoubleComplex* out, cuDoubleComplex* in, const int nx, const int ny);	//BDIMY * (BDIMX + IPAD)*sizeof(cuDoubleComplex)
__global__ void transposeSmemUnrollPad(cuDoubleComplex* out, cuDoubleComplex* in, int nx, int ny);			//BDIMY * (BDIMX * 2 + IPAD)*sizeof(cuDoubleComplex)	grid2.x = grid2.x / 2;
__global__ void transposeSmemUnrollPad_4(cuDoubleComplex* out, cuDoubleComplex* in, int nx, int ny);		//BDIMY * (BDIMX * 4 + IPAD)*sizeof(cuDoubleComplex)	grid2.x = grid2.x / 2;4
__global__ void transposeSmemUnrollPad_8(cuDoubleComplex* out, cuDoubleComplex* in, int nx, int ny);		//BDIMY * (BDIMX * 8 + IPAD)*sizeof(cuDoubleComplex)	grid2.x = grid2.x / 2;8




#endif // !TRANSPOSER_H


