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

__global__ void naiveGmem(double* out, double* in, int nx, int ny);				//for a M*N matric . nx is N, ny is M
__global__ void copyGmem(double* out, double* in, int nx, int ny);
__global__ void transposeSmem(double* out, double* in, const int nx, const int ny);
__global__ void transposeSmemPad(double* out, double* in, const int nx, const int ny);
__global__ void transposeSmemUnrollPad(double* out, double* in, int nx, int ny);
__global__ void transposeSmemUnrollPad_4(double* out, double* in, int nx, int ny);
__global__ void transposeSmemUnrollPad_8(double* out, double* in, int nx, int ny);



#endif // !TRANSPOSER_H


