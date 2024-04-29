#include"transposer.cuh"


//性能下界
__global__ void naiveGmem(cuDoubleComplex* out, cuDoubleComplex* in, int nx, int ny) {
	unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
	if (ix < nx && iy < ny) {
		out[ix * ny + iy] = in[iy * nx + ix];
	}
}

//性能上界（并未转置）
__global__ void copyGmem(cuDoubleComplex* out, cuDoubleComplex* in, int nx, int ny) {
	unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
	if (ix < nx && iy < ny) {
		out[iy * nx + ix] = in[iy * nx + ix];
	}
}


__global__ void transposeSmem(cuDoubleComplex* out, cuDoubleComplex* in, const int nx, const int ny) {

	//__shared__ cuDoubleComplex tile[BDIMY * BDIMX];
	extern __shared__ cuDoubleComplex tile[];
	//original
	unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
	//linear global memory index for original
	unsigned int ti = iy * nx + ix;
	//thread index in transposed block
	unsigned int bidx = threadIdx.y * blockDim.x + threadIdx.x;

	unsigned int irow = bidx / blockDim.y;
	unsigned int icol = bidx % blockDim.y;
	//coordinate in transposed matrix
	ix = blockIdx.y * blockDim.y + icol;
	iy = blockIdx.x * blockDim.x + irow;

	//linear global memory index for transposed matrix
	unsigned int to = iy * ny + ix;
	//printf(" %d  ", to);
	if (ix < ny && iy < nx) {														//因为ix和iy是out的ix和iy了, 所以要是ix<ny  iy<nx
		int aa = threadIdx.y * (BDIMX)+threadIdx.x;
		tile[aa] = in[ti];
		//tile[threadIdx.y][threadIdx.x] = in[ti];
		__syncthreads();
		int bb = icol * (BDIMX)+irow;
		out[to] = tile[bb];
		//tile[threadIdx.y][threadIdx.x] = in[ti];

		//printf("//////////col:%d  row: %d ti: %d\n", threadIdx.y, threadIdx.x, ti);
		//printf("//////////////////////////////////////////////////////////\n");
		//printf("in: %lf", in[ti]);
		__syncthreads();
		//out[to] = tile[icol][irow];
		//printf("col:%d  row: %d to: %d\n", icol, irow,to);
		//printf("out: %lf",out[to]);
		//printf("		%d\n", ti);
	}
}


__global__ void transposeSmemPad(cuDoubleComplex* out, cuDoubleComplex* in, const int nx, const int ny) {
	//__shared__ cuDoubleComplex tile[BDIMY * (BDIMX + IPAD)];
	extern __shared__ cuDoubleComplex tile[];
	//original
	unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
	//linear global memory index for original
	unsigned int ti = iy * nx + ix;
	//thread index in transposed block
	unsigned int bidx = threadIdx.y * blockDim.x + threadIdx.x;
	//块内转置
	unsigned int irow = bidx / blockDim.y;
	unsigned int icol = bidx % blockDim.y;
	//coordinate in transposed matrix
	//块外转置
	ix = blockIdx.y * blockDim.y + icol;
	iy = blockIdx.x * blockDim.x + irow;
	//linear global memory index for transposed matrix
	unsigned int to = iy * ny + ix;

	if (ix < ny && iy < nx) {
		int aa = threadIdx.y * (BDIMX + IPAD) + threadIdx.x;
		tile[aa] = in[ti];
		//tile[threadIdx.y][threadIdx.x] = in[ti];
		__syncthreads();
		int bb = icol * (BDIMX + IPAD) + irow;
		out[to] = tile[bb];
	}
}


__global__ void transposeSmemUnrollPad(cuDoubleComplex* out, cuDoubleComplex* in, int nx, int ny) {
	//__shared__ cuDoubleComplex tile[BDIMY * (BDIMX * 2 + IPAD)];
	extern __shared__ cuDoubleComplex tile[];
	unsigned int ix = 2 * blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

	unsigned int ti = iy * nx + ix;

	unsigned int bidx = blockDim.x * threadIdx.y + threadIdx.x;
	unsigned int irow = bidx / blockDim.y;
	unsigned int icol = bidx % blockDim.y;

	unsigned int ix2 = blockIdx.y * blockDim.y + icol;										//你看这就没毛病嘛
	unsigned int iy2 = 2 * blockIdx.x * blockDim.x + irow;

	unsigned int to = iy2 * ny + ix2;

	if ((ix + blockDim.x) < nx && iy < ny) {
		unsigned int row_idx = threadIdx.y * (blockDim.x * 2 + IPAD) + threadIdx.x;
		tile[row_idx] = in[ti];
		tile[row_idx + BDIMX] = in[ti + BDIMX];

		__syncthreads();

		unsigned int col_idx = icol * (blockDim.x * 2 + IPAD) + irow;
		out[to] = tile[col_idx];
		out[to + ny * BDIMX] = tile[col_idx + BDIMX];
	}
}

__global__ void transposeSmemUnrollPad_4(cuDoubleComplex* out, cuDoubleComplex* in, int nx, int ny) {
	extern __shared__ cuDoubleComplex tile[];
	//__shared__ cuDoubleComplex tile[BDIMY * (BDIMX * 4 + IPAD)];
	unsigned int ix = 4 * blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

	unsigned int ti = iy * nx + ix;

	unsigned int bidx = blockDim.x * threadIdx.y + threadIdx.x;
	unsigned int irow = bidx / blockDim.y;
	unsigned int icol = bidx % blockDim.y;

	unsigned int ix2 = blockIdx.y * blockDim.y + icol;
	unsigned int iy2 = 4 * blockIdx.x * blockDim.x + irow;

	unsigned int to = iy2 * ny + ix2;

	if ((ix + blockDim.x) < nx && iy < ny) {
		unsigned int row_idx = threadIdx.y * (blockDim.x * 4 + IPAD) + threadIdx.x;
		tile[row_idx] = in[ti];
		tile[row_idx + BDIMX] = in[ti + BDIMX];
		tile[row_idx + 2 * BDIMX] = in[ti + 2 * BDIMX];
		tile[row_idx + 3 * BDIMX] = in[ti + 3 * BDIMX];
		__syncthreads();

		unsigned int col_idx = icol * (blockDim.x * 4 + IPAD) + irow;
		out[to] = tile[col_idx];
		out[to + ny * BDIMX] = tile[col_idx + BDIMX];
		out[to + ny * 2 * BDIMX] = tile[col_idx + 2 * BDIMX];
		out[to + ny * 3 * BDIMX] = tile[col_idx + 3 * BDIMX];
	}
}



__global__ void transposeSmemUnrollPad_8(cuDoubleComplex* out, cuDoubleComplex* in, int nx, int ny) {
	extern __shared__ cuDoubleComplex tile[];
	//__shared__ cuDoubleComplex tile[BDIMY * (BDIMX * 8 + IPAD)];
	unsigned int ix = 8 * blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

	unsigned int ti = iy * nx + ix;

	unsigned int bidx = blockDim.x * threadIdx.y + threadIdx.x;
	unsigned int irow = bidx / blockDim.y;
	unsigned int icol = bidx % blockDim.y;

	unsigned int ix2 = blockIdx.y * blockDim.y + icol;
	unsigned int iy2 = 8 * blockIdx.x * blockDim.x + irow;

	unsigned int to = iy2 * ny + ix2;

	if ((ix + blockDim.x) < nx && iy < ny) {
		unsigned int row_idx = threadIdx.y * (blockDim.x * 8 + IPAD) + threadIdx.x;
		tile[row_idx] = in[ti];
		tile[row_idx + BDIMX] = in[ti + BDIMX];
		tile[row_idx + 2 * BDIMX] = in[ti + 2 * BDIMX];
		tile[row_idx + 3 * BDIMX] = in[ti + 3 * BDIMX];
		tile[row_idx + 4 * BDIMX] = in[ti + 4 * BDIMX];
		tile[row_idx + 5 * BDIMX] = in[ti + 5 * BDIMX];
		tile[row_idx + 6 * BDIMX] = in[ti + 6 * BDIMX];
		tile[row_idx + 7 * BDIMX] = in[ti + 7 * BDIMX];
		__syncthreads();

		unsigned int col_idx = icol * (blockDim.x * 8 + IPAD) + irow;
		out[to] = tile[col_idx];
		out[to + ny * BDIMX] = tile[col_idx + BDIMX];
		out[to + ny * 2 * BDIMX] = tile[col_idx + 2 * BDIMX];
		out[to + ny * 3 * BDIMX] = tile[col_idx + 3 * BDIMX];
		out[to + ny * 4 * BDIMX] = tile[col_idx + 4 * BDIMX];
		out[to + ny * 5 * BDIMX] = tile[col_idx + 5 * BDIMX];
		out[to + ny * 6 * BDIMX] = tile[col_idx + 6 * BDIMX];
		out[to + ny * 7 * BDIMX] = tile[col_idx + 7 * BDIMX];
	}
}