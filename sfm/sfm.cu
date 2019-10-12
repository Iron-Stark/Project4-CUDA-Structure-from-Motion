#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include<memory.h>
#include<iostream>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <cmath>
#include<string>
#include<algorithm>
#include "sfm.h"
#include<cudaSift/cudaSift.h>
#include<cudaSift/cudaImage.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <thrust/remove.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include<iomanip>
#include <cuda_runtime.h>
#include <random>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <glm/glm.hpp>

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, _LINE_)
#define scene_scale 100.0f
#define blockSize 128
#define TILE_DIM 32
#define BLOCK_ROWS 8
cublasHandle_t handle;

cusolverDnHandle_t cusolverH = NULL;
cudaStream_t stream = NULL;
gesvdjInfo_t gesvdj_params = NULL;

cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
cudaError_t cudaStat1 = cudaSuccess;
cudaError_t cudaStat2 = cudaSuccess;
cudaError_t cudaStat3 = cudaSuccess;
cudaError_t cudaStat4 = cudaSuccess;
cudaError_t cudaStat5 = cudaSuccess;

float residual = 0;
int executed_sweeps = 0;
const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
const float tol = 1.e-7;
const int max_sweeps = 15;
const int sort_svd = 1;
using namespace std;
glm::vec3 *dev_pos;
glm::vec3 *dev_correspond;
glm::vec3 *dev_vel1;
glm::vec3 *dev_vel2;

int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
glm::vec3 gridMinimum;

void checkCUDAErrorFn(const char *msg, const char *file, int line) {
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess == err) {
		return;
	}

	fprintf(stderr, "CUDA error");
	if (file) {
		fprintf(stderr, " (%s:%d)", file, line);
	}
	fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
	exit(EXIT_FAILURE);
}
namespace SFM {

	structure_from_motion::structure_from_motion(){
	}
	structure_from_motion::structure_from_motion(int num_images, int num_points) {
		cublasCreate_v2(&handle);
		status = cusolverDnCreate(&cusolverH);
		cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
		checkCUDAError("Could not create flags");
		cusolverDnSetStream(cusolverH, stream);
		checkCUDAError("Could not Set strea,");
		cusolverDnCreateGesvdjInfo(&gesvdj_params);
		checkCUDAError("Could not create GesvdjInfo");
		cusolverDnXgesvdjSetTolerance(
			gesvdj_params,
			tol);
		checkCUDAError("Could not SetTolerance");
		cusolverDnXgesvdjSetMaxSweeps(
			gesvdj_params,
			max_sweeps);
		checkCUDAError("Could not SetMaxSweeps");
		cusolverDnXgesvdjSetSortEig(
			gesvdj_params,
			sort_svd);
		checkCUDAError("Could not SetSortEigs");
		this->num_points = num_points;
		float *normalized_pts1;
		float *normalized_pts2;
		float *norm1;
		float *norm2;
		cudaMalloc((void **)&d_E, 3 * 3 * sizeof(float));
		// Canidate R, T
		cudaMalloc((void **)&d_P, 4 * 4 * 4 * sizeof(float));
		for (int i = 0; i < num_images; i++) {
			cudaMalloc((void**)&normalized_pts1, 3 * num_points * sizeof(float));
			norm_pts1.push_back(normalized_pts1);
			cudaMalloc((void**)&normalized_pts2, 3 * num_points * sizeof(float));
			norm_pts2.push_back(normalized_pts2);
			cudaMalloc((void**)&norm1, 3 * num_points * sizeof(float));
			norms1.push_back(norm1);
			cudaMalloc((void**)&norm2, 3 * num_points * sizeof(float));
			norms2.push_back(norm2);
		}
		cudaMalloc((void **)&d_final_points, 4 * num_points * sizeof(float));
	}
	__host__ __device__ unsigned int hash(unsigned int a) {
		a = (a + 0x7ed55d16) + (a << 12);
		a = (a ^ 0xc761c23c) ^ (a >> 19);
		a = (a + 0x165667b1) + (a << 5);
		a = (a + 0xd3a2646c) ^ (a << 9);
		a = (a + 0xfd7046c5) + (a << 3);
		a = (a ^ 0xb55a4f09) ^ (a >> 16);
		return a;
	}
	/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
	__global__
		void kernCopyPositionsToVBO(int N, float *pos, float *vbo, float s_scale) {
		int index = threadIdx.x + (blockIdx.x * blockDim.x);

		float c_scale = s_scale;

		if (index < N) {
			vbo[access2(index, x_pos, 4)] = pos[access2(x_pos, index, N)] * c_scale;
			vbo[access2(index, y_pos, 4)] = pos[access2(y_pos, index, N)] * c_scale;
			vbo[access2(index, z_pos, 4)] = pos[access2(z_pos, index, N)] * c_scale;
			vbo[access2(index, 3, 4)] = 1.0f;
		}
	}

	__global__
		void kernCopyVelocitiesToVBO(int N, float *vbo, float s_scale) {
		int index = threadIdx.x + (blockIdx.x * blockDim.x);

		if (index < N) {
			vbo[4 * index + 0] = 1;//vel[index].x + 0.3f;
			vbo[4 * index + 1] = 1;//vel[index].y + 0.3f;
			vbo[4 * index + 2] = 1;//vel[index].z + 0.3f;
			vbo[4 * index + 3] = 1.0f;
		}
	}

	void structure_from_motion::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
		dim3 fullBlocksPerGrid((num_points + blockSize - 1) / blockSize);
		checkCUDAErrorWithLine("Not copyBoidsToVBO failed!");
		kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> > (num_points, d_final_points, vbodptr_positions, 1);
		kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> > (num_points, vbodptr_velocities, 1);
		checkCUDAErrorWithLine("copyBoidsToVBO failed!");
		cudaDeviceSynchronize();
	}

	__global__ void copy_point(SiftPoint* data, int numPoints, float *U1, float *U2) {
		const int index_col = blockIdx.x*blockDim.x + threadIdx.x; // col is x to prevent warp divergence as much as possible in this naive implementation
		const int index_row = blockIdx.y*blockDim.y + threadIdx.y;
		if (index_row >= 3 || index_col >= numPoints)
			return;
		if (!index_row) {
			U1[access2(index_row, index_col, numPoints)] = data[index_col].xpos;
			U2[access2(index_row, index_col, numPoints)] = data[index_col].match_xpos;
		}
		else if (index_row == 1) {
			U1[access2(index_row, index_col, numPoints)] = data[index_col].ypos;
			U2[access2(index_row, index_col, numPoints)] = data[index_col].match_ypos;
		}
		else {
			U1[access2(index_row, index_col, numPoints)] = 1;
			U2[access2(index_row, index_col, numPoints)] = 1;
		}
	}
	__global__ void normalizeE(float *E, int ransac_iterations) {
		const int index = blockIdx.x*blockDim.x + threadIdx.x;
		if (index >= ransac_iterations)
			return;
		float u[9], d[9], v[9];
		svd(&(E[index * 3 * 3]), u, d, v);
		d[2 * 3 + 2] = 0;
		d[1 * 3 + 1] = 1;
		d[0] = 1;
		// E = U * D * V'
		float tmp_u[9];
		multAB(u, d, tmp_u);
		multABt(tmp_u, v, &(E[index * 3 * 3]));
	}
	__global__ void element_wise_mult(float *A, float *B, int size) {
		const int index = blockIdx.x*blockDim.x + threadIdx.x;
		if (index >= size)
			return;
		A[index] *= B[index];
	}
	__global__ void element_wise_div(float *A, float *B, int size) {
		const int index = blockIdx.x*blockDim.x + threadIdx.x;
		if (index >= size)
			return;
		float val = B[index];
		if (val == 0)
			A[index] = 0;
		else
			A[index] /= val;
	}
	__global__ void element_wise_sum(float *A, float *B, int size) {
		const int index = blockIdx.x*blockDim.x + threadIdx.x;
		if (index >= size)
			return;
		A[index] += B[index];
	}
	__global__
		void vecnorm(float *A, float *res, int row, int col, float exp, float final_pow) {
		int index = blockIdx.x*blockDim.x + threadIdx.x;
		if (index >= col)
			return;
		float tmp_vlaue = 0;
#pragma unroll
		for (int i = 0; i < row; i++) {
			tmp_vlaue += powf(A[access2(i, index, col)], exp);
		}
		// Now we can take the sqrt of exp and then rais to the final_pow
		if (exp == final_pow) {
			res[index] = tmp_vlaue;
			return;
		}
		res[index] = powf(tmp_vlaue, final_pow / exp);
	}
	__global__
		void threshold_count(float *A, int *count_res, int batch_size, int ransac_count, float threshold) {
		int index = blockIdx.x*blockDim.x + threadIdx.x;
		if (index >= ransac_count)
			return;
		int count = 0;
#pragma unroll
		for (int i = 0; i < batch_size; i++) {
			if (A[i + index * batch_size] < threshold)
				count++;
		}
		count_res[index] = count;
	}
	__global__ void canidate_kernels(float *d_P, const float *u, const float *v) {
		int index = blockIdx.x*blockDim.x + threadIdx.x;
		if (index >= 4) // only 4 canidate positions exist so fixed value
			return;
		float W[9] = { 0, -1, 0, 1, 0, 0, 0, 0, 1 }; // rotation about z axis
		float Wt[9]; transpose_copy3x3(W, Wt, 3, 3);
		float canidate_P[4 * 4];
		float tmp_prod[9], tmp_prod2[9], T[9];
		// T
		canidate_P[access2(x_pos, 3, 4)] = ((!index || index == 2) ? -1 : 1) * u[access2(x_pos, 2, 3)];
		canidate_P[access2(y_pos, 3, 4)] = ((!index || index == 2) ? -1 : 1) * u[access2(y_pos, 2, 3)];
		canidate_P[access2(z_pos, 3, 4)] = ((!index || index == 2) ? -1 : 1) * u[access2(z_pos, 2, 3)];
		// R
		if (index < 2)
			multABt(W, v, tmp_prod);
		else
			multABt(Wt, v, tmp_prod);
		multAB(u, tmp_prod, tmp_prod2); // 3x3 transpose
		transpose_copy3x3(tmp_prod2, canidate_P, 3, 4);
		// Now we copy from 2d to 3d into d_P
		//d_P[index] = index;
		memcpy(&(d_P[access3(0, 0, index, 4, 4)]), canidate_P, 4 * 4 * sizeof(float));
		d_P[access3(3, 0, index, 4, 4)] = 0; // Set last row maually
		d_P[access3(3, 1, index, 4, 4)] = 0;
		d_P[access3(3, 2, index, 4, 4)] = 0;
		d_P[access3(3, 3, index, 4, 4)] = 1;
	}
	__global__ void compute_linear_triangulation_A(float *A, const float *pt1, const float *pt2, const int count, const int num_points, const float *m1, const float *m2, int P_ind, bool canidate_m2) {
		// if canidate_m2,  we are computing 4 A's for different m2
		// Points are 3xN and Projection matrices are 4x4
		int index = blockIdx.x*blockDim.x + threadIdx.x;
		int row = blockIdx.y*blockDim.y + threadIdx.y; // 2 rows, x, y
		if (index >= count || row >= 2)
			return;
		float tmp_A[2 * 4], valx, valy;
		const float *correct_pt, *correct_m;
		if (canidate_m2) {
			assert(count == 4);
			if (!row) { // Slightly help with the warp divergence here
				correct_pt = pt1;
				correct_m = m1;
			}
			else {
				correct_pt = pt2;
				correct_m = &(m2[access3(0, 0, index, 4, 4)]);
			}
			valx = correct_pt[access2(x_pos, 0, num_points)]; // we only use the first point
			valy = correct_pt[access2(y_pos, 0, num_points)];
		}
		else {
			assert(P_ind < 4 && P_ind >= 0);
			if (!row) { // Slightly help with the warp divergence here
				correct_pt = pt1;
				correct_m = m1;
			}
			else {
				correct_pt = pt2;
				correct_m = &(m2[access3(0, 0, P_ind, 4, 4)]);;
			}
			valx = correct_pt[access2(x_pos, index, num_points)];
			valy = correct_pt[access2(y_pos, index, num_points)]; // Num points does not need to be the same as count
		}
#pragma unroll
		for (int i = 0; i < 4; i++) {
			tmp_A[access2(x_pos, i, 4)] = valx * correct_m[access2(2, i, 4)] - correct_m[access2(x_pos, i, 4)];
			tmp_A[access2(y_pos, i, 4)] = valy * correct_m[access2(2, i, 4)] - correct_m[access2(y_pos, i, 4)];
		}
		memcpy(&(A[access3(((!row) ? 0 : 2), 0, index, 4, 4)]), tmp_A, 4 * 2 * sizeof(float));
	}
	__global__
		void normalize_pt_kernal(float *v, float *converted_pt, int number_points) { // assumes size of converted_pt is 4xnum_points and v is 4x4xnum_points
		int index = blockIdx.x*blockDim.x + threadIdx.x; // one per num_points
		if (index >= number_points)
			return;
		float norm_value = v[access3(3, 3, index, 4, 4)];
		if (norm_value == 0 || abs(norm_value) > 10) {
			converted_pt[access2(x_pos, index, number_points)] = 1;
			converted_pt[access2(y_pos, index, number_points)] = 1;
			converted_pt[access2(z_pos, index, number_points)] = 1;
		}
		else {
			converted_pt[access2(x_pos, index, number_points)] = v[access3(3, x_pos, index, 4, 4)] / norm_value;
			converted_pt[access2(y_pos, index, number_points)] = v[access3(3, y_pos, index, 4, 4)] / norm_value;
			converted_pt[access2(z_pos, index, number_points)] = v[access3(3, z_pos, index, 4, 4)] / norm_value;
		}
		converted_pt[access2(3, index, number_points)] = 1;
	}
	template<typename T>
	int printMatrix(const T*A, int row, int col, int print_col, const char* name)
	{
		/// Prints first and last print_col values of A if A is a 2d matrix
		T *print_a = new T[col*row];
		cudaMemcpy(print_a, A, row* col * sizeof(T), cudaMemcpyDeviceToHost);
		cout << name << endl;
		cout << "{" << endl;
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				if (j < print_col || j > col - print_col - 1) {
					float Areg = print_a[i * col + j];
					cout << "\t" << Areg;
				}
				else if (j == print_col) {
					cout << "\t....";
				}
			}
			cout << endl;
		}
		cout << "}" << endl;
		delete[]print_a;
		return 0;
	}
	__global__ void fillData(float *normalized_pts, SiftPoint * sift1, int num_points, int match) {
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		if (row >= 3 || col >= num_points)
			return;
		if (row == 0)
			normalized_pts[row*num_points + col] = match == 0 ? sift1[col].xpos : sift1[col].match_xpos;
		else if (row == 1)
			normalized_pts[row*num_points + col] = match == 0 ? sift1[col].ypos : sift1[col].match_ypos;
		else
			normalized_pts[row*num_points + col] = 1;
	}
	__global__
		void kron_kernal(float*d1, float*d2, float *A, int *indices, const int ransac_iterations, int num_points) {
		const int index = blockIdx.x*blockDim.x + threadIdx.x;
		const int A_row = 8;
		const int A_col = 9;

		if (index > ransac_iterations)
			return;
#pragma unroll
		for (int i = 0; i < A_row; i++) {
			// begin
			A[access3(i, 0, index, A_row, A_col)] = d1[access2(x_pos, indices[index * A_row + i], num_points)] * d2[access2(x_pos, indices[index * A_row + i], num_points)];
			A[access3(i, 1, index, A_row, A_col)] = d1[access2(x_pos, indices[index * A_row + i], num_points)] * d2[access2(y_pos, indices[index * A_row + i], num_points)];
			A[access3(i, 2, index, A_row, A_col)] = d1[access2(x_pos, indices[index * A_row + i], num_points)] * d2[access2(z_pos, indices[index * A_row + i], num_points)];
			// second												  			    
			A[access3(i, 3, index, A_row, A_col)] = d1[access2(y_pos, indices[index * A_row + i], num_points)] * d2[access2(x_pos, indices[index * A_row + i], num_points)];
			A[access3(i, 4, index, A_row, A_col)] = d1[access2(y_pos, indices[index * A_row + i], num_points)] * d2[access2(y_pos, indices[index * A_row + i], num_points)];
			A[access3(i, 5, index, A_row, A_col)] = d1[access2(y_pos, indices[index * A_row + i], num_points)] * d2[access2(z_pos, indices[index * A_row + i], num_points)];
			//third													  			    
			A[access3(i, 6, index, A_row, A_col)] = d1[access2(z_pos, indices[index * A_row + i], num_points)] * d2[access2(x_pos, indices[index * A_row + i], num_points)];
			A[access3(i, 7, index, A_row, A_col)] = d1[access2(z_pos, indices[index * A_row + i], num_points)] * d2[access2(y_pos, indices[index * A_row + i], num_points)];
			A[access3(i, 8, index, A_row, A_col)] = d1[access2(z_pos, indices[index * A_row + i], num_points)] * d2[access2(z_pos, indices[index * A_row + i], num_points)];
		}
	}

	void printCuda(float *a1, int n, std::string name) {
		float *print_a = new float[n];
		std::cout << name.c_str() << std::endl;
		std::cout << "{" << std::endl;
		cudaMemcpy(print_a, a1, n * sizeof(float), cudaMemcpyDeviceToHost);
		for (int i = 0; i < n; i++) {
			std::cout << "\t" << print_a[i] << std::endl;
		}
		std::cout << "}" << std::endl;
		delete[]print_a;
	}
	void svd_device(float *src, float *VT, float *S, float *U, int m, int n, const int batchSize, int *d_info) {
		const int minmn = (m < n) ? m : n;
		const int lda = m; 
		const int ldu = m; 
		const int ldv = n;
		int lwork = 0;       /* size of workspace */
		float *d_work = NULL; /* device workspace for gesvdjBatched */
		cudaDeviceSynchronize();
		checkCUDAError("Could not Synchronize");
		cusolverDnSgesvdjBatched_bufferSize(cusolverH, jobz, m, n, src, lda, S, VT, ldu, U, ldv, &lwork, gesvdj_params, batchSize);
		checkCUDAError("Could not SgesvdjBatched_bufferSize");
		cudaMalloc((void**)&d_work, sizeof(float)*lwork);
		cusolverDnSgesvdjBatched(cusolverH, jobz, m, n, src, lda, S, VT, ldu, U, ldv, d_work, lwork, d_info, gesvdj_params, batchSize);
		checkCUDAError("Could not SgesvdjBatched");
		cudaDeviceSynchronize();
	}
	__global__ void transpose(float *odata, float* idata, int width, int height)
	{
		unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
		unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;

		if (xIndex < width && yIndex < height)
		{
			unsigned int index_in = xIndex + width * yIndex;
			unsigned int index_out = yIndex + height * xIndex;
			odata[index_out] = idata[index_in];
		}
	}
	void svd_device_transpose(float *src, float *UT, float *S, float *VT, int m, int n, const int batchSize, int *d_info) {
		float *d_A_trans = NULL;
		cudaMalloc((void **)&d_A_trans, 8 * 9 * batchSize * sizeof(float));
		for (int i = 0; i < batchSize; i++) {
			dim3 blocks(10, 10);
			dim3 fullBlocksPerGrid(1, 1);
			transpose << < fullBlocksPerGrid, blocks >> > (d_A_trans + i * 8 * 9, src + i * 8 * 9, 9, 8);
		}
		const int minmn = (m < n) ? m : n;
		const int lda = m;
		const int ldu = m;
		const int ldv = n;
		int lwork = 0;       /* size of workspace */
		float *d_work = NULL; /* device workspace for gesvdjBatched */
		cudaDeviceSynchronize();
		checkCUDAError("Could not Synchronize");
		cusolverDnSgesvdjBatched_bufferSize(cusolverH, jobz, m, n, d_A_trans, lda, S, UT, ldu, VT, ldv, &lwork, gesvdj_params, batchSize);
		checkCUDAError("Could not SgesvdjBatched_bufferSize");
		cudaMalloc((void**)&d_work, sizeof(float)*lwork);
		cusolverDnSgesvdjBatched(cusolverH, jobz, m, n, d_A_trans, lda, S, UT, ldu, VT, ldv, d_work, lwork, d_info, gesvdj_params, batchSize);
		checkCUDAError("Could not SgesvdjBatched");
		cudaDeviceSynchronize();
	}

	void invert_device(float *src, float *dst, int n) {
		int batchSize = 1;
		int *P, *INFO;
		cudaMalloc<int>(&P, n * batchSize * sizeof(int));
		cudaMalloc<int>(&INFO, batchSize * sizeof(int));
		int lda = n;
		float *A[] = {src};
		float ** A_d;
		cudaMalloc<float*>(&A_d, sizeof(A));
		cudaMemcpy(A_d, A, sizeof(A), cudaMemcpyHostToDevice);
		cublasSgetrfBatched(handle, n, A_d, lda, P, INFO, batchSize);
		int INFOh = 0;
		cudaMemcpy(&INFOh, INFO, sizeof(int), cudaMemcpyDeviceToHost);
		if (INFOh == 17) {
			fprintf(stderr, "Factorization Failed: Matrix is singular\n");
			cudaDeviceReset();
			exit(EXIT_FAILURE);
		}
		float* C[] = { dst };
		float** C_d;
		cudaMalloc<float*>(&C_d, sizeof(C));
		cudaMemcpy(C_d, C, sizeof(C), cudaMemcpyHostToDevice);

		cublasSgetriBatched(handle, n, A_d, lda, P, C_d, n, INFO, batchSize);

		cudaMemcpy(&INFOh, INFO, sizeof(int), cudaMemcpyDeviceToHost);

		if (INFOh != 0)
		{
			fprintf(stderr, "Inversion Failed: Matrix is singular\n");
			cudaDeviceReset();
			exit(EXIT_FAILURE);
		}

		cudaFree(P), cudaFree(INFO);
	}

	void invert(float *s, float *d, int n) {
		float *src;
		cudaMalloc<float>(&src, n * n * sizeof(float));
		cudaMemcpy(src, s, n * n * sizeof(float), cudaMemcpyHostToDevice);

		invert_device(src, d, n);
		cudaFree(src);
	}
	void mmul(const float* A, const float* B, float* C, const int m, const int k, const int n) {
		const float alf = 1;
		const float bet = 0;
		const float *alpha = &alf;
		const float *beta = &bet;
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, alpha, B, n, A, k, beta, C, n);
	}
	// For A'B
	// A is M by N and B is M by N
	//lda = num_col_A = num_row_AT = N;
	//ldb = num_col_B = num_row_BT = N;
	//ldc = num_row_C = N;
	//m = num_row_C = num_row_AT = num_col_A = N;
	//n = num_col_C = num_row_BT = num_col_B = N;
	//k = num_col_AT = num_row_B = M;
	void gpu_blas_mmul_batched(const float *A, const float *B, float *C, const int m, const int k, const int n, const int stride_A, const int stride_B, const int stride_C, const int batches,
		bool trans_a, bool trans_b) {
		assert(stride_A == 0 || stride_A == m * k);
		assert(stride_B == 0 || stride_B == n * k);
		assert(stride_C == 0 || stride_C == m * n);
		cublasHandle_t handle;
		cublasCreate(&handle);
		const float alf = 1; // gpu vs cpu
		const float bet = 0;
		const float *alpha = &alf;
		const float *beta = &bet;
		if(trans_a == 0 && trans_b == 0)
			cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, alpha, B, n, stride_B, A, k, stride_A, beta, C, n, stride_C, batches);
		else if(trans_a == 1 && trans_b == 0)
			cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, m, k, alpha, B, n, stride_B, A, m, stride_A, beta, C, n, stride_C, batches);
		else if(trans_a == 0 && trans_b == 1)
			cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, alpha, B, k, stride_B, A, m, stride_A, beta, C, k, stride_C, batches);
		cublasDestroy(handle);
	}
	template<typename T>
	T* cuda_alloc_copy(const T* host, int size) {
		T* data;
		cudaMalloc((void**)&data, size * sizeof(T));
		cudaMemcpy(data, host, size * sizeof(T), cudaMemcpyHostToDevice);
		return data;
	}
	int * structure_from_motion::calculateInliers(float *d_E_canidate, int ransac_iter) {
		/// This function calculates n1, d1, n2, d2 and then finds the number of residuals per E canidate in X[0] and X[1]
		// Init E1
		float E1[9] = { 0, -1, 0, 1, 0, 0, 0, 0, 0 };
		float *d_E1;
		cudaMalloc((void **)&d_E1, 9 * sizeof(float));
		cudaMemcpy(d_E1, E1, 9 * sizeof(float), cudaMemcpyHostToDevice);
		// Allocs
		float *x1_transformed, *x2_transformed;
		cudaMalloc((void**)&x1_transformed, 3 * num_points * ransac_iter * sizeof(float));
		cudaMalloc((void**)&x2_transformed, 3 * num_points * ransac_iter * sizeof(float));
		float *d1, *d2;
		cudaMalloc((void**)&d1, 3 * num_points * ransac_iter * sizeof(float));
		cudaMalloc((void**)&d2, 3 * num_points * ransac_iter * sizeof(float));
		float *n1, *n2;
		cudaMalloc((void **)&n1, 3 * num_points * ransac_iter * sizeof(float));
		cudaMalloc((void **)&n2, 3 * num_points * ransac_iter * sizeof(float));
		// Calculate x1 (from matlab code) {
		int m = 3, k = 3, n = num_points;
		gpu_blas_mmul_batched(d_E_canidate, norms1[0], x1_transformed, m, k, n, m * k, 0, m * n, ransac_iter, 0,0);

		//Compute n1 
		m = num_points, k = 3, n = 3; // these probably need to change because we need to transpose X[1]
		gpu_blas_mmul_batched(norms2[0], d_E_canidate, n1, m, k, n, 0, 3 * 3, m * n, ransac_iter, 1,0); // transpose X[1]
		int blocks = ceil((3 * num_points + blockSize - 1) / blockSize); // BUG!!! we need to make this batched
		element_wise_mult << <blocks, blockSize >> > (n1, norms1[0], 3 * num_points);
		// Compute d1
		// d1 = E1 * x1_transformed
		m = 3, k = 3, n = num_points;
		gpu_blas_mmul_batched(d_E_canidate, x1_transformed, d1, m, k, n, m*k, 0, m* n, ransac_iter, 0,0);
		// }
		// Now calculate x2_transformed, n2 and d2 {
		m = 3, k = 3, n = num_points;
		gpu_blas_mmul_batched(d_E_canidate, norms2[0], x2_transformed, m, k, n, m*k, 0, m* n, ransac_iter, 0,0);
		//Compute n2
		m = num_points, k = 3, n = 3; // these probably need to change because we need to transpose X[0]
		gpu_blas_mmul_batched(norms1[0], d_E_canidate, n2, m, k, n, 0, 3 * 3, m * n, ransac_iter, 1,0); // transpose X[0]
		blocks = ceil((3 * num_points + blockSize - 1) / blockSize);
		element_wise_mult << <blocks, blockSize >> > (n2, norms2[0], 3 * num_points);
		// Compute d2
		m = 3, k = 3, n = num_points;
		gpu_blas_mmul_batched(d_E_canidate, x2_transformed, d2, m, k, n, m*k, 0, m* n, ransac_iter, 0,0);
		// }
		// Now calculate the residual per canidate E{
		float *norm_n1, *norm_n2, *norm_d1, *norm_d2;
		int *inliers;
		int size = num_points * ransac_iter;
		cudaMalloc((void**)&norm_n1, size * sizeof(float));
		cudaMalloc((void**)&norm_n2, size * sizeof(float));
		cudaMalloc((void**)&norm_d1, size * sizeof(float));
		cudaMalloc((void**)&norm_d2, size * sizeof(float));
		cudaMalloc((void**)&inliers, ransac_iter * sizeof(int));
		blocks = ceil((num_points * ransac_iter + blockSize - 1) / blockSize);
		vecnorm << <blocks, blockSize >> > (n1, norm_n1, 3, size, 1, 2);
		vecnorm << <blocks, blockSize >> > (n2, norm_n2, 3, size, 1, 2);

		vecnorm << <blocks, blockSize >> > (d1, norm_d1, 3, size, 2, 2);
		vecnorm << <blocks, blockSize >> > (d1, norm_d1, 3, size, 2, 2);

		element_wise_div << <blocks, blockSize >> > (norm_n1, norm_d1, size);
		element_wise_div << <blocks, blockSize >> > (norm_n2, norm_d2, size);
		// have the residuals in norm_n1
		element_wise_sum << <blocks, blockSize >> > (norm_n1, norm_n2, size);
		// Calculate inliers per cell
		blocks = ceil((ransac_iter + blockSize - 1) / blockSize);
		threshold_count << <blocks, blockSize >> > (norm_n1, inliers, num_points, ransac_iter, 1e-3); // tested
		//}
		// Not sure if we should free
		cudaFree(n1);
		cudaFree(n2);
		cudaFree(d1);
		cudaFree(d2);
		cudaFree(x1_transformed);
		cudaFree(x2_transformed);
		// Free the norms!!!
		cudaFree(norm_n1);
		cudaFree(norm_n2);
		cudaFree(norm_d1);
		cudaFree(norm_d2);
		// 100% free
		cudaFree(d_E1);
		return inliers;
	}
	void GPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A) {
	     // Create a pseudo-random number generator
			     curandGenerator_t prng;
		    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
		
			     // Set the seed for the random number generator using the system clock
			     curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
		
			     // Fill the array with random numbers on the device
			     curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
		
	}
	void structure_from_motion::computePoseCandidates() {
		// Tested
		float E[9];// = { -0.211 , -0.798 , -0.561, -0.967 , 0.252  , 0.009, 0.046  , 0.047  , 0.039 }; // TODO remove this once testing is done
		cudaMemcpy(E, d_E, 3 * 3 * sizeof(float), cudaMemcpyDeviceToHost);
		float u[9], d[9], v[9], tmp[9];
		svd(E, u, d, v); // v is not transposed
		multABt(u, v, tmp); // u * v'
		if (det(tmp) < 0)
			neg(v);
		float *d_u, *d_v;
		d_u = cuda_alloc_copy(u, 3 * 3);
		d_v = cuda_alloc_copy(v, 3 * 3);
		canidate_kernels << <1, 32 >> > (d_P, d_u, d_v);
		cudaFree(d_u);
		cudaFree(d_v);
	}

	void structure_from_motion::linear_triangulation() {
		float P1[16] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 }; // I(4)
		float *d_P1 = cuda_alloc_copy(P1, 16);
		float *d_A, *d_u, *d_d, *d_vt;
		cudaMalloc((void **)&d_A, 4 * 4 * num_points * sizeof(float));
		cudaMalloc((void **)&d_u, 4 * 4 * num_points * sizeof(float));
		cudaMalloc((void **)&d_d, 4 * num_points * sizeof(float));
		cudaMalloc((void **)&d_vt, 4 * 4 * num_points * sizeof(float));
		// Create A
		dim3 grids(ceil((num_points * 2 + blockSize - 1) / blockSize), 1);
		dim3 block_sizes(blockSize / 2, 2);
		compute_linear_triangulation_A << <grids, block_sizes >> > (d_A, norms1[0], norms2[0], num_points, num_points, d_P1, d_P, P_ind, false);
		checkCUDAError("A computation error");
		// Assumes V isnt transposed, we need to take the last column
		int *d_info = NULL;
		cudaMalloc((void**)&d_info, 4 * sizeof(int));
		svd_device(d_A, d_vt, d_d, d_u, 4, 4, num_points, d_info);
		checkCUDAError("SVD error");
		dim3 grids2(ceil((num_points + blockSize - 1) / blockSize), 1);
		dim3 block_sizes2(blockSize, 4);
		// Normalize by using the last row of v'
		normalize_pt_kernal << <grids2, block_sizes2 >> > (d_vt, d_final_points, num_points);
		printMatrix(d_final_points, 3, num_points, 5, "Transformed points");
		cudaFree(d_P1);
		cudaFree(d_A);
		cudaFree(d_u);
		cudaFree(d_d);
		cudaFree(d_vt);
		cudaFree(d_info);
	}
	void structure_from_motion::choosePose() {
		
		float P1[16] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 }; // I(4)
		float *d_P1 = cuda_alloc_copy(P1, 16);
		float *d_A, *d_u, *d_d, *d_vt;
		cudaMalloc((void **)&d_A, 4 * 4 * 4 * sizeof(float));
		cudaMalloc((void **)&d_u, 4 * 4 * 4 * sizeof(float));
		cudaMalloc((void **)&d_d, 4 * 4 * sizeof(float));
		cudaMalloc((void **)&d_vt, 4 * 4 * 4 * sizeof(float));
		// Create A

		dim3 blocks(1, 1);
		dim3 block_sizes(4, 2);
		compute_linear_triangulation_A << <blocks, block_sizes >> > (d_A, norms1[0], norms2[0], 4, num_points, d_P1, d_P, -1, true);
		// We only care about V
		float *d_d1, *d_d2; // 3x4 batched
		cudaMalloc((void **)&d_d1, 4 * 4 * sizeof(float));
		cudaMalloc((void **)&d_d2, 4 * 4 * sizeof(float));
		// Assumes V isnt transposed, we need to take the last row
		// svd(d_A, d_u, d_d, d_v, 4 batches)
		checkCUDAErrorWithLine("Before SVD");
		int *d_info = NULL;
		cudaMalloc((void**)&d_info, 4 * sizeof(int));
		svd_device(d_A, d_vt, d_d, d_u, 4, 4, 4, d_info);
		checkCUDAErrorWithLine("SVD");
		normalize_pt_kernal << <1, 4 >> > (d_vt, d_d1, 4);
		printMatrix(d_d1, 4, 4, 4, "d1");
		float val_d1, val_d2;
		P_ind = 0;
		for (int i = 0; i < 4; i++) { // batched doesn't work for inverse + it is only 4, 4x4 matrices, should be easy
			invert(d_P + i * 4 * 4, d_P + i * 4 * 4, 4);
			int m = 4, k = 4, n = 4;
			mmul(d_P + i * 4 * 4, d_d1, d_d2, m, k, n);
			// Do the final testing on the host
			cudaMemcpy(&val_d1, &(d_d1[access2(2, i, 4)]), sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(&val_d2, &(d_d2[access2(2, i, 4)]), sizeof(float), cudaMemcpyDeviceToHost);
			// Now we do the final check on the cpu as well, because it is the same ease
			if (val_d1 > 0 && val_d2 > 0)
				P_ind = i;
		}
		cudaFree(d_P1);
		cudaFree(d_A);
		cudaFree(d_u);
		cudaFree(d_d);
		cudaFree(d_vt);
		cudaFree(d_d1);
		cudaFree(d_d2);
		cudaFree(d_info);
	}
	void structure_from_motion::struct_from_motion(SiftData siftData1, float *intrinsic, int n, int num_images){
		float *invert_intrinsic;
		cudaMalloc<float>(&invert_intrinsic, n * n * sizeof(float));
		invert(intrinsic, invert_intrinsic, n);
		// TODO: Improve this
		printMatrix(invert_intrinsic, 3, 3, 3, "inver");
		SiftPoint *sift1 = siftData1.d_data;
		for (int i = 0; i < num_images-1; i++) {
			dim3 block(blockSize, 3);
			dim3 fullBlocksPerGrid((siftData1.numPts + blockSize - 1) / blockSize, 1);
			fillData << <fullBlocksPerGrid, block >> > (norm_pts1[i], sift1, siftData1.numPts, 0);
			fillData << <fullBlocksPerGrid, block >> > (norm_pts2[i], sift1, siftData1.numPts, 1);
			mmul(invert_intrinsic, norm_pts1[i], norms1[i], 3, 3, siftData1.numPts);
			mmul(invert_intrinsic, norm_pts2[i], norms2[i], 3, 3, siftData1.numPts);
		}
		const int ransac_count = floor(num_points / 8);
		// Create random order of points (on cpu using std::shuffle)
		int *indices = new int[num_points];
		int *d_indices;
		cudaMalloc((void **)&d_indices, num_points * sizeof(int));
		for (int i = 0; i < num_points; indices[i] = i, i++);
		// Shufle data
		std::random_device rd;
		std::mt19937 g(rd());
		//shuffle(indices, indices + num_points, g); todo enable this 
		// Copy data to gpu
		cudaMemcpy(d_indices, indices, num_points * sizeof(int), cudaMemcpyHostToDevice);
		// Calculate all kron products correctly
		float *d_A;
		cudaMalloc((void **)&d_A, 8 * 9 * ransac_count * sizeof(float));
		checkCUDAErrorWithLine("A malloc failed!");
		int grids = ceil((ransac_count + blockSize - 1) / blockSize);
		kron_kernal << <grids, blockSize >> > (norms1[0], norms2[0], d_A, d_indices, ransac_count, num_points);
		checkCUDAErrorWithLine("Kron failed!");

		float *d_E_canidate;
		cudaMalloc((void **)&d_E_canidate, 3 * 3 * ransac_count * sizeof(float));
		// Calculate batch SVD of d_A
		float *d_ut, *d_vt, *d_s;
		cudaMalloc((void **)&d_ut, 8 * 8 * ransac_count * sizeof(float));
		cudaMalloc((void **)&d_vt, 9 * 9 * ransac_count * sizeof(float));
		cudaMalloc((void **)&d_s, 8 * ransac_count * sizeof(float));
		int *d_info = NULL;
		cudaMalloc((void**)&d_info, 4 * sizeof(int));
		svd_device_transpose(d_A, d_ut, d_s, d_vt, 8, 9, ransac_count, d_info);
		// Last column of V becomes E (row of v' in our case)
		int blocks = ceil((ransac_count + blockSize - 1) / blockSize);
		for (int i = 0; i < ransac_count; i++) {
			cudaMemcpy(d_E_canidate + 3 * 3 * i, d_vt + 9 * 9 * i + 8 * 9, 9 * sizeof(float), cudaMemcpyDeviceToDevice);
		}		// Calculate target E's
		normalizeE << <grids, blockSize >> > (d_E_canidate, ransac_count);

		// Calculate number of inliers for each E
		int *inliers = calculateInliers(d_E_canidate, ransac_count);
		// Pick best E and allocate d_E and E using thrust
		thrust::device_ptr<int> dv_in(inliers);
		auto iter = thrust::max_element(dv_in, dv_in + ransac_count);
		unsigned int best_pos = (iter - dv_in) - 1;
		// Assigne d_E
		cudaMemcpy(d_E, &(d_E_canidate[access3(0, 0, best_pos, 3, 3)]), 3 * 3 * sizeof(float), cudaMemcpyDeviceToDevice);
		// Free stuff
		cudaFree(inliers);
		cudaFree(d_A);
		// svd free
		cudaFree(d_ut);
		cudaFree(d_s);
		cudaFree(d_vt);
		cudaFree(d_info);
		cudaFree(d_indices);
		free(indices);
		cudaFree(d_E_canidate);
	}
	structure_from_motion::~structure_from_motion() {
		cublasDestroy(handle);
	}
}