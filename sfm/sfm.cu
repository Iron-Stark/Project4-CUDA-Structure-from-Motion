#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include<memory.h>
#include<iostream>
#include<string>
#include "sfm.h"
#include<cudaSift/cudaSift.h>
#include<cudaSift/cudaImage.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <thrust/remove.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, _LINE_)
#define blockSize 128
cublasHandle_t handle;
namespace SFM {

	structure_from_motion::structure_from_motion(int num_images, int num_points) {
		cublasCreate_v2(&handle);
		float *normalized_pts1;
		float *normalized_pts2;
		for (int i = 0; i < num_images; i++) {
			cudaMalloc((void**)&normalized_pts1, 3 * num_points * sizeof(float));
			norm_pts1.push_back(normalized_pts1);
			cudaMalloc((void**)&normalized_pts2, 3 * num_points * sizeof(float));
			norm_pts2.push_back(normalized_pts2);
		}
		cudaFree(normalized_pts1);
		cudaFree(normalized_pts2);
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

	void printMatrix(int m, int n, const double*A, int lda, const char* name)
	{
		for (int row = 0; row < m; row++) {
			for (int col = 0; col < n; col++) {
				double Areg = A[row + col * lda];
				printf("%s(%d,%d) = %20.16E\n", name, row + 1, col + 1, Areg);
			}
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
		int lda = m, ldb = k, ldc = m;
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	}

	struct rem_if
	{
		__host__ __device__
			bool operator()(const float x)
		{
			return x == -1.0f;
		}
	};
	void structure_from_motion::struct_from_motion(SiftData siftData1, float *intrinsic, int n, int num_images){
		float *invert_intrinsic;
		cudaMalloc<float>(&invert_intrinsic, n * n * sizeof(float));
		invert(intrinsic, invert_intrinsic, n);
		// TODO: Improve this
		SiftPoint *sift1 = siftData1.d_data;
		for (int i = 0; i < num_images-1; i++) {
			dim3 block(blockSize, 3);
			dim3 fullBlocksPerGrid((siftData1.numPts + blockSize - 1) / blockSize, 1);
			fillData << <fullBlocksPerGrid, block >> > (norm_pts1[i], sift1, siftData1.numPts, 0);
			fillData << <fullBlocksPerGrid, block >> > (norm_pts2[i], sift1, siftData1.numPts, 1);
			mmul(invert_intrinsic, norm_pts1[i], norm_pts1[i], 3, 3, siftData1.numPts);
			mmul(invert_intrinsic, norm_pts2[i], norm_pts2[i], 3, 3, siftData1.numPts);
		}

	}
	structure_from_motion::~structure_from_motion() {
		cublasDestroy(handle);
	}
}