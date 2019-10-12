#pragma once
#include<cublas_v2.h>
#include<vector>
#include<cudaSift/cudaSift.h>
#include<cudaSift/cudaImage.h>
#include "svd.h"
using namespace std;

namespace SFM {
	class structure_from_motion {
		vector<float*>norm_pts1;
		vector<float*>norm_pts2;
		vector<float*>norms1;
		vector<float*>norms2;
		float *d_E;
		int P_ind;
		float *d_P;
		float *d_final_points;
		int num_points;
	public:
		structure_from_motion();
		structure_from_motion(int num_images, int num_points);
		//int ImproveHomography(SiftData &data, float *homography, int numLoops, float minScore, float maxAmbiguity, float thresh);
		//void PrintMatchData(SiftData &siftData1, SiftData &siftData2, CudaImage &img);
		//void MatchAll(SiftData &siftData1, SiftData &siftData2, float *homography);
		//double ScaleUp(CudaImage &res, CudaImage &src);
		void struct_from_motion(SiftData sifData1, float *instrinsic, int n, int num_images);
		int *calculateInliers(float *d_E_canidate, int ransac_iter);
		void choosePose();
		void linear_triangulation();
		void computePoseCandidates();
		void copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities);
		~structure_from_motion();
	};

}