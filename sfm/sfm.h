#pragma once
#include<cublas_v2.h>
#include<vector>
#include<cudaSift/cudaSift.h>
#include<cudaSift/cudaImage.h>

using namespace std;

namespace SFM {
	class structure_from_motion {
		vector<float*>norm_pts1;
		vector<float*>norm_pts2;
	public:
		structure_from_motion(int num_images, int num_points);
		//int ImproveHomography(SiftData &data, float *homography, int numLoops, float minScore, float maxAmbiguity, float thresh);
		//void PrintMatchData(SiftData &siftData1, SiftData &siftData2, CudaImage &img);
		//void MatchAll(SiftData &siftData1, SiftData &siftData2, float *homography);
		//double ScaleUp(CudaImage &res, CudaImage &src);
		void struct_from_motion(SiftData sifData1, float *instrinsic, int n, int num_images);
		~structure_from_motion();
	};

}