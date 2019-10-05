/**
 * @file      main.cpp
 * @brief     Structure from motion
 * @authors   Dewang Sultania
 * @date      2019
 * @copyright University of Pennsylvania
 */

#include <cstdio>
#include "testing_helpers.hpp"

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cudaSift/cudaImage.h>
#include <cudaSift/cudaSift.h>


using namespace std;


// Steps:

// Detect 2D points  - SIFT
// Match 2D points across 2 images - Feature Matching Algorithms
// RANSAC 
// Epipolar geometry - 
// 3a.If both intrinsic and extrinsic camera parameters are known, reconstruct with projection matrices.
// 3b.If only the intrinsic parameters are known, normalize coordinates and calculate the essential matrix.
// 3c.If neither intrinsic nor extrinsic parameters are known, calculate the fundamental matrix.
// With fundamental or essential matrix, assume P1 = [I 0] and calulate parameters of camera 2.
// Triangulate knowing that x1 = P1 * X and x2 = P2 * X.
// Bundle adjustment to minimize reprojection errors and refine the 3D coordinates.
// Visualize

// cpu.cu is needed
// cpu.h is needed
// sift.cu
// sift.h
// ransac.cu
// ransac.h
// http://www.cs.cornell.edu/projects/bigsfm/
int main(int argc, char* argv[]) {
	SiftData siftData;
	InitSiftData(siftData, 25000, true, true);
	cv::Mat limg;
	cv::imread("../img/img_lights.jpg", 6).convertTo(limg, 1);
	CudaImage img;
	int w = limg.cols;
	int h = limg.rows;
	img.Allocate(w, h, iAlignUp(w,128), false, NULL, (float*)limg.data);
	img.Download();
	int numOctaves = 5;    /* Number of octaves in Gaussian pyramid */
	float initBlur = 1.0f; /* Amount of initial Gaussian blurring in standard deviations */
	float thresh = 3.5f;   /* Threshold on difference of Gaussians for feature pruning */
	float minScale = 0.0f; /* Minimum acceptable scale to remove fine-scale features */
	bool upScale = false;  /* Whether to upscale image before extraction */
	/* Extract SIFT features */
	ExtractSift(siftData, img, numOctaves, initBlur, thresh, minScale, upScale);
	/* Free space allocated from SIFT features */
	FreeSiftData(siftData);

	return 0;
}