#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include<memory.h>
#include<iostream>
#include<string>
#include "sfm.h"

#define blockSize 128

namespace SFM {
	sfm::sfm() {
	}
	sfm::~sfm() {
	}
}