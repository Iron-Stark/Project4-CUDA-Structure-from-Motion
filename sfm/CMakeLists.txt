set(SOURCE_FILES
    "sfm.h"
    "sfm.cu"
    )

cuda_add_library(sfm
    ${SOURCE_FILES}
    OPTIONS -arch=sm_61
    )
