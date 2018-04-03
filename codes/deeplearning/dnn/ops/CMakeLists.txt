cmake_minimum_required(VERSION 2.8)

project(tf_ops)

execute_process(COMMAND python -c "import tensorflow as tf; print(tf.sysconfig.get_include(),end='')" OUTPUT_VARIABLE TF_INC )
execute_process(COMMAND python -c "import tensorflow as tf; print(tf.sysconfig.get_lib(),end='')" OUTPUT_VARIABLE TF_LIB )

message(${TF_INC})
message("${TF_INC}/external/nsync/public")
message(${TF_LIB})

include_directories( ${TF_INC} )
include_directories( "${TF_INC}/external/nsync/public" )
link_directories( ${TF_LIB} )


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")

set(PACKAGES_LIBS "tensorflow_framework")

if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -undefined dynamic_lookup -Wno-deprecated")
    include_directories(./ops_cpu)
    add_subdirectory(./ops_cpu)
    set(PACKAGES_LIBS ops_cpu ${PACKAGES_LIBS})
else()
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    add_definitions(-DGOOGLE_CUDA=1)
	#add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -x cu")
    find_package(CUDA REQUIRED)
    include_directories(./ops_gpu)
    add_subdirectory(./ops_gpu)
    set(PACKAGES_LIBS ops_gpu ${PACKAGES_LIBS})
endif()

add_library( roi_pool SHARED roi_pool.cpp )
target_link_libraries( roi_pool ${PACKAGES_LIBS} )
set_target_properties( roi_pool PROPERTIES PREFIX "" SUFFIX ".so" )

add_library( proposal_encode SHARED proposal_encode.cpp )
target_link_libraries( proposal_encode ${PACKAGES_LIBS} )
set_target_properties( proposal_encode PROPERTIES PREFIX "" SUFFIX ".so" )

add_library( proposal_decode SHARED proposal_decode.cpp )
target_link_libraries( proposal_decode ${PACKAGES_LIBS} )
set_target_properties( proposal_decode PROPERTIES PREFIX "" SUFFIX ".so" )

add_library( overlaps SHARED overlaps.cpp )
target_link_libraries( overlaps ${PACKAGES_LIBS} )
set_target_properties( overlaps PROPERTIES PREFIX "" SUFFIX ".so" )

add_library( frcnn_prepare SHARED frcnn_prepare.cpp )
target_link_libraries( frcnn_prepare ${PACKAGES_LIBS} )
set_target_properties( frcnn_prepare PROPERTIES PREFIX "" SUFFIX ".so" )

add_library( proposal_decode_reshape SHARED proposal_decode_reshape.cpp )
target_link_libraries( proposal_decode_reshape ${PACKAGES_LIBS} )
set_target_properties( proposal_decode_reshape PROPERTIES PREFIX "" SUFFIX ".so" )

add_library( bbox_delta SHARED bbox_delta.cpp )
target_link_libraries( bbox_delta ${PACKAGES_LIBS} )
set_target_properties( bbox_delta PROPERTIES PREFIX "" SUFFIX ".so" )

add_library( smooth_L1 SHARED smooth_L1.cpp )
target_link_libraries( smooth_L1 ${PACKAGES_LIBS} )
set_target_properties( smooth_L1 PROPERTIES PREFIX "" SUFFIX ".so" )
