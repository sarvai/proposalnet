cmake_minimum_required(VERSION 2.8)

cuda_add_library(ops_gpu SHARED roi_pool_op_gpu.cu proposal_decode_op_gpu.cu
                                proposal_decode_reshape_op_gpu.cu
                                bbox_delta_gpu.cu frcnn_prepare_op_gpu.cu smooth_L1_gpu.cu
                            proposal_encode_op_gpu.cu overlaps_op_gpu.cu )
