cmake_minimum_required(VERSION 2.8)

add_library(ops_cpu roi_pool_op_cpu.cpp proposal_encode_op_cpu.cpp proposal_decode_op_cpu.cpp
            proposal_decode_reshape_op_cpu.cpp frcnn_prepare_op_cpu.cpp bbox_delta_cpu.cpp
            smooth_L1_cpu.cpp overlaps_op_cpu.cpp )
