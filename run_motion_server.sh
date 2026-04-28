#!/bin/bash

script_dir=$(dirname $(realpath $0))
# motion_file="${script_dir}/assets/example_motions/0807_yanjie_walk_005.pkl"
motion_file="${script_dir}/assets/example_motions/0807_yanjie_walk_008.pkl"


# Change to deploy_real directory
cd deploy_real

# by default we use our own laptop as the redis server
redis_ip="localhost"
# this is my unitree g1's ip in wifi
# redis_ip="192.168.110.24"


# Run the motion server
python server_motion_lib.py \
    --motion_file ${motion_file} \
#    --robot unitree_g1 \
    --robot unitree_g1_23dof \
    --vis \
    --redis_ip ${redis_ip}
    # --send_start_frame_as_end_frame \
    # --use_remote_control \
