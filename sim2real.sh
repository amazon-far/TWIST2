

source ~/miniconda3/bin/activate twist2
export PYTHONFAULTHANDLER=1
export PYTHONMALLOC=debug
export LD_LIBRARY_PATH=/home/wego/isaacgym/unitree_sdk2/thirdparty/lib/x86_64:$LD_LIBRARY_PATH

SCRIPT_DIR=$(dirname $(realpath $0))
ckpt_path=${SCRIPT_DIR}/assets/ckpts/twist2_1017_25k.onnx

# change the network interface name to your own that connects to the robot
# net=enp0s31f6
net=enp4s0

cd deploy_real


#gdb -ex=r --args python server_low_level_g1_real.py \
#    --policy ${ckpt_path} \
#    --net ${net} \
#    --device cuda


python server_low_level_g1_real.py \
    --policy ${ckpt_path} \
    --net ${net} \
    --device cuda \
    --use_hand \
    --smooth_body 0.5
#   --record_proprio \
