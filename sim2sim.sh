SCRIPT_DIR=$(dirname $(realpath $0))
ckpt_path=${SCRIPT_DIR}/assets/ckpts/twist2_1017_20k.onnx

cd deploy_real

python server_low_level_g1_sim.py \
    --xml ../assets/g1/g1_23dof.xml \
    --policy ${ckpt_path} \
    --device cuda \
    --measure_fps 1 \
    --policy_frequency 100 \
    --limit_fps 1 \
    # --record_proprio \
