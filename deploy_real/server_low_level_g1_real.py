#!/usr/bin/env python3
import argparse
import random
import time
import json
import numpy as np
import torch
import redis
from collections import deque
# from robot_control.common.remote_controller import KeyMap

from robot_control.g1_wrapper import G1RealWorldEnv
from robot_control.config import Config
import os
from data_utils.rot_utils import quatToEuler

from robot_control.dex_hand_wrapper import Dex3_1_Controller

try:
    import onnxruntime as ort
except ImportError:
    ort = None


class OnnxPolicyWrapper:
    """Minimal wrapper so ONNXRuntime policies mimic TorchScript call signature."""

    def __init__(self, session, input_name, output_index=0):
        self.session = session
        self.input_name = input_name
        self.output_index = output_index

    def __call__(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        if isinstance(obs_tensor, torch.Tensor):
            obs_np = obs_tensor.detach().cpu().numpy()
        else:
            obs_np = np.asarray(obs_tensor, dtype=np.float32)
        outputs = self.session.run(None, {self.input_name: obs_np})
        result = outputs[self.output_index]
        if not isinstance(result, np.ndarray):
            result = np.asarray(result, dtype=np.float32)
        return torch.from_numpy(result.astype(np.float32))


class EMASmoother:
    """Exponential Moving Average smoother for body actions."""
    
    def __init__(self, alpha=0.1, initial_value=None):
        """
        Args:
            alpha: Smoothing factor (0.0=no smoothing, 1.0=maximum smoothing)
            initial_value: Initial value for smoothing (if None, will use first input)
        """
        self.alpha = alpha
        self.initialized = False
        self.smoothed_value = initial_value
        
    def smooth(self, new_value):
        """Apply EMA smoothing to new value."""
        if not self.initialized:
            self.smoothed_value = new_value.copy() if hasattr(new_value, 'copy') else new_value
            self.initialized = True
            return self.smoothed_value
        
        # EMA formula: smoothed = alpha * new + (1 - alpha) * previous
        self.smoothed_value = self.alpha * new_value + (1 - self.alpha) * self.smoothed_value
        return self.smoothed_value
    
    def reset(self):
        """Reset the smoother to uninitialized state."""
        self.initialized = False
        self.smoothed_value = None


def load_onnx_policy(policy_path: str, device: str) -> OnnxPolicyWrapper:
    if ort is None:
        raise ImportError("onnxruntime is required for ONNX policy inference but is not installed.")
    providers = []
    available = ort.get_available_providers()
    if device.startswith('cuda'):
        if 'CUDAExecutionProvider' in available:
            providers.append('CUDAExecutionProvider')
        else:
            print("CUDAExecutionProvider not available in onnxruntime; falling back to CPUExecutionProvider.")
    providers.append('CPUExecutionProvider')
    session = ort.InferenceSession(policy_path, providers=providers)
    input_name = session.get_inputs()[0].name
    print(f"ONNX policy loaded from {policy_path} using providers: {session.get_providers()}")
    return OnnxPolicyWrapper(session, input_name)


class RealTimePolicyController(object):
    """
    Real robot controller for TWIST2 policy.
    Based on server_low_level_g1_real.py but adapted for TWIST2 architecture.
    """
    def __init__(self, 
                 policy_path,
                 config_path,
                 device='cuda',
                 net='enp4s0',
                 use_hand=False,
                 record_proprio=False,
                 smooth_body=0.0):
        self.redis_client = None
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
            self.redis_pipeline = self.redis_client.pipeline()
        except Exception as e:
            print(f"Error connecting to Redis: {e}")
            exit()
       
        self.config = Config(config_path)
        self.env = G1RealWorldEnv(net=net, config=self.config)
        self.use_hand = use_hand
        if use_hand:
            self.hand_ctrl = Dex3_1_Controller(net, re_init=False)

        self.device = device
        self.policy = load_onnx_policy(policy_path, device)

        self.num_actions = 29
        self.default_dof_pos = self.config.default_angles
        
        # scaling factors
        self.ang_vel_scale = 0.25
        self.dof_vel_scale = 0.05
        self.dof_pos_scale = 1.0
        self.ankle_idx = [4, 5, 10, 11]

        # TWIST2 observation structure
        self.n_mimic_obs = 35        # 6 + 29 (modified: root_vel_xy + root_pos_z + roll_pitch + yaw_ang_vel + dof_pos)
        self.n_proprio = 92          # from config analysis  
        self.n_obs_single = 127      # n_mimic_obs + n_proprio = 35 + 92 = 127
        self.history_len = 10
        
        self.total_obs_size = self.n_obs_single * (self.history_len + 1) + self.n_mimic_obs  # 127*11 + 35 = 1402
        
        print(f"TWIST2 Real Controller Configuration:")
        print(f"  n_mimic_obs: {self.n_mimic_obs}")
        print(f"  n_proprio: {self.n_proprio}")
        print(f"  n_obs_single: {self.n_obs_single}")
        print(f"  history_len: {self.history_len}")
        print(f"  total_obs_size: {self.total_obs_size}")

        self.proprio_history_buf = deque(maxlen=self.history_len)
        for _ in range(self.history_len):
            self.proprio_history_buf.append(np.zeros(self.n_obs_single, dtype=np.float32))

        self.last_action = np.zeros(self.num_actions, dtype=np.float32)

        self.control_dt = self.config.control_dt
        self.action_scale = self.config.action_scale
        
        self.record_proprio = record_proprio
        self.proprio_recordings = [] if record_proprio else None
        
        # Smoothing processing
        self.smooth_body = smooth_body
        if smooth_body > 0.0:
            self.body_smoother = EMASmoother(alpha=smooth_body)
            print(f"Body action smoothing enabled with alpha={smooth_body}")
        else:
            self.body_smoother = None

        
    def reset_robot(self):
        print("Press START on remote to move to default position ...")
        self.env.move_to_default_pos()

        print("Now in default position, press A to continue ...")
        self.env.default_pos_state()

        print("Robot will hold default pos. If needed, do other checks here.")

    def run(self):
        self.reset_robot()
        print("Begin main TWIST2 policy loop. Press [Select] on remote to exit.")

        try:
            while True:
                t_start = time.time()

                # Send remote control signals to Redis for motion server
                if self.redis_client:
                    # Send B button status (for motion start)
                    b_pressed = self.env.read_controller_input().keys == self.env.controller_mapping["B"]
                    self.redis_client.set("motion_start_signal", "1" if b_pressed else "0")
                    
                    # Send Select button status (for motion exit)
                    select_pressed = self.env.read_controller_input().keys == self.env.controller_mapping["select"]
                    self.redis_client.set("motion_exit_signal", "1" if select_pressed else "0")
                    
                if self.env.read_controller_input().keys == self.env.controller_mapping["select"]:
                    print("Select pressed, exiting main loop.")
                    break
                
                dof_pos, dof_vel, quat, ang_vel, dof_temp, dof_tau, dof_vol = self.env.get_robot_state()
                rpy = quatToEuler(quat)

                obs_dof_vel = dof_vel.copy()
                obs_dof_vel[self.ankle_idx] = 0.0

                obs_proprio = np.concatenate([
                    ang_vel * self.ang_vel_scale,
                    rpy[:2], # 只使用 roll 和 pitch
                    (dof_pos - self.default_dof_pos) * self.dof_pos_scale,
                    obs_dof_vel * self.dof_vel_scale,
                    self.last_action
                ])
                
                state_body = np.concatenate([
                    ang_vel,
                    rpy[:2],
                    dof_pos]) # 3+2+29 = 34 dims

                self.redis_pipeline.set("state_body_unitree_g1_with_hands", json.dumps(state_body.tolist()))
                
                if self.use_hand:
                    left_hand_state, right_hand_state = self.hand_ctrl.get_hand_state()
                    lh_pos, rh_pos, lh_temp, rh_temp, lh_tau, rh_tau = self.hand_ctrl.get_hand_all_state()
                    hand_left_json = json.dumps(left_hand_state.tolist())
                    hand_right_json = json.dumps(right_hand_state.tolist())
                    self.redis_pipeline.set("state_hand_left_unitree_g1_with_hands", hand_left_json)
                    self.redis_pipeline.set("state_hand_right_unitree_g1_with_hands", hand_right_json)
                
                # execute the pipeline once here for setting the keys
                self.redis_pipeline.execute()

                # 5. 从 Redis 接收模仿观察
                keys = ["action_body_unitree_g1_with_hands", "action_hand_left_unitree_g1_with_hands", "action_hand_right_unitree_g1_with_hands", "action_neck_unitree_g1_with_hands"]
                for key in keys:
                    self.redis_pipeline.get(key)
                redis_results = self.redis_pipeline.execute()
                action_mimic = json.loads(redis_results[0])
                action_hand_left = json.loads(redis_results[1])
                action_hand_right = json.loads(redis_results[2])
                action_neck = json.loads(redis_results[3])
                
                # Apply smoothing to body actions if enabled
                if self.body_smoother is not None:
                    action_mimic = self.body_smoother.smooth(np.array(action_mimic, dtype=np.float32))
                    action_mimic = action_mimic.tolist()
            
                
                if self.use_hand:
                    action_hand_left = np.array(action_hand_left, dtype=np.float32)
                    action_hand_right = np.array(action_hand_right, dtype=np.float32)
                else:
                    action_hand_left = np.zeros(7, dtype=np.float32)
                    action_hand_right = np.zeros(7, dtype=np.float32)

                obs_full = np.concatenate([action_mimic, obs_proprio])
                
                obs_hist = np.array(self.proprio_history_buf).flatten()
                self.proprio_history_buf.append(obs_full)
                
                future_obs = action_mimic.copy()
                
                obs_buf = np.concatenate([obs_full, obs_hist, future_obs])
                
                assert obs_buf.shape[0] == self.total_obs_size, f"Expected {self.total_obs_size} obs, got {obs_buf.shape[0]}"
                
                obs_tensor = torch.from_numpy(obs_buf).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    raw_action = self.policy(obs_tensor).cpu().numpy().squeeze()
                
                self.last_action = raw_action.copy()

                raw_action = np.clip(raw_action, -10.0, 10.0)
                target_dof_pos = self.default_dof_pos + raw_action * self.action_scale

                self.redis_client.set("action_low_level_unitree_g1", json.dumps(raw_action.tolist()))

                kp_scale = 1.0
                kd_scale = 1.0
                self.env.send_robot_action(target_dof_pos, kp_scale, kd_scale)
                
                if self.use_hand:
                    self.hand_ctrl.ctrl_dual_hand(action_hand_left, action_hand_right)
                
                elapsed = time.time() - t_start
                if elapsed < self.control_dt:
                    time.sleep(self.control_dt - elapsed)

                if self.record_proprio:
                    proprio_data = {
                        'timestamp': time.time(),
                        'body_dof_pos': dof_pos.tolist(),
                        'target_dof_pos': action_mimic.tolist()[-29:],
                        'temperature': dof_temp.tolist(),
                        'tau': dof_tau.tolist(),
                        'voltage': dof_vol.tolist(),
                    }
                    
                    if self.use_hand:
                        proprio_data['lh_pos'] = lh_pos.tolist()
                        proprio_data['rh_pos'] = rh_pos.tolist()
                        proprio_data['lh_temp'] = lh_temp.tolist()
                        proprio_data['rh_temp'] = rh_temp.tolist()
                        proprio_data['lh_tau'] = lh_tau.tolist()
                        proprio_data['rh_tau'] = rh_tau.tolist()
                    self.proprio_recordings.append(proprio_data)
                

        except Exception as e:
            print(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.record_proprio and self.proprio_recordings:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f'logs/twist2_real_recordings_{timestamp}.json'
                with open(filename, 'w') as f:
                    json.dump(self.proprio_recordings, f)
                print(f"Proprioceptive recordings saved as {filename}")

            self.env.close()
            if self.use_hand:
                self.hand_ctrl.close()
            print("TWIST2 real controller finished.")


def main():
    parser = argparse.ArgumentParser(description='Run TWIST2 policy on real G1 robot')
    parser.add_argument('--policy', type=str, required=True,
                        help='Path to TWIST2 ONNX policy file')
    parser.add_argument('--config', type=str, default="robot_control/configs/g1.yaml",
                        help='Path to robot configuration file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run policy on (cuda/cpu)')
    parser.add_argument('--net', type=str, default='wlp0s20f3',
                        help='Network interface for robot communication')
    parser.add_argument('--use_hand', action='store_true',
                        help='Enable hand control')
    parser.add_argument('--record_proprio', action='store_true',
                        help='Record proprioceptive data')
    parser.add_argument('--smooth_body', type=float, default=0.0,
                        help='Smoothing factor for body actions (0.0=no smoothing, 1.0=maximum smoothing)')
    
    args = parser.parse_args()

    
    # 验证文件存在
    if not os.path.exists(args.policy):
        print(f"Error: Policy file {args.policy} does not exist")
        return
    
    if not os.path.exists(args.config):
        print(f"Error: Config file {args.config} does not exist")
        return
    
    print(f"Starting TWIST2 real robot controller...")
    print(f"  Policy file: {args.policy}")
    print(f"  Config file: {args.config}")
    print(f"  Device: {args.device}")
    print(f"  Network interface: {args.net}")
    print(f"  Use hand: {args.use_hand}")
    print(f"  Record proprio: {args.record_proprio}")
    print(f"  Smooth body: {args.smooth_body}")
    
    # 安全提示
    print("\n" + "="*50)
    print("SAFETY WARNING:")
    print("You are about to run a policy on a real robot.")
    print("Make sure the robot is in a safe environment.")
    print("Press Ctrl+C to stop at any time.")
    print("Use the remote controller [Select] button to exit.")
    print("="*50 + "\n")
    
    controller = RealTimePolicyController(
        policy_path=args.policy,
        config_path=args.config,
        device=args.device,
        net=args.net,
        use_hand=args.use_hand,
        record_proprio=args.record_proprio,
        smooth_body=args.smooth_body,
    )
    
    controller.run()
    


if __name__ == "__main__":
    main()
