"""
conda activate gmr
sudo ufw disable
python xrobot_teleop_to_robot_w_hand.py --robot unitree_g1

State Machine Controls:
- Right controller key_one: Cycle through idle -> teleop -> pause -> teleop...
- Left controller key_one: Exit program from any state
- Left controller axis_click: Emergency stop - kills sim2real.sh process
- Left controller axis: Control root xy velocity and yaw velocity
- Right controller axis: Fine-tune root xy velocity and yaw velocity
- Auto-transition: idle -> teleop when motion data is available

States:
- idle: Waiting for input or data
- teleop: Processing motion retargeting with velocity control
- pause: Data received but not processing
- exit: Program will terminate

Whole-Body Teleop Features:
- Sends whole-body mode information to Redis
- 35-dimensional mimic observations
- Uses retargeted motion directly from the teleoperation stream
"""
import argparse
import json
import pathlib
import os
import subprocess
import sys
import time

import mujoco as mj
import mujoco.viewer as mjv
import numpy as np
from loop_rate_limiters import RateLimiter
from scipy.spatial.transform import Rotation as R
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import draw_frame
from general_motion_retargeting import ROBOT_XML_DICT, ROBOT_BASE_DICT
from general_motion_retargeting import human_head_to_robot_neck
from rich import print
from tqdm import tqdm
import cv2
import redis
from rich import print
from general_motion_retargeting import XRobotStreamer

from data_utils.params import DEFAULT_MIMIC_OBS, DEFAULT_HAND_POSE
from data_utils.rot_utils import euler_from_quaternion_np, quat_diff_np, quat_rotate_inverse_np
from data_utils.fps_monitor import FPSMonitor

def start_interpolation(state_machine, start_obs, end_obs, duration=1.0):
    """Start interpolation from start_obs to end_obs over given duration"""
    state_machine.is_interpolating = True
    state_machine.interpolation_start_time = time.time()
    state_machine.interpolation_duration = duration
    state_machine.interpolation_start_obs = start_obs.copy() if start_obs is not None else None
    state_machine.interpolation_target_obs = end_obs.copy() if end_obs is not None else None
    

def get_interpolated_obs(state_machine):
    """Get current interpolated observation, returns None if interpolation complete"""
    if (not state_machine.is_interpolating or 
        state_machine.interpolation_start_obs is None or 
        state_machine.interpolation_target_obs is None or 
        state_machine.interpolation_start_time is None):
        return None
    elapsed_time = time.time() - state_machine.interpolation_start_time
    progress = min(elapsed_time / state_machine.interpolation_duration, 1.0)
    
    # Linear interpolation
    interp_obs = state_machine.interpolation_start_obs + (state_machine.interpolation_target_obs - state_machine.interpolation_start_obs) * progress
    
    # Check if interpolation is complete
    if progress >= 1.0:
        state_machine.is_interpolating = False
        return state_machine.interpolation_target_obs
    
    return interp_obs

def extract_mimic_obs_whole_body(qpos, last_qpos, dt=1/30):
    """Extract whole body mimic observations from robot joint positions (35 dims)"""
    root_pos, last_root_pos = qpos[0:3], last_qpos[0:3]
    root_quat, last_root_quat = qpos[3:7], last_qpos[3:7]
    robot_joints = qpos[7:].copy()  # Make a copy to avoid modifying original
    base_vel = (root_pos - last_root_pos) / dt
    base_ang_vel = quat_diff_np(last_root_quat, root_quat, scalar_first=True) / dt
    roll, pitch, yaw = euler_from_quaternion_np(root_quat.reshape(1, -1), scalar_first=True)
    # convert root vel to local frame
    base_vel_local = quat_rotate_inverse_np(root_quat, base_vel, scalar_first=True)
    base_ang_vel_local = quat_rotate_inverse_np(root_quat, base_ang_vel, scalar_first=True)
    
    # Standard mimic observation (35 dims)
    height = root_pos[2:3]
    # print("height: ", height)
    mimic_obs = np.concatenate([
        base_vel_local[:2],  # xy velocity (2 dims)
        root_pos[2:3],       # z position (1 dim)
        roll, pitch,         # roll, pitch (2 dims)
        base_ang_vel_local[2:3],  # yaw angular velocity (1 dim)
        robot_joints         # joint positions (29 dims)
    ])
    
    return mimic_obs



class StateMachine:
    def __init__(self, enable_smooth=False, smooth_window_size=5, use_pinch=False):
        """
        State process for teleoperation:
        idle -> teleop -> pause -> teleop ... -> idle -> exit
        """
        self.state = "idle"
        self.previous_state = "idle"
        self.right_key_one_was_pressed = False
        self.left_key_one_was_pressed = False
        self.left_axis_click_was_pressed = False
        # Interpolation state
        self.is_interpolating = False
        self.interpolation_start_time = None
        self.interpolation_duration = 2.0  # seconds
        self.interpolation_start_obs = None
        self.interpolation_target_obs = None
        self.current_mimic_obs = None
        self.last_mimic_obs = None
        self.current_neck_data = None
        self.last_neck_data = None

        # Hand state - interpolation values (0.0 = open, 1.0 = closed)
        self.hand_left_position = 0.0  # 0.0 = fully open, 1.0 = fully closed
        self.hand_right_position = 0.0
        self.use_pinch = use_pinch
        # Hand control parameters
        self.hand_movement_step = 0.05  # 5% movement per press/hold
        
        # Velocity commands from joystick
        self.velocity_commands = np.array([0.0, 0.0, 0.0])  # [vx, vy, vyaw]
        
        # Smooth filtering
        self.enable_smooth = enable_smooth
        self.smooth_window_size = smooth_window_size
        self.smooth_history = []  # Store recent observations for sliding window

    def update(self, controller_data):
        """Update state machine with controller data"""
        # Store previous state
        self.previous_state = self.state
        
        # Get current button states
        right_key_current = controller_data.get('RightController', {}).get('key_one', False)
        left_key_current = controller_data.get('LeftController', {}).get('key_one', False)
        
        # Hand control - index_trig for close, grip for open
        right_index_trig_current = controller_data.get('RightController', {}).get('index_trig', False)
        left_index_trig_current = controller_data.get('LeftController', {}).get('index_trig', False)
        right_grip_current = controller_data.get('RightController', {}).get('grip', False)
        left_grip_current = controller_data.get('LeftController', {}).get('grip', False)

        # Emergency stop - left controller axis_click
        left_axis_click_current = controller_data.get('LeftController', {}).get('axis_click', False)

        # Detect button presses
        right_key_just_pressed = right_key_current and not self.right_key_one_was_pressed
        left_key_just_pressed = left_key_current and not self.left_key_one_was_pressed
        left_axis_click_just_pressed = left_axis_click_current and not self.left_axis_click_was_pressed

        # Handle left axis click - emergency stop
        if left_axis_click_just_pressed:
            self._emergency_stop()

        # Handle left key press - exit from any state
        if left_key_just_pressed:
            self.state = "exit"

        # Handle right key press - cycle between idle, teleop, pause
        elif right_key_just_pressed:
            if self.state == "idle":
                self.state = "teleop"
            elif self.state == "teleop":
                self.state = "pause"
            elif self.state == "pause":
                self.state = "teleop"

        # Handle hand control - continuous interpolation
        # Right hand control
        if right_index_trig_current:  # Close right hand
            new_position = min(1.0, self.hand_right_position + self.hand_movement_step)
            if new_position != self.hand_right_position:
                self.hand_right_position = new_position
                print(f"Right hand closing: {self.hand_right_position:.1f}")
        elif right_grip_current:  # Open right hand
            new_position = max(0.0, self.hand_right_position - self.hand_movement_step)
            if new_position != self.hand_right_position:
                self.hand_right_position = new_position
                print(f"Right hand opening: {self.hand_right_position:.1f}")
        
        # Left hand control
        if left_index_trig_current:  # Close left hand
            new_position = min(1.0, self.hand_left_position + self.hand_movement_step)
            if new_position != self.hand_left_position:
                self.hand_left_position = new_position
                print(f"Left hand closing: {self.hand_left_position:.1f}")
        elif left_grip_current:  # Open left hand
            new_position = max(0.0, self.hand_left_position - self.hand_movement_step)
            if new_position != self.hand_left_position:
                self.hand_left_position = new_position
                print(f"Left hand opening: {self.hand_left_position:.1f}")
        
        # Extract velocity commands from controller axes
        self._update_velocity_commands(controller_data)
        
        # Update button state tracking
        self.right_key_one_was_pressed = right_key_current
        self.left_key_one_was_pressed = left_key_current
        self.left_axis_click_was_pressed = left_axis_click_current
    
    def _update_velocity_commands(self, controller_data):
        """Update velocity commands from controller axes"""
        left_axis = controller_data.get('LeftController', {}).get('axis', [0.0, 0.0])
        right_axis = controller_data.get('RightController', {}).get('axis', [0.0, 0.0])
        
        # Use left stick for xy movement, right stick for yaw rotation
        if len(left_axis) >= 2 and len(right_axis) >= 2:
            # Scale factors for velocity commands
            xy_scale = 2.0  # m/s
            yaw_scale = 3.0  # rad/s
            
            self.velocity_commands[0] = left_axis[1] * xy_scale   # forward/backward (y axis inverted)
            self.velocity_commands[1] = -left_axis[0] * xy_scale  # left/right (x axis inverted)
            self.velocity_commands[2] = -right_axis[0] * yaw_scale  # yaw rotation (x axis inverted)
    
    def has_state_changed(self):
        """Check if state has changed since last update"""
        return self.state != self.previous_state
    
    
    def set_current_mimic_obs(self, mimic_obs):
        """Update current mimic obs"""
        self.current_mimic_obs = mimic_obs.copy() if mimic_obs is not None else None
        
    def set_last_mimic_obs(self, mimic_obs):
        """Update last mimic obs (used when entering pause)"""
        self.last_mimic_obs = mimic_obs.copy() if mimic_obs is not None else None
        
    def set_last_neck_data(self, neck_data):
        """Update last neck data (used when entering pause)"""
        self.last_neck_data = neck_data[:] if neck_data is not None else None
        
    def set_current_neck_data(self, neck_data):
        """Update current neck data"""
        self.current_neck_data = neck_data[:] if neck_data is not None else None
    
    def get_current_state(self):
        return self.state
    

    def get_velocity_commands(self):
        return self.velocity_commands.copy()
        
    def is_teleop_active(self):
        """Return True if currently in teleop state"""
        return self.state == "teleop"
        
    def should_exit(self):
        """Return True if should exit the program"""
        return self.state == "exit"
        
    def should_process_data(self):
        """Return True if should process motion data"""
        return self.state == "teleop" and not self.is_interpolating
    
    def get_hand_state(self):
        return self.hand_left_position, self.hand_right_position
    
    def get_hand_pose(self, robot_name):
        """Get interpolated hand poses based on current hand positions"""
        use_pinch = self.use_pinch
        # Get open and closed poses
        
        if not use_pinch:
            left_open = DEFAULT_HAND_POSE[robot_name]['left']['open']
            left_closed = DEFAULT_HAND_POSE[robot_name]['left']['close']
            right_open = DEFAULT_HAND_POSE[robot_name]['right']['open']
            right_closed = DEFAULT_HAND_POSE[robot_name]['right']['close']
        else:
            left_fully_open = DEFAULT_HAND_POSE[robot_name]['left']['open_pinch']
            left_fully_closed = DEFAULT_HAND_POSE[robot_name]['left']['close_pinch']
            right_fully_open = DEFAULT_HAND_POSE[robot_name]['right']['open_pinch']
            right_fully_closed = DEFAULT_HAND_POSE[robot_name]['right']['close_pinch']

            # compute the intermediate poses to shortern the distance betwen open and close
            # ratio * open + (1 - ratio) * closed
            ratio_open = 0.8
            ratio_closed = 0.0
            left_open =  left_fully_open * ratio_open + (1 - ratio_open) * left_fully_closed
            left_closed = left_fully_open * ratio_closed + (1 - ratio_closed) * left_fully_closed
            right_open = right_fully_open * ratio_open + (1 - ratio_open) * right_fully_closed
            right_closed = right_fully_open * ratio_closed + (1 - ratio_closed) * right_fully_closed
        
        # Interpolate between open and closed poses
        left_pose = left_open + (left_closed - left_open) * self.hand_left_position
        right_pose = right_open + (right_closed - right_open) * self.hand_right_position
        
        return left_pose, right_pose
    
    def apply_smooth(self, mimic_obs):
        """Apply sliding window smoothing to mimic observations"""
        if not self.enable_smooth or mimic_obs is None:
            return mimic_obs
            
        # Convert to numpy array if needed
        obs_array = np.array(mimic_obs) if not isinstance(mimic_obs, np.ndarray) else mimic_obs.copy()
        
        # Add current observation to history
        self.smooth_history.append(obs_array)
        
        # Keep only the recent window_size observations
        if len(self.smooth_history) > self.smooth_window_size:
            self.smooth_history.pop(0)
            
        # Apply sliding window average
        if len(self.smooth_history) >= 2:  # Need at least 2 observations for smoothing
            # Stack all observations in history
            history_stack = np.stack(self.smooth_history, axis=0)  # Shape: (history_len, obs_dim)
            # Compute mean across the time dimension
            smoothed_obs = np.mean(history_stack, axis=0)
            return smoothed_obs
        else:
            # Not enough history, return original observation
            return obs_array
    
    def reset_smooth_history(self):
        """Reset smooth history (call when transitioning states)"""
        self.smooth_history = []
    
    def _emergency_stop(self):
        """Emergency stop: kill sim2real.sh process (server_low_level_g1_real_future.py)"""
        try:
            print("[EMERGENCY STOP] Killing sim2real.sh process...")
            # Kill sim2real.sh which contains server_low_level_g1_real_future.py
            result = subprocess.run(['pkill', '-f', 'sim2real.sh'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("[EMERGENCY STOP] Successfully killed sim2real.sh process")
            else:
                print(f"[EMERGENCY STOP] pkill returned code {result.returncode}")

            # Also try to kill the specific server script directly as backup
            result2 = subprocess.run(['pkill', '-f', 'server_low_level_g1_real_future.py'], 
                                   capture_output=True, text=True, timeout=5)
            if result2.returncode == 0:
                print("[EMERGENCY STOP] Successfully killed server_low_level_g1_real_future.py process")
            else:
                print(f"[EMERGENCY STOP] pkill for server script returned code {result2.returncode}")
                
        except subprocess.TimeoutExpired:
            print("[EMERGENCY STOP] pkill command timed out")
        except Exception as e:
            print(f"[EMERGENCY STOP] Error executing pkill: {e}")

class XRobotTeleopToRobot:
    def __init__(self, args):
        self.args = args
        self.robot_name = args.robot
        self.xml_file = ROBOT_XML_DICT[args.robot]
        self.robot_base = ROBOT_BASE_DICT[args.robot]
        
        print(f"Pinch mode: {self.args.pinch_mode}")
        # Initialize state tracking
        self.last_qpos = None
        self.last_time = time.time()
        self.target_fps = args.target_fps
        self.measured_dt = 1/ self.target_fps # default fallback dt

        # Initialize components
        self.teleop_data_streamer = None
        self.redis_client = None
        self.retarget = None
        self.model = None
        self.data = None
        self.state_machine = StateMachine(
            enable_smooth=args.smooth,
            smooth_window_size=args.smooth_window_size,
            use_pinch=args.pinch_mode
        )
        self.rate = None
        
        # Video recording
        self.video_writer = None
        self.renderer = None
        
        # FPS monitoring
        self.fps_monitor = FPSMonitor(
            enable_detailed_stats=args.measure_fps,
            quick_print_interval=100,
            detailed_print_interval=1000,
            expected_fps=self.target_fps,
            name="Teleop Loop"
        )


    def setup_teleop_data_streamer(self):
        """Initialize and start the teleop data streamer"""
        self.teleop_data_streamer = XRobotStreamer()
        if self.teleop_data_streamer is not None:
            print("Teleop data streamer initialized")
        else:
            print("Failed to initialize teleop data streamer")
        
    def setup_redis_connection(self):
        """Setup Redis connection"""
        redis_ip = self.args.redis_ip
        self.redis_client = redis.Redis(host=redis_ip, port=6379, db=0)
        self.redis_pipeline = self.redis_client.pipeline()
        self.redis_client.ping()
        print("Redis connected successfully")

    def setup_retargeting_system(self):
        """Initialize the motion retargeting system"""
        self.retarget = GMR(
            src_human="xrobot",
            tgt_robot="unitree_g1",
            actual_human_height=self.args.actual_human_height,
        )
        print("Retargeting system initialized")
    
    def setup_mujoco_simulation(self):
        """Setup MuJoCo model and data"""
        self.model = mj.MjModel.from_xml_path(str(self.xml_file))
        self.data = mj.MjData(self.model)
        print("MuJoCo simulation initialized")
        
    def setup_video_recording(self):
        """Setup video recording if requested"""
        if not self.args.record_video:
            return
            
        self.video_writer = cv2.VideoWriter(
            'output.mp4', 
            cv2.VideoWriter_fourcc(*'mp4v'), 
            30, 
            (640, 480)
        )
        width, height = 640, 480
        self.renderer = mj.Renderer(self.model, height=height, width=width)
        print("Video recording setup completed")
        
    def setup_rate_limiter(self):
        """Setup rate limiter for consistent FPS"""
        self.rate = RateLimiter(frequency=self.target_fps, warn=False)
        print(f"Rate limiter setup for {self.target_fps} FPS")
        
    def get_teleop_data(self):
        """Get current teleop data from streamer"""
        if self.teleop_data_streamer is not None:
            return self.teleop_data_streamer.get_current_frame()
        return None, None, None, None, None
        
    def process_retargeting(self, smplx_data):
        """Process motion retargeting and return observations"""
        if smplx_data is None or self.retarget is None:
            return None, None
            
        # Measure dt between retarget calls
        current_time = time.time()
        self.measured_dt = current_time - self.last_time
        self.last_time = current_time
        
        # Retarget till convergence
        qpos = self.retarget.retarget(smplx_data, offset_to_ground=True)
        
        # Create mimic obs from retargeting
        if self.last_qpos is not None:
            current_retarget_obs = extract_mimic_obs_whole_body(qpos, self.last_qpos, dt=self.measured_dt)
        else:
            current_retarget_obs = DEFAULT_MIMIC_OBS[self.robot_name]
        
        self.last_qpos = qpos.copy()
        return qpos, current_retarget_obs
        
    def update_visualization(self, qpos, smplx_data, viewer):
        """Update MuJoCo visualization"""
        if qpos is None:
            return
            
        # Clean custom geometry
        if hasattr(viewer, 'user_scn') and viewer.user_scn is not None:
            viewer.user_scn.ngeom = 0
            
        # Draw the task targets for reference
        if smplx_data is not None and self.retarget is not None:
            for robot_link, ik_data in self.retarget.ik_match_table1.items():
                body_name = ik_data[0]
                if body_name not in smplx_data:
                    continue
                draw_frame(
                    self.retarget.scaled_human_data[body_name][0] - self.retarget.ground,
                    R.from_quat(smplx_data[body_name][1]).as_matrix(),
                    viewer,
                    0.1,
                    orientation_correction=R.from_quat(ik_data[-1]),
                )
                
        # Update the simulation
        if qpos is not None:
            self.data.qpos[:] = qpos.copy()
            mj.mj_forward(self.model, self.data)
            
            # Camera follow the pelvis
            self._update_camera_position(viewer)
        
    def _update_camera_position(self, viewer):
        """Update camera to follow the robot"""
        FOLLOW_CAMERA = True
        if FOLLOW_CAMERA:
            robot_base_pos = self.data.xpos[self.model.body(self.robot_base).id]
            viewer.cam.lookat = robot_base_pos
            viewer.cam.distance = 3.0
            
    def handle_state_transitions(self, current_retarget_obs):
        """Handle state machine transitions and interpolations"""
        if not self.state_machine.has_state_changed():
            return
            
        current_state = self.state_machine.get_current_state()
        previous_state = self.state_machine.previous_state
        
        print(f"State changed: {previous_state} -> {current_state}")
        
        if current_state == "teleop":
            self._handle_enter_teleop(previous_state, current_retarget_obs)
        elif current_state == "pause":
            self._handle_enter_pause()
            
    def _handle_enter_teleop(self, previous_state, current_retarget_obs):
        """Handle entering teleop state"""
        if previous_state in ["idle", "pause"]:
            self.state_machine.reset_smooth_history()
            print("Reset smooth history on entering teleop")

        if previous_state == "idle":
            if current_retarget_obs is not None:
                default_obs = DEFAULT_MIMIC_OBS[self.robot_name]
                start_interpolation(self.state_machine, default_obs, current_retarget_obs[:35])
                print("Interpolating from default to teleop...")
        elif previous_state == "pause":
            if (current_retarget_obs is not None and
                self.state_machine.last_mimic_obs is not None):
                last_obs_35d = self.state_machine.last_mimic_obs[:35] if len(self.state_machine.last_mimic_obs) > 35 else self.state_machine.last_mimic_obs
                start_interpolation(self.state_machine, last_obs_35d, current_retarget_obs[:35])
                print("Interpolating from pause to teleop...")
    def _handle_enter_pause(self):
        """Handle entering pause state"""
        if self.state_machine.current_mimic_obs is not None:
            self.state_machine.set_last_mimic_obs(self.state_machine.current_mimic_obs)
            print("Entered pause mode, storing last obs")
        if self.state_machine.current_neck_data is not None:
            self.state_machine.set_last_neck_data(self.state_machine.current_neck_data)
            print("Entered pause mode, storing last neck data")
            
    def determine_mimic_obs_to_send(self, current_retarget_obs):
        """Determine which mimic observation to send based on current state"""
        current_state = self.state_machine.get_current_state()

        if current_state == "idle":
            obs = DEFAULT_MIMIC_OBS[self.robot_name]
        elif current_state == "pause":
            if self.state_machine.last_mimic_obs is not None:
                obs = self.state_machine.last_mimic_obs[:35] if len(self.state_machine.last_mimic_obs) > 35 else self.state_machine.last_mimic_obs
            else:
                obs = DEFAULT_MIMIC_OBS[self.robot_name]
        elif current_state == "teleop":
            obs = self._get_teleop_mimic_obs(current_retarget_obs)
            obs = self.state_machine.apply_smooth(obs)
        else:
            obs = DEFAULT_MIMIC_OBS[self.robot_name]

        return obs
        
    def _get_teleop_mimic_obs(self, current_retarget_obs):
        """Get mimic obs for teleop state, handling interpolation"""
        if self.state_machine.is_interpolating:
            interp_obs = get_interpolated_obs(self.state_machine)
            if interp_obs is not None:
                self.state_machine.set_current_mimic_obs(interp_obs)
                return interp_obs
            return DEFAULT_MIMIC_OBS[self.robot_name]

        if current_retarget_obs is not None:
            obs_35d = current_retarget_obs[:35] if len(current_retarget_obs) > 35 else current_retarget_obs
            self.state_machine.set_current_mimic_obs(obs_35d)
            return obs_35d

        return DEFAULT_MIMIC_OBS[self.robot_name]
    
    def determine_neck_data_to_send(self, smplx_data):
        """Determine which neck data to send based on current state"""
       
        current_state = self.state_machine.get_current_state()
        
        # In non-teleop states, send default neck position [0, 0]
        if current_state in ["idle"]:
            return [0.0, 0.0]
        
        if current_state == "pause":
            # return [0.0, 0.0]
            # use last neck data
            if self.state_machine.last_neck_data is not None:
                return self.state_machine.last_neck_data
            else:
                return [0.0, 0.0]
            
        # In teleop state, extract neck data from smplx_data
        elif current_state == "teleop" and smplx_data is not None:
            scale = self.args.neck_retarget_scale
            neck_yaw, neck_pitch = human_head_to_robot_neck(smplx_data)
            return [neck_yaw * scale, neck_pitch * scale]
        
        # Default fallback
        return [0.0, 0.0]
            
    def send_to_redis(self, mimic_obs, neck_data=None):
        """Send mimic observations to Redis"""
        
        if self.redis_client is not None and mimic_obs is not None:
            # Expect 35D mimic observations
            assert len(mimic_obs) == 35, f"Expected 35 mimic obs dims, got {len(mimic_obs)}"
            # Send to both keys for compatibility
            self.redis_pipeline.set("action_body_unitree_g1_with_hands", json.dumps(mimic_obs.tolist()))
        
        # Send hand action to redis
        if self.redis_client is not None:
            hand_left_pose, hand_right_pose = self.state_machine.get_hand_pose(self.robot_name)
            self.redis_pipeline.set("action_hand_left_unitree_g1_with_hands", json.dumps(hand_left_pose.tolist()))
            self.redis_pipeline.set("action_hand_right_unitree_g1_with_hands", json.dumps(hand_right_pose.tolist()))
        
        # Send neck data to redis
        if neck_data is not None:
            self.redis_pipeline.set("action_neck_unitree_g1_with_hands", json.dumps(neck_data))
        
        # Send timestamp to redis
        t_action = int(time.time() * 1000) # current timestamp in ms
        self.redis_pipeline.set("t_action", t_action)

        # execute the pipeline once
        self.redis_pipeline.execute()

    
    def send_controller_data_to_redis(self, controller_data):
        """Send controller data to Redis"""
        if self.redis_client is not None and controller_data is not None:
            self.redis_client.set(
                f"controller_data", 
                json.dumps(controller_data)
            )
            
            
    def record_video_frame(self, viewer):
        """Record current frame to video if recording is enabled"""
        if not self.args.record_video or self.renderer is None:
            return
            
        self.renderer.update_scene(self.data, camera=viewer.cam)
        pixels = self.renderer.render()
        
        # Convert from RGB to BGR (OpenCV uses BGR)
        frame = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        self.video_writer.write(frame)
        
    def handle_exit_sequence(self, viewer):
        """Handle graceful exit with interpolation to default pose"""
        if self.state_machine.current_mimic_obs is not None:
            default_obs = DEFAULT_MIMIC_OBS[self.robot_name]
            current_obs = self.state_machine.current_mimic_obs[:35] if len(self.state_machine.current_mimic_obs) > 35 else self.state_machine.current_mimic_obs
            start_interpolation(self.state_machine, current_obs, default_obs)
            print("Interpolating to default pose before exit...")
            
            # Wait for interpolation to complete
            while self.state_machine.is_interpolating:
                interp_obs = get_interpolated_obs(self.state_machine)
                if interp_obs is not None:
                    # During exit sequence, send default neck position [0, 0]
                    neck_data_to_send = self.determine_neck_data_to_send(None)
                    self.send_to_redis(interp_obs, neck_data_to_send)
                viewer.sync()
                self.rate.sleep()
                


    def initialize_all_systems(self):
        """Initialize all required systems"""
        print("Initializing teleop systems...")
        self.setup_teleop_data_streamer()
        self.setup_redis_connection()
        self.setup_retargeting_system()
        self.setup_mujoco_simulation()
        self.setup_video_recording()
        self.setup_rate_limiter()

        print("Teleop state machine initialized. Controls:")
        print("- Right controller key_one: Cycle through idle -> teleop -> pause -> teleop...")
        print("- Left controller key_one: Exit program")
        print("- Left controller axis_click: Emergency stop - kills sim2real.sh process")
        print("- Left controller axis: Control root xy velocity")
        print("- Right controller axis: Control yaw velocity")
        print("- Publishes 35-dimensional mimic observations")
        print(f"Starting in state: {self.state_machine.get_current_state()}")

        if self.state_machine.enable_smooth:
            print(f"- Smooth filtering: ENABLED (window size: {self.state_machine.smooth_window_size} frames)")
        else:
            print("- Smooth filtering: DISABLED")
        
        if self.fps_monitor.enable_detailed_stats:
            print(f"- FPS measurement: ENABLED (detailed stats every {self.fps_monitor.detailed_print_interval} steps)")
        else:
            print(f"- FPS measurement: Quick stats only (every {self.fps_monitor.quick_print_interval} steps)")

        print("Ready to receive teleop data.")

    def run(self):
        """Main execution loop"""
        self.initialize_all_systems()
        
        # Start the viewer
        with mjv.launch_passive(
            model=self.model, 
            data=self.data, 
            show_left_ui=False, 
            show_right_ui=False
        ) as viewer:
            viewer.opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = 1
            
            while viewer.is_running():
                # Get current teleop data
                smplx_data, left_hand_data, right_hand_data, controller_data, headset_data = self.get_teleop_data()
                
                # Update state machine
                if controller_data is not None:
                    self.state_machine.update(controller_data)
                    self.send_controller_data_to_redis(controller_data)
                
                # Check if we should exit
                if self.state_machine.should_exit():
                    print("Exit requested via controller")
                    self.handle_exit_sequence(viewer)
                    break
                
                # Process retargeting if we have data
                qpos, current_retarget_obs = None, None
                if smplx_data is not None:
                    qpos, current_retarget_obs = self.process_retargeting(smplx_data)
                    self.update_visualization(qpos, smplx_data, viewer)
                
                # Handle state transitions
                self.handle_state_transitions(current_retarget_obs)
                
                # Determine and send mimic observations
                mimic_obs_to_send = self.determine_mimic_obs_to_send(current_retarget_obs)
                neck_data_to_send = self.determine_neck_data_to_send(smplx_data)
                
                # Store current neck data in state machine for pause state handling
                if neck_data_to_send is not None:
                    self.state_machine.set_current_neck_data(neck_data_to_send)
                
                self.send_to_redis(mimic_obs_to_send, neck_data_to_send)
                
                # Update visualization and record video
                viewer.sync()
                self.record_video_frame(viewer)
                
                # FPS monitoring
                self.fps_monitor.tick()
                
                self.rate.sleep()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands"],
        default="unitree_g1",
    )
    parser.add_argument(
        "--record_video",
        action="store_true",
        help="Whether to record the video.",
    )
    parser.add_argument(
        "--pinch_mode",
        action="store_true",
        help="Whether to use pinch mode for hand control.",
        default=False,
    )
    parser.add_argument(
        "--redis_ip",
        type=str,
        default="localhost",
        help="Redis IP",
    )
    parser.add_argument(
        "--actual_human_height",
        type=float,
        default=1.5,
        help="Actual human height for retargeting.",
    )   
    parser.add_argument(
        "--neck_retarget_scale",
        type=float,
        default=1.5,
        help="Scale factor for neck data.",
    )
    parser.add_argument(
        "--smooth",
        action="store_true",
        help="Enable smooth filtering for mimic observations in teleop mode.",
    )
    parser.add_argument(
        "--smooth_window_size",
        type=int,
        default=5,
        help="Window size for sliding window smoothing (default: 5 frames).",
    )
    parser.add_argument(
        "--target_fps",
        type=int,
        default=100,
        help="Target FPS for the teleop system.",
    )
    parser.add_argument(
        "--measure_fps",
        type=int,
        default=0,
        help="Measure and print detailed FPS statistics (0=disabled, 1=enabled).",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    teleop_robot = XRobotTeleopToRobot(args)
    teleop_robot.run()