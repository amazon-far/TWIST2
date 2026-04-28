# G1 23-DoF Teacher/Student 학습 파이프라인

> 대상: Unitree G1 23-DoF 휴머노이드 VR 텔레오퍼레이션
> 두 트랙 병행:
> - **(A) IsaacGym 트랙** — 기존 `legged_gym/` + `rsl_rl/` 구조 유지
> - **(B) Isaac Lab 트랙** — `isaaclab_train/` 신규 (Python 3.11 + CUDA 13)

---

## 0. 23-DoF 관절 매핑

29-DoF에서 6개 관절(`waist_roll`, `waist_pitch`, `L/R_wrist_pitch`, `L/R_wrist_yaw`)을 제거.

`keep_idx = list(range(0,13)) + list(range(15,20)) + list(range(22,27))`
근거: `deploy_real/server_motion_lib.py:61-63`

기본 자세(default joint angles)는 `legged_gym/legged_gym/envs/g1/g1_mimic_distill_config.py:76-110`에서 6개 관절을 빼고 그대로 사용.

---

## 1. 데이터 흐름 (공통)

```
AMASS/OMOMO (.npz, SMPL-X)
   │  GMR (offline batch retargeting, robot=unitree_g1_23dof)
   ▼
.pkl (base_translation, base_rotation, joint_positions[23])
   │  motion_data_configs/twist2_dataset_23dof.yaml 등록
   ▼
[Teacher 학습]  →  Teacher .pt 체크포인트
   │
   ▼
[Student 학습 (RL+BC distillation)]  →  Student .pt
   │  to_onnx
   ▼
.onnx  →  sim2sim / sim2real
```

GMR pkl은 `pose/utils/motion_lib_pkl.py:MotionLib`이 로드. 23-DoF pkl은 사용자가 GMR에서 이미 생성한 상태(요청에서 확인됨).

---

## 2. 트랙 (A) — IsaacGym 학습 코드 위치

### 2.1 진입점

| 단계 | 파일 |
|------|------|
| 학습 실행 | `train.sh` → `legged_gym/legged_gym/scripts/train.py` |
| ONNX 변환 | `to_onnx.sh` → `legged_gym/legged_gym/scripts/save_onnx.py` |
| Sim2Sim 검증 | `sim2sim.sh` → `deploy_real/server_low_level_g1_sim.py` |
| Sim2Real | `sim2real.sh` → `deploy_real/server_low_level_g1_real.py` |
| 모션 서버 | `run_motion_server.sh` → `deploy_real/server_motion_lib.py` |

### 2.2 Env / Config (`legged_gym/legged_gym/envs/g1/`)

| 파일 | 역할 |
|------|------|
| `g1_mimic.py`, `g1_mimic_config.py` | 기본 mimic env (PPO 단독) |
| `g1_mimic_distill.py`, `g1_mimic_distill_config.py` | **Teacher/Student 공유 env** (`obs_type='priv'` or `'student'`) |
| `g1_mimic_future.py`, `g1_mimic_future_config.py` | **Student + future-frame 입력** (실제로 배포되는 트랙) |

베이스: `legged_gym/legged_gym/envs/base/humanoid_mimic.py` (1064 lines). 모든 mimic 보상·종료·MotionLib 연동이 여기에 있음.

Task 등록: `legged_gym/legged_gym/envs/__init__.py:51-56`
- `g1_priv_mimic` — Teacher (PPO + privileged obs)
- `g1_stu_mimic` — Student (DAgger 순수 BC)
- `g1_stu_rl` — Student (RL+BC, current frame)
- `g1_stu_future` — Student (RL+BC, future-frame 입력) ← TWIST2 기본

### 2.3 Algorithm / Runner (`rsl_rl/rsl_rl/`)

| Teacher | Student |
|---------|---------|
| `algorithms/ppo.py` | `algorithms/dagger_ppo.py` (RL+BC), `algorithms/dagger.py` (순수 BC) |
| `runners/on_policy_runner_mimic.py` | `runners/on_policy_dagger_runner.py`, `runners/dagger_runner.py` |
| `modules/actor_critic_mimic.py` | `modules/actor_critic_future.py`, `modules/actor_critic_teleop.py`, `modules/dagger_actor.py` |

`train.py`는 `task_registry.make_alg_runner(...)`로 task별 runner 클래스(`runner_class_name`)를 동적으로 인스턴스화. config의 `runner.policy_class_name / algorithm_class_name / runner_class_name` 문자열로 결정됨.

### 2.4 23-DoF로 만들기 위해 복제·수정할 파일

새 파일(권장: `*_23dof_*` 접미):
1. `envs/g1/g1_23dof_mimic_distill_config.py` — `G1MimicPrivCfg` 복제, 다음 항목 변경:
   - `num_actions = 23`
   - `default_joint_angles` — 6개 관절 제거
   - `dof_err_w` — 23개로 축소
   - `dof_armature` — 23개로 축소
   - `asset.file = '...assets/g1/g1_23dof.urdf'` (없으면 `g1_23dof.xml`로부터 변환)
   - `policy.action_std = [0.7]*12 + [0.4]*1 + [0.5]*10` (waist_yaw만 1개, 팔은 5+5)
   - `n_priv_latent`, `n_proprio`, `n_mimic_obs_single = 6 + 23`
2. `envs/g1/g1_23dof_mimic_future_config.py` — `G1MimicStuFutureCfg` 복제, 동일 변경
3. `envs/g1/g1_23dof_mimic_distill.py`, `envs/g1/g1_23dof_mimic_future.py` — 위 두 env. 코드 자체는 거의 그대로(설정만 다르면 됨); 코드 안에 29 하드코딩이 있으면 23으로 교체.
4. `envs/__init__.py`에 task 4종 등록:
   - `g1_23dof_priv_mimic`, `g1_23dof_stu_mimic`, `g1_23dof_stu_rl`, `g1_23dof_stu_future`
5. `motion_data_configs/twist2_dataset_23dof.yaml` — 23-DoF pkl 경로

Asset 변환:
```bash
# g1_23dof.xml은 MJCF, IsaacGym은 URDF 선호.
# 필요 시: assets/g1/ 에 g1_23dof.urdf를 만들어 두기 (mujoco→URDF 변환 또는 기존 29dof URDF에서 6관절 fixed화)
```

### 2.5 학습 명령

```bash
# train.sh의 task_name을 g1_23dof_stu_future 로 교체 후
bash train.sh 1103_g1_23dof_teacher cuda:0   # Teacher (먼저)
# train.sh에서 --teacher_exptid 1103_g1_23dof_teacher 로 변경
bash train.sh 1103_g1_23dof_student cuda:0   # Student
```

(또는 본 저장소에 추가된 `gui_train.sh`로 버튼 클릭 실행 — 섹션 5 참조)

---

## 3. 트랙 (B) — Isaac Lab 학습 코드 위치 (`isaaclab_train/`)

`legged_gym/`을 그대로 두고 별도 디렉토리에 신규 작성. Python 3.11 + CUDA 13 환경(`env_isaaclab` conda).

### 3.1 디렉토리 구조

```
isaaclab_train/
├── README.md
├── twist2_g1_23dof/                          # Python 패키지
│   ├── __init__.py                           # gym.register(...)
│   ├── assets/
│   │   └── g1_23dof.py                       # ArticulationCfg (USD 경로 + actuators)
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── mdp/
│   │   │   ├── __init__.py
│   │   │   ├── motion_loader.py              # GMR pkl 로드 + 프레임 보간
│   │   │   ├── observations.py               # mimic_obs, proprio, future-frame obs
│   │   │   ├── rewards.py                    # tracking_joint_dof, root, keybody, ...
│   │   │   ├── terminations.py               # pose_termination
│   │   │   └── events.py                     # domain randomization, push
│   │   ├── mimic_env_cfg.py                  # 공통 ManagerBasedRLEnvCfg
│   │   ├── teacher_env_cfg.py                # privileged + future obs
│   │   └── student_env_cfg.py                # current frame only
│   └── agents/
│       ├── rsl_rl_teacher_cfg.py             # PPO RunnerCfg
│       └── rsl_rl_student_cfg.py             # Distillation RunnerCfg
├── scripts/
│   ├── train_teacher.sh
│   ├── train_student.sh
│   ├── play.sh
│   └── export_onnx.py
└── motion_data_configs/
    └── twist2_dataset_23dof.yaml             # legged_gym과 동일 포맷 재사용
```

### 3.2 Isaac Lab 측 매핑

Isaac Lab에는 이미 **DistillationRunner** / **rsl_rl 통합**이 내장되어 있어, 직접 DAgger runner를 다시 작성할 필요는 없음.

| TWIST2 IsaacGym 개념 | Isaac Lab 대응 |
|---------------------|----------------|
| `LeggedRobot` / `Humanoid` 베이스 | `ManagerBasedRLEnv` |
| Env config(class) | `ManagerBasedRLEnvCfg` (dataclass 기반) |
| 관측 정의 | `ObservationsCfg` + `ObsTerm`(`mdp/observations.py`) |
| 보상 정의 | `RewardsCfg` + `RewTerm`(`mdp/rewards.py`) |
| Termination | `TerminationsCfg` + `DoneTerm` |
| Domain Randomization | `EventCfg` + `EventTerm` |
| `MotionLib` | `MotionLoader`(직접 작성, pkl 호환) |
| Teacher/Student 분리 | 두 env cfg + 두 RunnerCfg |
| `OnPolicyDaggerRunner` | `rsl_rl.runners.DistillationRunner` (Isaac Lab 0.2+ 제공) |

### 3.3 등록 패턴 (Isaac Lab convention)

`twist2_g1_23dof/__init__.py`:
```python
import gymnasium as gym
from . import agents

gym.register(
    id="Isaac-G1-23DoF-Mimic-Teacher-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.envs.teacher_env_cfg:G1Mimic23DofTeacherCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_teacher_cfg:G1MimicTeacherPPORunnerCfg",
    },
)
gym.register(
    id="Isaac-G1-23DoF-Mimic-Student-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.envs.student_env_cfg:G1Mimic23DofStudentCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_student_cfg:G1MimicStudentDistillRunnerCfg",
    },
)
```

### 3.4 학습 실행

```bash
conda activate env_isaaclab
cd ~/isaacsim/IsaacLab

# Teacher
./isaaclab.sh -p ~/TWIST2/isaaclab_train/scripts/train.py \
    --task Isaac-G1-23DoF-Mimic-Teacher-v0 --headless --num_envs 4096

# Student (teacher_ckpt를 distillation cfg에서 참조)
./isaaclab.sh -p ~/TWIST2/isaaclab_train/scripts/train.py \
    --task Isaac-G1-23DoF-Mimic-Student-v0 --headless --num_envs 4096 \
    --teacher_ckpt <path_to_teacher.pt>
```

`train.py`는 Isaac Lab 표준 `scripts/reinforcement_learning/rsl_rl/train.py`를 그대로 호출하거나(대부분 task만 다르면 됨), 우리 패키지를 import해 등록만 하는 얇은 래퍼로 만든다.

### 3.5 G1 23-DoF Articulation

- Isaac Lab 내장 `G1_29DOF_CFG`(`isaaclab_assets/.../unitree.py:388`)와 USD 경로(`Robots/Unitree/G1/g1.usd`)를 참고
- 23-DoF용 USD가 없으면, `assets/g1/g1_23dof.xml`을 USD로 변환하거나, 29-DoF USD에서 6관절을 fixed로 만든 `G1_23DOF_CFG`를 신규 작성
- `actuators={"legs":..., "feet":..., "arms":...}` 패턴 그대로 사용, 단 waist_yaw만 / wrist_roll만 포함

---

## 4. Teacher / Student 차이 요약 (양 트랙 공통)

| | Teacher | Student |
|---|---------|---------|
| 관측 | 현재 + 미래 N프레임 + privileged | 현재 1프레임 + proprio history |
| 알고리즘 | PPO | RL+BC (DAgger) |
| Critic | 풍부한 priv obs | 동일 / 또는 학생 obs로 축소 |
| 출력 | 학습 시에만 사용 | 실기 배포 |
| Isaac Lab 클래스 | `OnPolicyRunner` (rsl_rl) | `DistillationRunner` (rsl_rl) |

---

## 5. GUI 학습기 (`gui_train.py`)

`/home/wego/TWIST2/gui_train.py` — Tkinter 기반 버튼 런처.
- 트랙(IsaacGym/IsaacLab), 단계(Teacher/Student/Sim2Sim/Sim2Real), exptid 입력
- 버튼 클릭 시 해당 bash 스크립트를 새 터미널 또는 백그라운드 프로세스로 실행
- 로그 tail 표시
- 의존성 없음(stdlib만)

실행: `bash gui_train.sh`

---

## 6. 작업 순서 체크리스트

### IsaacGym 트랙
- [ ] 23-DoF URDF 준비 (`assets/g1/g1_23dof.urdf`)
- [ ] `g1_23dof_mimic_distill_config.py` / `g1_23dof_mimic_future_config.py` 작성
- [ ] env 클래스 파일 두 개 작성 (대부분 상속만)
- [ ] `envs/__init__.py`에 task 4종 등록
- [ ] `twist2_dataset_23dof.yaml` 작성
- [ ] `train.sh` task_name 교체 → Teacher 학습
- [ ] Student 학습 (`--teacher_exptid`)
- [ ] `to_onnx.sh` → ONNX
- [ ] `sim2sim.sh` ckpt 교체 → 검증
- [ ] `sim2real.sh` 배포

### Isaac Lab 트랙
- [ ] `isaaclab_train/` 패키지 설치 (`pip install -e isaaclab_train/`)
- [ ] G1 23-DoF USD 또는 ArticulationCfg 작성
- [ ] `motion_loader.py`로 GMR pkl 로드 검증 (단일 모션 재생)
- [ ] Teacher env 학습 (소수 envs, headless 미사용으로 디버그)
- [ ] 4096 envs headless 본학습
- [ ] Student distillation 학습
- [ ] `export_onnx.py`로 ONNX 변환
- [ ] sim2sim/sim2real에 ONNX 투입 (이 단계는 IsaacGym 트랙과 공유)

---

## 7. 참고 파일 빠른 색인

| 항목 | 경로 |
|------|------|
| 29→23 매핑 로직 | `deploy_real/server_motion_lib.py:61-63` |
| 23-DoF MJCF | `assets/g1/g1_23dof.xml` |
| 29-DoF URDF (참고) | `assets/g1/g1_custom_collision_29dof.urdf` |
| GMR motion lib | `pose/pose/utils/motion_lib_pkl.py` |
| Mimic env 베이스 | `legged_gym/legged_gym/envs/base/humanoid_mimic.py` |
| Task 등록 | `legged_gym/legged_gym/envs/__init__.py` |
| Isaac Lab G1 cfg | `~/isaacsim/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/unitree.py:272-385` |
| Isaac Lab rsl_rl 진입점 | `~/isaacsim/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py` |
