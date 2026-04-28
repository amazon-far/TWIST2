# TWIST2 + GMR 시스템 분석 정리

> 출처:
> - TWIST2: https://github.com/amazon-far/TWIST2 (arXiv 2511.02832)
> - GMR: https://github.com/YanjieZe/GMR (arXiv 2510.02252)
> - TWIST: https://github.com/YanjieZe/TWIST (arXiv 2505.02833)

---

## 1. 시스템 전체 파이프라인

```
PICO VR 헤드셋 + 모션 트래커 (100Hz)
    │  사람의 실시간 전신 포즈 (상체/하체/손)
    ▼
GMR (gmr conda 환경, Python 3.10)
    │  온라인 리타겟팅: 사람 포즈 → 로봇 관절 목표값
    │  (base_translation, base_rotation, joint_positions)
    ▼
Redis 서버
    │  high-level 참조 모션(p_cmd)으로 전달
    ▼
RL Student 정책 (twist2 conda 환경, Python 3.8)
    │  참조 모션을 물리적으로 실현 가능한 관절 명령으로 변환
    ▼
로봇 (Unitree G1, 29 DoF + Dex31 핸드)
```

GMR이 직접 로봇을 제어하는 것이 아니라, GMR의 출력은 "이 자세를 취해라"라는 **목표(reference)**이고, RL 정책이 그 목표를 물리적으로 안정적으로 따라가는 **실제 관절 명령**을 생성하는 구조이다.

RL 정책이 중간에 필요한 이유는, GMR의 리타겟팅 결과를 그대로 관절에 보내면 로봇이 넘어지거나 물리적으로 불가능한 동작이 될 수 있기 때문이다. RL 정책이 균형 유지, 접촉력 분배, 동작 부드러움 등을 처리한다.

### 코드 실행 기준

- `teleop.sh`: GMR 환경에서 PICO 스트리밍 + 리타겟팅 실행
- `sim2real.sh` 또는 `sim2sim.sh`: RL 정책이 Redis에서 참조 모션을 읽어 로봇 제어
- 두 프로세스는 별도 터미널에서 실행

---

## 2. GMR의 역할

**GMR (General Motion Retargeting)**은 사람의 모션 데이터를 휴머노이드 로봇의 관절 동작으로 변환(리타겟팅)하는 도구이다.

사람과 로봇은 신체 구조(관절 수, 팔다리 길이, 자유도 등)가 다르기 때문에, 사람의 동작을 그대로 로봇에 적용할 수 없다. GMR은 이 차이를 IK(역운동학) 기반 솔버로 해결한다.

### 지원 입력 포맷

- SMPL-X (AMASS, OMOMO): `.npz` 파일
- BVH (LAFAN1, Nokov, Xsens)
- FBX (OptiTrack)
- PICO VR 실시간 스트리밍 (XRoboToolkit)

### 지원 로봇

17종 이상: Unitree G1/H1/H1-2, Booster T1/K1, Fourier N1/GR3, HighTorque Hi, Galaxea R1 Pro, Kuavo, PAL Talos 등

### TWIST2에서 GMR의 구체적 역할

1. **오프라인 모션 데이터 전처리**: AMASS 같은 대규모 인간 모션 데이터셋을 로봇용 학습 데이터(pkl)로 일괄 변환 → RL 정책 학습에 사용
2. **온라인 텔레오퍼레이션**: PICO VR 헤드셋으로 캡처한 사람의 실시간 포즈를 로봇 관절 명령으로 즉시 변환 → CPU에서 35~70 FPS로 실시간 처리

---

## 3. GMR의 실시간 리타겟팅 Input/Output 형식

### 3.1 Input: PICO VR에서 오는 인간 포즈 데이터

PICO 4 Ultra + XRoboToolkit SDK가 100Hz로 스트리밍하는 데이터이다.

각 프레임은 인간 신체 부위별 딕셔너리(dict)로, 각 부위마다 다음 정보를 담고 있다:

```python
{
    "human_body_name": [3D global translation (x,y,z) + global rotation (quaternion, wxyz)],
    ...
}
```

PICO가 추적하는 신체 부위:

- 머리 (VR 헤드셋)
- 양손 (핸드 컨트롤러)
- 양발 (PICO Motion Tracker × 2, 발목에 부착)
- 몸통/엉덩이 등 중간 관절은 PICO 내부 알고리즘이 추정

회전은 quaternion (wxyz 순서, MuJoCo 호환)으로 표현된다.

### 3.2 GMR 내부 처리 (2단계 최적화)

TWIST2에서는 PICO의 글로벌 위치 추정이 부정확하기 때문에 원래 GMR의 2단계 최적화를 수정한다:

1. **1단계**: 링크 회전 일관성(rotation consistency) 해결
2. **2단계 (수정됨)**:
   - 하체 → 위치 + 회전 제약조건 모두 최적화 (발 미끄러짐 방지)
   - 상체 → 회전 제약조건만 최적화 (팔 동작 정확도 우선)

### 3.3 Output: Redis로 전달되는 로봇 명령 벡터 (p_cmd)

논문 수식 (1)에 정의된 형식:

```
p_cmd = [ẋ_ref, ẏ_ref, z_ref, φ_ref, θ_ref, ψ̇_ref, q_ref]
```

| 요소 | 의미 | 단위/타입 |
|------|------|-----------|
| `ẋ_ref` | 루트 x축 병진 속도 (전후) | m/s (상대값) |
| `ẏ_ref` | 루트 y축 병진 속도 (좌우) | m/s (상대값) |
| `z_ref` | 루트 z 높이 | m |
| `φ_ref` | 루트 roll 각도 | rad |
| `θ_ref` | 루트 pitch 각도 | rad |
| `ψ̇_ref` | 루트 yaw 각속도 | rad/s (상대값) |
| `q_ref` | 전신 관절 목표 위치 | rad (29 DoF 벡터) |

**설계 의도:**

1. 루트 위치/방향은 절대값이 아닌 상대값(속도)으로 표현 → 글로벌 위치 추정에 의존하지 않아 장시간 텔레오퍼레이션에서 드리프트 누적 방지
2. 하체를 단순한 루트 속도 명령이 아닌 전신 관절 위치(q_ref)로 포함 → 다리 관절까지 직접 제어하므로 발차기, 춤 등 정밀한 하체 동작 가능

---

## 4. pkl 파일의 내용과 활용

### 4.1 pkl 파일 구조

GMR의 오프라인 리타겟팅 결과물인 `.pkl` (pickle) 파일에는 프레임별 로봇 모션 데이터가 저장된다. 각 프레임은 다음 세 가지 요소로 구성:

- `robot_base_translation`: 로봇 베이스의 3D 위치 (x, y, z)
- `robot_base_rotation`: 로봇 베이스의 회전 (quaternion, wxyz 순서)
- `robot_joint_positions`: 각 관절의 목표 각도값 (rad)

### 4.2 pkl 파일 활용

1. **RL 모션 트래커 학습 데이터**: pkl 파일들을 모아 `legged_gym/motion_data_configs/twist2_dataset.yaml`에 경로를 지정하면, `bash train.sh`로 실행되는 IsaacGym RL 환경에서 teacher/student 정책이 이 모션들을 추적(tracking)하도록 학습한다.
2. **Sim2Sim / Sim2Real 검증 시 참조 모션 스트리밍**: `run_motion_server.sh`로 오프라인 모션 서버를 실행하면, pkl 파일의 모션 데이터가 Redis를 통해 high-level 명령으로 스트리밍된다.
3. **다른 프레임워크로의 변환**: pkl → CSV 변환하여 BeyondMimic 등 다른 프레임워크에서도 활용 가능 (`scripts/batch_gmr_pkl_to_csv.py`)

### 4.3 파일 형식 정리

| 파일 | 확장자 | 설명 |
|------|--------|------|
| AMASS 모션 데이터 (`--smplx_file`) | `.npz` | 프레임별 body pose, global orient, transl 등 numpy 배열 |
| SMPL-X 바디 모델 | `.pkl` | 신체 형상 파라미터 (`assets/body_models/smplx/`) |
| GMR 리타겟팅 결과 (`--save_path`) | `.pkl` | 로봇 관절 궤적 (base_pos, base_rot, joint_pos) |

> **참고**: SMPL-X 라이브러리 설치 후, pkl 형식의 바디 모델을 사용하는 경우 `smplx/body_models.py`에서 `ext`를 `npz`에서 `pkl`로 변경해야 한다.

---

## 5. RL 모션 트래커 학습 상세

### 5.1 계층적 제어 구조 (Hierarchical Control)

TWIST2는 두 계층으로 분리된 제어 구조를 사용한다:

- **High-level (System 2)**: 텔레오퍼레이션이나 비주모터 정책이 "어떤 자세를 취할지"를 결정하여 참조 모션(reference motion)을 생성한다. GMR의 pkl 파일이나 실시간 스트리밍 데이터에 해당한다.
- **Low-level (System 1)**: RL로 학습된 모션 트래커가 참조 모션을 받아 물리적으로 실현 가능한 관절 토크/위치 명령으로 변환한다.

두 계층 사이의 통신은 Redis를 통해 이루어지며, high-level은 약 30~70Hz로 목표 자세를 보내고, low-level RL 정책은 약 50Hz로 관절 명령을 출력한다.

### 5.2 Teacher-Student 2단계 학습

#### Stage 1: Teacher 정책 학습 (순수 RL)

Teacher 정책은 privileged information(특권 정보)에 접근할 수 있는 전문가 정책이다.

- **핵심 특권 정보: 미래 모션 프레임(future reference motion frames)**
  - Teacher는 현재 프레임뿐 아니라 앞으로 수 프레임의 참조 모션도 관측할 수 있다
  - 미래 정보 덕분에 Teacher는 다음에 어떤 동작이 올지 미리 알고 더 부드럽고 자신감 있는 동작을 학습한다
  - 현재 프레임만 보면 정책이 "주저하는(hesitant)" 동작을 하게 되는데, 미래 프레임을 보면 이 문제가 해결된다
- **학습 환경**: IsaacGym (NVIDIA GPU 기반 병렬 물리 시뮬레이터)
- **학습 알고리즘**: PPO (Proximal Policy Optimization)
- **학습 데이터**: GMR로 리타겟팅된 pkl 파일들의 집합

```bash
# TWIST 기준
bash train_teacher.sh 0927_twist_teacher cuda:0

# TWIST2 기준 (teacher+student 통합)
bash train.sh 1021_twist2 cuda:0
```

#### Stage 2: Student 정책 학습 (RL + BC)

Student 정책은 실제 로봇에 배포될 정책으로, 특권 정보 없이 작동해야 한다.

- **관측 공간**: 현재 시점의 참조 모션 한 프레임만 + 로봇 고유감각(proprioception)
- **학습 방법: RL + BC (Behavior Cloning) 결합**
  - RL 부분: 환경에서 직접 보상을 받아 학습 → 일반화 능력 확보
  - BC 부분: Teacher의 행동을 모방하는 손실함수 추가 → Teacher가 미래 프레임을 보고 학습한 "부드러움"을 Student에게 전달(distillation)

RL + BC 결합이 중요한 이유:

- 순수 RL만: 발 미끄러짐(feet sliding) 현상 자주 발생 (미래 동작 예측 불가)
- 순수 BC만 (DAgger 등): 학습 데이터에 없던 새로운 모션에 대한 일반화가 약함
- RL + BC: 추적 정확도와 동작 부드러움 모두에서 최적의 성능

```bash
# TWIST 기준
bash train_student.sh 0927_twist_rlbcstu 0927_twist_teacher cuda:0
```

### 5.3 Teacher vs Student 비교

| | Teacher | Student |
|---|---------|---------|
| 참조 모션 | 현재 + 미래 N프레임 | 현재 1프레임만 |
| 학습 방법 | 순수 RL (PPO) | RL + BC (Teacher 모방) |
| 용도 | 학습 시 지식 전달용 | 실제 로봇 배포용 |
| 동작 품질 | 부드럽고 예측적 | Teacher에 근접 (BC 덕분) |
| 실행 환경 | 시뮬레이션만 | 시뮬레이션 + 실제 로봇 |

Teacher는 "답을 미리 알고 있는 선생님"이고, Student는 "선생님의 풀이 방식을 배워서 답을 모르는 상황에서도 잘 푸는 학생"이다.

### 5.4 관측 공간 (Observation Space)

| 구분 | Teacher | Student |
|------|---------|---------|
| 현재 참조 모션 (관절 위치, 베이스 자세) | ✅ | ✅ |
| 미래 참조 모션 프레임 (수 프레임) | ✅ (privileged) | ❌ |
| 로봇 고유감각 (관절 각도, 각속도, IMU) | ✅ | ✅ |
| 이전 행동(action history) | ✅ | ✅ |

참조 모션의 각 프레임은 GMR pkl에서 온 `(base_translation, base_rotation, joint_positions)`이다. 정책은 현재 로봇 상태와 참조 모션 사이의 차이(error)를 관측하여, 이 차이를 줄이는 방향으로 행동한다.

### 5.5 Low-level 정책의 수식 정의

**입력 (고유감각)**:

```
s = [ω, ω̇, q, q̇]
```

- `ω`: 루트 방향 (IMU)
- `ω̇`: 루트 각속도 (IMU)
- `q`: 관절 위치 (엔코더)
- `q̇`: 관절 속도 (엔코더)

**입력 (참조 명령)**:

```
p_cmd = [ẋ_ref, ẏ_ref, z_ref, φ_ref, θ_ref, ψ̇_ref, q_ref]
```

**출력**:

```
q_tgt = π_low(s, p_cmd)     # 목표 관절 위치 (50Hz)
```

**최종 토크**:

```
τ = K_P · (q_tgt - q) - K_D · q̇    # PD 제어기
```

### 5.6 보상 함수 (Reward Function)

TWIST 계열의 모션 트래커는 DeepMimic 계열의 표준적인 보상 항목들을 사용한다:

- 관절 각도 추적 보상: 참조 모션의 관절 각도와 실제 관절 각도 차이 최소화
- 베이스 위치/회전 추적 보상: 로봇 몸통의 글로벌 위치와 방향이 참조에 가까울수록 높은 보상
- 관절 속도 추적 보상: 각속도까지 매칭
- 엔드이펙터(손, 발) 위치 보상: 손과 발의 3D 위치가 참조와 일치
- 에너지 소모 페널티: 과도한 토크 사용 억제
- 동작 부드러움 페널티: 급격한 행동 변화 억제
- 안정성 관련 보상: 넘어지지 않기 등

### 5.7 Domain Randomization (도메인 랜덤화)

Sim-to-Real 전이를 위해 시뮬레이션에서 다양한 파라미터를 랜덤화한다:

- 로봇 질량, 마찰 계수, 관절 감쇠(damping)
- 관측 노이즈 (센서 부정확성 모사)
- 외부 힘 교란 (push disturbance)
- 모터 지연/강성(stiffness) 변화

TWIST는 자체 MoCap 데이터(노이즈가 있고 캘리브레이션 드리프트가 있는 데이터)를 학습에 포함시켜, 실제 텔레오퍼레이션 환경의 노이즈에 대한 강건성을 높였다.

### 5.8 학습 후 배포 파이프라인

```
학습 완료된 Student 정책 (.pt)
    ▼
ONNX 모델로 변환 (to_onnx.sh)
    ▼
Sim2Sim 검증 (sim2sim.sh)
    ▼
Sim2Real 배포 (sim2real.sh)
```

- TWIST: `.pt` → JIT 변환 / TWIST2: `.pt` → `.onnx` 변환
- 배포 시 Student 정책만 사용 (Teacher는 학습 시에만 사용)
- 약 50Hz로 실행, RTX 4090 한 장으로 1~2일이면 학습 완료

### 5.9 GMR pkl이 학습에 관여하는 흐름

```
AMASS/OMOMO 인간 모션 데이터 (SMPL-X, .npz 포맷)
    ▼  GMR 일괄 리타겟팅 (smplx_to_robot_dataset.py)
로봇용 pkl 파일들 (base_pos, base_rot, joint_pos 시퀀스)
    ▼  경로를 twist2_dataset.yaml에 등록
IsaacGym 환경에서 참조 모션으로 로드
    ▼  매 에피소드마다 랜덤하게 모션 클립 샘플링
RL 정책이 해당 모션을 추적하도록 학습
```

학습 시 수천~수만 개의 다양한 모션 클립을 병렬 환경에서 동시에 추적하게 하여, 걷기, 달리기, 춤, 물건 집기 등 **범용 모션 트래커(general motion tracker)**를 얻는다.

---

## 6. teleop.sh 스크립트 분석

```bash
source ~/miniconda3/bin/activate gmr

cd deploy_real

redis_ip="localhost"
actual_human_height=1.6

python xrobot_teleop_to_robot_w_hand.py --robot unitree_g1 \
    --actual_human_height $actual_human_height \
    --redis_ip $redis_ip \
    --target_fps 100 \
    --measure_fps 1 \
```

- `gmr` conda 환경 활성화 (Python 3.10)
- `xrobot_teleop_to_robot_w_hand.py`: PICO → GMR 리타겟팅 → Redis 전송을 수행하는 메인 스크립트
- `--actual_human_height`: PICO 추정 부정확성 보정을 위해 실제보다 작게 설정
- `--target_fps 100`: 목표 처리 속도
- `--redis_ip localhost`: 같은 PC에서 sim2sim 테스트 시 localhost 사용

---

## 7. 참고 논문 및 리포지토리

```bibtex
@article{ze2025twist2,
  title={TWIST2: Scalable, Portable, and Holistic Humanoid Data Collection System},
  author={Yanjie Ze and Siheng Zhao and Weizhuo Wang and Angjoo Kanazawa and Rocky Duan
          and Pieter Abbeel and Guanya Shi and Jiajun Wu and C. Karen Liu},
  year={2025},
  journal={arXiv preprint arXiv:2511.02832}
}

@article{ze2025twist,
  title={TWIST: Teleoperated Whole-Body Imitation System},
  author={Yanjie Ze and Zixuan Chen and João Pedro Araújo and Zi-ang Cao
          and Xue Bin Peng and Jiajun Wu and C. Karen Liu},
  year={2025},
  journal={arXiv preprint arXiv:2505.02833}
}

@article{joao2025gmr,
  title={Retargeting Matters: General Motion Retargeting for Humanoid Motion Tracking},
  author={Joao Pedro Araujo and Yanjie Ze and Pei Xu and Jiajun Wu and C. Karen Liu},
  year={2025},
  journal={arXiv preprint arXiv:2510.02252}
}
```
