# G1 23-DoF Teleoperation 학습 파이프라인

> 목표: 기존 29-DoF 정책(`assets/ckpts/twist2_1017_20k.onnx`)은 Unitree G1 29-DoF 구조(허리 3축 + 손목 3축)에 맞춰져 있어, **23-DoF G1**(허리 yaw만 + 손목 roll만)용 VR 텔레오퍼레이션을 위해서는 Teacher/Student 정책을 처음부터 재학습해야 한다.

---

## 0. 23-DoF vs 29-DoF DoF 차이

29-DoF 모션에서 23-DoF로 매핑할 때 제거되는 6개 관절 (참고: `deploy_real/server_motion_lib.py:61-63`):

| 제거 idx | 관절명 |
|---------|--------|
| 13 | waist_roll |
| 14 | waist_pitch |
| 20 | L_wrist_pitch |
| 21 | L_wrist_yaw |
| 27 | R_wrist_pitch |
| 28 | R_wrist_yaw |

유지 인덱스: `list(range(0,13)) + list(range(15,20)) + list(range(22,27))` → 총 23개.

---

## 1. 전체 파이프라인 개요

```
[1] AMASS/OMOMO 인간 모션 데이터 (.npz, SMPL-X)
        │
        │  GMR 오프라인 리타겟팅 (로봇 타겟: unitree_g1_23dof)
        ▼
[2] 23-DoF pkl 파일 dataset
        │
        │  twist2_dataset.yaml에 경로 등록
        ▼
[3] IsaacGym 환경에서 RL 학습
        │  - Stage 1: Teacher (privileged, future frames)
        │  - Stage 2: Student (RL + BC distillation)
        ▼
[4] Student .pt 체크포인트
        │
        │  save_onnx.py로 변환
        ▼
[5] .onnx 정책
        │
        ├─ Sim2Sim 검증 (sim2sim.sh, g1_23dof.xml)
        └─ Sim2Real 배포 (sim2real.sh)
```

---

## 2. Stage 0 — 에셋 및 코드 준비

### 2.1 로봇 에셋 확인
- MuJoCo XML: `assets/g1/g1_23dof.xml` (이미 존재)
- IsaacGym용 URDF/MJCF가 23-DoF 기준으로 존재하는지 `legged_gym/legged_gym/envs/g1/` 내 config에서 확인 필요.
  - 기존 파일: `g1_mimic_config.py`, `g1_mimic_future_config.py`, `g1_mimic_distill_config.py`
  - 29-DoF 전제라면 **23-DoF용 config/env 클래스를 신규 추가**해야 함 (예: `g1_23dof_mimic_future_config.py`).

### 2.2 수정이 필요한 주요 항목
| 항목 | 위치 | 변경점 |
|------|------|--------|
| asset 경로 (urdf/xml) | `g1_mimic*_config.py` | 23-DoF 로봇 asset으로 교체 |
| `num_actions` / `num_dofs` | config | 29 → 23 |
| `default_joint_angles` | config | 23개 관절에 맞춰 재정의 |
| PD gains (Kp/Kd) | config | 23개 관절 gain 테이블 |
| joint 이름 리스트 | config | waist_roll/pitch, wrist_pitch/yaw 제거 |
| 관측/행동 차원 | env | DoF 관련 텐서 shape 재계산 |
| task 등록명 | `envs/__init__.py` | 예: `g1_23dof_stu_future` |

---

## 3. Stage 1 — 모션 데이터셋 준비 (GMR 리타겟팅)

### 3.1 GMR로 AMASS/OMOMO → 23-DoF pkl 변환

`gmr` conda 환경에서 batch 리타겟팅 실행:

```bash
conda activate gmr
# GMR 레포에서:
python scripts/smplx_to_robot_dataset.py \
    --robot unitree_g1_23dof \
    --src_dir <AMASS_ROOT> \
    --save_dir <OUT_DIR_23DOF_PKL>
```

> 만약 GMR이 `unitree_g1_23dof`를 바로 지원하지 않으면:
> 1. GMR 로봇 config에 23-DoF 버전 추가, 또는
> 2. 기존 29-DoF pkl을 로드해 `keep_idx`(섹션 0)로 잘라 23-DoF pkl로 재저장하는 스크립트 작성.

### 3.2 dataset yaml 갱신
`legged_gym/motion_data_configs/twist2_dataset.yaml` 복제 → `twist2_dataset_23dof.yaml` 생성 후 경로를 23-DoF pkl 디렉토리로 지정. Config에서 이 yaml을 참조하도록 수정.

---

## 4. Stage 2 — Teacher 정책 학습 (privileged, future frames)

### 4.1 Env/Config 신규 생성
- `envs/g1/g1_23dof_mimic_future.py` (기존 `g1_mimic_future.py` 복제 후 23-DoF 수정)
- `envs/g1/g1_23dof_mimic_future_config.py`
- `envs/__init__.py`에 task 등록 (예: `g1_23dof_stu_future`)

Teacher 관측의 특권 정보:
- 현재 + 미래 N프레임 참조 모션(base trans/rot, joint_pos)
- 로봇 고유감각(q, q̇, ω, ω̇, action history)

### 4.2 학습 실행

```bash
conda activate twist2
bash train.sh <exptid_teacher> cuda:0
# 내부에서 task_name="g1_stu_future" 사용 중 → 23-DoF용으로 변경 필요
```

`train.sh`의 `task_name`, `proj_name`을 23-DoF용으로 교체:
```bash
task_name="g1_23dof_stu_future"
proj_name="g1_23dof_stu_future"
```

> TWIST2는 TWIST와 달리 teacher/student를 통합된 `train.py`로 돌리는 구조(섹션 5.2 of 분석정리). 필요시 `--teacher_exptid None`을 teacher 단독 모드로 사용.

### 4.3 체크: RTX 4090 한 장 기준 1~2일 소요.

---

## 5. Stage 3 — Student 정책 학습 (RL + BC distillation)

### 5.1 Student config
- 관측: **현재 1프레임 참조 모션**만 + proprioception
- 손실: RL reward + BC(teacher 행동 모방)

### 5.2 학습 실행

```bash
bash train.sh <exptid_student> cuda:0
# train.sh에서 --teacher_exptid <exptid_teacher> 로 지정
```

기존 `train.sh`의 주석 처리된 부분을 활용:
```bash
python train.py --task "g1_23dof_stu_future" \
    --proj_name "g1_23dof_stu_future" \
    --exptid "${exptid}" \
    --device "${device}" \
    --teacher_exptid "<teacher_exptid>"
```

---

## 6. Stage 4 — ONNX 변환

```bash
bash to_onnx.sh <student_ckpt_path>
# 내부: legged_gym/legged_gym/scripts/save_onnx.py
```

출력: `*.onnx` → `assets/ckpts/` 하위에 저장 권장 (예: `twist2_g1_23dof_vXX.onnx`).

---

## 7. Stage 5 — Sim2Sim 검증

### 7.1 모션 서버 실행 (Terminal 1)
이미 23-DoF 지원됨 (`run_motion_server.sh`):
```bash
bash run_motion_server.sh
# 내부: --robot unitree_g1_23dof, --vis, xml=g1_23dof.xml
```

### 7.2 Low-level 정책 실행 (Terminal 2)
`sim2sim.sh`의 `ckpt_path`를 새 23-DoF ONNX로 교체:
```bash
ckpt_path=${SCRIPT_DIR}/assets/ckpts/twist2_g1_23dof_vXX.onnx
python server_low_level_g1_sim.py \
    --xml ../assets/g1/g1_23dof.xml \
    --policy ${ckpt_path} ...
```

추가로 `server_low_level_g1_sim.py` 내부에 29-DoF 하드코딩(action dim, joint idx, default pose 등)이 있다면 23-DoF로 수정 필요.

---

## 8. Stage 6 — VR Teleoperation (Sim) 검증

```bash
# Terminal A (gmr 환경): PICO → GMR → Redis
bash teleop.sh   # --robot unitree_g1_23dof 로 변경
# teleop.sh 내부 xrobot_teleop_to_robot_w_hand.py --robot unitree_g1 → unitree_g1_23dof

# Terminal B (twist2 환경): 23-DoF 정책
bash sim2sim.sh
```

`xrobot_teleop_to_robot_w_hand.py`가 `unitree_g1_23dof`를 지원하는지 확인하고, 지원하지 않으면 GMR 리타겟팅 target과 Redis publish 포맷을 23-DoF에 맞게 수정.

---

## 9. Stage 7 — Sim2Real 배포

1. `sim2real.sh`의 ckpt_path를 23-DoF ONNX로 교체
2. `server_low_level_g1_real.py`에서 29-DoF 가정이 있는 부분을 23-DoF로 수정 (action dim, motor id 매핑, default pose, DEFAULT_MIMIC_OBS 등)
3. `deploy_real/data_utils/params.py`의 `DEFAULT_MIMIC_OBS["unitree_g1_23dof"]` 존재 확인 (없으면 추가)
4. 실제 G1 23-DoF 로봇에 배포

---

## 10. 체크리스트 요약

- [ ] GMR이 `unitree_g1_23dof` 리타겟팅을 지원하도록 설정/패치
- [ ] 23-DoF pkl dataset 생성 및 yaml 등록
- [ ] `legged_gym` 에 23-DoF env/config 추가, task 등록
- [ ] `train.sh` task_name 교체 후 Teacher 학습
- [ ] Student 학습 (RL+BC, teacher_exptid 지정)
- [ ] `to_onnx.sh`로 ONNX 변환
- [ ] `server_low_level_g1_sim.py` / `server_low_level_g1_real.py` 23-DoF 호환성 점검
- [ ] `run_motion_server.sh`로 Sim2Sim 검증
- [ ] `teleop.sh` + `sim2sim.sh`로 VR 텔레오퍼레이션 검증
- [ ] `sim2real.sh`로 실기 배포

---

## 11. 예상 리스크 및 대응

| 리스크 | 대응 |
|--------|------|
| GMR에 23-DoF 로봇 정의 없음 | 29-DoF pkl을 keep_idx로 잘라 23-DoF pkl로 후처리 |
| URDF/MJCF 불일치로 IsaacGym 로드 실패 | `g1_23dof.xml` 기반 URDF 확보 또는 변환 |
| 허리 roll/pitch 부재로 상체 기울기 추적 저하 | 보상에서 허리 자세 추적 가중치 조정, 어깨로 대체 표현 |
| 손목 자유도 감소로 손 포즈 오차 증가 | end-effector 위치 보상 비중 증가, 핸드 정책은 별도 처리 |
| DEFAULT_MIMIC_OBS shape 불일치 | `deploy_real/data_utils/params.py`에 23-DoF 엔트리 추가 |

---

## 12. 참고 파일 경로

- 모션 서버: `deploy_real/server_motion_lib.py`
- Low-level sim: `deploy_real/server_low_level_g1_sim.py`
- Low-level real: `deploy_real/server_low_level_g1_real.py`
- Teleop: `deploy_real/xrobot_teleop_to_robot_w_hand.py`
- 학습 스크립트: `legged_gym/legged_gym/scripts/train.py`
- ONNX 변환: `legged_gym/legged_gym/scripts/save_onnx.py`
- 기존 env (29-DoF): `legged_gym/legged_gym/envs/g1/g1_mimic_future*.py`
- 모션 dataset config: `legged_gym/motion_data_configs/twist2_dataset.yaml`
- 23-DoF XML: `assets/g1/g1_23dof.xml`
