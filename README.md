# TYRO — 듀얼암 타이어 마운팅 학습 환경

UR10 (Robot A) + Franka Panda (Robot B) 두 팔이 트럭 휠 허브에 타이어를 끼우는
PyBullet 시뮬레이션 환경과 PPO 학습 파이프라인입니다. 현재 **Phase 1** —
UR10이 타이어를 픽업해 허브에 마운트하고 다시 거치대로 복귀시키는 단일 로봇
시퀀스를 학습합니다 (Panda는 HOME에 동결).

- 시뮬레이터: PyBullet, 240 Hz physics / 20 Hz control
- 좌표계: **Robot B (Panda) base 중심** — 모든 절대 좌표가 Panda 베이스 기준
- 관측 / 액션 (Phase 1): **obs 85-d / action 6-d** (PPO **잔차(residual)** 6, gripper는 FSM이 자동 제어)
  - obs 채널: 로봇 joints(qA·dqA, qB·dqB) + EE/타이어/허브/볼트 상대 pose + prev_action + mount tail 3 + hub_guide 3
  - legacy 체크포인트 호환: `(obs 82, act 6)` v6 이전, `(obs 83, act 7)` v5 sharp 시대 — `eval.py`가 자동 감지
- 제어 (2026-06-01 기본): **Min-Jerk 플래너 + PPO 잔차** — FSM 단계마다 명목 EE 궤적을 생성하고, 정책은 ±0.15 m 잔차만 학습 (`use_planner_residual=True`)
- 알고리즘: PPO + MlpPolicy `[256, 256]`, vec env 12개 (SubprocVecEnv)
- **기본 학습 목표 (2026-06-01)**: `terminate_on=mount` + attached hot-start → Stage 1(운반/마운트) 집중; 전체 4-stage는 `--terminate-on never`로 전환

> 모든 명령은 `tyro` conda 환경 활성화 상태에서 실행한다고 가정합니다
> (`conda activate tyro`).

---

## 설치

```bash
git clone https://github.com/MinziiK/TYRO.git
cd TYRO

conda create -n tyro python=3.10 -y
conda activate tyro

pip install -r requirements.txt
```

---

## 빠른 시작

### 학습

```powershell
# PowerShell (권장)
.\scripts\train.ps1

# cmd.exe
scripts\train.bat
```

기본 설정 = `--stage 3 --phase 1 --num-envs 12 --total-steps 2_000_000`,
run-name은 `phase1_<yyyymmdd-HHmm>` 자동 타임스탬프.
스크립트 뒤에 붙이는 모든 인자는 `python -m src.train`에 그대로 전달됩니다:

```powershell
.\scripts\train.ps1 --total-steps 500000           # 짧게 시험
.\scripts\train.ps1 --num-envs 8 --run-name test   # envs / 이름 override
.\scripts\train.ps1 --resume runs\phase1_xxx\final.zip  # 이어 학습
```

### 시각 검증

```powershell
# 정적 씬 GUI 확인
python -m src.test --render --action-scale 0

# FSM 5-step + random 5-ep 자동 스모크
python scripts\smoke_fsm.py

# Min-Jerk 플래너 + attached hot-start + mount 종료 검증 (권장)
python scripts\smoke_planner_residual.py

# 픽업 / 리프트 / 마운트 3단계 정답지 이미지 생성
python scripts\render_phase1_goal.py --out runs\phase1_goal.png
```

### 평가 (체크포인트 로드)

```powershell
python -m src.eval runs\phase1_xxx\final.zip --render --episodes 5
python -m src.eval runs\phase1_xxx\best\best_model.zip --episodes 20  # 헤드리스
```

### TensorBoard

```powershell
tensorboard --logdir runs/
```

---

## 프로젝트 구조

```
TYRO/
├── src/
│   ├── config.py           # 모든 환경 / 보상 / 액션 / 커리큘럼 파라미터 (단일 진실 원천)
│   ├── train.py            # PPO 학습 진입점 (커리큘럼 콜백 포함)
│   ├── eval.py             # 체크포인트 로드 + N 에피소드 평가
│   ├── sweep.py            # Optuna 하이퍼파라미터 sweep
│   ├── test.py             # GUI / 정적 스모크
│   └── env/
│       ├── tyro_env.py     # Gymnasium env (reset / step / reward / FSM)
│       ├── scene.py        # 트럭 휠 스테이션 + 타이어 거치대
│       ├── models.py       # 메시 없는 타이어 + 휠 디스크 빌더
│       ├── robots.py       # UR10 / Panda 래퍼 (IK, EE pose, joint state)
│       ├── rewards.py      # 보상 항 + aggregate
│       └── utils.py        # 쿼터니언 / 축 헬퍼
├── scripts/
│   ├── train.ps1 / train.bat              # 학습 런처
│   ├── smoke_fsm.py                       # FSM 스모크
│   ├── smoke_planner_residual.py          # 플래너+잔차+attached hot-start 스모크
│   ├── render_phase1_goal.py              # 3단계 정답지 렌더
│   ├── verify_home.py                     # UR10 HOME / AABB / grasp 거리 검증
│   ├── summarize_log.py                   # 학습 로그 요약 테이블
│   └── generate_truck_wheel_station_urdf.py
├── data/urdf/
│   ├── truck_assembly/truck_wheel_station.urdf   # 허브 + 10 볼트 + 브레이크
│   └── ur10_robot/...                            # UR10 + Robotiq 그리퍼
└── runs/                                  # 학습 산출물 (체크포인트 / tb / monitor)
```

---

## Phase 1 — FSM 4단계 (v6+)

`task_stage` 0 → 1 → 2 → 3 → done 순으로 진행. 각 전이 시 `info`에
`picked_up` / `mounted` / `demounted` / `landed` 이벤트가 떨어지고, 해당 단계
일회성 sparse 보너스가 더해집니다 (`src/env/tyro_env.py:_try_stage_transitions`).

| Stage | 이름 | 핵심 dense 보상 | 전이 게이트 | Sparse 보너스 |
|---|---|---|---|---|
| **0** | 접근 / 픽업 | `3.0·exp(-d/0.15)` + `2.0·exp(-d/0.2)` + PB `5·Δd` | `‖EE − grasp_target‖ < approach_radius_tol` (기본 0.08 m) | **R_pickup = +300** |
| **1** | 운반 / 마운트 | `guide_A = 8·exp(-d_A/0.5)` + PB carry `10·Δd_A` − `sync_joint_A` 0.02 | `‖tire − hub‖ < mount_radius_tol` (기본 0.04 m) AND `θ_tire↔hub < δ_A` | **R_mount = +300** |
| **2** | 디마운트 (백트래킹) | PB demount `20·Δd_hub` + `3·(1 − exp(-d_hub/0.20))` (반전 kernel) | `d_hub > demount_axial_distance` (기본 0.30 m), `demount_stall_steps ≥ 20` 이후 | **R_demount = +150** |
| **3** | 거치대 복귀 / 안착 | `3.0·exp(-d_rack/0.5)` + PB return `30·Δd_return` − `landing_speed · 0.5·|v_z|` | `‖tire − pickup‖ < 0.05 m` AND `|v_z| < 0.10 m/s` | **R_return = +300** (성공) |

- 항상 작동 패널티: 충돌 / 작업공간 이탈 / action L2 / jerk / contact-force
- Stage 1에서는 vertical-pose 패널티 면제 (보어 정합을 위한 자유 회전 허용)
- Stage 2 디마운트는 마운트 후 `demount_stall_steps` 스톨 동안 `d_hub` 게이트가
  비활성화 — 가상 fastener-release 인터벌
- 실패 종료 (충돌·workspace·contact-force) 시 `R_fail = −50` 일회성

### terminate_on — 조기 종료 게이트

전체 사이클 학습 외에 단계별 종료를 지원합니다 (`cfg.terminate_on`,
`eval.py --terminate-on`).

| 값 | 종료 시점 | 용도 |
|---|---|---|
| `never` | Stage 3 landed 또는 `max_steps` | 전체 4-stage 사이클 학습 |
| `pickup` | Stage 0 → 1 전이 (R_pickup 지급 후) | Stage 0 단독 학습 / 디버깅 |
| `mount` (**2026-06-01 기본**) | Stage 1 → 2 전이 | Mount-only / 플래너+잔차 초기 학습 |
| `demount` | Stage 2 → 3 전이 | 디마운트 분리 학습 |

### 픽업 게이트 타임스텝 커리큘럼 (opt-in)

`approach_radius_tol`을 학습 진행에 따라 점진적으로 조여 R_pickup 신호를
초반부터 받을 수 있게 합니다 (`src/train.py:ApproachTolCurriculumCallback`).

**기본 OFF** — mount-only + attached hot-start에서는 Stage 0을 건너뛰므로
불필요합니다. full cycle 또는 pickup 단독 학습 시 `--approach-curriculum`으로
활성화:

- **t ≤ 100k**: `approach_tol_soft = 0.10 m` (soft 홀드)
- **100k < t ≤ 400k**: smoothstep 감쇠 (`0.10 → 0.08`, 램프 폭 300k)
- **t > 400k**: `approach_radius_tol = 0.08 m` (hard cap)

매 rollout 종료 시 `env_method("set_approach_tol", v)`로 모든 sub-env에
전파. TensorBoard에 `curriculum/approach_tol`, `curriculum/approach_tol_frac` 기록.

### 마운트 게이트 커리큘럼 (`MountTolCurriculumCallback`)

`mount_radius_tol`(반경)과 `mount_angle_tol`(각도)을 단계적으로 조여
Stage 1 → 2 전이를 학습 초반에는 느슨하게, 후반에는 정밀하게:

- soft: `(0.30 m, mount_angle_tol_soft_rad)` → hard: `(0.04 m, reward.delta_A)`
- 램프 폭은 `mount_tol_ramp_steps`로 제어

### 시작 자세 커리큘럼 (`start_pos`)

매 reset마다 UR10 EE를 HOME ↔ grasp_anchor 사이에서 보간 (mix 모드는
Bernoulli `start_pos_easy_prob = 1.0` — attached hot-start와 함께 항상 easy).
lerp 모드일 때만 `StartPosCurriculumCallback`이 등록됩니다 (100k 홀드 → 300k 램프).

---

## 보상 stage 토글 (× DR phase)

위의 4-stage FSM (task_stage 0~3)과 별개로, `make_reward_config(stage)`는
**보상 항 자체의 활성화 여부**를 단계적으로 제어합니다 (`src/config.py`):

| Stage | 활성화되는 보상 항 |
|---|---|
| 1 | `align_A`, `reach_B` (per-robot task 만) |
| 2 | + `coop`, `sync_joint_A` (협력 항 추가) |
| 3 | + `success` + 충돌 / workspace / action / jerk 패널티 (전체 dense — **현재 학습 기본**) |
| 4 | potential shaping + 약한 거리 baseline (mix 20/80) |

| Phase | 도메인 랜덤화 |
|---|---|
| 1 | 정적 — 고정 좌표 (Panda 동결, 현재 학습 기본) |
| 2 | ±2 cm 허브 XY 노이즈 |
| 3 | ±5 cm |

추가로 **정적 스폰 DR** 플래그 (`USE_DOMAIN_RANDOMIZATION`)가 별도로 있어
`reset()`에서 hub / cargo XY에 균등 노이즈 (`RANDOM_POSITION_RANGE`, 기본 2 cm)를
주입할 수 있습니다. **기본값 False** — Phase 1 안정화 후 True로 전환 권장.

Phase 1에서는 `freeze_robot_b=True`로 인해 자동으로:
- action dim 13 → 6 (UR10 Δpose 6, Panda 채널 제거), obs dim 92 → 85
- `_compute_obs`에서 Panda 관련 obs 채널 (qB, eeB, bolt, ΔeeB-bolt) 0-마스킹
- `action_penalty` / `jerk_penalty`에서 Panda 채널 마스킹

---

## 씬 자산 (참고)

- **UR10**: `data/urdf/ur10_robot/ur10_robot.urdf` (검색 경로 = URDF 디렉터리)
- **트럭 휠 스테이션** (허브 ø420 mm, ISO 335 mm PCD, 10볼트):

  ```bash
  python scripts/generate_truck_wheel_station_urdf.py \
      --bolt-collision-radius-factor 0.95 \
      --bolt-pattern-phase-deg 0 \
      --output data/urdf/truck_assembly/truck_wheel_station.urdf
  ```

- **타이어 + 휠 디스크**: 메시 없이 primitives만 사용
  (`src/env/models.py:create_tire_wheel_multibody`). 트레드는 박스 토러스,
  휠 디스크는 PCD/위상이 URDF 생성기와 정합되도록 구성.
- **카고 (휠 웰)**: AABB 셀 분할 + 허브 동축 원통 cutout으로 타이어 진입 통로
  확보 (`cargo_use_wheel_well_cutout`, `cargo_wheel_well_radius_yz`).

---

## 좌표 요약 (Robot B-centric)

| 객체 | 위치 |
|---|---|
| Robot B (Panda) base | `(0, 0, 0)` ← 원점 |
| Robot A (UR10) base | `(−0.80, 0, −0.30)` — 30 cm 플린스 |
| 허브 중심 | `(0, +0.80, +0.22)` |
| 타이어 픽업 COM | `(−1.90, 0, +0.225)` |
| 타이어 거치대 (Y-split) | rail top Z = `−0.30` (UR10 base 평면과 동일) |
| 카고 박스 중심 | `(0, +1.05, +0.78)` |
| 바닥 평면 Z | `−0.60` |

---

## Min-Jerk 플래너 + PPO 잔차 제어 (2026-06-01)

이전(v11c)에는 PPO가 **원시 Δ-EE-pose 6-d**로 전체 궤적을 직접 합성했습니다. 마운트 구간(≈1.9 m 이동 + 90° 타이어 회전)에서 탐색 공간이 너무 커서 `d_A≈2 m` 정체·hover-lockin·collision-suicide가 반복되었습니다.

**변경 내용** (`src/env/tyro_env.py`, `src/config.py`, `src/env/robots.py`):

| 구성요소 | 동작 |
|---|---|
| `_generate_nominal_trajectory` | FSM reset/전이마다 시작 EE → 단계 종료 pose로 **5차 Min-Jerk** (위치) + **SLERP** (자세), 기본 100 step |
| `_apply_action` | `final_pos = nominal[idx] + action[0:3] × planner_pos_offset_scale` (기본 0.15 m); 회전 잔차는 기본 **비활성** (`planner_enable_rot_offset=False`) |
| `UR10Robot.apply_absolute_ee` | 플래너 모드 전용 6-DOF IK; `ur10_lock_tool_up` 우회 (플래너가 자세 담당) |
| `use_planner_residual=False` | v11c 이전 체크포인트·레거시 eval용 **raw Δ-EE** 경로 유지 |

**왜 잔차만 학습하는가**: 명목 궤적이 Phase 1 고정 씬에서 충돌 없는 coarse 경로를 제공하므로, 정책은 랙/카고 회피·정밀 정합만 보정하면 됩니다. PPO가 “어디로 갈지”가 아니라 “명목에서 얼마나 벗어날지”를 학습합니다.

---

## Attached hot-start (Mount-only 빠른 루프)

| 플래그 | 기본 (2026-06-01) | 효과 |
|---|---|---|
| `start_pos_easy_prob` | `1.0` | 매 reset이 easy 분기 |
| `attached_spawn_when_easy` | `True` | 타이어 cradle 고정 + grasp 연결 + `task_stage=1`, Stage 0 스킵 |
| `terminate_on` | `mount` | Stage 1→2 시 `R_mount` + `is_success=True`로 조기 종료 |

Stage 2(디마운트)·Stage 3(복귀) 코드는 env에 그대로 있으나, 위 기본값에서는 **학습 루프에서 호출되지 않음**. 마운트 수렴 후 `--terminate-on never`로 전체 사이클을 켭니다.

```powershell
# 권장: 플래너+잔차 Mount-only (EnvConfig 기본과 동일)
.\scripts\train.ps1 --run-name phase1_planner_mount_v1

# 전체 4-stage 사이클 (pickup부터)
.\scripts\train.ps1 --terminate-on never --start-pos-easy-prob 0.75 `
    --no-attached-spawn-when-easy

# v11c 역커리큘럼 (허브 hot-start → easy mix → HOME)
.\scripts\train.ps1 --reverse-curriculum --terminate-on never --run-name phase1_rev_v1
```

> attached hot-start 끄기: `--no-attached-spawn-when-easy`  
> 레거시 raw Δ-EE: `--no-use-planner-residual`

---

## v11 역커리큘럼 (Backtracking) — 선택 기능

`--reverse-curriculum` 시 `ReverseCurriculumCallback`이 global step에 따라 reset 경로를 바꿉니다 (`reverse_curriculum_enable=False`가 기본).

| Phase | step 구간 (기본) | reset 동작 |
|---|---|---|
| **A** | 0 ~ 250k | 허브 정렬 hot-start, `task_stage=1`, 소 jitter (0.01 m / 0.5°) |
| **A↔B** | 250k ~ 300k | Bernoulli 75% A / 25% B |
| **B** | 300k ~ 750k | legacy easy/HOME mix (`start_pos_easy_prob`) |
| **C** | 750k+ | pure HOME |

**v11c 패치가 필요했던 이유** (실험 로그 기반):

- **v11c1**: Phase A에서 `terminate_on=mount` + 1-step ep → PyBullet `reset()` 폭주, fps ≈ 12 → `reverse_phase_a_terminate_on_mount=False` (R_mount는 step 1에 지급, ep는 계속)
- **v11c2**: Phase A 첫 step 물리 폭주로 safety 종료 → `safety_terminations_enabled` 토글
- **v11c4/v11c5**: Mount tol이 hard(0.04 m)로 조여지면 `fsm_bonus` 붕괴 → Phase A/B에서 soft tol(0.30 m) **lock**

---

## UR10 / 씬 튜닝 요약 (최근)

| 항목 | 변경 | 이유 |
|---|---|---|
| V-cradle rack | Y-gap 70 cm, rail 높이 60 cm, COM Z ≈ 0.391 | 6시 grasp anchor가 HOME EE Z와 ≈3 mm — pickup을 거의 평면 도달로 |
| `tire_spawn_rpy` | bore axis → world **+X** (Robot A 방향) | bore가 로봇을 향해 보이도록; Stage 1에서 hub(−Y) 정렬 시 90° 회전 허용 |
| `HOME_POSE` | palm-up cradle | 이전 HOME이 타이어 bore 내부(접촉력 즉시 종료) |
| `apply_delta_ee` | `last_target_pos` 누적 (측정 EE 비누적) | 중력 sag가 IK 목표를 매 step 끌어내려 drift |
| joint `forces` | 400/400/300/60/60/60 N·m | uniform 150 N·m → shoulder sag |
| `collision_terminates` | True (플래너 모드) | 명목 궤적이 충돌-free이면 접촉 = 정책 오류 |
| grasp 중 contact-force | tire↔UR10 제외 | JOINT_FIXED 의도 접촉이 step-1 kill 방지 |

---

## 추가 학습 옵션

```powershell
# W&B 로깅
.\scripts\train.ps1 --wandb tyro --tags v1

# 솔버 / 물리 미세조정
.\scripts\train.ps1 --physics-num-sub-steps 6 --contact-force-done 2500

# Stage / Phase 변경
.\scripts\train.ps1 --stage 4 --phase 2

# Mount-only → full cycle 전환 (체크포인트 이어하기)
.\scripts\train.ps1 --resume runs\phase1_xxx\final.zip --terminate-on never `
    --reset-timesteps --run-name phase1_full_cycle_v1

# 커리큘럼 노브 (override)
.\scripts\train.ps1 `
    --approach-tol-soft 0.60 `
    --approach-tol-hard 0.45 `
    --approach-tol-curriculum-steps 150000 `
    --approach-tol-ramp-steps 250000

# eval: 플래너 기본 env + easy spawn 강제
python -m src.eval runs\phase1_xxx\best\best_model.zip --render --easy-start --terminate-on mount
```

---

## 하이퍼파라미터 sweep

Optuna 기반 PPO + 협력 보상 가중치 sweep (TPE + median pruning, SQLite 저장).

```bash
python -m src.sweep --study smoke --smoke-test         # 1 trial × 5k step 스모크
python -m src.sweep --study tyro1 --n-trials 30 --steps-per-trial 200000
python -m src.sweep --study tyro1 --n-trials 20         # 이어서 진행
```

탐색 공간은 `src/sweep.py:suggest_hparams`에서 직접 편집 가능.

---

## 학습 파이프라인 (mount-only 기본)

### 기본 활성 콜백

| 항목 | 이유 |
|---|---|
| `MountTolCurriculumCallback` (기본 ON) | mount tol soft→hard; 역커리큘럼 Phase A/B lock과 공존 |
| `RewardBreakdownCallback` | TensorBoard `reward/*`, `fsm_bonus` 디버깅 |
| `mix_dense=0.3` / `mix_sparse=0.7` | sparse FSM 보너스가 value를 주도 (hover 방지와 병행) |
| `w_step_alive=0.15` | mix를 우회하는 per-step 생존 비용 — 장기 hover 억제 |
| Stage 2/3 env 코드 (`terminate_on=mount` 시 미실행) | full cycle 전환 시 `--terminate-on never`만 변경 |

### opt-in / legacy (mount-only에서 비활성)

| 항목 | 활성화 방법 |
|---|---|
| `ApproachTolCurriculumCallback` | `--approach-curriculum` (pickup / full cycle 재학습 시) |
| `StartPosCurriculumCallback` | `--start-pos-curriculum --start-pos-mode lerp` |
| `ReverseCurriculumCallback` | `--reverse-curriculum` |
| `terminate_on_pickup` (legacy) | `terminate_on`이 우선 — `--terminate-on`만 사용 권장 |

### TensorBoard / monitor에서 볼 지표

- `rollout/success_rate` — mount-only면 초반 급상승 목표
- `reward/fsm_bonus` — sparse mount/pickup 신호 (≈0이면 tol/gate 문제)
- `reward/guide_A`, `reward/pb_carry` — Stage 1 dense
- `curriculum/mount_radius_tol` — tol ramp 진행
- `env/ik_residual_A_mean` — 장기 >0.02 m이면 reach/DR 점검

### 정리 완료 (2026-06-01)

- **`runs/`** — legacy 실험 run·로그·체크포인트 전부 삭제 (`.gitignore` 유지, git 미추적)
- **`scripts/`** — 유지: `train.ps1/bat`, `smoke_fsm.py`, `smoke_planner_residual.py`,
  `replay_planner.py`, `diag_v4_eval.py`, `render_phase1_goal.py`, `verify_home.py`,
  `summarize_log.py`, `generate_truck_wheel_station_urdf.py`
- 제거됨: `check_*`, `dump_scene.py`, `calibrate_home_pose.py`, 버전별 `smoke_v5_*`~`smoke_v11_*`,
  v11c supervisor `.ps1` (일회성 디버그)

---

## 시뮬 안정화 · 물리 장착 · 경로 시각화 (2026-06-03)

Phase 1 **학습 설계는 EE 명목 궤적 + PPO 잔차** (`_traj_pos` + `action[0:3]`)입니다.  
아래 패치는 **GUI 떨림**, **허브 물리 결합**, **계획 vs 실제 경로 불일치**를 다룹니다.

### GUI 떨림 — 원인과 대응

| 원인 | 설명 |
|---|---|
| 매 스텝 HOME warm-start IK | `apply_palm_up_pose`가 `arm.rest`로 IK → 관절해가 스텝마다 튐 |
| 키네마틱 upright lock | `_sync_grasped_tire_upright`가 EE 떨림을 타이어에 매 스텝 텔레포트 |
| 관절 bake vs PD | bake `_traj_q`는 부드러운데 PD가 못 따라가면 실제 EE가 계획선과 어긋남 |

| 설정 / 코드 | 기본 (학습) | replay / 시연 |
|---|---|---|
| `planner_precompute_joint_traj` | `True` | 동일 — **action≈0일 때만** 관절 재생; PPO는 `apply_palm_up_pose(EE)` |
| `apply_palm_up_pose` warm-start | **현재 관절** (`robots.py`) | HOME 아님 |
| `kinematic_tire_sync_alpha` | `0.65` | EMA로 타이어 pose 부드럽게 (1.0=즉시 스냅) |
| `ur10_joint_*_smooth_*` / PD gains | 꺼짐 / 1.0 | `replay_planner.py`에서만 조정 가능 |

**권장**: 학습 일관성을 위해 장기적으로 `planner_precompute_joint_traj=False`로 **매 스텝 EE→IK**만 쓰는 것도 검토.  
replay는 `scripts/replay_planner.py` (플래너만, action=0).

### 허브 물리 장착 + 홀드 (Robot B 볼트 체결 전)

| 플래그 | 기본 | 효과 |
|---|---|---|
| `pin_tire_on_mount` | `True` | 마운트 시 타이어 ↔ **허브 URDF** `JOINT_FIXED` (world-pin 대신) |
| `mount_hold_steps` | `0` (학습) | replay 스크립트는 `40` (~2 s) — UR10 관절 동결 + 타이어 seated pose 클램프 |

- Stage 2(빼기)·3(복귀)는 **`terminate_on=mount`** 기본 때문에 **학습/기본 replay에서 실행 안 됨**.
- 전체 사이클: `--terminate-on never` (train.ps1 / eval).

### 플래너 경로 GUI (`scripts/replay_planner.py`)

```powershell
conda activate tyro
python scripts/replay_planner.py --render --easy-start
# 전체 4-stage: config override로 terminate_on=never 필요 시 train.ps1 참고
```

| 선 | 의미 |
|---|---|
| 파랑 / 주황 / 자홍 / 초록 | stage 0~3 **명목 EE** 궤적 (`compute_all_stage_trajectories`) |
| 노란 | **실제 EE**가 지나간 자취 (매 step FK) |

색깔 선(이상적 EE)과 노란 선(관절+IK+PD+키네마틱 실측)이 벌어지는 것은 위 표의 구조적 이유 때문입니다.

### 학습 run 참고 (phase1_grad_v7)

- **Best ckpt**: `runs/phase1_grad_v7/best/best_model.zip` — det. EASY 5/5 full cycle (S1 설정).
- `max_steps=600`, `w_step_alive=0.15`, D1/D4 planner 패치 포함.
- 2M step까지 학습 시 easy_prob 하락으로 **regression** 가능 → 배포는 **best** 사용 권장.

```powershell
python -m src.eval runs/phase1_grad_v7/best/best_model.zip --render --phase 1 --easy-start
```

---

## 변경 이력 (CHANGELOG) — 왜 바꿨는지

기존 README·코드 주석에 흩어져 있던 실험 근거를 한곳에 모았습니다. **날짜 = 커밋/실험 기준**, 세부 수치는 `src/config.py` 주석이 단일 진실 원천입니다.

### 2026-06-04 — "오락가락" 근본 원인 규명 + 삽입 속도 제한 (측정 기반)

**근본 원인 = 기하 특이점.** carry를 4분면으로 분해하면 점프가 어디 있는지 명확:

| 구간 (step) | max jump | 큰 점프(>15cm) |
|---|---|---|
| 0–51 (carry 전반) | ~5 cm | 0 — **이미 부드러움** |
| 74–99 (허브 삽입) | **70 cm** | 11 — 여기서만 터짐 |

- baked 관절 명령은 그 구간에서도 \|Δq\|≈0.12 rad로 **매끄러움** → IK/계획 문제 아님. 허브가 UR10 reach의 **88–96%** 라 stiff PD가 특이점에서 팔을 **채찍질**하는 것.
- **DLS Cartesian 서보** 구현(`UR10Robot.drive_ee_servo_dls`, `use_dls_cartesian_servo`): 동작은 매우 부드럽게(mean 1.05 cm, 큰 점프 0) 만들지만 **같은 특이점 때문에 타이어를 4 cm 게이트에 못 꽂음** → **기본 OFF** (opt-in 레버로 보존).
- **베이스/허브 재배치 검토(공격적 레이아웃, reach 60%)**: zero-action carry가 **70→8 cm** 로 완전 평활화됨을 입증(근본 원인 재확인). 그러나 허브 이동이 **cargo 박스를 carry 통로로 끌어들이고**(가드 동결), −Y 삽입 standoff를 넣자 **팔 링크가 트럭 본체와 충돌(step 80 폭발)**. mount/삽입 서브시스템 전체가 원래 기하에 정밀 설계돼 있어 다층 재튜닝 필요 → **레이아웃 원복**. (`planner_stage1_approach_standoff` 다구간 경로 코드는 향후 재설계용으로 보존, 기본 0.)
- **채택한 해법 = `ur10_motor_max_velocity_rad_s = 1.0`** (PyBullet POSITION_CONTROL maxVelocity). 특이점에서의 PD 채찍질만 잘라냄:
  - 최악 삽입 점프 **70 → 25 cm**, mean **6.3 → 3.2 cm**, 큰 점프 **11 → 2**.
  - 커리큘럼 시작 게이트(0.12 m)에서 **mount@112**, hard 게이트(0.04 m)에선 타이어 허브 5.8 cm 까지(마지막 ~2 cm는 정책 residual이 마무리).
  - 1.5는 미시팅, 2.0은 mount되나 노이즈(max 38 cm) → **1.0이 최적**. 0이면 legacy(mount@104, 70 cm snap).
- **학습**: maxVel이 env dynamics를 바꾸므로 pre-2026-06-04 ckpt는 OOD → `scripts/train_v8_smooth.ps1` 신규 학습.
- **정직한 한계**: 88% reach 특이점 자체는 남아 있음(완전 평활은 씬 재설계 필요). maxVel은 그 안에서 **위험 없이 3배 개선**하는 실용 해법.
- **yaw 진동**: EE yaw가 carry 중 24회 역회전(step 20–70 정체) 후 특이점에서 wrap. 허브 근처 회전은 cargo 충돌, 거치대 선회전은 도달 불가 → **spawn bore −Y** 또는 **특이점 제거**만이 근본 해법.
- **대형 로봇 / 재배치**: reach 2.6 m(FANUC R-2000iC) 또는 약한 재배치(hub Y 0.65)로 특이점 제거 가능. R-2000iD URDF는 미지원 → iC/210F 대체. **100 kg**은 payload OK, PyBullet PD·contact 재튜닝 필요.
- **전체 조사 기록**: [`docs/INVESTIGATION_2026-06-04.md`](docs/INVESTIGATION_2026-06-04.md)
- **FANUC PoC (1단계)**: `python scripts/poc_fanuc_urdf.py --fetch` → `python scripts/poc_fanuc_urdf.py --load [--gui]`

### 2026-06-03 — 시뮬 떨림 · 허브 물리 결합 · 경로 시각화

- **문제**: 플래너-only replay도 떨림; 마운트 후 타이어가 EE에 붙어 허브에 안착한 것처럼 보이지 않음; GUI 계획선 vs 실제 경로 괴리.
- **조치**: palm-up IK current warm-start; bake joint traj + tilt-lock at replan; kinematic tire EMA; hub `JOINT_FIXED` + mount hold; `replay_planner.py` (다색 명목 경로 + 노란 실측); `compute_all_stage_trajectories()`.
- **학습**: 정책 잔차는 여전히 EE; `mount_hold_steps=0`, `terminate_on=mount` 유지.

### 2026-06-01 — Min-Jerk 플래너 + PPO 잔차 + Mount-only 기본

- **문제**: raw Δ-EE만으로는 carry/mount 탐색이 수렴하지 않음; v11c5까지도 `d_A` 장거리 정체.
- **조치**: 명목 궤적 + 0.15 m 잔차; `attached_spawn_when_easy`로 Stage 1부터 시작; `terminate_on=mount`.
- **부수**: `apply_absolute_ee`, grasp 중 contact-force 필터, `collision_terminates=True` 복원.

### 2026-05-31 — v11 역커리큘럼 + 4-stage FSM 완성

- **문제**: pickup→mount→demount→return 전체를 한 번에 학습하면 credit assignment 붕괴.
- **조치**: Phase A 허브 hot-start로 mount endgame 먼저 맛보기 → B easy mix → C HOME.
- **v11c1~c5**: 1-step ep fps 붕괴, safety kill, mount tol hard lock 등을 패치 (README §v11 역커리큘럼).

### 2026-05-30 — v6/v7 4-stage + vector-guided carry

- Stage 2 demount, Stage 3 return 추가; Stage 1 dense를 `guide_A` + `pb_carry` 양의 커널로 교체 (음의 `align_A`만으로 collision-suicide 유발하던 문제).
- `hub_guide_vector` obs 3-d 추가 (v7).
- `terminate_on` enum으로 단계별 조기 성공.

### 2026-05-29 — hover-lockin / 1-step pickup 버그

- **hover**: `approach_decay` 0.15 m, `R_pickup` 300, `w_step_alive` 도입 — dense만 먹고 pickup 안 하는 Nash 균형 제거.
- **1-step pickup**: `approach_tol_soft` 0.35→0.10 m — easy spawn에서 gate가 너무 넓어 step 1 종료·PPO gradient 소실.

### 2026-05-28 — Robot B-centric + V-cradle + 6-d action

- Panda base를 world origin에 두어 obs/좌표 단순화.
- gripper_A action 제거 (FSM auto-grasp); Phase 1 action 7→6.
- UR10 HOME palm-up, tire bore +X spawn.

### 그 이전

- Phase 1 single-arm (`freeze_robot_b`), hub URDF, curriculum stage 1→3 — `git log` 및 초기 README 설치 섹션 참고.

---

## 실험 run 네이밍

신규 학습 권장 이름: `phase1_planner_mount_v1` (또는 `train.ps1` 기본 타임스탬프).

산출물은 `runs/<run-name>/` (`final.zip`, `best/`, `ckpts/`, `monitor.csv`, `tb/`).
`.gitignore`에 포함되어 git에는 올리지 않습니다.

| 접두 (legacy 참고) | 의미 |
|---|---|
| `phase1_fsm_v4/v5` | 3-stage → 4-stage FSM 전환 |
| `phase1_fsm_v6_*` | mount curriculum / vector-guided |
| `phase1_fsm_v11*` | reverse curriculum 실험 |
| `phase1_planner_mount_*` | Min-Jerk 플래너 + 잔차 (2026-06-01~) |
