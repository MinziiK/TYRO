# TYRO — 듀얼암 타이어 마운팅 학습 환경

UR10 (Robot A) + Franka Panda (Robot B) 두 팔이 트럭 휠 허브에 타이어를 끼우는
PyBullet 시뮬레이션 환경과 PPO 학습 파이프라인입니다. 현재 **Phase 1** —
UR10이 타이어를 픽업해 허브에 마운트하고 다시 거치대로 복귀시키는 단일 로봇
시퀀스를 학습합니다 (Panda는 HOME에 동결).

- 시뮬레이터: PyBullet, 240 Hz physics / 20 Hz control
- 좌표계: **Robot B (Panda) base 중심** — 모든 절대 좌표가 Panda 베이스 기준
- 관측 / 액션 (Phase 1): **obs 83-d / action 7-d** (UR10 Δpose 6 + gripper_A 1)
- 알고리즘: PPO + MlpPolicy `[256, 256]`, vec env 12개 (SubprocVecEnv)

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
│   ├── render_phase1_goal.py              # 3단계 정답지 렌더
│   └── generate_truck_wheel_station_urdf.py
├── data/urdf/
│   ├── truck_assembly/truck_wheel_station.urdf   # 허브 + 10 볼트 + 브레이크
│   └── ur10_robot/...                            # UR10 + Robotiq 그리퍼
└── runs/                                  # 학습 산출물 (체크포인트 / tb / monitor)
```

---

## Phase 1 — FSM 3단계

| 단계 | 이름 | 핵심 보상 | 전이 조건 | 보너스 |
|---|---|---|---|---|
| **Stage 0** | 접근 / 픽업 | `w_approach · exp(-d / 1.5)` + PB shaping | `‖EE − grasp_target‖ < approach_radius_tol` | **R_pickup = +25** |
| **Stage 1** | 운반 / 마운트 | `−w_d_A · d_A − w_θ_A · θ_A` (타이어→허브) | `‖tire − hub‖ < 0.01 m` AND 마운트 정합 게이트 | **R_mount = +50** |
| **Stage 2** | 복귀 / 안착 | `w_return · exp(-d / 0.8)` + soft landing | `‖tire − pickup‖ < 0.05 m` AND `|v_z| < 0.1 m/s` | **R_return = +100** (성공) |

- 항상 작동 패널티: 충돌 / 작업공간 이탈 / action L2 / jerk
- Stage 1에서는 vertical-pose 패널티 면제 (마운트를 위해 보어 회전 자유)
- 실패 종료 시 `R_fail = −50` 일회성

### 픽업 게이트 타임스텝 커리큘럼

`approach_radius_tol`을 학습 진행에 따라 점진적으로 조여 R_pickup 신호를
초반부터 받을 수 있게 합니다.

- **t ≤ 100k**: `0.58 m` (soft 유지)
- **100k < t ≤ 300k**: 선형 감쇠 (`0.58 → 0.50`)
- **t > 300k**: `0.50 m` (hard cap)

`src/train.py`의 `ApproachTolCurriculumCallback`이 매 rollout 종료마다
`env_method("set_approach_tol", v)`를 호출하여 모든 sub-env에 전파합니다.
TensorBoard에 `curriculum/approach_tol`, `curriculum/approach_tol_frac` 기록.

---

## 커리큘럼 (stage × phase)

| Stage | 활성화되는 보상 항 |
|---|---|
| 1 | `align_A`, `reach_B` |
| 2 | + `coop`, `sync_joint_A` |
| 3 | + `success` + 충돌 / action / jerk 패널티 (전체 dense — **현재 학습 기본**) |
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
- action / obs dim이 13/89 → 7/83으로 축소
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

## 추가 학습 옵션

```powershell
# W&B 로깅
.\scripts\train.ps1 --wandb tyro --tags v1

# 솔버 / 물리 미세조정
.\scripts\train.ps1 --physics-num-sub-steps 6 --contact-force-done 2500

# Stage / Phase 변경
.\scripts\train.ps1 --stage 4 --phase 2

# 커리큘럼 노브 (override)
.\scripts\train.ps1 `
    --approach-tol-soft 0.60 `
    --approach-tol-hard 0.45 `
    --approach-tol-curriculum-steps 150000 `
    --approach-tol-ramp-steps 250000
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
