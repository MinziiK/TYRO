# TYRO — 듀얼암 타이어 마운팅 (PyBullet + PPO)

UR10 (Robot A) + Franka Panda (Robot B)가 협조하여 트럭 휠 허브에 타이어를
끼우는 PyBullet 시뮬레이션을 PPO로 학습합니다.

- 시뮬레이터: PyBullet, 240 Hz 물리 / 20 Hz 제어
- 좌표계: **Panda 베이스 = 원점** (모든 절대 좌표는 Panda 기준)
- 정책: PPO + MlpPolicy `[256, 256]`, vec env 12개 (SubprocVecEnv)
- obs **85-d** / action **6-d** (PPO는 UR10 Cartesian 잔차만 학습)

> **동작하는 구성 = 듀얼암 협조 운반** (`--dual-arm-coop`). 단일 UR10으로는
> 트럭 타이어를 다룰 수 없어서(도달거리 + 90° 재배향 한계), UR10이 다룰 수
> 있는 작은 타이어를 **양팔**이 함께 운반합니다. 레거시 단일팔 planner 경로도
> 남아 있으나 수렴하지 않으므로 `--dual-arm-coop`를 권장합니다.

모든 명령은 `tyro` conda 환경 활성화 상태를 가정합니다 (`conda activate tyro`).

---

## 설치

```bash
conda create -n tyro python=3.10 -y
conda activate tyro
pip install -r requirements.txt
```

---

## 빠른 시작

```bash
# 학습 (듀얼암 협조 운반, 권장)
python -m src.train --dual-arm-coop --num-envs 12 --total-steps 2_000_000

# 체크포인트 평가
python -m src.eval runs/<run>/best/best_model.zip --episodes 20

# GUI 진단 (양팔 + 타이어를 직접 조작)
python -m scripts.teleop_dualarm

# TensorBoard
tensorboard --logdir runs/
```

`train.py`는 추가 인자를 PPO/env에 그대로 전달합니다. `--run-name` 생략 시
자동 타임스탬프, 산출물은 `runs/<run>/`에 저장됩니다 (git 미추적).

---

## 동작 원리

**듀얼암 협조 운반** (mount-only):
1. 작은 URDF 타이어(외경 0.30 m, 보어 0.23 m)가 허브 앞 **양팔이 닿는 개방
   영역**(`coop_spawn_pos`)에 보어를 허브축에 맞춘 채 스폰됩니다.
2. 양팔이 파지: **UR10는 아래(6시)를 강체 파지**, **Panda는 위(12시)를 점
   파지** — 상하로 분리해 팔끼리 충돌을 방지합니다.
3. Min-Jerk 플래너가 짧은 lift + 평행이동(운반 중 재배향 없음)으로 타이어를
   허브에 올립니다. PPO는 UR10의 매 스텝 Cartesian **잔차**만 출력하고
   (`planner_pos_offset_scale`), Panda는 정책이 아니라 플래너가 지지합니다.
4. 타이어 COM이 허브에서 `mount_radius_tol` 이내이고 보어축이 정렬되면 마운트
   성공. 커리큘럼이 게이트를 **soft → hard**(0.30 m / 35° → 0.04 m / 5°)로
   조이며, `terminate_on=mount`로 마운트 시 에피소드를 종료합니다.

모든 치수·보상 가중치·임계값의 단일 진실 원천은 `src/config.py`입니다.

---

## 프로젝트 구조

```
TYRO/
├── src/
│   ├── config.py        # env / 보상 / 액션 / 커리큘럼 파라미터 전부
│   ├── train.py         # PPO 학습 진입점 (+ 커리큘럼 콜백)
│   ├── eval.py          # 체크포인트 로드 + N 에피소드 평가
│   ├── sweep.py         # Optuna 하이퍼파라미터 sweep
│   └── env/
│       ├── tyro_env.py  # Gymnasium env (reset / step / 보상 / FSM / 플래너)
│       ├── scene.py     # 트럭 허브 + 카고 + 타이어 스폰
│       ├── models.py    # 메시 없는 절차적 타이어 + 휠 디스크
│       ├── robots.py    # UR10 / Panda 래퍼 (IK, EE pose)
│       ├── rewards.py   # 보상 항
│       └── utils.py     # 쿼터니언 / 축 헬퍼
├── scripts/
│   ├── generate_tire_urdf.py   # UR10용 타이어 URDF 생성기
│   ├── teleop_dualarm.py       # GUI 텔레옵 / 도달성 진단
│   ├── smoke_planner_residual.py
│   └── generate_truck_wheel_station_urdf.py
└── data/urdf/
    ├── tire/tire_ur10.urdf          # 생성된 작은 타이어
    ├── truck_assembly/...           # 허브 + 10 볼트
    └── ur10_robot/...               # UR10 + Robotiq 그리퍼
```

---

## 주요 좌표 (Panda 기준)

| 객체 | 위치 |
|---|---|
| Panda (Robot B) 베이스 | `(0, 0, 0)` — 원점 |
| UR10 (Robot A) 베이스 | `(-0.60, 0.15, -0.30)` |
| 허브 중심 | `(0, 0.80, 0.22)`, 축 = world −Y |
| 협조 타이어 스폰 COM | `(-0.20, 0.50, 0.20)` |
| 바닥 평면 Z | `-0.60` |

타이어(UR10용): 외경 0.30 m, 보어 0.23 m, 폭 0.16 m, 1.5 kg.
재생성 / 크기 변경:

```bash
python -m scripts.generate_tire_urdf --outer 0.30 --inner 0.23 --width 0.16
```

---

## 플래그

```bash
--dual-arm-coop            # 협조 운반 (작은 URDF 타이어 자동 포함)
--num-envs / --total-steps # 병렬 수 / 학습 길이
--mount-curriculum         # soft→hard 마운트 게이트 (기본 ON)
--resume <ckpt.zip>        # 체크포인트 이어 학습
--wandb <project>          # W&B 로깅
```

하이퍼파라미터 sweep (Optuna, SQLite 저장):

```bash
python -m src.sweep --study smoke --smoke-test
python -m src.sweep --study tyro1 --n-trials 30 --steps-per-trial 200000
```
