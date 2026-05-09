# Tyro team

dual-arm tyre / bolt alignment 환경(UR10 + Franka Panda)과 PPO 학습 스크립트입니다. 아래 순서는 **반드시 `conda activate tyro` 환경**에서 실행한다고 가정합니다.

## 설치 (`tyro` 가상환경)

```bash
git clone https://github.com/MinziiK/TYRO.git
cd TYRO

conda create -n tyro python=3.10 -y
conda activate tyro

# Windows 등에서 pip로 빌드된 pybullet이 실패하면 conda-forge 권장
conda install -c conda-forge pybullet -y
pip install -r requirements.txt
```

---

## 워크플로 개요

### 1단계 — 환경 및 하드웨어 에셋

- **UR10**  
  - `data/urdf/ur10_robot/ur10_robot.urdf`  
  - 메쉬는 `ur_description/meshes/ur10/visual/*.dae`처럼 **URDF 디렉터리 기준 상대 경로**로 참조됩니다(PyBullet 검색 경로: `data/urdf/ur10_robot`). ROS `package://` 접두사는 레포에서는 사용하지 않습니다.
- **트럭 휠 스테이션 URDF**(ISO 335 mm PCD · **볼트 10**개 · 허브 약 ø420 mm, 볼트 +Z 로컬축, 월드는 `hub_base_rpy`) 재생성:

  ```bash
  python scripts/generate_truck_wheel_station_urdf.py \
      --bolt-collision-radius-factor 0.95 \
      --bolt-pattern-phase-deg 0 \
      --output data/urdf/truck_assembly/truck_wheel_station.urdf
  ```

  비주얼 반지름은 유지하고, **충돌 실린더만 줄여**(약 5% 게이프) 간섭 여유를 둡니다.  
  같은 스크립트로 **허브 파일럿 보스**(휠 센터 보어 슬라이딩)·플랜지 **역방향(−Z)** 의 **브레이크 로터·캘리퍼 단순 충돌 프록시**도 기본 포함됩니다(끄기: `--no-hub-pilot`, `--no-brake-proxy` 및 `--hub-pilot-radius` 등).

- **카고(휠 웰 프록시)**  
  - `EnvConfig.spawn_vehicle_primitive_box=True`(별칭: `spawn_cargo_box`).  
  - 정비 레이아웃: 카고 월드 중심 `vehicle_center_world`, 반변 `vehicle_half_extents`(허브 대비 +X 깊이·펜더 높이는 `EnvConfig` 주석 참고). 커리큘럼으로 허브가 흔들리면 카고도 같은 편향을 따라 움직임.

실제 장면 빌드는 `src/env/scene.py`에서 `use_truck_hub_urdf`로 선택합니다(Urdf 또는 구 프리미티브).

- **타이어+휠(단일 바디 근사)**  
  `src/env/models.py`의 `create_tire_wheel_multibody`가 **메쉬 없이** 프리미티브만으로 만듭니다. **검은 타이어 트레드**는 원환 박스 링(`tire_annulus_collision_segments`, `tire_inner_radius` 기본 0.282 m), **은색 휠 디스크**는 기본값 **`tire_wheel_disk_style="three_piece"`** 일 때 **러그 홀마다 은색 박스 3개**(측벽 2 + 안쪽 실)로 원형 홀을 근사하고, Bullet **~16개 시각 프리미티브 상한**을 넘기지 않도록 **휠 디스크만 고정 자식 링크**로 쪼갭니다. 레거시 **`inter_lug_wedge`** 는 단일 compound 방사 쐐기 10개입니다. PCD·위상·홀 반지름은 `bolt_circle_radius`, `wheel_disk_bolt_phase_rad`, `wheel_disk_bolt_hole_radius`·URDF 생성기와 맞춥니다.
- **카고 휠 멀 컷아웃**  
  기본 활성 시 바디 AABB를 셀로 나누고, 허브 중심을 지나 **`+world X`** 축 평행 원통 안의 셀은 충돌·비주얼에서 빼 타어가 들어갈 빈 칸을 냅니다 (`cargo_use_wheel_well_cutout`, `cargo_wheel_well_radius_yz`, `cargo_wheel_well_x_range_from_hub`, `cargo_collision_subdiv`).

---

### 2단계 — 좌표계 및 보상

- 허브 루트 자세: URDF에서는 스터드가 링크 **로컬 +Z**. `hub_base_rpy` 기본 **pitch −π/2** 가 이 축을 **월드 −X**(로봇 A/B가 있는 쪽, 베이스 ≈ −X)으로 향하게 합니다(`EnvConfig`). 시뮬에서 볼트가 하늘/옆을 보면 부호만 **`+π/2`** 로 바꿔 보세요 — URDF 생성기가 아니라 **`src/config.py`의 ``hub_base_rpy``·``hub_axis_world``** 에서 장면 회전을 잡습니다.
- Panda EE와 볼트 목표점은 **`getLinkState`의 월드 링크 프레임(인덱스 4 — 위치, 5 — 자세)**를 사용합니다(로봇 EE와 동일).

- 보상은 `src/env/rewards.py` / `TyroEnv`에서 구성합니다.  
  - **Reach / Align**: 타이어–허브, 그리퍼–볼트 거리 및 축 각 (`angle_between`).  
  - **관측 89차원**: 마지막 3값은 허브 법선 방향 분리량·횡오프셋·정규화된 럭 각 오차(**`wheel_disk`** vs **`bolt_0`** 기준 ray, `RewardConfig.success_*`, `lug_spin_tolerance_rad` 등과 함께 튜닝).  
  - **Cooperative sync**: Robot A 관절 속도 크기 페널티(`sync_joint_A`)로 B의 정밀 작업 여지 확보(stage ≥ 2부터 가중치 ON).  
  - **Dense / Sparse 비율**: 기본값 **dense 30%**, **sparse success 70%**(`RewardConfig.mix_dense`, `mix_sparse_success`).  
  - **Stage 1**(Phase‑1 학습): `R_success=0` 이라 스파스 성공 종료는 꺼지고, **`align_reward`/`reach_reward` 밀도**로만 유도됩니다(허브 밀착 스파스 종료가 필요하면 **stage ≥ 3** 또는 `make_reward_config` 조정).

- **접촉력 과다 종료**  
  - `contact_force_done`: 시뮬 전역 접촉점의 노멀 포스 최대값이 `contact_force_terminate_above` 이상이면 종료(`termination`: `contact_force`).  
  - 비활성화하려면 `<= 0`으로 설정합니다.

물리 안정 설정은 `EnvConfig`(및 학습 스크립트 CLI 오버라이드)와 `TyroEnv.reset()`의 `setPhysicsEngineParameter`에 있습니다.

---

### 3단계 — 정적 정합 검증

```bash
python scripts/check_alignment.py --render
python scripts/check_alignment.py --render --stage 1
python scripts/check_cargo_reachability.py --render
```

- 첫 스크립트: 그리퍼 Z축과 볼트 축 정렬 각도 확인(디버그 라인 표시). 옵션: `--stage`, `--phase`, `--no-truck-hub-urdf`.
- 두 번째: 카고 프록시 ON 상태에서 허브 근접 IK의 잔차·Robot A 자기충돌·카고 접촉 개수 확인.

---

### 4단계 — 물리 및 PPO 설정

환경 기본값(`EnvConfig`): `physics_num_sub_steps=6`(4–8 범위 권장), `contact_erp≈0.15`, `contact_cfm≈1e-5`(실제로는 PyBullet `globalCFM`에 전달됨).

학습 스크립트 기본값: `batch_size=128`, `gamma=0.995`, `n_steps=2048`.

물리 종료 역치 등은 학습 시 덮어쓸 수 있습니다:

```bash
python -m src.train --physics-num-sub-steps 6 --contact-force-done 2500 ...
```

텐서보드에는 `reward/*` 평균과 `env/contact_force_mean`이 기록됩니다.

---

## GUI 스모크

```bash
python -m src.test --render
python -m src.test --render --action-scale 0
```

---

## 학습 / 평가

```bash
# Phase 1 (도달·정렬 워밍업) 별칭 ↔ stage 1 + curriculum phase 1
python -m src.train --task phase1 --num-envs 8 --total-steps 1_000_000

python -m src.train --stage 1 --phase 1 --total-steps 1_000_000 --num-envs 8
python -m src.train --stage 3 --phase 1 --total-steps 3_000_000 --num-envs 8
tensorboard --logdir runs/

python -m src.eval runs/stage3_phase3_*/best/best_model.zip --render --episodes 5
```
