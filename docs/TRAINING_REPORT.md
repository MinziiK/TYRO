# TYRO 이중 로봇 강화학습 — 전체 학습 보고서

> **작성일:** 2026-06-19 (최종 갱신)  
> **프로젝트:** TYRO — 듀얼암 타이어 마운팅 시뮬레이션 + PPO 학습  
> **배포 스택:** Robot A `phase1_mount_v3_dr` · Robot B `nut_fastening_v24_dr_stageB3` (1.75M ckpt)

> **제어 구조 (현재):** Robot A는 **Min-Jerk 명목 궤적 + PPO 잔차**(6-DOF).
> Robot B(v24)는 **pure-RL 3-DOF 위치 제어 + clean-branch 스크립트 매크로**
> (플래너 잔차 없음, coaxial 방향 락). E2E는 두 에피소드를 순차 실행한다.

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [시뮬레이션 환경](#2-시뮬레이션-환경)
3. [물체 모델링](#3-물체-모델링)
4. [씬 레이아웃 및 좌표계](#4-씬-레이아웃-및-좌표계)
5. [로봇 모델 및 제어](#5-로봇-모델-및-제어)
6. [관측 공간 (Observation)](#6-관측-공간-observation)
7. [액션 공간 (Action)](#7-액션-공간-action)
8. [제어 구조 (A: Planner+잔차 / B: Pure-RL+매크로)](#8-제어-구조)
9. [Robot A — Phase A 타이어 마운트 학습](#9-robot-a--phase-a-타이어-마운트-학습)
10. [Robot B — 너트 체결 학습](#10-robot-b--너트-체결-학습)
11. [PPO 학습 설정 (공통)](#11-ppo-학습-설정-공통)
12. [커리큘럼 및 도메인 랜덤화](#12-커리큘럼-및-도메인-랜덤화)
13. [종료 조건 및 안전 게이트](#13-종료-조건-및-안전-게이트)
14. [학습 파이프라인 및 실행 방법](#14-학습-파이프라인-및-실행-방법)
15. [설계 근거](#15-설계-근거)
16. [현재 결과 및 E2E 실측](#16-현재-결과-및-e2e-실측)
17. [E2E 평가 — 보상·설정·지표](#17-e2e-평가--보상설정지표)

---

## 1. 프로젝트 개요

### 1.1 목표

두 대의 산업용 로봇이 협동하여 **트럭 휠 허브에 타이어를 마운트하고 볼트를 체결**하는 작업을 강화학습으로 학습한다.

| 단계 | 로봇 | 작업 | 상태 |
|------|------|------|------|
| **Phase A** | Robot A (FANUC) | 타이어 픽업 → 허브 마운트 | ✅ `phase1_mount_v3_dr` (±5 cm DR) |
| **Phase A'** | Robot B (UR10e) | A 지지 중 10볼트 순차 체결 | ✅ v24 chain → DR Stage B3 |
| **E2E** | A + B | 마운트 후 동일 허브 offset에서 B 체결 | ✅ 100 시나리오 실측 (§16, §17) |
| **Phase B** | A + B | 6단계 풀사이클 (픽업→…→거치) | ⏳ 미착수 |

> Robot A는 명목+잔차, Robot B(v24)는 **pure-RL 접근 + env 매크로 체결**.
> B의 장거리 reach는 학습으로, INSERT/HOLD/RETRACT는 clean-branch 매크로가 담당한다(§10, §15).

### 1.2 기술 스택

| 항목 | 값 |
|------|-----|
| 시뮬레이터 | PyBullet |
| 알고리즘 | PPO (Proximal Policy Optimization) |
| 프레임워크 | Stable-Baselines3 |
| 정책 네트워크 | MlpPolicy `[256, 256]` |
| 언어 | Python 3.10 |
| 환경 | conda `tyro` |

### 1.3 전체 작업 흐름

```
[타이어 거치] ──S0 픽업──→ [A가 타이어 잡음]
                              │
                         S1 운반/마운트
                              │
                         [타이어 허브 안착] ──→ [A가 타이어 지지 (frozen)]
                                                    │
                                              B가 10개 볼트 순차 체결
                                                    │
                                              [체결 완료 = 에피소드 성공]
```

---

## 2. 시뮬레이션 환경

### 2.1 물리 파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `sim_freq_hz` | **240 Hz** | 물리 시뮬레이션 주파수 |
| `control_freq_hz` | **20 Hz** | 정책 제어 주파수 (decimation = 12) |
| `physics_num_sub_steps` | **12** (fanuc_spacious) | 서브스텝 수 (무거운 타이어 안정성) |
| `contact_erp` | 0.2 | 접촉 Error Reduction Parameter |
| `contact_cfm` | 2e-5 | Constraint Force Mixing |
| `gravity` | (0, 0, −9.81) m/s² | |
| `max_steps` | 2000 (A mount train/eval) / 2000–4000 (B v24 train) / 2500 (E2E B) | 태스크별 horizon |

### 2.2 환경 구성 요소

| 요소 | 타입 | 설명 |
|------|------|------|
| 바닥 | 무한 평면 + pit rim | 로봇 베이스 아래 pit 구조 |
| 허브 | 실린더 + 볼트 | ø420mm 플랜지, 10개 볼트 돌출 |
| 타이어 | 복합 multibody | 트레드 링 + 휠 디스크 (10개 lug hole) |
| **카고/차량 본체** | 정적 box + 휠웰 컷아웃 | 트럭 섀시, 작업공간 제약 + 진입 통로 |
| **카고 백월** | 정적 슬랩 | 타이어 허브 관통 방지 |
| 타이어 거치대 | 2개 정적 블록 | V-cradle (Y-split) |
| Robot A | URDF (FANUC) | 6-DOF + wheel gripper |
| Robot B | URDF (UR10e + nut tool) | 6-DOF + 30cm 너트러너 소켓 |

---

## 3. 물체 모델링

### 3.1 타이어 + 휠 디스크

실제 메시(STL) 대신 **프리미티브(box/cylinder) 조합**으로 모델링한다. PyBullet의 compound shape 제한(~16개/body)을 고려하여 multi-link 구조를 사용한다.

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `tire_outer_radius` | **0.525 m** | 외경 (295/80R22.5 근사) |
| `tire_inner_radius` | **0.282 m** | 보어(허브 pilot) 내경 |
| `tire_thickness` | **0.30 m** | 트레드 두께 (보어축 방향) |
| `tire_mass` | **100 kg** (heavy) | fanuc_spacious에서 heavy 모드 |
| `tire_inertia_heavy` | (18, 32, 32) kg·m² | 명시적 관성 (Bullet 자동 추정 방지) |
| `tire_hollow_collision` | True | 속이 빈 트레드 링 (실린더 아님) |
| `tire_annulus_collision_segments` | 16 | XY 평면 box 타일링 수 |

**트레드 링:** `tire_annulus_boxes()` — XY 평면에 N개 box를 원형 배치하여 속이 빈 링 형상 구현.

**휠 디스크:** `wheel_disk_three_boxes_per_hole()` — 10개 lug hole × 3 box(양쪽 flank + inner sill) = 30개 silver box. lug hole clearance를 물리적으로 표현.

**스폰 자세:** 수직 (보어축 = 월드 +X), V-cradle 거치대 위에 안착. 위치 `(-1.90, 0.0, 0.3913)`.

### 3.2 허브 (Hub)

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `hub_radius` | **0.21 m** | ø420mm 플랜지 |
| `hub_thickness` | **0.06 m** | 플랜지 두께 |
| `hub_pos_nominal` | (0, 0, 0) | fanuc_spacious: **허브 = 월드 원점** |
| `hub_base_rpy` | (0, −π/2, π/2) | 보어축 → 월드 −Y |
| `hub_axis_world` | (0, −1, 0) | 허브 법선 = 월드 −Y |

### 3.3 볼트 (Studs)

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `n_bolts` | **10** | lug circle 상 균등 배치 |
| `bolt_circle_radius` | **0.1675 m** | PCD (Pitch Circle Diameter) |
| `bolt_length` | **0.10 m** | 스터드 길이 (허브면 → 자유단) |
| `bolt_radius` | **0.011 m** | ø22mm (M22 스터드 근사) |
| 볼트 축 방향 | **월드 −Y** | 허브면에서 Robot B 방향으로 돌출 |
| 볼트 간 피치 | **≈10.4 cm** | 원주 ÷ 10 |

> **중요:** 10개 볼트 모두 동일한 축 방향(−Y)을 가지며, Robot B의 너트러너 공구 +Z가 이 축에 정렬되어야 체결 가능.

### 3.4 카고 / 차량 본체 (Cargo / Vehicle Body)

허브는 단독으로 떠 있지 않고 **트럭 섀시(카고 박스)에 장착**된 형태로 모델링한다. 카고는 (1) 작업 공간을 물리적으로 제약하고, (2) 휠 웰(wheel-well) 컷아웃으로 타이어 진입 통로를 만들며, (3) 타이어가 허브를 관통해 차량 내부로 밀려 들어가지 않도록 막는다.

#### 3.4.1 카고 박스 (Vehicle Primitive Box)

| 파라미터 | 값 (fanuc_spacious) | 설명 |
|----------|---------------------|------|
| `spawn_vehicle_primitive_box` | True | 카고 박스 활성 |
| `vehicle_half_extents` | (0.25, 1.0, 0.5) m | local 반치수 |
| `vehicle_base_rpy` | (0, 0, +π/2) | yaw +90° → 섀시가 X 주행축에 정렬 |
| `vehicle_center_world` | **(0.0, 0.25, 0.56)** | 허브(원점) 뒤·위로 추적 |
| 월드 footprint | 2.00 m (X) × 0.50 m (Y) × 1.00 m (Z) | 회전 후 |

> yaw +π/2로 인해 local Y(2m) → 월드 X(긴 섀시), local X(0.5m) → 월드 Y(두께). 카고의 −Y 면이 허브 플랜지 평면과 마주본다(실제 트럭처럼 허브 양옆을 감쌈).

#### 3.4.2 휠 웰 컷아웃 (Wheel-Well Cutout)

타이어가 진입할 수 있도록 카고 면에 **아치형 개구부**를 뚫는다.

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `cargo_use_wheel_well_cutout` | True | 컷아웃 활성 |
| `cargo_wheel_well_axis` | "y" | 허브축(−Y)과 동축 |
| `cargo_wheel_well_radius` | 0.85 m | 아치 반경 |
| `cargo_wheel_well_radius_yz` | 0.85 m | YZ 평면 반경 |
| `cargo_wheel_well_along_range_from_hub` | (−0.50, 0.50) | 허브 기준 축방향 범위 |
| `cargo_wheel_well_x_range_from_hub` | (−0.65, 0.85) | 허브 기준 X 범위 |
| `cargo_collision_subdiv` | (1, 6, 3) | local (nx, ny, nz) 분할 |

> 컷아웃 구현: 카고를 6×3 셀로 분할(subdiv)한 뒤, 휠 웰 실린더가 지나는 중앙·하단 셀을 제거 → **양쪽 기둥(pillar) + 지붕(roof)** 구조가 남아 타이어 둘레로 아치가 열린다.

#### 3.4.3 카고 백월 (Cargo Back Wall)

타이어가 허브 플랜지를 **관통해 차량 내부로 밀려 들어가는 것을 방지**하는 별도 정적 슬랩. (PyBullet compound primitive 한계를 피하기 위해 독립 body로 구성.)

| 파라미터 | 값 (fanuc_spacious) | 설명 |
|----------|---------------------|------|
| `spawn_cargo_back_wall` | True | 백월 활성 |
| `cargo_back_wall_y_offset` | **0.35 m** | 허브 중심에서 +Y 오프셋 |
| `cargo_back_wall_half_extents` | (1.0, 0.02, 0.50) m | 얇은 슬랩 (Y 두께 2cm) |
| `cargo_back_wall_center_z` | **0.56 m** | 카고 바닥과 정렬 |
| `cargo_back_wall_rgba` | (0.45, 0.30, 0.30, 1.0) | 갈색 |

> **물리적 역할:** 타이어가 마운트될 때 far face가 hub_y + 0.15에 도달. 백월은 그보다 더 뒤(허브 +0.35)에 위치하여, 마운트 시 약간의 overshoot은 허용하되 실제 관통은 막는다. `_sync_grasped_tire_upright`의 카고 관통 가드와 함께 작동 — kinematic 업데이트가 타이어를 카고/백월로 밀어넣으려 하면 마지막 안전 pose로 되돌린다.

#### 3.4.4 카고 충돌 처리

| 검사 대상 | 방식 |
|-----------|------|
| 로봇 arm ↔ 카고 | `_in_bad_collision` (충돌 페널티 / 종료) |
| kinematic 타이어 ↔ 카고/백월 | `_sync_grasped_tire_upright` penetration guard (안전 pose 복원) |
| Stage 1 carry arch | 0.35 m (카고 바닥 클리어) |

> **주의:** Robot B 너트 체결 task에서는 `collision_terminates=False` + socket↔hub/tire 필터로 충돌 종료를 끄지만, 카고 박스/백월은 정적 구조물로 씬에 그대로 존재한다.

### 3.5 타이어 거치대 (Tire Rack)

Phase A에서 타이어를 수직으로 지지하는 V-cradle.

| 파라미터 | 값 (fanuc_spacious) | 설명 |
|----------|---------------------|------|
| `tire_rack_half_extents` | (0.10, 0.05, 0.025) m | 레일 반치수 |
| `tire_rack_inner_center` | (−2.20, −1.10, −0.716) | 안쪽 레일 |
| `tire_rack_outer_center` | (−2.20, −1.90, −0.716) | 바깥 레일 |
| `tire_rack_support_posts` | True | 지지 기둥 |
| 구조 | Y-split V-cradle | 두 레일 위에 타이어 트레드가 안착 |

### 3.6 Robot B 너트러너 도구

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| URDF | `ur10e_with_nut_tool.urdf` | UR10e + 30cm 소켓 extension |
| `ur10e_nut_tool_length` | **0.30 m** | 플랜지 → tool_tip |
| tool_tip | `tool_tip` link | IK/관측의 EE 기준점 |
| 공구 +Z 방향 | **월드 +Y** (= 볼트축 반대) | 소켓이 볼트를 향해 삽입 |

---

## 4. 씬 레이아웃 및 좌표계

### 4.1 fanuc_spacious 레이아웃

```
                    +X (Robot A 방향)
                     ↑
                     │
    [타이어 거치] ←──┼──→ [허브 (원점)]
    (-1.9, 0, 0.39)  │
                     │
    ─────────────────┼─────────────→ +Y
                     │
                     ↓
              [Robot B base]
              (0.90, -0.95, -0.30)
                     │
                     ↓ −Y (볼트 축)
```

| 요소 | 월드 좌표 (x, y, z) | 비고 |
|------|---------------------|------|
| **허브 중심** | (0, 0, 0) | **월드 원점 = obs 기준점** |
| Robot A base | fanuc_spacious 오프셋 적용 | 대형 6축 |
| Robot B base | (0.90, −0.95, −0.30) | UR10e + 30cm tool |
| 타이어 거치 | (−1.90, 0.0, 0.3913) | V-cradle 위 |
| 볼트 0 | (0, −0.08, +0.168) | 12시 방향 |
| 볼트 4 | (−0.10, −0.08, −0.136) | 8시 방향 (먼 호) |

### 4.2 좌표계 규약

- **관측 기준점:** `obs_reference_pos = (0, 0, 0)` = 허브 중심
- **모든 위치 관측:** `body_pos − obs_reference_pos` (허브 기준 상대 좌표)
- **볼트 축:** 월드 −Y (모든 볼트 동일)
- **Robot B HOME EE:** (1.591, −0.776, 0.077) — 볼트 0까지 ≈1.7m

---

## 5. 로봇 모델 및 제어

### 5.1 Robot A — FANUC R2000iC

| 항목 | 값 |
|------|-----|
| URDF | `r2000ic210f_wheeltool.urdf` (wheel gripper 부착) |
| DOF | 6 (shoulder_pan, shoulder_lift, elbow, wrist_1/2/3) |
| EE link | `ee_link` |
| Palm-up lock | **ON** (tool +Z = world +Z) |
| Motor max velocity | 1.4 rad/s |
| Torque scale | 1.5× |
| Position/velocity gain | 0.85 / 1.15 |
| Post-step palm-up enforcement | ON (threshold 0.9998) |

**타이어 운반:** kinematic tire lock — 매 스텝 타이어 pose를 EE+offset으로 직접 설정. 100kg 타이어를 PD로 끌면 arm이 saturation되어 마운트 도달 불가 → kinematic teleport로 해결.

### 5.2 Robot B — UR10e + Nut Runner

| 항목 | 값 |
|------|-----|
| URDF | `ur10e_with_nut_tool.urdf` |
| DOF | 6 |
| EE link | `tool_tip` (소켓 끝) |
| Palm-up lock | **OFF** (자유 orientation, roll-free IK) |
| Motor forces (default) | [400, 400, 300, 60, 60, 60] N·m |
| Motor forces (nut task) | **[6000, 6000, 4000, 1000, 1000, 1000] N·m** |
| IK warm-start | **현재 관절** (nut task, branch continuity) |

> **중요 (중력 sag 대응):** 먼 호 볼트(4~6)에서 arm이 거의 완전히 뻗은 자세 → elbow 중력 토크 > 기본 300 N·m → arm sag 36.8cm. nut task에서 force cap을 20× 상향하여 해결(§15.1).

---

## 6. Observaton (관측 공간)

### 6.1 Phase A (Robot A 마운트) — 85차원

| 채널 | 차원 | 설명 |
|------|------|------|
| qA (normalized) | 6 | Robot A 관절 위치 [−1, 1] |
| dqA (normalized) | 6 | Robot A 관절 속도 |
| qB (normalized) | 7 | Robot B 관절 (frozen, 0으로 마스킹) |
| dqB (normalized) | 7 | Robot B 관절 속도 (frozen) |
| eeA_pos / ws | 3 | A EE 위치 (허브 기준) |
| eeA_orn | 4 | A EE quaternion |
| eeB_pos / ws | 3 | B EE 위치 (frozen, 마스킹) |
| eeB_orn | 4 | B EE quaternion (frozen) |
| tire_pos / ws | 3 | 타이어 COM 위치 |
| tire_orn | 4 | 타이어 quaternion |
| hub_pos / ws | 3 | 허브 중심 위치 |
| hub_orn | 4 | 허브 quaternion |
| bolt_pos / ws | 3 | 현재 타겟 볼트 위치 |
| bolt_orn | 4 | 볼트 quaternion |
| rel_th_pos / ws | 3 | 타이어→허브 위치 차 |
| rel_th_rot / π | 3 | 타이어→허브 회전 차 |
| rel_eb_pos / ws | 3 | B EE→볼트 위치 차 |
| rel_eb_rot / π | 3 | B EE→볼트 회전 차 |
| prev_action | 6 | 이전 스텝 액션 |
| mount_tail | 3 | (axial, lateral, lug_spin) 마운트 잔차 |
| hub_guide_vector | 3 | (hub − eeA) / ws — S1 운반 방향 cue |
| **합계** | **85** | |

### 6.2 Nut Task (Robot B, v24 pure-RL) — 94차원

Phase A **85차원** 베이스(6-DOF `prev_action` + `hub_guide_vector`)에서 B frozen 마스킹 후,
**+12차원 nut 전용 블록**:

| 채널 | 차원 | 설명 |
|------|------|------|
| vec_to_staging / ws | 3 | (staging_point − tool_tip) / ws |
| bolt_axis_unit | 3 | 볼트 축 단위벡터 (월드 −Y) |
| theta_normalized | 1 | tool +Z ↔ bolt axis / (π/2) |
| nut_axial, nut_lateral | 2 | 소켓 축방향 깊이·횡편차 (m, 정규화) |
| nut_subphase, nut_macro_stage | 2 | APPROACH(0) vs MACRO(1), 매크로 leg index |
| axial_err_to_target | 1 | 현재 매크로 leg 목표 대비 축 오차 |

> Robot A 채널은 **frozen + obs 마스킹**(0). 정책은 B의 **3-DOF Δposition**만 출력.

---

## 7. Action (액션 공간)

### 7.1 Phase A — 6차원

| 채널 | 범위 | 스케일 | 설명 |
|------|------|--------|------|
| Δpos_A [0:3] | [−1, 1] | × 0.03 m | EE 위치 잔차 (planner 위에 추가) |
| Δrot_A [3:6] | [−1, 1] | × 0.05 rad | EE 회전 잔차 (planner lock 시 무시) |

### 7.2 Nut Task (v24 pure-RL) — 3차원

| 채널 | 범위 | 스케일 | 설명 |
|------|------|--------|------|
| **Δpos_B [0:3]** | [−1, 1] | per-step EE Δ (pure-RL) | **유일한 활성 제어** — coaxial 락으로 방향 고정 |

> v24에서는 `nut_b_solo_action=True` → action space **3-DOF**. A 채널·B 회전 채널 없음.
> APPROACH(sub=0)에서 정책이 EE 위치를 직접 구동; MACRO(sub=1) INSERT는 axial servo가
> env 구동(정책 action 무시). HOLD/RETRACT는 매크로 joint lerp.

---

## 8. 제어 구조

### 8.1 Robot A — Min-Jerk Planner + PPO Residual

```
FSM 전이 → Min-Jerk baked traj (200 step) → nominal EE[idx]
         → + action[0:3]×0.03m (pos_only, palm-up lock) → IK → motors
```

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `use_planner_residual` | True | Master switch |
| `planner_pos_offset_scale` | **0.03 m** (train) / **0.06 m** (A DR fine-tune) | 잔차 위치 스케일 |
| `planner_enable_rot_offset` | False | 회전 잔차 비활성 |
| `planner_lock_palm_up` | True | tool +Z = world +Z |
| `planner_traj_steps` | 200 | baked trajectory 길이 |

### 8.2 Robot B (v24) — Pure-RL APPROACH + Clean-Branch Macro

```
APPROACH: policy Δpos (3-DOF) + nut_b_lock_coaxial (방향 고정)
       → arrive gate → clean-branch PREP/INSERT/HOLD/RETRACT (env joint lerp)
       → bolt fastened → next bolt (순서 nut_bolt_order)
```

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `nut_pure_rl` | True | 플래너 잔차 OFF |
| `nut_b_clean_branch_insert` | True | 타이어 충돌 없는 seat branch |
| `nut_clean_shortest_macro` | True | ±2π winding 정리 (스핀 제거) |
| `nut_clean_macro_smooth` | True | PREP smooth lerp |
| `nut_clean_prep_len` / `plunge_len` | **30 / 25** (train default) | E2E·배포 eval: **72 / 45** (`e2e_eval.py`) |
| `nut_b_axial_insert_servo` | True | INSERT 축방향 env servo |
| `nut_b_align_servo` | True | arrive 전 축 정렬 servo |
| `nut_arrive_lat_tol` | 0.015 m | arrive 횡편차 게이트 |
| `nut_arrive_pos_tol` | 0.08 m | staging 캡처 구 |

---

## 9. Robot A — Phase A 타이어 마운트 학습

### 9.1 FSM (4-Stage)

```
S0 (Approach)  ──pickup──→  S1 (Carry/Mount)  ──mount──→  S2 (Demount)  ──demount──→  S3 (Return)
  EE → 타이어 6시          타이어 → 허브           허브에서 인출           크래들 복귀
  그립 anchor               동축 마운트              축방향 pull             소프트 안착
  R_pickup=300              R_mount=300              R_demount=150           R_success=300
```

### 9.2 마운트 게이트

| 게이트 | 소프트 (시작) | 하드 (최종) | 램프 |
|--------|-------------|------------|------|
| 반경 (`mount_radius_tol`) | 0.55 m | 0.12 m | 2.5M step |
| 각도 (`mount_angle_tol`) | 45° | 5° | 2.5M step |

> 소프트 게이트(0.55m)는 `planner_stage1_approach_standoff`(0.70m)보다 **좁게** 설정하여, coaxial insertion segment(+Y leg)가 먼저 실행되도록 강제.

### 9.3 보상 구조

#### Dense (매 step, mix_dense=0.3)

| Stage | 보상 항 | 가중치 | 설명 |
|-------|---------|--------|------|
| S0 | `w_approach × exp(−d/decay)` | 3.0 / 0.15m | EE→그립 anchor 접근 |
| S0 | `w_approach_close × exp(−d/decay)` | 2.0 / 0.2m | 근거리 보너스 |
| S0 | `w_pb_approach × Δd` | 5.0 | Potential-based shaping |
| S1 | `w_guide × exp(−‖hub−ee‖/decay)` | 8.0 / 0.5m | EE→허브 벡터 유도 |
| S1 | `w_pb_carry × Δd_A` | 10.0 | 타이어→허브 거리 PB |
| S2 | `w_pull_demount × (1−exp(−d/decay))` | 3.0 / 0.2m | 디마운트 인출 |
| S2 | `w_pb_demount × Δd_hub` | 20.0 | 허브 이탈 PB |
| S3 | `w_return × exp(−d/decay)` | 3.0 / 0.5m | 크래들 복귀 |
| S3 | `w_pb_return × Δd` | 30.0 | 복귀 PB |
| All | `w_vertical × err` | 1.0 | 타이어 수직 유지 |
| All | `w_step_alive` | **−0.15/step** | hover 방지 (mix bypass, 매 스텝 차감) |
| All | `w_collision` | 10.0 | 충돌 페널티 |
| All | `w_sync_joint_a` | 0.005 | A 관절 속도 페널티 |

#### Sparse (FSM 이벤트, mix_sparse=0.7)

| 이벤트 | 보너스 |
|--------|--------|
| R_pickup (S0→S1) | +300 |
| R_mount (S1→S2) | +300 |
| R_demount (S2→S3) | +150 |
| R_success (S3→Done) | +300 |
| R_fail (실패 종료) | −50 |

### 9.4 학습 설정

#### Phase A 초기 학습 (`run_phase1_pipeline.sh`)

```bash
python -u -m src.train \
  --stage 3 --phase 1 --scene-layout fanuc_spacious \
  --num-envs 72 --n-steps 341 --batch-size 1024 \
  --device cpu --log-std-init -0.5 \
  --start-pos-easy-prob-schedule-mid 0.7 --...-end 0.6 \
  --mount-radius-soft 0.55 --mount-angle-soft-deg 45 \
  --mount-tol-ramp-steps 2500000 \
  --total-steps 2000000 --run-name phase1_mount_v2
```

#### Phase A 파인튜닝 (`run_phase1_mount_ft03.sh`)

```bash
python -u -m src.train \
  --stage 3 --phase 1 --scene-layout fanuc_spacious \
  --num-envs 72 --n-steps 341 --batch-size 1024 \
  --device cpu --log-std-init -0.5 \
  --start-pos-easy-prob-schedule-mid 0.85 --...-end 0.8 \
  --mount-radius-soft 0.55 --mount-angle-soft-deg 45 \
  --mount-tol-ramp-steps 2500000 \
  --total-steps 3000000 --run-name phase1_mount_v2_ft03 \
  --resume runs/phase1_mount_v2/best/best_model.zip \
  --resume-mode full
```

**결과:** v2 수렴 → ft03 best 확보. 결정론적 eval에서 타이어 중심 d ≈ 0 (~0.4 cm) 허브 안착.

#### Phase A DR fine-tune (`run_phase_a_dr_finetune.sh` → `phase1_mount_v3_dr`)

| 항목 | 값 |
|------|-----|
| resume | `phase1_mount_v2_ft03/best/best_model.zip` (policy-only, reset steps) |
| hub DR | 0→5 cm (hold 200k, ramp 1M) |
| `planner_pos_offset_scale` | **0.06 m** |
| easy prob | 0.85→0.8 |
| `total_steps` | 2M |
| 배포 | `runs/phase1_mount_v3_dr/final.zip` |

> E2E·배포 eval은 `terminate_on=mount`, `max_steps=2000`, `mix_easy_prob=0.8`(§17.2).
> full 4-stage train(`terminate_on=never`)과 달리 **R_demount/R_success 미발생**.

---

## 10. Robot B — 너트 체결 학습 (v24)

### 10.1 태스크 정의

- Robot A: 타이어 **허브에 pre-seat** + mount-hold 자세로 **frozen fixture**
- Robot B: **10볼트 순차 체결** — 순서 `(0, 5, 7, 2, 3, 8, 9, 4, 6, 1)`
- 시작 볼트: **항상 0번** (`scene.py` target_idx=0, premark 없음)
- 기하학적 체결: 너트 바디/토크 없음, coaxial 접근 + env 매크로 삽입/유지/후퇴
- 볼트 축 = 월드 **−Y**; 공구 +Z = 삽입 방향 (+Y)

### 10.2 서브 FSM (APPROACH + MACRO)

```
APPROACH (sub=0) — pure-RL 3-DOF, coaxial lock
  policy가 HOME → staging까지 EE 이동
  arrive: d_stage < 8cm, θ < 5°, lateral < 1.5cm
       ↓
MACRO (sub=1) — clean-branch env script
  PREP (optional smooth lerp) → INSERT (axial servo) → HOLD (6 step)
  → RETRACT → fastened → 다음 볼트
       ↓ (10/10)
  all_fastened → is_success
```

### 10.3 보상 구조 (farm-proof)

**공통 mix:** `total = 0.3×dense + 0.7×fsm_bonus − 0.15` (`step_alive`)

#### APPROACH (sub=0) — dense (×0.3)

| 항목 | 공식 | 가중치 | 설명 |
|------|------|--------|------|
| **pb_nut** | `25×(d_prev−d_now)` | w=25 | staging까지 거리 **감소분만** (유일한 주요 양수) |
| **nut_path** | `−8×max(0, Y_excursion)` | w=8, margin 2cm | 허브쪽(+Y) 평면 이탈 페널티 |
| **nut_joint_vel** | `−0.02×‖dq_B‖` | w=0.02 | 관절 속도 페널티 |
| **collision** | `−40` | w_nut_collision=40 | A-B mesh 접촉 |
| **action, jerk** | L2 | w=0.01 | 활성 3채널만 |

standing exp 커널(align/reach/lateral/axial/path/ba_clear)은 **전부 0**.

#### Sparse (×0.7 실효)

| 이벤트 | 설정값 | **실효** | 시점 |
|--------|--------|---------|------|
| `R_arrive` | +40 | **+28** | staging 도달 → 매크로 진입 |
| `R_insert` | +60 | **+42** | INSERT+HOLD 완료 |
| `R_fasten` | +120 | **+84** | 볼트 1개 RETRACT 완료 |
| `R_all_fastened` | +500 | **+350** | 10/10 → `is_success=True` |

볼트 1개 희소 합(실효): **154**. 10/10 완벽 시 희소만 **~1,890** + APPROACH PB 누적.

#### MACRO (sub=1)

INSERT axial servo 구간: 정책 action/jerk/pb **0** (env 구동). HOLD/RETRACT는 매크로 joint lerp.

### 10.4 v24 학습 파이프라인 (4단계)

| Stage | 스크립트 | run-name | 목표 | total steps |
|-------|---------|----------|------|-------------|
| **A** | `run_b_nut_v24_chain_recover.sh` | `nut_fastening_v24_chain` | 스핀-free 연속 10/10 (DR off) | 4M |
| **B** | `run_b_nut_v24_dr_stageB.sh` | `nut_fastening_v24_dr_stageB` | 허브 DR 0→5 cm | 3M |
| **B2** | `run_b_nut_v24_dr_stageB2.sh` | `nut_fastening_v24_dr_stageB2` | 1.25M ckpt에서 DR 3.5→5 cm | 2M |
| **B3** | `run_b_nut_v24_dr_stageB3.sh` | `nut_fastening_v24_dr_stageB3` | 5 cm 코너 집중 (4.5→5 cm) | 2M |

**배포 ckpt:** `runs/nut_fastening_v24_dr_stageB3/ckpts/ppo_1749440_steps.zip` (1.75M)

> Stage B `final.zip`은 붕괴(3/10). 중간 ckpt 스캔으로 **1.25M(B) → B2 → B3** 경로 채택.

### 10.5 B-side PPO 하이퍼파라미터 (v24 공통 베이스)

| 파라미터 | Stage A (chain) | Stage B | Stage B2 | Stage B3 |
|----------|----------------|---------|----------|----------|
| `num_envs` | 88 | 88 | 88 | 88 |
| `n_steps` | 279 | 279 | 279 | 279 |
| `batch_size` | 1024 | 1024 | 1024 | 1024 |
| `lr` | **1e-4** | **1e-4** | **5e-5** | **3e-5** |
| `ent_coef` | **0.001** | **0.001** | **0.0005** | **0.0005** |
| `log_std_init` | −1.0 | −1.0 | −1.0 | −1.2 |
| `max_steps` | 2000 | 4000 | 4000 | 4000 |
| `eval_episodes` | 30 | 20 | 20 | 20 |
| resume | v22 stage2 policy-only | v24_chain policy-only | B@1.25M policy-only | B2@1.5M policy-only |

SB3 공통: `gamma=0.995`, `gae_lambda=0.95`, `clip_range=0.2`, `vf_coef=0.5`,
`max_grad_norm=0.5`, `n_epochs=10`, `net_arch=[256,256]`, `device=cpu`.

### 10.6 B 커리큘럼 (v24)

| 커리큘럼 | Stage A | Stage B/B2/B3 |
|----------|---------|---------------|
| hot-start alpha | 0.3→0 (hold 200k, ramp 1.5M) | 0.15→0 (B) / 0.05→0 (B2/B3) |
| arrive ang tol | **5° 고정** | **5° 고정** |
| arrive pos tol | **8 cm 고정** | **8 cm 고정** |
| DR hub range | off | 0→5 cm (B) / 3.5→5 (B2) / 4.5→5 (B3) |
| A hold jitter | — | ±6° (train) / **0° (E2E)** |

---

## 11. PPO 학습 설정

### 11.1 Stable-Baselines3 기본값 (`src/train.py`)

| 파라미터 | 기본값 | CLI |
|----------|--------|-----|
| Algorithm | PPO | — |
| Policy | MlpPolicy | — |
| `net_arch` | [256, 256] | `--net-arch 256,256` |
| `learning_rate` | 3e-4 | `--lr` |
| `gamma` | 0.995 | `--gamma` |
| `gae_lambda` | 0.95 | `--gae-lambda` |
| `clip_range` | 0.2 | `--clip-range` |
| `ent_coef` | **0.0** | `--ent-coef` |
| `vf_coef` | 0.5 | `--vf-coef` |
| `max_grad_norm` | 0.5 | `--max-grad-norm` |
| `n_epochs` | 10 | `--n-epochs` |
| `n_steps` | 2048 | `--n-steps` |
| `batch_size` | 128 | `--batch-size` |
| `log_std_init` | 0.0 | `--log-std-init` |

> 실제 run은 아래 파이프라인 표처럼 CLI로 override한다.

### 11.2 파이프라인별 실측 하이퍼파라미터

#### Robot A — mount (`run_phase1_pipeline.sh` → `run_phase_a_dr_finetune.sh`)

| 파라미터 | v2 / ft03 | v3_dr (DR fine-tune) |
|----------|-----------|----------------------|
| `num_envs` | 72 | 72 |
| `n_steps` | 341 | 341 |
| `batch_size` | 1024 | 1024 |
| `lr` | 3e-4 (default) | 3e-4 |
| `ent_coef` | 0.0 | 0.0 |
| `log_std_init` | −0.5 | −0.5 |
| `eval_episodes` | 5 | 10 |
| `total_steps` | 2M + 3M (ft03) | 2M |
| `planner_pos_offset_scale` | 0.03 m | **0.06 m** |
| `max_steps` | 600 (train default) | 600 |

#### Robot B — v24 (§10.5 표 참조)

Stage A/B/B2/B3 각각 `lr`, `ent_coef`, `log_std_init`, DR curriculum이 다름.
공통: `num_envs=88`, `n_steps=279`, `batch_size=1024`, `device=cpu`,
`eval_freq=250000`, `terminate_on=never`.
**rollout batch** = 88×279 = **24,552** env-step / update (A mount: 72×341 = 24,552).

### 11.3 Rollout 로깅 지표 (`RewardBreakdownCallback`)

| 로그 키 | 의미 |
|---------|------|
| `rollout/success_rate` | 에피소드 `is_success` 비율 |
| `reward/{term}` | 보상 항목별 **스텝 평균** (pb_nut, fsm_bonus, guide_A, …) |
| `env/contact_force_mean` | 접촉력 |
| `env/ik_residual_A/B_mean` | IK 잔차 |
| `curriculum/*` | DR range, mount tol, hot-start alpha, … |

**B 핵심 지표:** `n_fastened_policy`(premark 제외), `rollout/success_rate`(10/10).

### 11.4 CPU / threading

env collection이 wall-time ~98%. `OMP/MKL/OPENBLAS=1`, `CUDA_VISIBLE_DEVICES=""`.
72–88 SubprocVecEnv worker oversubscription 방지.

---

## 12. 커리큘럼 및 도메인 랜덤화

### 12.1 Robot A 커리큘럼

| 커리큘럼 | 파라미터 | 설명 |
|-----------|---------|------|
| **시작 위치 easy prob** | v2: 0.7→0.6 / ft03·v3_dr: **0.85→0.8** | 타이어가 허브 근처에서 시작할 확률 |
| **마운트 tol ramp** | soft(0.55m,45°) → hard(0.12m,5°) over 2.5M | 점진적 정밀도 요구 |
| **Hub DR (v3_dr)** | 0→5 cm, hold 200k, ramp 1M | E2E용 A robustness |
| **Attached hot-start** | 타이어 pre-attached | Stage 0 pickup skip, Stage 1부터 |

### 12.2 Robot B 커리큘럼 (v24)

| 커리큘럼 | 설정 |
|-----------|------|
| Hot-start alpha | 0.3→0 (Stage A) / 0.05→0 (B2/B3) |
| arrive ang/pos tol | **5° / 8 cm 고정** (v24) |
| DR hub range | Stage B: 0→5 cm → B2: 3.5→5 → B3: 4.5→5 cm |
| Per-bolt random start | OFF (bolt 0 cold-start) |

### 12.3 도메인 랜덤화 (현재 배포)

| 로봇 | run | DR 범위 | cargo DR |
|------|-----|---------|----------|
| A | `phase1_mount_v3_dr` | 학습·E2E: 허브 XY **0→5 cm** | OFF (`--no-dr-cargo`) |
| B | `v24_dr_stageB3` | **학습** 4.5→5 cm; **E2E eval** uniform ±5 cm | OFF |

> 카고/백월은 허브와 **동행 drift** → 상대 geometry 유지. 독립 cargo jitter는 사용하지 않음.

---

## 13. 종료 조건 및 안전 게이트

### 13.1 Robot A

| 조건 | 기본 | fanuc_spacious | 설명 |
|------|------|----------------|------|
| Success (mount) | `terminate_on=mount` | mount event | 마운트 완료 |
| Success (full) | `terminate_on=never` | landed event | 4-stage 완료 |
| Vertical violation | ON | ON (nut task: OFF) | 타이어 수직 이탈 (>15°) |
| Collision | OFF (penalty only) | OFF | per-step −10 페널티 |
| Workspace | ON | ON | EE workspace 이탈 |
| Contact force | 2500N / 50000N | 50000N | 과도 접촉력 |
| Max steps | 2000 (E2E/eval) | 2000 | mount horizon |

### 13.2 Robot B (Nut Task)

| 조건 | 값 | 설명 |
|------|-----|------|
| **Success** | all_fastened (10/10) | 전체 체결 |
| Collision | **OFF** | 기하 체결, socket↔hub/tire filtered |
| Contact force | **OFF** (0.0) | |
| Vertical | **OFF** | 타이어 mounted 상태 |
| Max steps | 4000 (train) / 2500 (E2E) | timeout |

---

## 14. 학습 파이프라인 및 실행 방법

### 14.1 전체 파이프라인 (현재)

```
1. Robot A mount
   run_phase1_pipeline.sh (2M) → run_phase1_mount_ft03.sh (3M resume)

2. Robot A DR
   run_phase_a_dr_finetune.sh → phase1_mount_v3_dr/ (2M, hub 0→5cm)

3. Robot B v24 chain (DR off, spin-free)
   run_b_nut_v24_chain_recover.sh → nut_fastening_v24_chain/ (4M)

4. Robot B DR stages
   run_b_nut_v24_dr_stageB.sh (3M)
   → run_b_nut_v24_dr_stageB2.sh (2M, from 1.25M ckpt)
   → run_b_nut_v24_dr_stageB3.sh (2M, 5cm corner)

5. E2E eval
   scripts/run_e2e_eval.sh (100 scenarios, ±5cm)
   scripts/e2e_hub_capture.py (허브 PNG 캡처)
```

### 14.2 실행 예시

```bash
conda activate tyro

# A DR fine-tune
bash scripts/run_phase_a_dr_finetune.sh

# B v24 chain → DR
bash scripts/run_b_nut_v24_chain_recover.sh
bash scripts/run_b_nut_v24_dr_stageB.sh
bash scripts/run_b_nut_v24_dr_stageB2.sh
bash scripts/run_b_nut_v24_dr_stageB3.sh

# E2E 100 시나리오
bash scripts/run_e2e_eval.sh

# GUI 연속 뷰
bash scripts/run_e2e_view_v24.sh
```

### 14.3 배포 체크포인트

| Run | Deploy |
|-----|--------|
| `phase1_mount_v3_dr` | `runs/phase1_mount_v3_dr/final.zip` |
| `nut_fastening_v24_dr_stageB3` | `runs/nut_fastening_v24_dr_stageB3/ckpts/ppo_1749440_steps.zip` |

---

## 15. 설계 근거

현재 v24 스택의 핵심 설계 원칙 4가지.

### 15.1 원칙 ① — 물리적 한계는 학습이 아니라 물리로 푼다

먼 호 볼트(4·5·6)는 UR10e가 거의 완전히 뻗은 자세에서 도달한다. 이때 elbow 정적 중력
토크가 기본 motor force cap(300 N·m)을 초과하면 arm이 staging pose를 못 버티고 EE가
36.8 cm sag → macro trigger 자체가 불가능하다. 이는 **보상/커리큘럼으로 해결할 수 없는
물리적 벽**이다.

**설계:** nut task motor force cap을 `[400,400,300,60,60,60]` →
`[6000,6000,4000,1000,1000,1000]` N·m으로 상향(B는 payload 없어 안전). sag 36.8→0~2.5 cm.

### 15.2 원칙 ② — 진척 지표는 정직해야 한다

reverse-curriculum이 앞 볼트를 미리 fastened로 표시(premark)하면 `n_fastened`가 부풀려져
실제 정책 성능과 무관해진다. **설계:** premark를 제외한 `n_fastened_policy` 지표 +
`eval/success_rate`로 측정하고, 항상 bolt 0 cold-start(random-bolt OFF)로 고정한다.

### 15.3 원칙 ③ — 보상은 farm 불가능해야 한다

`exp(−거리)` 형태의 항상-양수 standing 커널은 정책이 볼트 옆 ~12 cm에 주차해 per-step
income을 빨아먹는 farm을 만든다(체결보다 이득). 하나를 막으면 다른 커널이 새 farm
소득원이 되는 whack-a-mole이 발생한다. **설계:** standing 커널을 **전부 0**으로 하고,
전진분만 보상하는 potential-based shaping(PB) + corridor 페널티만 유지한다.

### 15.4 원칙 ④ — Robot A: 명목+잔차 / Robot B: pure-RL + 매크로

Robot A는 Min-Jerk 명목+잔차로 장거리 carry/mount를 해결한다.

Robot B(v24)는 **접근(APPROACH)만 RL**로 학습하고, INSERT/HOLD/RETRACT는
**clean-branch 매크로**가 담당한다. v14 planner+잔차는 DR/연속 체인에서 한계가 있어
폐기. 스핀 제거(`nut_clean_shortest_macro`)와 DR 학습은 **단계 분리** 필수
(Stage A: chain 회복 → Stage B/B2/B3: DR only).

### 15.5 핵심 교훈

> A는 명목+잔차, B는 **접근 RL + env 체결 매크로**. DR·매크로·체인 변경을 동시에
> 학습하면 chain 붕괴(v23). E2E는 A mount 성공 × B 10/10의 **곱**이므로 B DR가 병목.

---

## 16. 현재 결과 및 E2E 실측

### 16.1 Robot A (`phase1_mount_v3_dr`)

| 항목 | 결과 |
|------|------|
| 학습 | v2 2M + ft03 3M + **v3_dr 2M** (hub 0→5 cm) |
| 배포 | `runs/phase1_mount_v3_dr/final.zip` |
| E2E (100 sc, ±5 cm) | **A success 79%** |

### 16.2 Robot B (`v24_dr_stageB3` @ 1.75M)

| 항목 | 결과 |
|------|------|
| Stage A chain | 명목 연속 **10/10**, 스핀 제거 |
| Stage B | final 붕괴 → **1.25M ckpt** 채택 |
| Stage B2 | 0 cm **9.6/10**, 2 cm **10.0/10** (mean) |
| Stage B3 | 5 cm corner 보강 (4.5→5 cm 집중) |
| E2E (100 sc, ±5 cm) | **B success 17%** (10/10) |

### 16.3 E2E 100 시나리오 실측 (2026-06-16)

**하니스:** `scripts/e2e_eval.py --v24 --scenarios 100 --dr-range-cm 5`

| 지표 | 값 |
|------|-----|
| A mount rate | **79%** (79/100) |
| B 10/10 rate | **17%** (17/100) |
| **E2E success** | **15%** (15/100) |
| B `n_fastened` mean | **3.14** / 10 (partial 체결 다수) |

**병목:** B DR @ ±5 cm. A는 ~80%로 안정, B가 E2E 곱의 제한 요인.

**E2E 성공 중 가장 먼 허브:** scenario 11 (x=+4.7 cm, y=+3.9 cm, max|축|=4.7 cm).

> JSON: `runs/e2e_eval/e2e_100sc_5cm_20260615_184724.json`

### 16.4 v24 진화 요약

| 단계 | 핵심 | 결과 |
|------|------|------|
| v22 pure-RL + clean-branch | 타이어 충돌 회피 seating | 명목 10/10 |
| v23 (폐기) | approach-seed IK + DR 동시 | chain 1–3/10 |
| v24 Stage A | `nut_clean_shortest_macro` | 스핀-free 10/10 |
| v24 Stage B→B3 | DR only, ckpt 스캔 | B2 1.5M deploy → B3 1.75M |

### 16.5 향후

1. B 5 cm corner 추가 보강 (horizon / arrive tol / workspace edge)
2. E2E `b_max_steps`·A jitter ablation
3. Phase B 6-stage full cycle (`--remount-cycle`)

---

## 17. E2E 평가 — 보상·설정·지표

E2E는 **두 개의 독립 에피소드**다. 보상은 이어지지 않으며 `a_reward`, `b_reward`로
각각 합산된다. 성공 판정은 `is_success` 플래그(A mount, B all_fastened).

### 17.1 공통 보상 mix

```
total = 0.3 × dense + 0.7 × fsm_bonus − 0.15   (step_alive, 매 스텝)
```

희소 보너스 **실효값 = 설정값 × 0.7**.

### 17.2 Phase A — Robot A 마운트 (`terminate_on=mount`)

**설정** (`scripts/e2e_eval.py` mount_overrides):

| 항목 | 값 |
|------|-----|
| `terminate_on` | `"mount"` — **R_mount 직후 종료** |
| `mix_easy_prob` | 0.8 |
| `mount_radius_tol` | 0.55 m (soft) |
| `planner_pos_offset_scale` | 0.06 m |
| `max_steps` | 2000 |
| DR | ±5 cm (`RANDOM_POSITION_RANGE`) |

**FSM (E2E에서 도달 가능):**

```
S0 Approach ──pickup──→ S1 Carry/Mount ──mount──→ [종료]
```

`R_demount`, `R_success`는 **발생하지 않음**.

#### Dense (×0.3)

| Stage | 항목 | 가중치 |
|-------|------|--------|
| S0 | `approach_A` (exp + close) | 3.0 + 2.0 |
| S0 | `pb_approach` | 5.0 × Δd |
| S1 | `guide_A` | 8.0 × exp(−‖hub−ee‖/0.5) |
| S1 | `pb_carry` | 10.0 × Δd_A |
| All | `vertical_pen`, `collision`, `action`, `jerk`, `sync_joint_a` | §9.3 |
| All | `step_alive` | **−0.15/step** |

#### Sparse (×0.7 실효)

| 이벤트 | 설정 | **실효** |
|--------|------|---------|
| `R_pickup` | +300 | **+210** (S0→S1, easy start에서 생략 가능) |
| `R_mount` | +300 | **+210** (S1→S2, **에피소드 종료**) |
| `R_fail` | −50 | **−35** |

**A 에피소드 희소 합:** pickup+mount ≈ **420** (pickup 포함 시) / **210** (mount만).

### 17.3 Phase B — Robot B 너트 (v24)

**설정** (`_nut_overrides_v24`):

| 항목 | E2E train | E2E eval |
|------|-----------|----------|
| `nut_pure_rl` | True | True |
| `nut_clean_shortest_macro` | True | True |
| `nut_clean_prep_len` / `plunge_len` | train default 30/25 | **E2E eval 72/45** |
| `nut_a_hold_jitter_rad` | 6° (train) | **0°** (deterministic fixture) |
| `max_steps` | 4000 (train) | **2500** |
| `terminate_on` | never | never |

reset: 타이어 **허브 pre-seat**, A mount-hold IK, B bolt 0부터 cold-start.

> **매크로 타이밍:** 학습 기본값은 `prep=30`, `plunge=25`. E2E harness는 B3 검증
> 설정과 동일하게 **72/45**를 강제한다(`e2e_eval.py` `_nut_overrides_v24`).
> 배포 eval·GUI는 harness와 동일 wiring을 쓸 것.

#### APPROACH dense (×0.3) — §10.3과 동일

`pb_nut`(25×Δd), `nut_path`(−8×Y_excursion), `nut_joint_vel`, `collision`(−40), `action`/`jerk`.

#### Sparse per bolt (×0.7)

| 이벤트 | 설정 | **실효** |
|--------|------|---------|
| `R_arrive` | +40 | +28 |
| `R_insert` | +60 | +42 |
| `R_fasten` | +120 | +84 |
| `R_all_fastened` | +500 | +350 (10/10) |

**10/10 완벽 시 희소 합(실효):** 10×154 + 350 ≈ **1,890** + APPROACH PB − 페널티.

#### MACRO 구간

INSERT axial servo: 정책 `action`/`pb_nut`/`jerk` = **0**. env clean-branch lerp.

### 17.4 E2E eval 출력 지표

| 필드 | 의미 |
|------|------|
| `a_success` / `b_success` | 각 `is_success` |
| `e2e_success` | A ∧ B (10/10) |
| `a_reward` / `b_reward` | 에피소드 **총 step reward** (스케일 다름, 디버그용) |
| `b_n_fastened` | 체결 볼트 수 (0–10) |
| `hub_offset_*` | 시나리오별 DR offset |

**실행:**

```bash
bash scripts/run_e2e_eval.sh   # A=v3_dr, B=B3_1.75M, 100 sc, ±5cm
```

### 17.5 E2E reward 스케일 참고 (실측)

A·B 총 reward는 **스케일·horizon이 달라 직접 비교 불가**. 성공 판정은 `is_success`만 사용.

| 케이스 | `a_reward` (예) | `b_reward` (예) | 비고 |
|--------|-----------------|-----------------|------|
| E2E 성공 (sc0) | ~232 | ~1,659 | A 176 step, B 1798 step |
| A만 성공 | ~200–400 | — | mount sparse ≈210–420 |
| B partial (7/10) | — | ~800–1200 | 희소 7×154 미만 + PB |

---

## 부록 A: 파일 구조

```
TYRO/
├── src/
│   ├── config.py          # 모든 설정 (EnvConfig, RewardConfig, ...)
│   ├── train.py           # PPO 학습 CLI + curriculum callbacks
│   └── env/
│       ├── tyro_env.py    # 핵심 환경 (FSM, reward, obs, nut task)
│       ├── robots.py      # Robot A/B 클래스 (IK, motor control)
│       ├── scene.py       # 씬 빌드 (hub, tire, bolts, floor)
│       ├── models.py      # 타이어/휠 프리미티브 모델링
│       └── rewards.py     # RewardBreakdown dataclass
├── scripts/
│   ├── run_phase1_pipeline.sh           # A mount v2 (2M)
│   ├── run_phase1_mount_ft03.sh         # A mount ft03 (3M)
│   ├── run_phase_a_dr_finetune.sh       # A DR v3_dr (2M)
│   ├── run_b_nut_v24_chain_recover.sh   # B v24 Stage A
│   ├── run_b_nut_v24_dr_stageB.sh       # B DR Stage B
│   ├── run_b_nut_v24_dr_stageB2.sh      # B DR Stage B2
│   ├── run_b_nut_v24_dr_stageB3.sh      # B DR Stage B3 (deploy)
│   ├── run_e2e_eval.sh                  # E2E 100 sc headless
│   ├── run_e2e_view_v24.sh              # E2E GUI
│   ├── e2e_eval.py                      # E2E harness
│   ├── e2e_hub_capture.py               # 허브 before/after PNG
│   ├── e2e_view_continuous.py           # 단일 창 A→B 연속 GUI
│   └── view_nut.py                      # B 단독 GUI (--v24)
├── data/
│   ├── urdf/              # 로봇 URDF (메시 별도 fetch)
│   └── nut_mount_endpose.npz  # A 지지 자세
└── runs/                  # 학습 로그 + 체크포인트
```

## 부록 B: 주요 설정 파일 위치

| 설정 | 파일 | 함수/클래스 |
|------|------|------------|
| 물리/씬/로봇 | `src/config.py` | `EnvConfig` |
| 보상 가중치 | `src/config.py` | `RewardConfig` |
| fanuc_spacious | `src/config.py` | `apply_fanuc_spacious_layout()` |
| nut task override | `src/config.py` | `make_env_config()` |
| 카고/차량/거치대 빌드 | `src/env/scene.py` | `Scene.build()` |
| 타이어/휠 프리미티브 | `src/env/models.py` | `create_tire_wheel_multibody()` |
| FSM/reward/obs | `src/env/tyro_env.py` | `TyroEnv` |
| PPO CLI | `src/train.py` | `main()` |

---

*본 문서는 TYRO 프로젝트의 Robot A/B 학습·E2E 평가에 사용된 설정, 보상, PPO 하이퍼파라미터,
v24 파이프라인, E2E 실측(2026-06)을 정리한 것입니다.*
