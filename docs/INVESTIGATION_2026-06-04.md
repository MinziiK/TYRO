# TYRO Phase 1 — 부드러운 carry / mount 조사 기록 (2026-06-04)

이 문서는 2026-06-04 세션에서 수행한 **측정 기반** 조사 전체를 기록합니다.  
코드 기본값·수치의 단일 진실 원천은 `src/config.py` 주석과 CHANGELOG(`README.md`)입니다.

---

## 1. 문제 정의

- **증상**: carry→허브 구간에서 EE가 "오락가락"(큰 점프, yaw 진동), 실로봇처럼 부드럽지 않음.
- **목표**: 흔들림 없이 안정적으로 허브에 시팅. 학습 전에 **물리/기하**가 타당해야 함.

---

## 2. 근본 원인 (확정)

### 2.1 carry는 이미 부드러움 — 삽입 구간만 문제

| 구간 (stage-1 step) | max EE jump | jumps > 15 cm |
|---|---|---|
| 0–51 (carry 전반) | ~5 cm | 0 |
| 74–99 (허브 삽입) | **70 cm** | 11 |

- **baked 관절 명령**은 삽입 구간에서도 max \|Δq\| ≈ 0.12 rad → **계획/IK 분기 문제 아님**.
- **허브 mount EE**가 UR10 베이스에서 **~1.13 m = reach 1.30 m의 87%** → **reach 포화 특이점**.
- 허브 도착 시 **manipulability w = √(det(JJᵀ)) → 0** (완전 특이점).

### 2.2 증상의 두 가지

1. **위치**: stiff PD가 특이점 근처에서 EE를 **채찹질** (70 cm/step).
2. **yaw**: EE yaw가 단조 90° 회전이 아니라 **24회 역회전** 후 step ~80에서 wrap → GUI에서 "허브 앞에서 각도 틀어진 채 정체".

---

## 3. 시도한 레버와 결과 (zero-action 기준)

### 3.1 실패 또는 기대 미달

| 레버 | 결과 |
|---|---|
| `planner_residual_warmstart_from_baked` | 효과 없음 (IK 200-iter 분기 독립) → 기본 OFF |
| `planner_smooth_baked_window` | mount 실패 (4 cm 게이트 이탈) → 기본 0 |
| waypoint 도착 게이트 (baked) | 허브 미도달 → 기본 OFF |
| `planner_precompute_joint_traj=False` | 허브 ~0.85 m 미도달 |
| **DLS Cartesian (고정 λ)** | 매우 부드러움(j>15=0) but **mount 실패** (특이점 감쇠로 미도달) |
| **DLS + adaptive λ** | mount 가능하나 j>15 여전히 2–14 |
| **허브만 Y 당김** (0.80→0.55) | cargo 통로 침범, 가드 동결 / mount 실패 |
| **공격적 재배치** (pickup−1.45, hub−0.15/0.55) | carry **70→8 cm** but cargo·팔 충돌 → mount 실패 |
| **front-load 강화** (k=6, 20) | yaw 선회전 → **허브 미도달** (회전 자세가 멀리서 도달 불가) |
| **rotate-first @ cradle** | mount 실패 (76–124 cm 정체) |
| **back-load @ hub** | cargo 충돌 (사용자 지적: 허브 근처 회전 불가) |
| **EE endpoint pullback** | mount 실패, 점프 유지 |

### 3.2 부분 개선 (채택)

| 레버 | 결과 |
|---|---|
| **`ur10_motor_max_velocity_rad_s = 1.0`** | 최악 점프 **70→25 cm**, j>15: **11→2**, mount@112 (gate 0.12 m) |
| `planner_pos_offset_scale` 0.15→0.10 | 잔차 권한 축소 (도달성 무관) |

**정직한 한계**: 88% reach 특이점은 남음. maxVel은 **반창고**이지 완전 해결 아님.

---

## 4. yaw / 회전 타이밍

- Stage-1 종착: palm_up + **−90° world Z** → bore +X → −Y (허브 축).
- 타이어 yaw = EE yaw delta (`_upright_tire_quat_for_ee`).
- **현재 k=2.5 front-load**: zero-action에서도 step 20–70에 theta **70–82°** 정체, step 80 이후 수렴 → mount@102–104.
- **허브 근처에서만 회전** → cargo/트럭 충돌.
- **거치대에서 선회전** → 회전된 자세로 허브 평행이동 **도달 불가** (또 다른 특이점 조합).

**유일한 깨끗한 yaw 해법 후보**: spawn bore를 처음부터 −Y로 (90° carry 회전 제거) — grasp/거치대 재설계 필요.

---

## 5. reach / 재배치 / 대형 로봇 (기하)

EE 분리 (pickup-EE ↔ mount-EE) ≈ **2.07 m** → 최적 베이스에서도 max reach need ≈ **1.13 m**.

| 방안 | UR10 1.30 m | reach 1.65 m | R-2000iC ~2.65 m |
|---|---|---|---|
| 현재 기하 | **87%** 특이점 | 68% OK | **43%** OK |
| 약한 재배치 (hub Y 0.65) | **74%** OK | 66% | 53% |
| 강한 재배치 | 60% | 47% | 43% |

---

## 6. FANUC R-2000iD / 100 kg 검토 (2026-06-04)

### 6.1 URDF

| 모델 | ROS-Industrial | 비고 |
|---|---|---|
| **R-2000iD** | ❌ 미지원 | 사용자 요청 모델 |
| **R-2000iC/210F** | ✅ `fanuc_r2000ic_support` | reach/payload iD와 유사, **대체 후보** |
| R-2000iB/210F | ✅ `fanuc_r2000ib_support` | 동급 |

### 6.2 효과 (기하만)

- reach 2.6 m → 현재 씬에서 **~43% 신전** → **reach 포화 특이점 제거** (위치 점프 + yaw wobble 해결 가능).

### 6.3 100 kg 시뮬

- payload 210 kg → **토크 여유 충분** (실물).
- PyBullet: `tire_mass` 0.5 kg는 PD 공진 회피용 튜닝값 → **100 kg는 force cap·solver·contact 전면 재튜닝**.
- grasp는 kinematic lock → carry 중 무게 영향 작음; **mount 접촉 순간**이 관건.

### 6.4 통합 작업량 (PoC 이후)

1. URDF xacro→URDF + mesh 경로 (Windows, ROS 없을 수 있음)
2. `FanucRobot` 클래스 (joint 이름, EE link, HOME, FINAL_LOCK_QUATERNION)
3. 그리퍼 없음 → EE-only 또는 별도 tool URDF
4. 100 kg 물리 튜닝
5. 씬 재배치 (대형 base)
6. **전량 재학습**

### 6.5 PoC (1단계)

- `scripts/poc_fanuc_urdf.py` — iC/210F URDF 확보·변환·PyBullet 로드 검증.
- 산출: `data/urdf/fanuc_r2000ic/` (clone 후 처리).

---

## 7. 학습 정책

- **부드러움 미해결 상태에서 학습 중단** (사용자 결정).
- `phase1_smooth_v1` (74k step)은 **maxVel 미적용** 구 env → OOD.
- 신규 학습: `scripts/train_v8_smooth.ps1` (`ur10_motor_max_velocity_rad_s=1.0`, easy_prob=1.0).
- pre-2026-06-04 ckpt는 maxVel·env 변경으로 **재사용 비권장**.

---

## 8. 코드에 남긴 opt-in 레버 (기본 OFF unless noted)

| Config | 기본 | 용도 |
|---|---|---|
| `ur10_motor_max_velocity_rad_s` | **1.0** | 삽입 PD 채찹질 완화 |
| `use_dls_cartesian_servo` | False | DLS resolved-rate |
| `planner_dls_adaptive` | True (if DLS on) | manipulability-scheduled λ |
| `planner_stage1_approach_standoff` | 0 | −Y 삽입 via-point (재배치용) |
| `_multi_min_jerk_positions` | — | 다구간 경로 (standoff>0 시) |

---

## 9. 권고 로드맵

1. **단기 (UR10 유지)**: maxVel=1.0 + 학습 → 정책이 residual로 특이점 우회하는지 확인.
2. **중기**: 약한 재배치 (hub Y 0.65) + cargo 약한 재튜닝 → UR10 74% without 대형 로봇.
3. **장기**: FANUC R-2000iC PoC → 통과 시 단계적 통합 + 100 kg 물리.

---

## 10. 재현 명령

```powershell
conda activate tyro
# 현재 배치 GUI
python scripts/replay_planner.py --render --easy-start

# maxVel 검증은 일회성 스크립트 삭제됨 — config 기본값 1.0 사용

# FANUC PoC
python scripts/poc_fanuc_urdf.py --fetch
python scripts/poc_fanuc_urdf.py --load --gui
```
