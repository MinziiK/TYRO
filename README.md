# Tyro team

## Installation
### 1) Clone the repository
```
git clone https://github.com/MinziiK/TYRO.git
```
### 2) Install libraries
```
# create conda env (Python 3.10)
conda create -n tyro python=3.10 -y
conda activate tyro

# install dependencies
pip install -r requirements.txt
```

## Check GUI
```
# 기본 GUI: 200 step 실시간 재생 (10초), 끝나면 창 유지
python -m src.test --render

# 정적 씬만 둘러보기 (랜덤 액션 없음, 카메라 돌려가며 확인)
python -m src.test --render --action-scale 0

# 자동 종료
python -m src.test --render --no-hold

# 더 긴 롤아웃
python -m src.test --render --steps 500
```

## Train
```
# 1) Stage 1, phase 1 — 두 로봇이 각자 task 수행하는지 (~30~60분 / 1M step)
python -m src.train --stage 1 --phase 1 --total-steps 1_000_000 --num-envs 8

# 2) Stage 3, phase 1 — coop + sparse + penalty 합쳐서 본격 학습 (~3시간 / 3M step)
python -m src.train --stage 3 --phase 1 --total-steps 3_000_000 --num-envs 8 \
    --resume runs/stage1_phase1_*/final.zip

# 3) Phase 진행 — DR 강화
python -m src.train --stage 3 --phase 3 --total-steps 2_000_000 --num-envs 8 \
    --resume runs/stage3_phase1_*/final.zip

# 평가 (GUI 로 확인)
python -m src.eval runs/stage3_phase3_*/best/best_model.zip --render --episodes 5

# 학습 곡선
tensorboard --logdir runs/
```