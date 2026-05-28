@echo off
:: Phase 1 PPO training launcher (cmd.exe).
::
:: Usage (defaults: 12 envs, 2M steps, timestamped run name):
::   scripts\train.bat
::
:: Override any flag inline:
::   scripts\train.bat --total-steps 500000
::   scripts\train.bat --num-envs 8 --run-name quicktest

setlocal
set PYTHONIOENCODING=utf-8
set TYRO_PY=C:\Users\nhdkweon\miniconda3\envs\tyro\python.exe

if not exist "%TYRO_PY%" (
    echo [train.bat] tyro python not found at %TYRO_PY%
    exit /b 1
)

:: Build a timestamped default run name: phase1_yyyymmdd-HHMM
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd-HHmm"') do set TS=%%i
set DEFAULT_RUN=phase1_%TS%

"%TYRO_PY%" -m src.train ^
    --stage 3 --phase 1 ^
    --num-envs 12 ^
    --total-steps 2000000 ^
    --run-name %DEFAULT_RUN% ^
    %*

endlocal
