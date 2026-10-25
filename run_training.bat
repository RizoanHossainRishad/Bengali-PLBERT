@echo off
REM Start or resume training. Run setup.bat first.
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo The virtual environment is missing. Run setup.bat first.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
python train_plbert.py --config config.yml

echo.
echo Training stopped. Run this file again to resume from the last checkpoint.
pause
