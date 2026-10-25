@echo off
REM One-time setup. Creates the venv, installs everything, then runs the
REM preflight check. Safe to run again -- it reuses an existing venv.
setlocal
cd /d "%~dp0"

echo ============================================================
echo  Bengali PL-BERT - setup
echo ============================================================
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: "python" is not on PATH.
    echo Install Python 3.12 ^(64-bit^) from python.org and tick
    echo "Add python.exe to PATH" during installation.
    goto :end
)

python -c "import sys; raise SystemExit(0 if (3,9) <= sys.version_info[:2] <= (3,12) else 1)"
if errorlevel 1 (
    echo ERROR: this needs Python 3.9 - 3.12. You have:
    python --version
    echo Python 3.13 has no Windows wheels for tokenizers/numpy.
    goto :end
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creating virtual environment ...
    python -m venv .venv
    if errorlevel 1 goto :end
) else (
    echo [1/3] Reusing existing .venv
)

echo [2/3] Installing packages ^(this downloads ~2.5 GB, be patient^) ...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Install failed. Check your internet connection and run setup.bat again.
    goto :end
)

echo.
echo [3/3] Running the preflight check ...
echo.
python check_setup.py
if errorlevel 1 (
    echo.
    echo Setup finished but some checks failed - read the messages above.
    goto :end
)

echo.
echo Ready. Start training with run_training.bat
:end
echo.
pause
