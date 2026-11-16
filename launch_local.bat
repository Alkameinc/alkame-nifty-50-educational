@echo off
setlocal enabledelayedexpansion

title Alkame Nifty 50 Launcher
cls
echo =====================================================================
echo           ALKAME NIFTY 50 - ONE-CLICK LOCAL LAUNCHER
echo =====================================================================
echo.

cd /d "%~dp0"

:: 1. Check Python Environment
echo [*] Checking Python environment...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not found in PATH. Please install Python or activate your virtual environment.
    pause
    exit /b 1
)

:: 2. Check and Sync Market Data (Real-time update)
echo [*] Checking data freshness and updating real-time market data...
python automation\check_and_update_data.py
if errorlevel 1 (
    echo [WARNING] Data sync encountered an issue. Proceeding with existing cache...
)
echo.

:: 3. Launch FastAPI Backend in Background
echo [*] Starting REST API Server in background on http://localhost:8000 ...
start /b "" uvicorn api:app --host 127.0.0.1 --port 8000 > logs\backend_api.log 2>&1
timeout /t 3 /nobreak >nul

:: 4. Launch Background Scheduler (Optional / Market monitoring loop)
echo [*] Starting Signal Scheduler in background...
start /b "" python scheduler.py > logs\scheduler_runtime.log 2>&1
timeout /t 2 /nobreak >nul

:: 5. Launch and Display Streamlit Dashboard
echo.
echo =====================================================================
echo [*] All background services started successfully!
echo [*] Launching Dashboard on http://localhost:8501 ...
echo =====================================================================
echo.
streamlit run app.py

pause
