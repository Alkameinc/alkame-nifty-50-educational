@echo off
title Alkame Platform Launcher
cls
echo =====================================================================
echo           ALKAME NIFTY 50 - ONE-CLICK LAUNCHER
echo =====================================================================
echo.

cd /d "%~dp0"

:: 1. Check Python
echo [*] Checking Python environment...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not found in PATH. Please activate your virtual environment.
    pause
    exit /b 1
)

:: 2. Check Node.js
echo [*] Checking Node.js environment...
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js is not found in PATH. Node is required for the React frontend.
    pause
    exit /b 1
)

:: 3. Run Data Sync
echo [*] Checking data freshness and updating real-time market data...
if exist automation\check_and_update_data.py (
    python automation\check_and_update_data.py
)
echo.

:: 4. Start Services in separate labeled windows
echo [*] Starting Backend API Server (Port 8000)...
start "Alkame Backend API" cmd /k "uvicorn api:app --host 127.0.0.1 --port 8000"

echo [*] Starting Signal Scheduler...
start "Alkame Scheduler" cmd /k "python scheduler.py"

echo [*] Starting React Frontend (Port 3000)...
start "Alkame React Frontend" cmd /k "cd frontend && npm install && npm run dev"

echo [*] Starting Streamlit Dashboard (Optional)...
if exist app.py (
    start "Alkame Streamlit" cmd /k "streamlit run app.py"
)

echo.
echo =====================================================================
echo [*] All services have been launched in separate windows!
echo [*] 
echo [*] - React Frontend:   http://localhost:3000
echo [*] - API Docs:         http://localhost:8000/docs
echo [*] - Streamlit Dash:   http://localhost:8501
echo [*] 
echo [*] To stop the project, simply close the opened command prompt windows.
echo =====================================================================
echo.
pause
