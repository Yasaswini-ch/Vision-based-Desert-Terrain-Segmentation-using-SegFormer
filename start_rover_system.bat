@echo off
title Desert Rover System Launcher
echo ========================================================
echo       🏜️  DESERT ROVER INTELLIGENCE SYSTEM  🏜️
echo ========================================================
echo.

:: Check for virtual environment
if exist venv (
    echo [SYSTEM] Activating virtual environment...
    call venv\Scripts\activate
) else (
    echo [WARNING] No 'venv' folder found. Running with global Python.
)

:: 1. Start Inference Server (Brain)
echo [1/3] Starting Inference Server (Brain)...
start "Desert Brain (API)" cmd /c "python inference_server.py"
timeout /t 8 /nobreak > nul

:: 2. Start Streamlit Frontend (Monitor)
echo [2/3] Starting Monitor Dashboard (Streamlit)...
start "Desert Monitor (UI)" cmd /c "streamlit run app.py"
timeout /t 5 /nobreak > nul

:: 3. Inform User
echo.
echo ========================================================
echo   ✅ SYSTEM INITIALIZED
echo ========================================================
echo.
echo   1. Inference Server: http://localhost:8000
echo   2. Monitor UI:        Automatic (Check browser)
echo.
echo   To test the system, choose an option:
echo   [S] Run Rover Simulator (Default Demo)
echo   [F] Run FalconSim Bridge
echo   [X] Exit
echo.

set /p choice="Enter choice (S/F/X): "

if /i "%choice%"=="S" (
    echo [SYSTEM] Launching Rover Simulator...
    start "Desert Rover (Sim)" cmd /c "python rover_simulator.py"
) else if /i "%choice%"=="F" (
    echo [SYSTEM] Launching FalconSim Bridge...
    start "FalconSim Bridge" cmd /c "python falconsim_bridge.py"
) else (
    echo [SYSTEM] Closing launcher.
)

exit
