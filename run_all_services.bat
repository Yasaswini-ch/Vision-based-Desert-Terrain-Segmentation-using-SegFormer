# Run all services for the Desert Segmentation project
# This batch file starts the inference server, rover simulator, and Streamlit UI in detached windows.
# It allows the processes to keep running after you close this command prompt.

@echo off
setlocal

REM ==== Configuration ==== 
set PROJECT_ROOT=%~dp0
set PYTHON_EXEC=python

REM ==== Start Inference Server (Flask) ==== 
start "Inference Server" cmd /c "%PYTHON_EXEC% %PROJECT_ROOT%inference_server.py"

REM ==== Start Rover Simulator ==== 
start "Rover Simulator" cmd /c "%PYTHON_EXEC% %PROJECT_ROOT%rover_simulator.py"

REM ==== Start Streamlit UI ==== 
REM Use 0.0.0.0 to be reachable from other machines on the LAN
start "Streamlit UI" cmd /c "streamlit run %PROJECT_ROOT%app.py --server.port 8501 --server.address 0.0.0.0"

echo All services have been launched in separate windows.
exit /b
