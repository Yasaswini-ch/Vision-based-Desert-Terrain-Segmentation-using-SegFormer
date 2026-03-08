@echo off

:: Check if conda is available in the system
where conda >nul 2>nul
IF ERRORLEVEL 1 (
    echo "Conda is not found in your system. Please install Miniconda or Anaconda first."
    exit /b 1
)

:: Create the environment
echo Creating the Conda environment 'EDU' with Python 3.10...
conda create --name EDU python=3.10 -y



