@echo off

:: Call the first batch file
echo Running create_env.bat...
call create_env.bat

:: Call the second batch file after the first completes
echo Running install_packages.bat...
call install_packages.bat

echo All tasks completed.
pause

