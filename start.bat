@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
echo.
echo ========================================
echo   AI Vision - virtual environment ready
echo ========================================
echo.
echo  Available scripts:
echo    python 01_view_camera.py   ^(color stream only^)
echo    python 02_view_depth.py    ^(color + depth + distance^)
echo.
cmd /k