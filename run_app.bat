@echo off
echo 🏋️‍♂️ AI Exercise Form Analyzer
echo ==============================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

REM Check if main_app.py exists
if not exist "main_app.py" (
    echo ❌ main_app.py not found!
    echo Make sure you're running this from the repsAI directory
    pause
    exit /b 1
)

echo 🚀 Launching AI Exercise Form Analyzer...
echo.
echo 💡 Tip: The application will open in your default web browser
echo    If it doesn't open automatically, go to: http://localhost:8501
echo.

REM Launch the application
python launch.py

echo.
echo 👋 Application closed. Press any key to exit...
pause >nul
