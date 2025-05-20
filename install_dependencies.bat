@echo off
echo Plant Identification System - Dependencies Installer
echo ===================================================
echo.

echo This script will install all required Python packages for the plant identification system.
echo Required packages: tensorflow, pillow, fastapi, uvicorn, numpy
echo.

:CONFIRM
set /P CONFIRM=Do you want to continue with installation? (Y/N): 
if /I "%CONFIRM%" EQU "Y" goto INSTALL
if /I "%CONFIRM%" EQU "N" goto CANCEL
echo Invalid input. Please enter Y or N.
goto CONFIRM

:INSTALL
echo.
echo Checking for Python installation...
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python is not installed or not in the PATH.
    echo Please install Python 3.7 or later from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    goto END
)

echo Python is installed.
echo.
echo Checking pip installation...
pip --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: pip is not installed or not in the PATH.
    echo Please ensure pip is properly installed with Python.
    goto END
)

echo pip is installed.
echo.
echo Installing required packages...
echo.

echo Installing tensorflow (this may take a few minutes)...
pip install tensorflow

echo.
echo Installing pillow (PIL)...
pip install pillow

echo.
echo Installing FastAPI and Uvicorn...
pip install fastapi uvicorn

echo.
echo Installing additional dependencies...
pip install python-multipart matplotlib tqdm scikit-learn

echo.
echo All dependencies have been installed successfully!
echo.
echo You can now run the plant identification system.
echo Try the following:
echo 1. Start the web app: run_web_app.bat
echo 2. Download datasets: run_data_download.bat
echo 3. Train the model: run_training.bat
goto END

:CANCEL
echo.
echo Installation canceled by user.

:END
echo.
pause