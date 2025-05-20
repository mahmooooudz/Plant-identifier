@echo off
echo Plant Identification - Dataset Downloader
echo ========================================
echo.

echo This script will download training datasets for plant identification.
echo The data will be saved to the data/train directory.
echo.

:CONFIRM
set /P CONFIRM=Do you want to continue? (Y/N): 
if /I "%CONFIRM%" EQU "Y" goto DOWNLOAD
if /I "%CONFIRM%" EQU "N" goto CANCEL
echo Invalid input. Please enter Y or N.
goto CONFIRM

:DOWNLOAD
echo.
echo Starting dataset download...

python data/download_datasets.py --output data/train --max 5000

echo.
echo Download complete!
goto END

:CANCEL
echo.
echo Download canceled by user.

:END
echo.
pause