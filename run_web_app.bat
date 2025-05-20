@echo off
echo Plant Identification Web App
echo ==========================
echo.

echo Creating necessary directories...
if not exist "static" mkdir static
if not exist "uploads" mkdir uploads

echo.
echo Starting the server...
echo.
echo When the server starts, open your web browser and go to:
echo http://localhost:8000
echo.
echo Press Ctrl+C to stop the server when you're done.
echo.

cd web_app
python app.py

pause