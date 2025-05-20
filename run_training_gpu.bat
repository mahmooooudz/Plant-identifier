@echo off
echo Plant Identification - GPU-Optimized Training
echo ==========================================
echo.

echo This script will train the plant identification model using your GPU.
echo Optimized for RTX 4060 with 8GB VRAM.
echo.

:: Set the correct data path with double "data" directory
set DATA_DIR="W:\Upwork\Plant identifier V2\data\data\train"
set MODEL_DIR="W:\Upwork\Plant identifier V2\models\layer1_is_plant"

echo Using data directory: %DATA_DIR%
echo Using model directory: %MODEL_DIR%
echo.

:: Check if data directory exists
if not exist %DATA_DIR% (
    echo ERROR: Could not find data directory at %DATA_DIR%
    echo Please verify the path and try again.
    goto END
)

:: Create model directory if it doesn't exist
if not exist %MODEL_DIR% (
    echo Creating model directory...
    mkdir %MODEL_DIR%
)

:CONFIRM
set /P CONFIRM=Do you want to continue? (Y/N): 
if /I "%CONFIRM%" EQU "Y" goto TRAIN
if /I "%CONFIRM%" EQU "N" goto CANCEL
echo Invalid input. Please enter Y or N.
goto CONFIRM

:TRAIN
echo.
echo Starting GPU-optimized training...

:: Set environment variables to control TensorFlow GPU memory usage
set TF_GPU_ALLOCATOR=cuda_malloc_async
set TF_FORCE_GPU_ALLOW_GROWTH=true

:: Run the training with optimized parameters for RTX 4060
python scripts\train_layer1.py --data_dir %DATA_DIR% --model_dir %MODEL_DIR% ^
--epochs 30 --fine_tune_epochs 15 --batch_size 24 --class_weight ^
--dropout_rate 0.3 --l2_reg 0.001 --unfreeze_layers 80

echo.
echo Training complete!
goto END

:CANCEL
echo.
echo Training canceled by user.

:END
echo.
pause