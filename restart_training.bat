@echo off
echo Plant Identification - RESTART TRAINING WITH FIXED CODE
echo ==============================================
echo.

echo This script will retrain the model with the fixed code.
echo It will produce the best possible performance.
echo.

set DATA_DIR="W:\Upwork\Plant identifier V2\data\data\train"
set MODEL_DIR="W:\Upwork\Plant identifier V2\models\layer1_is_plant"

echo Using data directory: %DATA_DIR%
echo Using model directory: %MODEL_DIR%
echo.
echo IMPORTANT: Make sure you've replaced train_layer1.py with the fixed version!
echo.

:CONFIRM
set /P CONFIRM=Do you want to continue? (Y/N): 
if /I "%CONFIRM%" EQU "Y" goto TRAIN
if /I "%CONFIRM%" EQU "N" goto CANCEL
echo Invalid input. Please enter Y or N.
goto CONFIRM

:TRAIN
echo.
echo Clearing old model files to ensure clean training...

if exist %MODEL_DIR%\plant_classifier (
    echo Removing old plant_classifier directory...
    rmdir /s /q %MODEL_DIR%\plant_classifier
)

if exist %MODEL_DIR%\plant_classifier.h5 (
    echo Removing old plant_classifier.h5...
    del %MODEL_DIR%\plant_classifier.h5
)

echo Starting training with fixed code...

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