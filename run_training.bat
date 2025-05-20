@echo off
echo Plant Identification - Model Training with Enhanced Data Augmentation
echo ==================================================================
echo.

echo This script will train the plant identification model with enhanced data augmentation.
echo This will help maximize the value of your limited training data.
echo.

:CONFIRM
set /P CONFIRM=Do you want to continue? (Y/N): 
if /I "%CONFIRM%" EQU "Y" goto TRAIN
if /I "%CONFIRM%" EQU "N" goto CANCEL
echo Invalid input. Please enter Y or N.
goto CONFIRM

:TRAIN
echo.
echo Starting model training with enhanced augmentation...

python scripts/train_layer1.py --data_dir data/train --model_dir models/layer1_is_plant ^
--epochs 30 --fine_tune_epochs 15 --batch_size 16 --class_weight ^
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