@echo off
echo ============================================================
echo  AI-Based Freshwater Quality Assessment System
echo  Setup and Run Script
echo ============================================================
echo.

REM Copy dataset to data folder
echo [1/6] Setting up directories...
if not exist "data" mkdir data
if not exist "models" mkdir models
if not exist "outputs" mkdir outputs
if not exist "notebooks" mkdir notebooks
copy /Y ground_water_quality.csv data\ground_water_quality.csv >nul 2>&1
echo      Done.

REM Install dependencies
echo.
echo [2/6] Installing dependencies...
pip install -r requirements.txt
echo      Done.

REM Run preprocessing
echo.
echo [3/6] Running preprocessing pipeline...
python src/preprocessing.py
echo      Done.

REM Train models
echo.
echo [4/6] Training models (this may take a few minutes)...
python src/train_model.py
echo      Done.

REM Evaluate
echo.
echo [5/6] Evaluating model and generating visualizations...
python src/evaluate.py
echo      Done.

REM Test prediction
echo.
echo [6/6] Testing prediction module...
python src/predict.py
echo      Done.

echo.
echo ============================================================
echo  All steps completed! 
echo.
echo  To launch the web app, run:
echo    python app/app.py
echo ============================================================
pause
