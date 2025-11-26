@echo off
REM CrowdVision Development Commands for Windows

if "%1"=="" goto help
if "%1"=="help" goto help
if "%1"=="setup" goto setup
if "%1"=="install" goto install
if "%1"=="test" goto test
if "%1"=="clean" goto clean
if "%1"=="run-api" goto run-api
if "%1"=="run-mlflow" goto run-mlflow
if "%1"=="init-mlflow" goto init-mlflow
goto help

:help
echo CrowdVision Development Commands
echo =================================
echo run.bat setup       - Set up development environment
echo run.bat install     - Install dependencies
echo run.bat test        - Run tests
echo run.bat clean       - Clean temporary files
echo run.bat run-api     - Start FastAPI server
echo run.bat run-mlflow  - Start MLflow server
echo run.bat init-mlflow - Initialize MLflow experiments
goto end

:setup
python scripts\setup_env.py
goto end

:install
pip install -r requirements.txt
goto end

:test
pytest tests\ -v
goto end

:clean
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
for /d /r . %%d in (*.egg-info) do @if exist "%%d" rd /s /q "%%d"
for /d /r . %%d in (.pytest_cache) do @if exist "%%d" rd /s /q "%%d"
for /d /r . %%d in (.hypothesis) do @if exist "%%d" rd /s /q "%%d"
del /s /q *.pyc 2>nul
del /s /q *.pyo 2>nul
goto end

:run-api
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
goto end

:run-mlflow
mlflow server --host 0.0.0.0 --port 5000
goto end

:init-mlflow
python scripts\init_mlflow.py
goto end

:end
