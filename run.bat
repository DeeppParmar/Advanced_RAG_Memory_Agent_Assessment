@echo off
cd /d "%~dp0"

:: Ensure .env exists
if not exist .env (
    echo [.env] not found, copying from .env.example...
    copy .env.example .env >nul
)

echo ========================================================
echo  Advanced RAG Agent with Context Compression and Memory
echo ========================================================

if "%1"=="test" goto run_tests
if "%1"=="server" goto run_server

echo [1] Start FastAPI Server
echo [2] Run Pytest Automated Verification Suite
set /p choice="Choose [1/2] (Default: 1): "

if "%choice%"=="2" goto run_tests

:run_server
echo.
echo Starting FastAPI server at http://localhost:8000 ...
echo Swagger UI documentation: http://localhost:8000/docs
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
goto end

:run_tests
echo.
echo Executing pytest test suite...
python -m pytest -v
goto end

:end
