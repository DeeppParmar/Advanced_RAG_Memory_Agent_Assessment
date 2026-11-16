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
echo.
echo [1] Start Server and Open Web UI  (Default - Just press ENTER)
echo [2] Run Pytest Automated Tests   (Type 2 and press ENTER)
echo.

if "%1"=="test" goto run_tests
if "%1"=="server" goto run_server

set choice=1
set /p choice="Enter option [1 or 2, default is 1]: "

if "%choice%"=="2" goto run_tests

:run_server
echo.
echo ========================================================
echo  FastAPI Server Starting...
echo  - Interactive UI: http://localhost:8000/
echo  - Swagger Docs:   http://localhost:8000/docs
echo ========================================================
echo.
echo Automatically opening UI in your default browser...
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://localhost:8000/"
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
goto end

:run_tests
echo.
echo Executing pytest test suite...
python -m pytest -v
goto end

:end
