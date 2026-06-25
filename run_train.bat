@echo off
REM Train the PPO agent inside the project venv. Double-click this file.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python train.py
echo.
echo ---- training ended (close this window or press a key) ----
pause
