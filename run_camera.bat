@echo off
REM Launch camera mode (hand + emotion control) inside the project venv. Double-click this file.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python camera_play.py
echo.
echo ---- script ended (close this window or press a key) ----
pause
