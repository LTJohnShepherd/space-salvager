@echo off
REM Launch the trained AI agent inside the project venv. Double-click this file.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python ai_play.py
echo.
echo ---- script ended (close this window or press a key) ----
pause
