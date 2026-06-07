@echo off
powershell -ExecutionPolicy Bypass -Command "Start-Process 'wsl' -ArgumentList 'bash /home/suyoung1031/start_web_recognition.sh' -WindowStyle Minimized; Start-Sleep 8; Start-Process 'http://localhost:5001'"
