@echo off
echo ========================================
echo AI Financial Fraud Detection System v2.0
echo ========================================
echo.
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo Starting server...
echo.
echo Access the application at: http://127.0.0.1:5000
echo.
echo Demo Login Credentials:
echo - Username: admin    | Password: demo123 | Role: Admin
echo - Username: analyst  | Password: demo123 | Role: Analyst
echo - Username: reviewer | Password: demo123 | Role: Reviewer
echo.
echo Press Ctrl+C to stop the server
echo.
python app_new.py
pause
